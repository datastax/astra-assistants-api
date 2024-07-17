import asyncio
import logging
from typing import Callable, Sequence, Union, Any

import httpx
import openai
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Summary, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator.metrics import Info
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from impl.background import background_task_set
from impl.rate_limiter import limiter
from impl.routes import stateless, assistants, files, health, threads
from impl.routes_v2 import assistants_v2, threads_v2, vector_stores

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s (%(module)s:%(filename)s:%(lineno)d)',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('cassandra')
logger.setLevel(logging.WARN)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Astra Assistants API",
    description="Drop in replacement for OpenAI Assistants API.",
    version="2.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("shutting down server")
    for task in background_task_set:
        task.cancel()
        try:
            await task  # Give the task a chance to finish
        except asyncio.CancelledError:
            pass  # Handle cancellation if needed


app.include_router(health.router, prefix="/v1")
app.include_router(stateless.router, prefix="/v1")
app.include_router(assistants.router, prefix="/v1")
app.include_router(files.router, prefix="/v1")
app.include_router(threads.router, prefix="/v1")

app.include_router(stateless.router, prefix="/v2")
app.include_router(assistants_v2.router, prefix="/v2")
app.include_router(files.router, prefix="/v2")
app.include_router(threads_v2.router, prefix="/v2")
app.include_router(vector_stores.router, prefix="/v2")


class APIVersionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        version_header = request.headers.get("OpenAI-Beta")
        if version_header is None or version_header == "assistants=v1":
            response = await call_next(request)
            return response
        if version_header == "assistants=v2":
            request.scope['path'] = request.scope['path'].replace("v1", "v2")
            if 'raw_path' in request.scope:
                request.scope['raw_path'] = request.scope['raw_path'].replace(b'v1', b'v2')
            try:
                response = await call_next(request)
                return response
            except Exception as e:
                print(e)
                raise e

        else:
            return Response(
                f"Unsupported version: {version_header})",
                status_code=400)


app.add_middleware(APIVersionMiddleware)


def count_dbid(
        latency_lowr_buckets: Sequence[Union[float, str]] = (0.1, 0.5, 1),
) -> Callable[[Info], None]:
    DB_TOTAL = Counter(
        name="dbid_requests_total",
        documentation="Total number requests by dbid.",
        labelnames=(
            "dbid",
            "handler",
            "method",
            "status",
        )
    )
    TOTAL = Counter(
        name="http_requests_total",
        documentation="Total number of requests by method, status and handler.",
        labelnames=(
            "handler",
            "method",
            "status",
        )
    )
    IN_SIZE = Summary(
        name="http_request_size_bytes",
        documentation=(
            "Content length of incoming requests by handler. "
            "Only value of header is respected. Otherwise ignored. "
            "No percentile calculated. "
        ),
        labelnames=("handler",),
    )

    OUT_SIZE = Summary(
        name="http_response_size_bytes",
        documentation=(
            "Content length of outgoing responses by handler. "
            "Only value of header is respected. Otherwise ignored. "
            "No percentile calculated. "
        ),
        labelnames=("handler",),
    )

    LATENCY = Histogram(
        name="http_request_duration_seconds",
        documentation=(
            "Latency with only few buckets by handler. "
            "Made to be only used if aggregation by handler is important. "
        ),
        buckets=latency_lowr_buckets,
        labelnames=(
            "handler",
            "method",
        ),
    )

    def instrumentation(info: Info) -> None:
        if info.request and hasattr(info.request.state, "dbid"):
            DB_TOTAL.labels(info.request.state.dbid, info.modified_handler, info.method, info.modified_status).inc()

        TOTAL.labels(info.modified_handler, info.method, info.modified_status).inc()
        IN_SIZE.labels(info.modified_handler).observe(
            int(info.request.headers.get("Content-Length", 0))
        )

        if info.response and hasattr(info.response, "headers"):
            OUT_SIZE.labels(info.modified_handler).observe(
                int(info.response.headers.get("Content-Length", 0))
            )
        else:
            OUT_SIZE.labels(info.modified_handler).observe(0)

        LATENCY.labels(info.modified_handler, info.method).observe(
            info.modified_duration
        )

    return instrumentation


instrumentator = Instrumentator()
instrumentator.add().add(count_dbid())
instrumentator.instrument(app).expose(
    app,
    endpoint="/metrics",
)


@app.on_event("startup")
async def startup_event():
    dummy_openai_client = openai.OpenAI(api_key="fake_key")
    client = httpx.AsyncClient(
        base_url=dummy_openai_client.base_url,
        # TODO: Evaluate this value and if we want to make it dynamic based on request
        timeout=300,
    )
    app.state.client = client


@app.on_event("shutdown")
async def shutdown_event():
    client = app.state.client
    await client.aclose()


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log the error
    logger.error(f"Unexpected error: {exc} for request url {request.url} request method {request.method} request path params {request.path_params}  request query params {request.query_params} request body {request.body()} base_url {request.base_url}")

    if isinstance(exc, HTTPException):
        raise exec
    # Return an error response, not sure if we want to return all errors but at least this surfaces things like bad embedding model. Though that should be a 4xx error?
    return JSONResponse(
        status_code=500, content={"message": str(exc)}
    )


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE"],
)
async def unimplemented(request: Request, full_path: str):
    logger.info(f"Unmatched route accessed: {request.method} {request.url.path} {full_path} for request {request}")
    return JSONResponse(
        status_code=501, content={"message": "This feature is not yet implemented"}
    )

# if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)
