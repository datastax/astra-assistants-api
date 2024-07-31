import logging
import os
import re
from json import JSONDecodeError
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Type

from async_lru import alru_cache
from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from fastapi.security.utils import get_authorization_scheme_param
import httpx
from litellm import get_llm_provider, BadRequestError
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse
from typing_extensions import Annotated

from impl.astra_vector import AstraVectorDataStore, CassandraClient

logger = logging.getLogger(__name__)


def verify_server_admin(api_key: str = Depends(APIKeyHeader(name="api-key"))) -> bool:
    internal_key = os.getenv("ADMIN_API_KEY")
    if internal_key is not None and api_key != internal_key:
        raise HTTPException(
            status_code=400,
            detail="Not Authenticated",
        )
    return True


class OptionalHTTPBearer(HTTPBearer):
    async def __call__(self, request: Request):
        authorization: str = request.headers.get("Authorization")
        scheme, param = get_authorization_scheme_param(authorization)
        if not authorization or scheme.lower() != "bearer":
            return None
        return HTTPAuthorizationCredentials(scheme=scheme, credentials=param)


openai_token_bearer = OptionalHTTPBearer()


async def verify_openai_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(openai_token_bearer)
):
    if credentials is None:
        return None
    if credentials.scheme != "Bearer":
        raise HTTPException(status_code=401, detail="Missing token")
    token = credentials.credentials
    return token


LLM_HEADER_PREFIX = "LLM-PARAM-"
EMBEDDING_HEADER_PREFIX = "EMBEDDING-PARAM-"


async def get_body(request: Request) -> Any:
    content_type = request.headers.get('content-type', "")
    if 'multipart/form-data' in content_type:
        if not hasattr(request.state, 'body'):
            request.state.body = await request.form()
    if not hasattr(request.state, 'body'):
        try:
            request.state.body = await request.json()
        except (JSONDecodeError):
            request.state.body = {}
    return request.state.body

async def get_litellm_kwargs(
        request: Request,
        api_key: Annotated[Optional[str], Header()] = None,
        base_url: Annotated[Optional[str], Header()] = None,
        api_version: Annotated[Optional[str], Header()] = None,
        custom_llm_provider: Annotated[Optional[str], Header()] = None,
        openai_token: str = Depends(verify_openai_token),
        astra_api_token: Annotated[Optional[str], Header()] = None,
        astra_db_id: Annotated[Optional[str], Header()] = None,
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Dependency to get kwargs for litellm completion"""
    # NOTE: If api_key header is present, it will override the openai_token
    lite_llm_kwargs = dict(
        api_key=api_key or openai_token,
        base_url=base_url,
        api_version=api_version,
        custom_llm_provider=custom_llm_provider,
    )

    embedding_lite_llm_kwargs = dict(
        api_key=api_key or openai_token,
        base_url=base_url,
        api_version=api_version,
        custom_llm_provider=custom_llm_provider,
    )

    non_openai_embedding = False

    # security
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
    # Add any additional headers passed with prefix as kwargs
    for key, value in request.headers.items():
        if key == "vertexai-project":
            os.environ["VERTEXAI_PROJECT"] = value
        if key == "vertexai-location":
            os.environ["VERTEXAI_LOCATION"] = value
        if key.lower().startswith(LLM_HEADER_PREFIX.lower()):
            lite_llm_kwargs[key[len(LLM_HEADER_PREFIX):].replace("-", "_")] = value
            # Are there cases where we give the embedding the llm params?
            # embedding_lite_llm_kwargs[key[len(LITELLM_HEADER_PREFIX):].replace("-","_")] = value
        if key.lower().startswith(EMBEDDING_HEADER_PREFIX.lower()):
            non_openai_embedding = True
            embedding_lite_llm_kwargs[key[len(EMBEDDING_HEADER_PREFIX):].replace("-","_")] = value

    if non_openai_embedding:
        if embedding_lite_llm_kwargs["api_key"] == openai_token:
            # handle aws case where api_key should not be passed if aws tokens are used.
            embedding_lite_llm_kwargs.pop("api_key")

    # custom header for google
    if "google-application-credentials-file-id" in request.headers.keys():
        astradb: CassandraClient = await verify_db_client(request, astra_api_token, astra_db_id)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = astradb.load_auth_file(request.headers.get("google-application-credentials-file-id"))

    if "model" in embedding_lite_llm_kwargs:
        embedding_lite_llm_kwargs.pop("model")
    return lite_llm_kwargs, embedding_lite_llm_kwargs


async def check_if_using_openai(
        body: Any = Depends(get_body),
        litellm_kwargs: tuple[Dict[str, Any]] = Depends(get_litellm_kwargs),
        openai_token: str = Depends(verify_openai_token),
) -> bool:
    llm_is_openai = await is_using_openai(body, litellm_kwargs[0], openai_token)
    embedding_is_openai = await is_using_openai(body, litellm_kwargs[1], openai_token)
    return llm_is_openai or embedding_is_openai


async def is_using_openai(
        body: Any,
        litellm_kwargs: Dict[str, str],
        openai_token: str,
) -> bool:
    """Dependency to check if using OpenAI API based on headers"""
    # Kind of hack to get model from request, since can't declare multiple Body fields
    # without breaking things
    model = None
    if body != {}:
        if "model" in body:
            model = body["model"]

    if model is not None:
        model = model.replace("-edit-", "-")

        # Can use LiteLLM get_llm_provider that uses model name
        try:
            is_openai = "openai" in get_llm_provider(
                model=model,
                custom_llm_provider=litellm_kwargs.get("custom_llm_provider"),
                api_base=litellm_kwargs.get("base_url"),
            )[1]
            return is_openai
        except ValueError:
            # LiteLLM can't parse with the args provided
            return True
        except BadRequestError as e:
            # LiteLLM doesn't recognize the model
            if "tts" in model:
                return True
            raise HTTPException(
                status_code=400,
                detail="Model not recognized",
            )

    else:
        # Our custom logic
        if litellm_kwargs.get("custom_llm_provider") is not None and litellm_kwargs.get("custom_llm_provider") != "openai":
            return False

        if litellm_kwargs["api_key"] == openai_token:
            # Using OpenAI API key
            return True

        if len(litellm_kwargs.keys()) > 4:
            # Passed some custom LiteLLM headers, probably doing something custom
            return False

        if litellm_kwargs["base_url"] is not None:
            # Using custom base_url, probably not OpenAI
            return False

        return True


def infer_embedding_model(
        embedding_model: Annotated[Optional[str], Header()] = None,
        using_openai: bool = Depends(check_if_using_openai),
) -> str:
    """Dependency to infer embedding model based on headers"""
    if embedding_model is not None:
        return embedding_model
    elif using_openai:
        return "openai/text-embedding-3-large"
    else:
        # TODO: do we want other defaults?
        return None


def infer_embedding_api_key(
        embedding_model: str = Depends(infer_embedding_model),
        openai_token: str = Depends(verify_openai_token),
        embedding_param_api_key: Annotated[Optional[str], Header()] = None,
        api_key: Annotated[Optional[str], Header()] = None,
) -> str:
    if embedding_model is None:
        return None
    triple = get_llm_provider(embedding_model)
    provider = triple[1]
    if provider == "openai":
        return openai_token
    else:
        if embedding_param_api_key is not None:
            return embedding_param_api_key
        else:
            logger.warning("Probably should never reach here")
            return api_key


async def forward_request(request: Request) -> StreamingResponse:
    """Forwards the request for a stateless endpoint

    NOTE: Currently only supports OpenAI - in future could support more providers
    """
    logger.info(request.method + " " + request.url.path + " - Sending to OpenAI")

    httpx_client = request.app.state.client

    # Remove the version from the path, otherwise we get /v1/v1/...
    compatible_path = re.sub(r"/v\d+", "", request.url.path)
    url = httpx.URL(path=compatible_path, query=request.url.query.encode('utf-8'))
    raw_headers = request.headers
    cooked_headers = []
    for header in raw_headers:
        if header != "astra-api-token" and header != "astra-db-id":
            cooked_headers.append((header, raw_headers[header]))

    req = httpx_client.build_request(
        request.method, url, headers=cooked_headers, content=request.stream()
    )
    req.headers["Host"] = httpx_client.base_url.host

    resp = await httpx_client.send(req, stream=True)
    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers=resp.headers,
        background=BackgroundTask(resp.aclose)
    )


# Since we may have no control over client, using global DB connections cache
# keyed by DB ID and token instead of sessions
@alru_cache(maxsize=128)
async def datastore_cache(
    token: str,
    dbid: str,
) -> CassandraClient:
    logger.debug(f"Client not in cache for DB: {dbid}")
    datastore = AstraVectorDataStore()
    client = await datastore.setupSession(token, dbid)
    return client


async def verify_db_client(
    request: Request,
    astra_api_token: Annotated[Optional[str], Header()] = None,
    astra_db_id: Annotated[Optional[str], Header()] = None,
) -> CassandraClient:
    if astra_api_token is None:
        logger.error("Must pass an astradb token in the astra-api-token header for this request")
        raise HTTPException(
            status_code=403,
            detail="Must pass an astradb token in the astra-api-token header",
        )
    client = await datastore_cache(
        astra_api_token,
        astra_db_id
    )
    request.state.dbid = client.dbid  # Store the dbid in the request state

    # log requests and dbids
    request_scope_limited_headers = {}

    for key, value in request.scope.items():
        if key == 'headers':
            limited_headers = [
                (header_key, header_value[:15].decode('utf-8') + '...' if len(header_value) > 10 else header_value.decode('utf-8'))
                for header_key, header_value in value
            ]
            request_scope_limited_headers[key] = limited_headers
        else:
            request_scope_limited_headers[key] = value

    logger.info(f"dbid: {request.state.dbid}, request: {request_scope_limited_headers}")
    return client
