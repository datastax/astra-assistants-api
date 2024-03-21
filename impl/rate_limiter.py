from fastapi import Request
from slowapi import Limiter


def get_dbid(request: Request) -> str:
    return request.headers.get("astra-api-token", "unknown")

limiter = Limiter(key_func=get_dbid)
