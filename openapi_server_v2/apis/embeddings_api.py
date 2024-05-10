# coding: utf-8

from typing import Dict, List  # noqa: F401
import importlib
import pkgutil

from openapi_server_v2.apis.embeddings_api_base import BaseEmbeddingsApi
import impl

from fastapi import (  # noqa: F401
    APIRouter,
    Body,
    Cookie,
    Depends,
    Form,
    Header,
    Path,
    Query,
    Response,
    Security,
    status,
)

from openapi_server_v2.models.extra_models import TokenModel  # noqa: F401
from openapi_server_v2.models.create_embedding_request import CreateEmbeddingRequest
from openapi_server_v2.models.create_embedding_response import CreateEmbeddingResponse
from openapi_server_v2.security_api import get_token_ApiKeyAuth

router = APIRouter()

ns_pkg = impl
for _, name, _ in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + "."):
    importlib.import_module(name)


@router.post(
    "/embeddings",
    responses={
        200: {"model": CreateEmbeddingResponse, "description": "OK"},
    },
    tags=["Embeddings"],
    summary="Creates an embedding vector representing the input text.",
    response_model_by_alias=True,
)
async def create_embedding(
    create_embedding_request: CreateEmbeddingRequest = Body(None, description="")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> CreateEmbeddingResponse:
    ...
