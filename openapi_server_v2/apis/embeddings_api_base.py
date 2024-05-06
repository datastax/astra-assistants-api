# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from openapi_server.models.create_embedding_request import CreateEmbeddingRequest
from openapi_server.models.create_embedding_response import CreateEmbeddingResponse
from openapi_server.security_api import get_token_ApiKeyAuth

class BaseEmbeddingsApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseEmbeddingsApi.subclasses = BaseEmbeddingsApi.subclasses + (cls,)
    def create_embedding(
        self,
        create_embedding_request: CreateEmbeddingRequest,
    ) -> CreateEmbeddingResponse:
        ...
