from typing import Union, List, Optional

from pydantic import Field

from openapi_server.models.create_embedding_request import CreateEmbeddingRequest as CreateEmbeddingRequestGenerated


class CreateEmbeddingRequest(CreateEmbeddingRequestGenerated):
    input: Union[str, List[str], List[int], List[List[int]]] = Field(alias="input")
    model: str = Field(alias="model")
    user: Optional[str] = Field(alias="user", default=None)
