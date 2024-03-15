from typing import Any

from pydantic import Field

from openapi_server.models.embedding import Embedding as EmbeddingGenerated


class Embedding(EmbeddingGenerated):
    embedding: Any = Field(description="The embedding vector, which is a list of floats. The length of vector depends on the model as listed in the [embedding guide](/docs/guides/embeddings). ")
