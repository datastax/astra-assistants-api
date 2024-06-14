from __future__ import annotations

from typing import Optional

from pydantic import StrictStr, Field

from openapi_server_v2.models.vector_store_object import VectorStoreObject as VectorStoreObjectGenerated

class VectorStoreObject(VectorStoreObjectGenerated):
    name: Optional[StrictStr] = Field(description="The name of the vector store.")
