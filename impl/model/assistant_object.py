from typing import Optional, Annotated, List

from pydantic import Field, StrictStr

from impl.model.assistant_object_tools_inner import AssistantObjectToolsInner
from openapi_server.models.assistant_object import AssistantObject as AssistantObjectGenerated


class AssistantObject(AssistantObjectGenerated):
    tools: Annotated[List[AssistantObjectToolsInner], Field(max_length=20)] = Field(description="The list of tools that the [assistant](/docs/api-reference/assistants) used for this run.")
    file_ids: Annotated[List[StrictStr], Field(max_length=1000)] = Field(description="A list of [file](/docs/api-reference/files) IDs attached to this assistant. There can be a maximum of 20 files attached to the assistant. Files are ordered by their creation date in ascending order. ")
