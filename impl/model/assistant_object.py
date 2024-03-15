from typing import Optional, Annotated, List

from pydantic import Field

from impl.model.assistant_object_tools_inner import AssistantObjectToolsInner
from openapi_server.models.assistant_object import AssistantObject as AssistantObjectGenerated


class AssistantObject(AssistantObjectGenerated):
    tools: Annotated[List[AssistantObjectToolsInner], Field(max_length=20)] = Field(description="The list of tools that the [assistant](/docs/api-reference/assistants) used for this run.")
