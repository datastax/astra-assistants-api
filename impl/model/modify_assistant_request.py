from typing import Optional, Annotated, List

from pydantic import Field

from impl.model.assistant_object_tools_inner import AssistantObjectToolsInner
from openapi_server.models.modify_assistant_request import ModifyAssistantRequest as ModifyAssistantRequestGenerated


class ModifyAssistantRequest(ModifyAssistantRequestGenerated):
    tools: Optional[Annotated[List[AssistantObjectToolsInner], Field(max_length=128)]] = Field(default=None, description="assistant_tools_param_description")