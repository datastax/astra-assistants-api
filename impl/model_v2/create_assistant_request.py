from typing import Optional, Annotated, List, Any

from pydantic import Field

from impl.model.assistant_object_tools_inner import AssistantObjectToolsInner
from openapi_server_v2.models.create_assistant_request import CreateAssistantRequest as CreateAssistantRequestGenerated

class CreateAssistantRequest(CreateAssistantRequestGenerated):
    model: str
    tools: Optional[Annotated[List[AssistantObjectToolsInner], Field(max_length=128)]] = Field(default=None, description="assistant_tools_param_description")
    response_format: Optional[Any] = None
