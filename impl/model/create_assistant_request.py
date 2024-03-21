from typing import Optional, Annotated, List

from pydantic import Field, StrictStr

from impl.model.assistant_object_tools_inner import AssistantObjectToolsInner
from openapi_server.models.create_assistant_request import CreateAssistantRequest as CreateAssistantRequestGenerated

MAX_FILE_IDS = 10000

class CreateAssistantRequest(CreateAssistantRequestGenerated):
    tools: Optional[Annotated[List[AssistantObjectToolsInner], Field(max_length=128)]] = Field(default=None, description="assistant_tools_param_description")
    file_ids: Optional[Annotated[List[StrictStr], Field(max_length=MAX_FILE_IDS)]] = Field(default=None, description="assistant_file_param_description")
