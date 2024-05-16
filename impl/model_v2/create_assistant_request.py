from typing import Optional, Annotated, List

from pydantic import Field, StrictStr

from impl.model.assistant_object_tools_inner import AssistantObjectToolsInner
from openapi_server_v2.models.create_assistant_request import CreateAssistantRequest as CreateAssistantRequestGenerated

class CreateAssistantRequest(CreateAssistantRequestGenerated):
    model: str
