from typing import Optional

from impl.model_v2.assistants_api_tool_choice_option import AssistantsApiToolChoiceOption
from openapi_server_v2.models.create_run_request import CreateRunRequest as GeneratedCreateRunRequest


class CreateRunRequest(GeneratedCreateRunRequest):
    tool_choice: Optional[AssistantsApiToolChoiceOption] = None
