from typing import Optional, Any

from openapi_server_v2.models.create_run_request import CreateRunRequest as GeneratedCreateRunRequest


class CreateRunRequest(GeneratedCreateRunRequest):
    tool_choice: Optional[Any] = None
