from typing import Optional
from openapi_server.models.submit_tool_outputs_run_request import SubmitToolOutputsRunRequest as SubmitToolOutputsRunRequestGenerated

class SubmitToolOutputsRunRequest(SubmitToolOutputsRunRequestGenerated):
    stream: Optional[bool] = None