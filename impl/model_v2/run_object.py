from typing import Optional, Annotated, List, Literal

from pydantic import Field

from openapi_server_v2.models.run_completion_usage import RunCompletionUsage
from openapi_server_v2.models.run_object import RunObject as RunObjectGenerated


class RunObject(RunObjectGenerated):
    usage: Optional[RunCompletionUsage] = None
    status: str
