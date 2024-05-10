from typing import Optional, Annotated, List, Literal

from pydantic import Field

from impl.model.assistant_object_tools_inner import AssistantObjectToolsInner
from openapi_server_v2.models.run_completion_usage import RunCompletionUsage
from openapi_server_v2.models.run_object import RunObject as RunObjectGenerated


class RunObject(RunObjectGenerated):
    usage: Optional[RunCompletionUsage] = None
    tools: Annotated[List[AssistantObjectToolsInner], Field(max_length=20)] = Field(description="The list of tools that the [assistant](/docs/api-reference/assistants) used for this run.")
    status: Literal[
        "queued", "in_progress", "requires_action", "cancelling", "cancelled", "failed", "completed", "expired", "generating"
    ]
