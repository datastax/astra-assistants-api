from typing import Literal

from openai.types.beta.threads import Run as ClientRun


class Run(ClientRun):
    status: Literal[
        "queued", "in_progress", "requires_action", "cancelling", "cancelled", "failed", "completed", "expired", "generating"
    ]
