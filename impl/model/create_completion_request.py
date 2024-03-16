from typing import Optional, Annotated, List

from pydantic import Field

from openapi_server.models.create_completion_request import CreateCompletionRequest as CreateCompletionRequestGenerated


class CreateCompletionRequest(CreateCompletionRequestGenerated):
    model: str = Field(alias="model")
    prompt: str = Field(alias="prompt")