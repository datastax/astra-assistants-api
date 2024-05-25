from typing import Optional, Annotated

from pydantic import Field, StrictStr

from openapi_server_v2.models.create_message_request import CreateMessageRequest


class ModifyMessageRequest(CreateMessageRequest):
    content: Optional[str] = Field(default=None, min_length=1, strict=True, max_length=32768, description="The content of the message.")
    role: Optional[StrictStr] = Field(default=None, description="The role of the entity that is creating the message. Currently only `user` is supported.")
