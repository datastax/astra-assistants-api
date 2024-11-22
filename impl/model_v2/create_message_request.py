from typing import Optional, List

from pydantic import Field

from model_v2.message_object_attatchments_inner import MessageObjectAttachmentsInner
from openapi_server_v2.models.create_message_request import CreateMessageRequest as CreateMessageRequestGenerated


class CreateMessageRequest(CreateMessageRequestGenerated):
    attachments: Optional[List[MessageObjectAttachmentsInner]] = Field(default=None, description="A list of files attached to the message, and the tools they should be added to.")
