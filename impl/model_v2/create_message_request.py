from typing import Optional, List

from pydantic import Field

from impl.model_v2.message_object_attachments_inner import MessageObjectAttachmentsInner
from openapi_server_v2.models.create_message_request import CreateMessageRequest as CreateMessageRequestGenerated


class CreateMessageRequest(CreateMessageRequestGenerated):
    attachments: Optional[List[MessageObjectAttachmentsInner]] = Field(default=None, description="A list of files attached to the message, and the tools they should be added to.")
