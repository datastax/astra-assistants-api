from typing import Optional, Any, List

from pydantic import Field

from openapi_server_v2.models.message_object_attachments_inner import MessageObjectAttachmentsInner as MessageObjectAttachmentsInnerGenerated


class MessageObjectAttachmentsInner(MessageObjectAttachmentsInnerGenerated):
    tools: Optional[List[Any]] = Field(default=None, description="The tools to add this file to.")
