from __future__ import annotations

from typing import List

from openapi_server_v2.models.message_content_text_object import MessageContentTextObject
from openapi_server_v2.models.message_object import MessageObject as MessageObjectGenerated

class MessageObject(MessageObjectGenerated):
    content: List[MessageContentTextObject]