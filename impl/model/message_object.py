from typing import List

from pydantic import Field

from openapi_server.models.message_content_text_object import MessageContentTextObject
from openapi_server.models.message_object import MessageObject as MessageObjectGenerated

class MessageObject(MessageObjectGenerated):
    content: List[MessageContentTextObject]
