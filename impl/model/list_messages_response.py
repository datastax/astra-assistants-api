from typing import List

from impl.model.message_object import MessageObject
from openapi_server.models.list_messages_response import ListMessagesResponse as ListMessagesResponseGenerated

class ListMessagesResponse(ListMessagesResponseGenerated):
    data: List[MessageObject]
