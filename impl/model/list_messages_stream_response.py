try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from typing import List, Dict


from pydantic import Field

from impl.model.message_stream_response_object import MessageStreamResponseObject
from openapi_server.models.list_messages_stream_response import ListMessagesStreamResponse as ListMessagesStreamResponseGenerated


class ListMessagesStreamResponse(ListMessagesStreamResponseGenerated):
    data: List[MessageStreamResponseObject] = Field(description="The streamed chunks of messages, each representing a part of a message or a full message.")

    @classmethod
    def from_dict(cls, obj: Dict) -> Self:
        """Create an instance of ListMessagesStreamResponse from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "object": obj.get("object"),
            "data": [MessageStreamResponseObject.from_dict(_item) for _item in obj.get("data")] if obj.get("data") is not None else None,
            "first_id": obj.get("first_id"),
            "last_id": obj.get("last_id")
        })
        return _obj
