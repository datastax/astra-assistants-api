from typing import Annotated, List

from pydantic import StrictStr, Field

from openapi_server.models.message_stream_response_object import MessageStreamResponseObject as MessageStreamResponseObjectGenerated


class MessageStreamResponseObject(MessageStreamResponseObjectGenerated):
    file_ids: Annotated[List[StrictStr], Field(max_length=1000)] = Field(description="A list of [file](/docs/api-reference/files) IDs that the assistant should use. Useful for tools like retrieval and code_interpreter that can access files. A maximum of 10 files can be attached to a message.")