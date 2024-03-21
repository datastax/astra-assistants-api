from typing import Annotated, List

from pydantic import StrictStr, Field

from impl.model.create_assistant_request import MAX_FILE_IDS
from openapi_server.models.message_stream_response_object import MessageStreamResponseObject as MessageStreamResponseObjectGenerated


class MessageStreamResponseObject(MessageStreamResponseObjectGenerated):
    file_ids: Annotated[List[StrictStr], Field(max_length=MAX_FILE_IDS)] = Field(description="A list of [file](/docs/api-reference/files) IDs that the assistant should use. Useful for tools like retrieval and code_interpreter that can access files. A maximum of 10 files can be attached to a message.")