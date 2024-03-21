from typing import List, Annotated

from pydantic import Field, StrictStr

from impl.model.create_assistant_request import MAX_FILE_IDS
from openapi_server.models.message_content_text_object import MessageContentTextObject
from openapi_server.models.message_object import MessageObject as MessageObjectGenerated

class MessageObject(MessageObjectGenerated):
    content: List[MessageContentTextObject]
    file_ids: Annotated[List[StrictStr], Field(max_length=MAX_FILE_IDS)] = Field(description="A list of [file](/docs/api-reference/files) IDs that the assistant should use. Useful for tools like retrieval and code_interpreter that can access files. A maximum of 10 files can be attached to a message.")
