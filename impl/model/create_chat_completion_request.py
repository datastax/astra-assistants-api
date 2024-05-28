from typing import Optional, Annotated, List, Any

from pydantic import Field

from impl.model.chat_completion_request_message import ChatCompletionRequestMessage
from openapi_server_v2.models.create_chat_completion_request import CreateChatCompletionRequest as CreateChatCompletionRequestGenerated


class CreateChatCompletionRequest(CreateChatCompletionRequestGenerated):
    messages: Annotated[List[ChatCompletionRequestMessage], Field(min_length=1)] = Field(description="A list of messages comprising the conversation so far. [Example Python code](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models).")
    model: str
    tool_choice: Optional[Any] = None
#    user: Optional[str] = Field(alias="user", default=None)