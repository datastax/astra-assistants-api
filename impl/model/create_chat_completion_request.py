from typing import Optional, Annotated, List

from pydantic import Field

from impl.model.chat_completion_request_message import ChatCompletionRequestMessage
from openapi_server.models.chat_completion_functions import ChatCompletionFunctions
from openapi_server.models.create_chat_completion_request import CreateChatCompletionRequest as CreateChatCompletionRequestGenerated


class CreateChatCompletionRequest(CreateChatCompletionRequestGenerated):
    model: str
    functions: Optional[Annotated[List[ChatCompletionFunctions], Field(min_length=0, max_length=128)]] = Field(default=None, description="Deprecated in favor of `tools`.  A list of functions the model may generate JSON inputs for. ")
    messages: Annotated[List[ChatCompletionRequestMessage], Field(min_length=1)] = Field(description="A list of messages comprising the conversation so far. [Example Python code](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models).")
    user: Optional[str] = Field(alias="user", default=None)