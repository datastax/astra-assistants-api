import time
from openai import OpenAI
from dotenv import load_dotenv
from astra_assistants import patch
from openai.lib.streaming import AssistantEventHandler
from typing_extensions import override
from openai.types.beta.threads.runs import ToolCall
import logging

from astra_assistants.astra_assistants_event_handler import AstraEventHandler
from astra_assistants.tools.astra_data_api import AstraDataAPITool

logger = logging.getLogger(__name__)

load_dotenv("./.env")

client = patch(OpenAI())

# Ensure the right environment variables are configured for the model you are using
model="gpt-4-1106-preview"
#model="anthropic/claude-3-opus-20240229"
#model="anthropic/claude-3-sonnet-20240229"
#model="gpt-3.5-turbo"
#model="cohere_chat/command-r"
#model="perplexity/mixtral-8x7b-instruct"
#model="perplexity/pplx-70b-online"
#model="anthropic.claude-v2"
#model="gemini/gemini-1.5-pro-latest"
#model = "meta.llama2-13b-chat-v1"


print(f"making assistant for model {model}")

# get url from the astradb UI
db_url = "https://<db_id>-<region>.apps.astra.datastax.com"
collection_name = "movie_reviews"
namespace = "default"
data_api_tool = AstraDataAPITool(
    db_url=db_url,
    collection_name=collection_name,
    namespace=namespace,
    vectorize=False,
    openai_client=client,
    embedding_model="text-embedding-ada-002",
)

# Create the assistant
assistant = client.beta.assistants.create(
    name="Smart bot",
    instructions="You are a bot. Use the provided functions to answer questions about movies.",
    model="gpt-3.5-turbo",
    tools=[data_api_tool.to_function()],
)

event_handler = AstraEventHandler(client)
event_handler.register_tool(data_api_tool)


thread = client.beta.threads.create()

client.beta.threads.messages.create(thread.id, content="What's a good, short kids movie?", role="user")

# Run the assistant
with client.beta.threads.runs.create_and_stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        event_handler=event_handler,
        tool_choice=data_api_tool.tool_choice_object(),
) as stream:
    for text in stream.text_deltas:
        print(text, end="", flush=True)
    print()
    print(f"tool_outputs: {event_handler.tool_outputs}")