from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput

from astra_assistants.astra_assistants_event_handler import AstraEventHandler
from astra_assistants.astra_assistants_manager import AssistantManager
from astra_assistants.tools.astra_data_api import AstraDataAPITool

import pytest
import logging

logger = logging.getLogger(__name__)

db_url = "https://8a79c989-8b12-4154-8a93-063d597e78fd-us-east1.apps.astra.datastax.com"
collection_name = "movie_reviews"
namespace = "default"

@pytest.mark.asyncio
def test_astra_data_api(patched_openai_client):
   data_api_tool = AstraDataAPITool(
        db_url=db_url,
        collection_name=collection_name,
        namespace=namespace,
        vectorize=False,
        openai_client=patched_openai_client,
        embedding_model="text-embedding-ada-002",
    )

   # Create the assistant
   assistant = patched_openai_client.beta.assistants.create(
       name="Smart bot",
       instructions="You are a bot. Use the provided functions to answer questions about movies.",
       model="gpt-3.5-turbo",
       tools=[data_api_tool.to_function()],
   )

   event_handler = AstraEventHandler(patched_openai_client)
   event_handler.register_tool(data_api_tool)


   thread = patched_openai_client.beta.threads.create()

   patched_openai_client.beta.threads.messages.create(thread.id, content="What's a good, short kids movie?", role="user")

   # Run the assistant
   with patched_openai_client.beta.threads.runs.create_and_stream(
           thread_id=thread.id,
           assistant_id=assistant.id,
           event_handler=event_handler,
           tool_choice=data_api_tool.tool_choice_object(),
   ) as stream:
       for text in stream.text_deltas:
           print(text, end="", flush=True)
       print()
       print(f"tool_output: {event_handler.tool_output}")

@pytest.mark.asyncio
async def test_astra_data_api_with_manager(patched_openai_client):
    data_api_tool = AstraDataAPITool(
        db_url=db_url,
        collection_name=collection_name,
        namespace=namespace,
        vectorize=False,
        openai_client=patched_openai_client,
        embedding_model="text-embedding-ada-002",
    )

    tools = [data_api_tool]

    assistant_manager = AssistantManager(
        instructions="You are a bot. Use the provided functions to answer questions about movies.",
        tools=tools,
        model="gpt-3.5-turbo",
        name="Smart bot",
    )

    content="What's a good, short kids movie?"
    result: ToolOutput = await assistant_manager.run_thread(
        content=content,
        tool=data_api_tool
    )
    print(f"tool_output: {result}")
