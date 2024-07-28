import pytest
import logging
import traceback

from openai.lib.streaming import AsyncAssistantEventHandler
from openai.types.beta.threads.runs import ToolCall
from typing_extensions import override

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_function_calling_gpt_4o(async_patched_openai_client):
    model="gpt-4o-mini"
    await function_calling(model, async_patched_openai_client)

async def function_calling(model, client):
    logger.info(f"making assistant for model {model}")
    assistant = await client.beta.assistants.create(
        name=f"Weather Bot {model}",
        instructions="You are a weather bot. Use the provided functions to answer questions.",
        model=model,
        tools=[{
            "type": "function",
            "function": {
                "name": "getCurrentWeather",
                "description": "Get the weather in location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
                        "unit": {"type": "string", "enum": ["c", "f"]}
                    },
                    "required": ["location"]
                }
            }
        }, {
            "type": "function",
            "function": {
                "name": "getNickname",
                "description": "Get the nickname of a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
                    },
                    "required": ["location"]
                }
            }
        }]
    )
    logger.info(assistant)

    logger.info("generating thread")
    user_message="What's the weather like in Miami today?"
    thread = await client.beta.threads.create()

    await client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )

    class EventHandler(AsyncAssistantEventHandler):
        def __init__(self):
            super().__init__()
            self.tool_called = 0

        @override
        async def on_exception(self, exception: Exception):
            logger.error(exception)
            trace = traceback.format_exc()
            logger.error(trace)
            raise exception

        @override
        async def on_tool_call_done(self, toolCall: ToolCall):
            # Increment the counter each time the method is called
            self.tool_called += 1
            logger.info(toolCall)
            tool_outputs = []
            assert toolCall.function.name == "getCurrentWeather"
            logger.info(f'arguments: {toolCall.function.arguments}')
            tool_outputs.append({"tool_call_id": toolCall.id, "output": "75 degrees F and sunny"})

            async with client.beta.threads.runs.submit_tool_outputs_stream(
                thread_id=self.current_run.thread_id,
                run_id=self.current_run.id,
                tool_outputs=tool_outputs,
                event_handler=EventHandler(),
            ) as stream:
                i = 0
                async for text in stream.text_deltas:
                    i += 1
                    logger.info(text)
                    print(text, end="", flush=True)
                assert i > 0

    event_handler = EventHandler()

    try:
        created_stream = client.beta.threads.runs.create_and_stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=event_handler,
        )
        async with created_stream as stream:
            await stream.until_done()
    except Exception as e:
        logger.error(e)
        tb_str = traceback.format_exc()
        logger.error(tb_str)
        print(e)

    assert event_handler.tool_called > 0

    logger.info(thread)