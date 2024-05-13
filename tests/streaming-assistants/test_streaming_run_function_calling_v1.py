import time
import pytest
import logging

from openai.lib.streaming import AssistantEventHandler
from openai.types.beta.threads.runs import ToolCall
from typing_extensions import override

logger = logging.getLogger(__name__)

def test_function_calling_gpt_4o(patched_openai_client):
    model="gpt-4o"
    function_calling(model, patched_openai_client)

def test_function_calling_gpt_3_5(patched_openai_client):
    model="gpt-3.5-turbo"
    function_calling(model, patched_openai_client)

def test_function_calling_cohere(patched_openai_client):
    model="cohere_chat/command-r"
    function_calling(model, patched_openai_client)

def test_function_calling_pplx_mix(patched_openai_client):
    model="perplexity/mixtral-8x7b-instruct"
    function_calling(model, patched_openai_client)

@pytest.mark.skip(reason="pplx_online just looks up the weather and doesn't do the function call")
def test_function_calling_pplx_online(patched_openai_client):
    model="perplexity/pplx-70b-online"
    function_calling(model, patched_openai_client)

@pytest.mark.skip(reason="claude does not consistently work with function calling, skip")
def test_function_calling_claude(patched_openai_client):
    model="claude-3-haiku-20240307"
    function_calling(model, patched_openai_client)

@pytest.mark.skip(reason="litellm does not use the latest gemini tool support yet and gemini refuses without it, skip")
def test_function_calling_gemini(patched_openai_client):
    model="gemini/gemini-pro"
    function_calling(model, patched_openai_client)

def test_function_calling_groq_llama3(patched_openai_client):
    model="groq/llama3-8b-8192"
    function_calling(model, patched_openai_client)

@pytest.mark.skip(reason="llama does not consistently work with function calling, skip")
def test_function_calling_llama(patched_openai_client):
    model = "meta.llama2-13b-chat-v1"
    function_calling(model, patched_openai_client)


def function_calling(model, client):
    logger.info(f"making assistant for model {model}")
    assistant = client.beta.assistants.create(
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
    thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )

    class EventHandler(AssistantEventHandler):
        def __init__(self):
            super().__init__()
            self.tool_called = 0

        @override
        def on_exception(self, exception: Exception):
            logger.error(exception)
            raise exception

        @override
        def on_tool_call_done(self, toolCall: ToolCall):
            # Increment the counter each time the method is called
            self.tool_called += 1
            logger.info(toolCall)
            tool_outputs = []
            assert toolCall.function.name == "getCurrentWeather"
            logger.info(f'arguments: {toolCall.function.arguments}')
            tool_outputs.append({"tool_call_id": toolCall.id, "output": "75 degrees F and sunny"})

            with client.beta.threads.runs.submit_tool_outputs_stream(
                thread_id=self.current_run.thread_id,
                run_id=self.current_run.id,
                tool_outputs=tool_outputs,
                event_handler=EventHandler(),
            ) as stream:
                i = 0
                #for part in stream:
                #    logger.info(part)
                for text in stream.text_deltas:
                    i += 1
                    print(text, end="", flush=True)
                print()
                assert i > 0


    event_handler = EventHandler()

    i = 0
    with client.beta.threads.runs.create_and_stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        event_handler=event_handler
    ) as stream:
        #stream.until_done()
        for part in stream:
            logger.info(part)

    assert event_handler.tool_called > 0

    logger.info(thread)