import time
import pytest
import logging

logger = logging.getLogger(__name__)

def test_function_calling_gpt_4(patched_openai_client):
    model="gpt-4-1106-preview"
    function_calling(model, patched_openai_client)

def test_function_calling_gpt_4o_mini(patched_openai_client):
    model="gpt-4o-mini"
    function_calling(model, patched_openai_client)

def test_function_calling_cohere(patched_openai_client):
    model="cohere_chat/command-r"
    function_calling(model, patched_openai_client)

def test_function_calling_groq(patched_openai_client):
    model="groq/llama3-8b-8192"
    function_calling(model, patched_openai_client)

#TODO: bisect litellm versions to find when this started failing
@pytest.mark.skip(reason="for some reason this no longer works consistently with modern litellm, skip")
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

@pytest.mark.skip(reason="gemini does not consistently work with function calling, skip")
def test_function_calling_gemini(patched_openai_client):
    model="gemini/gemini-1.5-pro-latest"
    function_calling(model, patched_openai_client)

@pytest.mark.skip(reason="llama does not consistently work with function calling, skip")
def test_function_calling_llama(patched_openai_client):
    model = "meta.llama2-13b-chat-v1"
    function_calling(model, patched_openai_client)


def function_calling(model, client):
    logger.info(f"making assistant for model {model}")
    functions_list = [{
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
    },
        {
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
        }
    ]

    assistant = client.beta.assistants.create(
        name=f"Weather Bot {model}",
        instructions="You are a weather bot. Use the provided functions to answer questions.",
        model=model,
        tools=functions_list,
    )
    logger.info(assistant)

    logger.info("generating thread")
    user_message="What's the weather like in Miami today?"
    thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    logger.info(thread)
    logger.info(run)

    def wait_on_run(run, thread):
        while run.status == "queued" or run.status == "in_progress":
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)
        logger.info(f"run {run}")
        return run


    run = wait_on_run(run, thread)
    if run.required_action is not None:
        logger.info(run.required_action)
        tool_outputs = []
        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
            tool_outputs.append({"tool_call_id": tool_call.id, "output": "75 and sunny"})

        try:
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
            run = wait_on_run(run, thread)
        except Exception as e:
            logger.info(f"run {run}")
            logger.error(e)
            raise e

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    logger.info(f"{model}-->")
    logger.info(messages.data[0].content[0].text.value)
    assert messages.data[0].created_at >= messages.data[1].created_at, f"messages should be listed by created_at desc by default {messages}"
    assert len(messages.data) == 3, "should have 3 messages in the thread"