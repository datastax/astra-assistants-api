import time
import pytest
import logging

logger = logging.getLogger(__name__)

def test_function_calling_gpt_4(openai_client):
    model="gpt-4-1106-preview"
    function_calling(model, openai_client)

def test_function_calling_gpt_3_5(openai_client):
    model="gpt-3.5-turbo"
    function_calling(model, openai_client)

@pytest.mark.skip(reason="claude does not consistently work with function calling, skip")
def test_function_calling_cohere(openai_client):
    model="cohere/command"
    function_calling(model, openai_client)

def test_function_calling_pplx_mix(openai_client):
    model="perplexity/mixtral-8x7b-instruct"
    function_calling(model, openai_client)

@pytest.mark.skip(reason="pplx_online just looks up the weather and doesn't do the function call")
def test_function_calling_pplx_online(openai_client):
    model="perplexity/pplx-70b-online"
    function_calling(model, openai_client)

@pytest.mark.skip(reason="claude does not consistently work with function calling, skip")
def test_function_calling_claude(openai_client):
    model="anthropic.claude-v2"
    function_calling(model, openai_client)

def test_function_calling_gemini(openai_client):
    model="gemini/gemini-pro"
    function_calling(model, openai_client)

@pytest.mark.skip(reason="llama does not consistently work with function calling, skip")
def test_function_calling_llama(openai_client):
    model = "meta.llama2-13b-chat-v1"
    function_calling(model, openai_client)


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
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    logger.info(thread)
    logger.info(run)

    def wait_on_run(run, thread):
        while run.status == "queued" or run.status == "in_progress" or run.status == "generating":
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

    response = client.beta.threads.messages.list(thread_id=thread.id, stream=True)
    for part in response:
        logger.info(f"streamed response: {part.data[0].content[0].delta.value}")