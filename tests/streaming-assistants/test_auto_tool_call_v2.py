
import time
import pytest
import logging

logger = logging.getLogger(__name__)

def function_calling(model, client):
    assistant = client.beta.assistants.create(
        name=f"Math bot with weather skills {model}",
        instructions="You are a math bot. Use the provided functions to answer questions.",
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
    user_message="what is 2 + 2?"
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
    assert run.required_action is None, "Run should not have required action"

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    logger.info(f"{model}-->")
    logger.info(messages.data[0].content[0].text.value)


def test_function_calling_gpt_4(patched_openai_client):
    model="gpt-4-1106-preview"
    function_calling(model, patched_openai_client)

