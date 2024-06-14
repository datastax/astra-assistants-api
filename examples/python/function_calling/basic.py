import time
from openai import OpenAI
from dotenv import load_dotenv
from astra_assistants import patch
from openai.lib.streaming import AssistantEventHandler
from typing_extensions import override
from openai.types.beta.threads.runs import ToolCall
import logging

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
assert messages.data[0].created_at > messages.data[1].created_at, "messages should be listed by created_at desc by default"
assert len(messages.data) == 3, "should have 3 messages in the thread"
print(messages.data[0].content[0].text.value)
