import time
from openai import OpenAI
from dotenv import load_dotenv
from streaming_assistants import patch
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
#model="cohere/command"
#model="perplexity/mixtral-8x7b-instruct"
#model="perplexity/pplx-70b-online"
#model="anthropic.claude-v2"
#model="gemini/gemini-pro"
#model = "meta.llama2-13b-chat-v1"


print("make assistant")
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
print(assistant)

print("generating thread")
user_message="What's the weather like in Miami today?"
thread = client.beta.threads.create()

client.beta.threads.messages.create(
    thread_id=thread.id, role="user", content=user_message
)

class EventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()

    @override
    def on_exception(self, exception: Exception):
        logger.error(exception)
        raise exception

    @override
    def on_tool_call_done(self, toolCall: ToolCall):
        logger.debug(toolCall)
        tool_outputs = []
        tool_outputs.append({"tool_call_id": toolCall.id, "output": "75 degrees F and sunny"})

        with client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(),
        ) as stream:
            #for part in stream:
            #    logger.info(part)
            for text in stream.text_deltas:
                print(text, end="", flush=True)
            print()

with client.beta.threads.runs.create_and_stream(
    thread_id=thread.id,
    assistant_id=assistant.id,
    event_handler=EventHandler()
) as stream:
    stream.until_done()
    #for part in stream:
    #    logger.info(f"event: {part}")

logger.info(thread)
