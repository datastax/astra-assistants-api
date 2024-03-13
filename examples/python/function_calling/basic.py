import time
from openai import OpenAI
from dotenv import load_dotenv
from streaming_assistants import patch

load_dotenv("./.env")

client = patch(OpenAI())

# Ensure the right environment variables are configured for the model you are using
#model="gpt-4-1106-preview"
#model="anthropic/claude-3-opus-20240229"
model="anthropic/claude-3-sonet-20240229"
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
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
print(thread)
print(run)

def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress" or run.status == "generating":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    print(f"run {run}")
    return run


run = wait_on_run(run, thread)
if run.required_action is not None:
    print(run.required_action)
    tool_outputs = []
    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
        tool_outputs.append({"tool_call_id": tool_call.id, "output": "75 and sunny"})

    run = client.beta.threads.runs.submit_tool_outputs(
      thread_id=thread.id,
      run_id=run.id,
      tool_outputs=tool_outputs
    )
    run = wait_on_run(run, thread)

messages = client.beta.threads.messages.list(thread_id=thread.id)
print(f"{model}-->")
print(messages.data[0].content[0].text.value)

response = client.beta.threads.messages.list(thread_id=thread.id, stream=True)
for part in response:
    print(f"streamed response: {part.data[0].content[0].delta.value}")
