import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("./.env")

# you do have to pass a key because the client requires it but it doesn't have to be valid since we're using a third party LLM
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
ASTRA_DB_TOKEN=os.getenv("ASTRA_DB_TOKEN")

GOOGLE_JSON_PATH=os.getenv("GOOGLE_JSON_PATH")
GOOGLE_PROJECT_ID=os.getenv("GOOGLE_PROJECT_ID")

base_url=os.getenv("base_url", "https://open-assistant-ai.astra.datastax.com/v1")

print(f"token {ASTRA_DB_TOKEN}")
print(f"google json path {GOOGLE_JSON_PATH}")
print(f"google project id {GOOGLE_PROJECT_ID}")

client = OpenAI(
    base_url=base_url,
    api_key=OPENAI_API_KEY,
    default_headers={
        "astra-api-token": ASTRA_DB_TOKEN,
        "VERTEXAI-PROJECT": GOOGLE_PROJECT_ID,
    }
)

print(client)
print("Uploading file:")

# Upload the JSON auth file
# this will get stored in your astradb as plain text, make sure your db adequately secured.
# you only need to do this once, then you can save your file.id and use it to connect your openai client
file = client.files.create(
    file=open(
        GOOGLE_JSON_PATH,
        "rb",
    ),
    purpose="auth",
)

print(file.id)

del client

client = OpenAI(
    base_url=base_url,
    api_key=OPENAI_API_KEY,
    default_headers={
        "astra-api-token": ASTRA_DB_TOKEN,
        "VERTEXAI-PROJECT": GOOGLE_PROJECT_ID,
        "google-application-credentials-file-id": file.id
    }
)

model="gemini-pro"


print("generating assistants")
assistant = client.beta.assistants.create(
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

def submit_message(assistant_id, thread, user_message):
    print("create message")
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    print("create run")
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

def create_thread_and_run(user_input, assistant_id):
    print("create thread")
    thread = client.beta.threads.create()
    run = submit_message(assistant_id, thread, user_input)
    return thread, run


print("generating thread")
thread, run = create_thread_and_run(
    "What's the weather like in Miami today?",
    assistant.id
)

print(thread)
print(run)

def get_response(thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="desc")

def pretty_print(messages):
    print("# Messages")
    for m in messages.data:
        print(f"{m.role}: {m.content[0].text.value}")
    print()


# Waiting in a loop
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

pretty_print(get_response(thread))
