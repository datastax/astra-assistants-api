import os
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv("./.env")


OPENAI_API_KEY="fakekey"
ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
PERPLEXITY_API_KEY=os.getenv("PERPLEXITY_API_KEY")
base_url=os.getenv("base_url", "https://open-assistant-ai.astra.datastax.com/v1")

client = OpenAI(
    base_url=base_url,
    api_key=OPENAI_API_KEY,
    default_headers={
        "astra-api-token": ASTRA_DB_APPLICATION_TOKEN,
        "api-key": PERPLEXITY_API_KEY,
        "custom_llm_provider": "perplexity",
    }
)


response = client.chat.completions.create(
    model="perplexity/llama-3.1-70b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you give me some travel tips for Japan?"}
    ]
)

# Print the response
print(response)


thread = client.beta.threads.create()
my_thread = client.beta.threads.retrieve(thread.id)
updated = client.beta.threads.update(thread.id, metadata={"hi": "there"})

client.beta.threads.messages.create(thread_id=thread.id, content="some content", role="user")
deleted = client.beta.threads.delete(thread.id)
print(my_thread)

#mod = client.moderations.create(input="puppies")
#print(mod)

assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
    model="perplexity/llama-3.1-70b-instruct",
)

print(assistant)


def submit_message(assistant_id, thread, user_message):
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

def create_thread_and_run(user_input, assistant_id):
    thread = client.beta.threads.create()
    run = submit_message(assistant_id, thread, user_input)
    return thread, run


thread, run = create_thread_and_run(
    "What is the average speed of an unladen swallow?",
    assistant.id
)
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
    return run

run = wait_on_run(run, thread)
pretty_print(get_response(thread))
