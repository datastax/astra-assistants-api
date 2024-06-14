import time
from openai import OpenAI
from dotenv import load_dotenv
from astra_assistants import patch

load_dotenv("./.env")

def run_with_assistant(assistant, client):
    print(f"created assistant: {assistant.name}")
    print("Uploading file:")
    # Upload the file
    file = client.files.create(
        file=open(
            "./examples/python/language_models_are_unsupervised_multitask_learners.pdf",
            "rb",
        ),
        purpose="assistants",
    )
    print("adding file id to assistant")
    # Update Assistant
    assistant = client.beta.assistants.update(
        assistant.id,
        tools=[{"type": "retrieval"}],
        file_ids=[file.id],
    )
    user_message = "What are some cool math concepts behind this ML paper pdf? Explain in two sentences."
    print("creating persistent thread and message")
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    print(f"> {user_message}")

    print(f"creating run")
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    # Waiting in a loop
    while True:
        if run.status == 'failed':
            raise ValueError("Run is in failed state")
        if run.status == 'completed':
            print(f"run status: {run.status}")
            break
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)

    print("-->", end="")
    response = client.beta.threads.messages.list(thread_id=thread.id)
    print(f"{response.data[0].content[0].text.value}", end="")
    print("\n")


client = patch(OpenAI())

instructions = "You are a personal math tutor. Answer thoroughly. The system will provide relevant context from files, use the context to respond."

model = "gpt-3.5-turbo"
name = f"{model} Math Tutor"

gpt3_assistant = client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
)
run_with_assistant(gpt3_assistant, client)

model="cohere_chat/command-r"
name = f"{model} Math Tutor"

cohere_assistant = client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
)
run_with_assistant(cohere_assistant, client)

model="groq/llama3-8b-8192"
name = f"{model} Math Tutor"

cohere_assistant = client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
)
run_with_assistant(cohere_assistant, client)

model = "perplexity/mixtral-8x7b-instruct"
name = f"{model} Math Tutor"

perplexity_assistant = client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
)
run_with_assistant(perplexity_assistant, client)

model = "anthropic.claude-v2"
name = f"{model} Math Tutor"

claude_assistant = client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
)
run_with_assistant(claude_assistant, client)

model = "gemini/gemini-1.5-pro-latest"
name = f"{model} Math Tutor"

gemini_assistant = client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
)
run_with_assistant(gemini_assistant, client)
