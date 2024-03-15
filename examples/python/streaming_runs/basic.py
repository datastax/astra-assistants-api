import time
from openai import OpenAI
from dotenv import load_dotenv
from openai.types.beta.assistant_stream_event import ThreadMessageDelta
from streaming_assistants import patch
from openai.lib.streaming import AssistantEventHandler
from typing_extensions import override


load_dotenv("./.env")
load_dotenv("../../../.env")

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

    class EventHandler(AssistantEventHandler):
            @override
            def on_text_delta(self, delta, snapshot):
                # Increment the counter each time the method is called
                print(delta.value, end="", flush=True)

    print(f"creating run")
    with client.beta.threads.runs.create_and_stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        event_handler=EventHandler(),
    ) as stream:
        for part in stream:
            if not isinstance(part, ThreadMessageDelta):
                print(f'received event: {part}\n')

    print("\n")


client = patch(OpenAI())

instructions = "You are a personal math tutor. Answer thoroughly. The system will provide relevant context from files, use the context to respond and share the exact snippets from the file at the end of your response."

model = "gpt-3.5-turbo"
name = f"{model} Math Tutor"

gpt3_assistant = client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
)
run_with_assistant(gpt3_assistant, client)

model = "cohere/command"
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

model = "gemini/gemini-pro"
name = f"{model} Math Tutor"

gemini_assistant = client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
)
run_with_assistant(gemini_assistant, client)
