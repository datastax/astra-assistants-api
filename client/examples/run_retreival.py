import json
import time
from openai import OpenAI
from openai.lib.streaming import AssistantEventHandler
from typing_extensions import override

from astra_assistants import patch

class EventHandler(AssistantEventHandler):
    @override
    def on_run_step_done(self, run_step) -> None:
        print("retrieval")
        matches = []
        for tool_call in run_step.step_details.tool_calls:
            matches = tool_call.retrieval
            print(json.dumps(tool_call.retrieval))
    @override
    def on_text_created(self, text) -> None:
        # Increment the counter each time the method is called
        print(f"\nassistant > {text}", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        # Increment the counter each time the method is called
        print(delta.value, end="", flush=True)


def run_with_assistant(assistant, client):
    print(f"created assistant: {assistant.name}")
    print("Uploading file:")
    # Upload the file
    file = client.files.create(
        file=open(
            "./tests/language_models_are_unsupervised_multitask_learners.pdf",
            "rb",
        ),
        purpose="assistants",
        embedding_model="text-embedding-3-large",
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

    print(f"create and stream run")
    with client.beta.threads.runs.create_and_stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        event_handler=EventHandler(),
    ) as stream:
        for part in stream:
            pass
        #    print(part)



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

model = "cohere/command"
name = f"{model} Math Tutor"

cohere_assistant = client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
)
run_with_assistant(cohere_assistant, client)

model="perplexity/llama-3.1-70b-instruct"
name = f"{model} Math Tutor"

perplexity_assistant = client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
)
run_with_assistant(perplexity_assistant, client)

model = "claude-3-haiku-20240307"
name = f"{model} Math Tutor"

claude_assistant = client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
)
run_with_assistant(claude_assistant, client)

model = gemini/gemini-1.5-flash
name = f"{model} Math Tutor"

gemini_assistant = client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
)
run_with_assistant(gemini_assistant, client)