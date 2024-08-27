import json
import logging

from openai import OpenAI
from dotenv import load_dotenv
from openai.lib.streaming import AssistantEventHandler
from typing_extensions import override

from astra_assistants import patch


load_dotenv("./.env")
load_dotenv("../../../.env")

def run_with_assistant(assistant, client):
    print(f"using assistant: {assistant}")
    print("Uploading file:")
    # Upload the file
    file = client.files.create(
        file=open(
            "./tests/fixtures/language_models_are_unsupervised_multitask_learners.pdf",
            "rb",
        ),
        purpose="assistants",
    )

    vector_store = client.beta.vector_stores.create(
        name="papers",
        file_ids=[file.id]
    )

    print("adding vector_store id to assistant")
    # Update Assistant
    assistant = client.beta.assistants.update(
        assistant.id,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    print(f"updated assistant: {assistant}")
    user_message = "What are some cool math concepts behind this ML paper pdf? Explain in two sentences."
    print("creating persistent thread and message")
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    print(f"> {user_message}")

    class EventHandler(AssistantEventHandler):
        def __init__(self):
            super().__init__()
            self.on_text_created_count = 0
            self.on_text_delta_count = 0

        @override
        def on_run_step_done(self, run_step) -> None:
            print("file_search")
            matches = []
            for tool_call in run_step.step_details.tool_calls:
                matches = tool_call.file_search
                print(json.dumps(tool_call.file_search))
            assert len(matches) > 0, "No matches found"

    event_handler = EventHandler()

    print(f"creating run")
    with client.beta.threads.runs.create_and_stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=event_handler,
    ) as stream:
        for part in stream.text_deltas:
            print(part, end="", flush=True)


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

model="groq/llama3-8b-8192"
name = f"{model} Math Tutor"

groq_assistant = client.beta.assistants.create(
    name=name,
    instructions=instructions,
    model=model,
)
run_with_assistant(groq_assistant, client)


model="cohere_chat/command-r"
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
