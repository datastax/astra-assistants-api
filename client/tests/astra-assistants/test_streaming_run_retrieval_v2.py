import json

import pytest
import time

from openai.lib.streaming import AssistantEventHandler
from openai.types.beta.threads.message_create_params import Attachment
from typing_extensions import override

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

    # TODO support  vector store file creation
    #file = client.beta.vector_stores.files.create_and_poll(
    #    vector_store_id=vector_store.id,
    #    file_id=file2.id
    #)

    # TODO support batch
    # Ready the files for upload to OpenAI
    #file_paths = ["edgar/goog-10k.pdf", "edgar/brka-10k.txt"]
    #file_streams = [open(path, "rb") for path in file_paths]

    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    #file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    #    vector_store_id=vector_store.id, files=file_streams
    #)

    # You can print the status and the file counts of the batch to see the result of this operation.
    #print(file_batch.status)
    #print(file_batch.file_counts)


    print("adding vector_store id to assistant")
    # Update Assistant
    assistant = client.beta.assistants.update(
        assistant.id,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    print(f"updated assistant: {assistant}")
    print("creating persistent thread and message")
    thread = client.beta.threads.create()
    
    # Create a message with an attachment that has file_search enabled
    file2 = client.files.create(
        file=open(
            "./tests/fixtures/hudson.txt",
            "rb",
        ),
        purpose="assistants",
    )

    user_message = "What are some cool math concepts behind this ML paper pdf? Explain in two sentences."
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message,
        attachments=[
            Attachment(file_id=file2.id, tools=[{"type": "file_search"}]),
        ]
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
                print(tool_call.file_search)
            assert len(matches.chunks) > 0, "No matches found"

        @override
        def on_text_created(self, text) -> None:
            # Increment the counter each time the method is called
            self.on_text_created_count += 1
            print(f"\nassistant > {text}", end="", flush=True)

        @override
        def on_text_delta(self, delta, snapshot):
            # Increment the counter each time the method is called
            self.on_text_delta_count += 1
            print(delta.value, end="", flush=True)

    event_handler = EventHandler()

    print(f"creating run")
    with client.beta.threads.runs.create_and_stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=event_handler,
    ) as stream:
        for part in stream:
            print(part)

    assert event_handler.on_text_created_count > 0, "No text created"
    assert event_handler.on_text_delta_count > 0, "No text delta"

    user_message = "What is the name of my dog"
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_message,
        attachments=[
            Attachment(file_id=file2.id, tools=[{"type": "file_search"}]),
        ]
    )
    print(f"> {user_message}")

    event_handler = EventHandler()
    print(f"creating run")
    with client.beta.threads.runs.create_and_stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=event_handler,
    ) as stream:
        for part in stream.text_deltas:
            print(part)

    assert event_handler.on_text_created_count > 0, "No text created"
    assert event_handler.on_text_delta_count > 0, "No text delta"



instructions = "You are a personal math tutor. Answer thoroughly. The system will provide relevant context from files, use the context to respond."

def test_run_gpt_4o_mini(patched_openai_client):
    model = "gpt-4o-mini"
    name = f"{model} Math Tutor"

    gpt3_assistant = patched_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(gpt3_assistant, patched_openai_client)

def test_run_cohere(patched_openai_client):
    model = "cohere_chat/command-r"
    name = f"{model} Math Tutor"

    cohere_assistant = patched_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(cohere_assistant, patched_openai_client)

@pytest.mark.skip
def test_run_perp(patched_openai_client):
    model="perplexity/llama-3.1-70b-instruct"
    name = f"{model} Math Tutor"

    perplexity_assistant = patched_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(perplexity_assistant, patched_openai_client)

def test_run_claude(patched_openai_client):
    model = "claude-3-haiku-20240307"
    name = f"{model} Math Tutor"

    claude_assistant = patched_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(claude_assistant, patched_openai_client)

@pytest.mark.skip
def test_run_gemini(patched_openai_client):
    #model = "gemini/gemini-1.5-pro-latest"
    model = "gemini/gemini-1.5-flash"
    name = f"{model} Math Tutor"

    gemini_assistant = patched_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(gemini_assistant, patched_openai_client)