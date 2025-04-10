import json

import pytest
import logging
import time

from openai.lib.streaming import AssistantEventHandler
from typing_extensions import override

logger = logging.getLogger(__name__)

def run_with_assistant(assistant, client):
    logger.info(f"using assistant: {assistant}")
    logger.info("Uploading file:")
    # Upload the file
    file = client.files.create(
        file=open(
            "./tests/fixtures/language_models_are_unsupervised_multitask_learners.pdf",
            "rb",
        ),
        purpose="assistants",
    )
    logger.info("adding file id to assistant")
    # Update Assistant
    assistant = client.beta.assistants.update(
        assistant.id,
        tools=[{"type": "retrieval"}],
        file_ids=[file.id],
    )
    logger.info(f"updated assistant: {assistant}")
    user_message = "What are some cool math concepts behind this ML paper pdf? Explain in two sentences."
    logger.info("creating persistent thread and message")
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    logger.info(f"> {user_message}")

    class EventHandler(AssistantEventHandler):
        def __init__(self):
            super().__init__()
            self.on_text_created_count = 0
            self.on_text_delta_count = 0

        @override
        def on_run_step_done(self, run_step) -> None:
            print("retrieval")
            matches = []
            for tool_call in run_step.step_details.tool_calls:
                matches = tool_call.retrieval
                print(json.dumps(tool_call.retrieval))
            assert len(matches) > 0, "No matches found"

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

    logger.info(f"creating run")
    with client.beta.threads.runs.create_and_stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=event_handler,
    ) as stream:
        for part in stream:
            print(part)

    assert event_handler.on_text_created_count > 0, "No text created"
    assert event_handler.on_text_delta_count > 0, "No text delta"


instructions = "You are a personal math tutor. Answer thoroughly. The system will provide relevant context from files, use the context to respond."

def test_run_gpt_4o_mini(streaming_assistants_openai_client):
    model = "gpt-4o-mini"
    name = f"{model} Math Tutor"

    gpt3_assistant = streaming_assistants_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(gpt3_assistant, streaming_assistants_openai_client)

def test_run_cohere(streaming_assistants_openai_client):
    model = "cohere_chat/command-r"
    name = f"{model} Math Tutor"

    cohere_assistant = streaming_assistants_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(cohere_assistant, streaming_assistants_openai_client)

@pytest.mark.skip
def test_run_perp(streaming_assistants_openai_client):
    model="perplexity/llama-3.1-70b-instruct"
    name = f"{model} Math Tutor"

    perplexity_assistant = streaming_assistants_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(perplexity_assistant, streaming_assistants_openai_client)

@pytest.mark.skip(reason="fix astra-assistants aws with openai embedding issue")
def test_run_claude(streaming_assistants_openai_client):
    model = "claude-3-haiku-20240307"
    name = f"{model} Math Tutor"

    claude_assistant = streaming_assistants_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(claude_assistant, streaming_assistants_openai_client)

@pytest.mark.skip(reason="flaky")
def test_run_gemini(streaming_assistants_openai_client):
    model = "gemini/gemini-1.5-flash"
    name = f"{model} Math Tutor"

    gemini_assistant = streaming_assistants_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(gemini_assistant, streaming_assistants_openai_client)