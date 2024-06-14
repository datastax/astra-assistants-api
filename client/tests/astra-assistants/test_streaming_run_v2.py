import time
import logging

from openai.lib.streaming import AssistantEventHandler
from typing_extensions import override

logger = logging.getLogger(__name__)
def run_with_assistant(assistant, client):
    user_message = "What's your favorite animal."

    thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )

    class EventHandler(AssistantEventHandler):
        def __init__(self):
            super().__init__()
            self.on_message_created_count = 0
            self.on_text_created_count = 0
            self.on_text_delta_count = 0

        @override
        def on_message_created(self, message) -> None:
            # Increment the counter each time the method is called
            self.on_message_created_count += 1
            print(message.id)

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

    with client.beta.threads.runs.create_and_stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            #instructions="Speak in spanish",
            event_handler=event_handler,
    ) as stream:
        for part in stream:
            print(part)

    assert event_handler.on_text_created_count > 0
    assert event_handler.on_text_delta_count > 0




instructions="You're an animal expert who gives very long winded answers with flowery prose. Keep answers below 3 sentences."
def test_run_gpt3_5(patched_openai_client):
    gpt3_assistant = patched_openai_client.beta.assistants.create(
        name="GPT3 Animal Tutor",
        instructions=instructions,
        model="gpt-3.5-turbo",
    )

    assistant = patched_openai_client.beta.assistants.retrieve(gpt3_assistant.id)
    logger.info(assistant)

    run_with_assistant(gpt3_assistant, patched_openai_client)

def test_run_groq_llama3(patched_openai_client):
    groq_assistant = patched_openai_client.beta.assistants.create(
        name="Groq Llama3 Animal Tutor",
        instructions=instructions,
        model="groq/llama3-8b-8192",
    )
    run_with_assistant(groq_assistant, patched_openai_client)


def test_run_cohere(patched_openai_client):
    cohere_assistant = patched_openai_client.beta.assistants.create(
        name="Cohere Animal Tutor",
        instructions=instructions,
        model="cohere_chat/command-r"
    )
    run_with_assistant(cohere_assistant, patched_openai_client)

def test_run_perp(patched_openai_client):
    perplexity_assistant = patched_openai_client.beta.assistants.create(
        name="Perplexity/Mixtral Animal Tutor",
        instructions=instructions,
        model="perplexity/mixtral-8x7b-instruct",
    )
    run_with_assistant(perplexity_assistant, patched_openai_client)

def test_run_claude(patched_openai_client):
    claude_assistant = patched_openai_client.beta.assistants.create(
        name="Claude Animal Tutor",
        instructions=instructions,
        model="claude-3-haiku-20240307",
    )
    run_with_assistant(claude_assistant, patched_openai_client)

def test_run_gemini(patched_openai_client):
    gemini_assistant = patched_openai_client.beta.assistants.create(
        name="Gemini Animal Tutor",
        instructions=instructions,
        model="gemini/gemini-1.5-pro-latest",
    )
    run_with_assistant(gemini_assistant, patched_openai_client)