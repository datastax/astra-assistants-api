import time
import logging

import pytest

logger = logging.getLogger(__name__)
def run_with_assistant(assistant, client):
    user_message = "What's your favorite animal."

    thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        temperature=0,
    )

    # Waiting in a loop
    while True:
        if run.status == 'failed':
            raise ValueError("Run is in failed state")
        if run.status == 'completed' or run.status == 'generating':
            logger.info(f"run status: {run.status}")
            break
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)


    logger.info(f"thread.id {thread.id}")
    response = client.beta.threads.messages.list(
        thread_id=thread.id,
        stream=True
    )

    logger.info(f"{assistant.model} - streaming=>")
    i = 0
    for part in response:
        i += 1
        logger.info(part.data[0].content[0].delta.value)

    assert i > 0

    #Note, we can list now that the run is completed, we know the run is completed because we finished streaming
    logger.info(f"{assistant.model} no streaming=>")
    response = client.beta.threads.messages.list(thread_id=thread.id)
    logger.info(response.data[0].content[0].text.value)




instructions="You're an animal expert who gives very long winded answers with flowery prose. Keep answers below 3 sentences."
def test_run_gpt_4o_mini(streaming_assistants_openai_client):
    gpt3_assistant = streaming_assistants_openai_client.beta.assistants.create(
        name="GPT3 Animal Tutor",
        instructions=instructions,
        model="gpt-4o-mini",
    )

    assistant = streaming_assistants_openai_client.beta.assistants.retrieve(gpt3_assistant.id)
    logger.info(assistant)

    run_with_assistant(gpt3_assistant, streaming_assistants_openai_client)

def test_run_cohere(streaming_assistants_openai_client):
    cohere_assistant = streaming_assistants_openai_client.beta.assistants.create(
        name="Cohere Animal Tutor",
        instructions=instructions,
        model="cohere_chat/command-r"
    )
    run_with_assistant(cohere_assistant, streaming_assistants_openai_client)

def test_run_groq(streaming_assistants_openai_client):
    cohere_assistant = streaming_assistants_openai_client.beta.assistants.create(
        name="Groq Animal Tutor",
        instructions=instructions,
        model="groq/llama3-8b-8192"
    )
    run_with_assistant(cohere_assistant, streaming_assistants_openai_client)

def test_run_perp(streaming_assistants_openai_client):
    perplexity_assistant = streaming_assistants_openai_client.beta.assistants.create(
        name="Perplexity/Mixtral Animal Tutor",
        instructions=instructions,
        model="perplexity/llama-3.1-70b-instruct",
    )
    run_with_assistant(perplexity_assistant, streaming_assistants_openai_client)

def test_run_claude(streaming_assistants_openai_client):
    claude_assistant = streaming_assistants_openai_client.beta.assistants.create(
        name="Claude Animal Tutor",
        instructions=instructions,
        model="claude-3-haiku-20240307",
    )
    run_with_assistant(claude_assistant, streaming_assistants_openai_client)

@pytest.mark.skip(reason="flaky")
def test_run_gemini(streaming_assistants_openai_client):
    gemini_assistant = streaming_assistants_openai_client.beta.assistants.create(
        name="Gemini Animal Tutor",
        instructions=instructions,
        model="gemini/gemini-1.5-flash",
    )
    run_with_assistant(gemini_assistant, streaming_assistants_openai_client)