import pytest
import logging

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

    logger.info(f"creating run")
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
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
    logger.info(f"streaming messages")
    logger.info("-->")
    response = client.beta.threads.messages.list(thread_id=thread.id, stream=True)

    i = 0
    for part in response:
        i += 1
        logger.info(f"{part.data[0].content[0].delta.value}")
    assert i > 0
    logger.info("\n")



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

def test_run_perp(streaming_assistants_openai_client):
    model="perplexity/llama-3.1-70b-instruct"
    name = f"{model} Math Tutor"

    perplexity_assistant = streaming_assistants_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(perplexity_assistant, streaming_assistants_openai_client)

@pytest.mark.skip(reason="fix astra-assistants aws with patched_openai embedding issue")
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