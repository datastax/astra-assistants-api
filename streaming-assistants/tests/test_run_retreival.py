import time

import pytest


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
    )
    vector_store = client.beta.vector_stores.create(
        name="papers",
        file_ids=[file.id]
    )

    print("adding file id to assistant")
    # Update Assistant
    assistant = client.beta.assistants.update(
        assistant.id,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
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
        if run.status == 'completed' or run.status == 'generating':
            print(f"run status: {run.status}")
            break
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)

    response = client.beta.threads.messages.list(thread_id=thread.id)

    print(f"{response.data[0].content[0].text.value}")
    print("\n")



instructions = "You are a personal math tutor. Answer thoroughly. The system will provide relevant context from files, use the context to respond."

def test_run_gpt3_5(openai_client):
    model = "gpt-3.5-turbo"
    name = f"{model} Math Tutor"

    gpt3_assistant = openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(gpt3_assistant, openai_client)

def test_run_groq(openai_client):
    model = "groq/llama3-8b-8192"
    name = f"{model} Math Tutor"

    gpt3_assistant = openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(gpt3_assistant, openai_client)

def test_run_cohere(openai_client):
    model = "command-r"
    name = f"{model} Math Tutor"

    cohere_assistant = openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(cohere_assistant, openai_client)

@pytest.mark.skip(reason="flaky")
def test_run_perp(openai_client):
    model = "perplexity/mixtral-8x7b-instruct"
    name = f"{model} Math Tutor"

    perplexity_assistant = openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(perplexity_assistant, openai_client)

@pytest.mark.skip(reason="fix streaming-assistants aws with openai embedding issue")
def test_run_claude(openai_client):
    model = "anthropic.claude-v2"
    name = f"{model} Math Tutor"

    claude_assistant = openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(claude_assistant, openai_client)

def test_run_gemini(openai_client):
    model = "gemini/gemini-pro"
    name = f"{model} Math Tutor"

    gemini_assistant = openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(gemini_assistant, openai_client)