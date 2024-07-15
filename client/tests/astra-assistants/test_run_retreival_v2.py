import time

import pytest
import logging

logger = logging.getLogger(__name__)

def run_with_assistant(assistant, client, file_path, embedding_model):
    logger.info(f"using assistant: {assistant}")
    logger.info("Uploading file:")
    # Upload the file
    file = client.files.create(
        file=open(
            file_path,
            "rb",
        ),
        purpose="assistants",
        embedding_model=embedding_model,
    )
    try:
        client.files.create(
            file=open(
                "./tests/fixtures/language_models_are_unsupervised_multitask_learners.pdf",
                "rb",
            ),
            purpose="assistants",
            embedding_model="text-embedding-3-small",
        )
    except Exception as e:
        pass


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

    logger.info("adding vector_store id to assistant")
    # Update Assistant
    assistant = client.beta.assistants.update(
        assistant.id,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
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
        if run.status == 'completed':
            logger.info(f"run status: {run.status}")
            break
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)

    logger.info(f"thread.id {thread.id}")
    logger.info(f"{assistant.model} =>")
    response = client.beta.threads.messages.list(thread_id=thread.id)
    logger.info(response.data[0].content[0].text.value)



instructions = "You are a personal math tutor. Answer thoroughly. The system will provide relevant context from files, use the context to respond."

file1_path = "./tests/fixtures/language_models_are_unsupervised_multitask_learners.pdf"
embedding_model1 ="text-embedding-3-large"

file2_path = "./tests/fixtures/language_models_are_unsupervised_multitask_learners_2.pdf"
embedding_model2 ="embed-english-light-v3.0"

def test_run_gpt3_5(patched_openai_client):
    model = "gpt-3.5-turbo"
    name = f"{model} Math Tutor"

    gpt3_assistant = patched_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
        response_format='auto',
    )
    run_with_assistant(gpt3_assistant, patched_openai_client, file2_path, embedding_model2)

def test_run_cohere(patched_openai_client):
    model = "cohere_chat/command-r"
    name = f"{model} Math Tutor"

    cohere_assistant = patched_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(cohere_assistant, patched_openai_client, file2_path, embedding_model2)

def test_run_perp(patched_openai_client):
    model = "perplexity/mixtral-8x7b-instruct"
    name = f"{model} Math Tutor"

    perplexity_assistant = patched_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(perplexity_assistant, patched_openai_client, file1_path, embedding_model1)

def test_run_claude(patched_openai_client):
    model = "claude-3-haiku-20240307"
    name = f"{model} Math Tutor"

    claude_assistant = patched_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(claude_assistant, patched_openai_client, file1_path, embedding_model1)

def test_run_gemini(patched_openai_client):
    model = "gemini/gemini-1.5-pro-latest"
    name = f"{model} Math Tutor"

    gemini_assistant = patched_openai_client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model,
    )
    run_with_assistant(gemini_assistant, patched_openai_client, file1_path, embedding_model1)