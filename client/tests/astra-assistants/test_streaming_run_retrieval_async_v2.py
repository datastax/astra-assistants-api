import asyncio
import json
import traceback

import pytest
import logging
import time

from openai.lib.streaming import AsyncAssistantEventHandler
from typing_extensions import override

logger = logging.getLogger(__name__)

async def run_with_assistant(assistant, client):
    logger.info(f"using assistant: {assistant}")
    logger.info("Uploading file:")
    # Upload the file
    file = await client.files.create(
        file=open(
            "./tests/fixtures/language_models_are_unsupervised_multitask_learners.pdf",
            "rb",
        ),
        purpose="assistants",
    )

    vector_store = await client.beta.vector_stores.create(
        name="papers",
        file_ids=[file.id]
    )

    # TODO support  vector store file creation
    #file = await client.beta.vector_stores.files.create_and_poll(
    #    vector_store_id=vector_store.id,
    #    file_id=file2.id
    #)

    # TODO support batch
    # Ready the files for upload to OpenAI
    #file_paths = ["edgar/goog-10k.pdf", "edgar/brka-10k.txt"]
    #file_streams = [open(path, "rb") for path in file_paths]

    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    #file_batch = await client.beta.vector_stores.file_batches.upload_and_poll(
    #    vector_store_id=vector_store.id, files=file_streams
    #)

    # You can print the status and the file counts of the batch to see the result of this operation.
    #print(file_batch.status)
    #print(file_batch.file_counts)


    logger.info("adding vector_store id to assistant")
    # Update Assistant
    assistant = await client.beta.assistants.update(
        assistant.id,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    logger.info(f"updated assistant: {assistant}")
    user_message = "What are some cool math concepts behind this ML paper pdf? Explain in two sentences."
    logger.info("creating persistent thread and message")
    thread = await client.beta.threads.create()
    await client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    logger.info(f"> {user_message}")

    class EventHandler(AsyncAssistantEventHandler):
        def __init__(self):
            super().__init__()
            self.on_text_created_count = 0
            self.on_text_delta_count = 0

        @override
        async def on_exception(self, exception: Exception):
            logger.error(exception)
            trace = traceback.format_exc()
            logger.error(trace)
            raise exception

        @override
        async def on_run_step_done(self, run_step) -> None:
            print("file_search")
            matches = []
            for tool_call in run_step.step_details.tool_calls:
                matches = tool_call.file_search
                print(json.dumps(tool_call.file_search))
            assert len(matches) > 0, "No matches found"

        @override
        async def on_text_created(self, text) -> None:
            # Increment the counter each time the method is called
            self.on_text_created_count += 1
            print(f"\nassistant > {text}", end="", flush=True)

        @override
        async def on_text_delta(self, delta, snapshot):
            # Increment the counter each time the method is called
            self.on_text_delta_count += 1
            print(delta.value, end="", flush=True)

    event_handler = EventHandler()

    logger.info(f"creating run")
    try:
        async with client.beta.threads.runs.create_and_stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=event_handler,
        ) as stream:
            async for part in stream:
                print(part)

        assert event_handler.on_text_created_count > 0, "No text created"
        assert event_handler.on_text_delta_count > 0, "No text delta"
        run_id = event_handler.current_run.id
        run = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run_id)
        print(run)
        user_message = "Thanks, what's 1 + 1. Be terse."
        logger.info("creating persistent thread and message")
        thread = await client.beta.threads.create()
        await client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=user_message
        )

        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )
        run_id = run.id

        while True:
            run = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run_id)
            if run.status == "completed":
                break
            time.sleep(0.5)

        assert run.status == "completed", "Run should be complete"

    except Exception as e:
        print(e)
        tcb = traceback.format_exc()
        print(tcb)
        raise e


instructions = "You are a personal math tutor. Answer thoroughly. The system will provide relevant context from files, use the context to respond."

@pytest.mark.asyncio
async def test_run_gpt_4o_mini(async_patched_openai_client):
    model="gpt-4o-mini"
    name = f"{model} Math Tutor"

    try:
        gpt_4o_mini_assistant = await async_patched_openai_client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model=model,
        )
        await run_with_assistant(gpt_4o_mini_assistant, async_patched_openai_client)
    except Exception as e:
        print(e)
        tcb = traceback.format_exc()
        print(tcb)
        raise e

@pytest.mark.asyncio
async def test_run_claude_haiku(async_patched_openai_client):
    model="claude-3-haiku-20240307"
    name = f"{model} Math Tutor"

    try:
        claude_assistant = await async_patched_openai_client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model=model,
        )
        await run_with_assistant(claude_assistant, async_patched_openai_client)

    except Exception as e:
        print(e)
        tcb = traceback.format_exc()
        print(tcb)
        raise e

@pytest.mark.asyncio
async def test_run_two_same_provider(async_patched_openai_client):
    model1="gpt-4o-mini"
    name1= f"{model1} Math Tutor"

    model2="gpt-4o-mini"
    name2= f"{model2} Math Tutor"

    assistant_task_1 = async_patched_openai_client.beta.assistants.create(
        name=name1,
        instructions=instructions,
        model=model1,
    )
    assistant_task_2 = async_patched_openai_client.beta.assistants.create(
        name=name2,
        instructions=instructions,
        model=model2,
    )

    # wait for both assistants
    claude_assistant, gpt_assistant = await asyncio.gather(
        assistant_task_1,
        assistant_task_2
    )

    run_1 = run_with_assistant(claude_assistant, async_patched_openai_client)
    run_2 = run_with_assistant(gpt_assistant, async_patched_openai_client)

    await asyncio.gather(run_1, run_2)

@pytest.mark.asyncio
async def test_run_two_differnet_providers(async_patched_openai_client):
    model1="gpt-4o-mini"
    name1= f"{model1} Math Tutor"

    model2="claude-3-haiku-20240307"
    name2= f"{model2} Math Tutor"

    assistant_task_1 = async_patched_openai_client.beta.assistants.create(
        name=name1,
        instructions=instructions,
        model=model1,
    )
    assistant_task_2 = async_patched_openai_client.beta.assistants.create(
        name=name2,
        instructions=instructions,
        model=model2,
    )

    # wait for both assistants
    claude_assistant, gpt_assistant = await asyncio.gather(
        assistant_task_1,
        assistant_task_2
    )

    run_1 = run_with_assistant(claude_assistant, async_patched_openai_client)
    run_2 = run_with_assistant(gpt_assistant, async_patched_openai_client)

    await asyncio.gather(run_1, run_2)


@pytest.mark.asyncio
async def test_run_two_differnet_providers_two_clients(async_patched_openai_client, async_patched_openai_client_2):
    model1="gpt-4o-mini"
    name1= f"{model1} Math Tutor"

    model2="claude-3-haiku-20240307"
    name2= f"{model2} Math Tutor"

    assistant_task_1 = async_patched_openai_client.beta.assistants.create(
        name=name1,
        instructions=instructions,
        model=model1,
    )
    assistant_task_2 = async_patched_openai_client_2.beta.assistants.create(
        name=name2,
        instructions=instructions,
        model=model2,
    )

    # wait for both assistants
    claude_assistant, gpt_assistant = await asyncio.gather(
        assistant_task_1,
        assistant_task_2
    )

    run_1 = run_with_assistant(claude_assistant, async_patched_openai_client)
    run_2 = run_with_assistant(gpt_assistant, async_patched_openai_client_2)

    await asyncio.gather(run_1, run_2)