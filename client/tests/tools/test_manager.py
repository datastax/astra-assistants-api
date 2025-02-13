import pytest
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput

import logging
from astra_assistants.astra_assistants_manager import AssistantManager
from astra_assistants.mcp_openai_adapter import MCPRepresentationStdio

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
#async def test_mcp(openai_client):
#    client = openai_client
async def test_mcp(patched_openai_client):
    client = patched_openai_client

    mcps = [
        MCPRepresentationStdio(
            type="stdio",
            command="uvx",
            arguments=[
                "mcp-server-time"
            ]
        )
    ]

    assistant_manager = AssistantManager(
        instructions="you are a useful assistant",
        model="gpt-4o-mini",
        client=client,
        mcp_represenations=mcps
    )

    content = "what time is it?"
    result: ToolOutput = await assistant_manager.run_thread(
        content=content,
    )
    print(result)

    assistant_manager.shutdown()


@pytest.mark.asyncio
async def test_structured_code_with_manager(patched_openai_client):
    client = patched_openai_client

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

    assistant_manager = AssistantManager(
        instructions="you are a useful assistant",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
        model="gpt-4o",
    )

    content = "what's some cool math from the paper?"
    result: ToolOutput = await assistant_manager.run_thread(
        content=content,
    )
    print(result)