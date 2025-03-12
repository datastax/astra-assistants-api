from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput

from astra_assistants.astra_assistants_event_handler import AstraEventHandler
from astra_assistants.astra_assistants_manager import AssistantManager
from astra_assistants.tools.astra_data_api import AstraDataAPITool

import pytest
import logging

from astra_assistants.tools.e2b_code_interpreter import E2BCodeInterpreter

logger = logging.getLogger(__name__)

instructions = """
## your job & context
you are a python data scientist. you are given tasks to complete and you run python code to solve them.
- the python code runs in jupyter notebook.
- every time you call `execute_python` tool, the python code is executed in a separate cell. it's okay to multiple calls to `execute_python`.
- display visualizations using matplotlib or any other visualization library directly in the notebook. don't worry about saving the visualizations to a file.
- you have access to the internet and can make api requests.
- you also have access to the filesystem and can read/write files.
- you can install any pip package (if it exists) if you need to but the usual packages for data analysis are already preinstalled.
- you can run any python code you want, everything is running in a secure sandbox environment.
"""

@pytest.mark.asyncio
def test_code_interpreter(patched_openai_client):
    code_interpreter_tool = E2BCodeInterpreter()

    # Create the assistant
    assistant = patched_openai_client.beta.assistants.create(
        name="Smart bot",
        instructions=instructions,
        model="gpt-4o",
        tools=[code_interpreter_tool.to_function()],
    )
    print(f"Assistant created: {assistant.id}")

    event_handler = AstraEventHandler(patched_openai_client)
    event_handler.register_tool(code_interpreter_tool)


    thread = patched_openai_client.beta.threads.create()

    patched_openai_client.beta.threads.messages.create(thread.id, content="what's 2^6 + 16", role="user")

    # Run the assistant
    with patched_openai_client.beta.threads.runs.create_and_stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=event_handler,
            tool_choice=code_interpreter_tool.tool_choice_object(),
    ) as stream:
        for text in stream.text_deltas:
            print(text, end="", flush=True)
        print()
        print(f"tool_output: {event_handler.tool_output}")


@pytest.mark.asyncio
async def test_code_interpreter_with_helper(patched_openai_client):
    code_interpreter_tool = E2BCodeInterpreter()

    tools = [code_interpreter_tool]

    assistant_manager = AssistantManager(
        instructions=instructions,
        tools=tools,
        model="gpt-4o",
    )

    content="what's 2^6 + 16"
    result: ToolOutput = await assistant_manager.run_thread(
        content=content,
        tool=code_interpreter_tool
    )
    print(f"tool_output: {result}")
