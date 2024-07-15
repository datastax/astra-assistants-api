from typing import Dict, List

import pytest
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput

from astra_assistants.astra_assistants_event_handler import AstraEventHandler

import logging


from astra_assistants.astra_assistants_manager import AssistantManager
from astra_assistants.tools.structured_code import StructuredProgram, StructuredCodeGenerator, StructuredCodeEditor

logger = logging.getLogger(__name__)


def test_structured_code_raw(patched_openai_client):
    programs: List[Dict[str, StructuredProgram]] = []
    code_generator = StructuredCodeGenerator(programs)
    code_editor = StructuredCodeEditor(programs)

    # Create the assistant
    assistant = patched_openai_client.beta.assistants.create(
        name="Smart bot",
        instructions="""
use the structured code tool to generate code to help the user.
""",
        model="gpt-4o",
        tools=[code_generator.to_function(), code_editor.to_function()],
    )
    print(f"Assistant created: {assistant.id}")

    event_handler = AstraEventHandler(patched_openai_client)
    event_handler.register_tool(code_generator)

    thread = patched_openai_client.beta.threads.create()

    patched_openai_client.beta.threads.messages.create(thread.id, content="make a calculator web app that supports simple arithmetic, stick everything in one file.", role="user")

    # Generate the program
    with patched_openai_client.beta.threads.runs.create_and_stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=event_handler,
            tool_choice=code_generator.tool_choice_object(),
    ) as stream:
        for text in stream.text_deltas:
            print(text, end="", flush=True)
        print()
        print(f"tool_outputs: {event_handler.tool_outputs}")

    # Edit the program
    event_handler = AstraEventHandler(patched_openai_client)
    event_handler.register_tool(code_editor)
    program_id = programs[0]['program_id']
    program = programs[0]['program']
    patched_openai_client.beta.threads.messages.create(thread.id, content=f"nice, now add trigonometric functions to program_id {program_id}: \n{program.to_string()}" , role="user")
    with patched_openai_client.beta.threads.runs.create_and_stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=event_handler,
            tool_choice=code_editor.tool_choice_object(),
    ) as stream:
        for text in stream.text_deltas:
            print(text, end="", flush=True)
        print()
        programs = event_handler.tool_outputs
        print(f"tool_outputs: {event_handler.tool_outputs}")


@pytest.mark.asyncio
async def test_structured_code_with_manager(patched_openai_client):
    programs: List[Dict[str, StructuredProgram]] = []
    code_generator = StructuredCodeGenerator(programs)
    code_editor = StructuredCodeEditor(programs)
    tools = [code_generator, code_editor]

    assistant_manager = AssistantManager(
        instructions="use the structured code tool to generate code to help the user.",
        tools=tools,
        model="gpt-4o",
    )

    content = "make a calculator web app that supports simple arithmetic, stick everything in one file."
    result: ToolOutput = await assistant_manager.run_thread(
        content=content,
        tool=code_generator
    )
    content = f"nice, now add trigonometric functions to program_id {result['program_id']}: \n{result['output'].to_string()}"
    result = await assistant_manager.run_thread(
        content=content,
        tool=code_editor
    )

    return result
