from typing import Dict, List
from uuid import uuid1

import pytest
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from pydantic import BaseModel

from astra_assistants.astra_assistants_event_handler import AstraEventHandler

import logging


from astra_assistants.astra_assistants_manager import AssistantManager
from astra_assistants.tools.structured_code import StructuredProgram, StructuredCodeGenerator, StructuredCodeEditor, \
    StructuredCodeRewrite, program_str_to_program, add_program_to_cache, StructuredCodeIndentLeft, add_chunks_to_cache

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
        print(f"tool_output: {event_handler.tool_output}")

    # Edit the program
    event_handler = AstraEventHandler(patched_openai_client)
    event_handler.register_tool(code_editor)
    program_id = programs[0]['program_id']
    program = programs[0]['output']
    patched_openai_client.beta.threads.messages.create(thread.id, content=f"nice, now add trigonometric functions to program_id {program_id}: \n{program.to_string()}" , role="user")
    code_editor.set_program_id(program_id)
    with patched_openai_client.beta.threads.runs.create_and_stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=event_handler,
            tool_choice=code_editor.tool_choice_object(),
    ) as stream:
        for text in stream.text_deltas:
            print(text, end="", flush=True)
        print()
        programs = event_handler.tool_output
        print(f"tool_output: {event_handler.tool_output}")


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
    code_editor.set_program_id(result['program_id'])
    result = await assistant_manager.run_thread(
        content=content,
        tool=code_editor
    )

    return result

def test_structured_rewrite_with_manager(patched_openai_client):
    programs: List[Dict[str, StructuredProgram]] = []
    program_content = """
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

number = int(input("Enter a number: "))
print("Factorial:", factorial(number))
"""

    program = program_str_to_program(
        program_content,
        language="python",
        filename="factorial.py",
        description="Calculate the factorial of a number",
        tags=["math", "factorial"]
    )
    program_id = add_program_to_cache(program, programs)

    code_rewriter = StructuredCodeRewrite(programs)
    code_rewriter.set_program_id(program_id)
    tools = [code_rewriter]

    assistant_manager = AssistantManager(
        instructions="use the structured code tool to generate code to help the user.",
        tools=tools,
        model="gpt-4o",
    )

    chunks: ToolOutput = assistant_manager.stream_thread(
        content="Rewrite to use memoization.",
        #tool_choice=code_rewriter
        tool_choice="auto"
    )

    text = ""
    chunk = next(chunks)
    if not isinstance(chunk, str):
        tool_call_results_dict = chunk["output"]
        print(f"tool_call_results_dict: {tool_call_results_dict}")
    else:
        print(chunk, end="", flush=True)
        text += chunk

    for chunk in chunks:
        print(chunk, end="", flush=True)
        text += chunk

    program = program_str_to_program(
        text,
        program.language,
        program.filename,
        program.tags,
        program.description
    )
    program_id = add_program_to_cache(program, programs)
    print(program_id)



def test_structured_rewrite_and_edit_with_manager(patched_openai_client):
    programs: List[Dict[str, StructuredProgram]] = []
    program_content = """
    def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

number = int(input("Enter a number: "))
print("Factorial:", factorial(number))
"""

    program = program_str_to_program(
        program_content,
        language="python",
        filename="factorial.py",
        description="Calculate the factorial of a number",
        tags=["math", "factorial"]
    )
    program_id = add_program_to_cache(program, programs)

    code_rewriter = StructuredCodeRewrite(programs)
    code_indent_left = StructuredCodeIndentLeft(programs)
    tools = [code_rewriter, code_indent_left]

    assistant_manager = AssistantManager(
        instructions="use the structured code tool to generate code to help the user.",
        tools=tools,
        model="openai/gpt-4o-2024-08-06",
    )

    #code_indent_left.set_program_id(program_id)
    try:
        chunks: ToolOutput = assistant_manager.stream_thread(
            content="Fix the indentation.",
            tool_choice=code_indent_left
        )
        for chunk in chunks:
            pass
        assert False, "stream_thread should fail"
    except Exception as e:
        print(e)

    assert len(programs) == 1
    code_indent_left.set_program_id(program_id)
    chunks: ToolOutput = assistant_manager.stream_thread(
        content="Fix the indentation.",
        tool_choice=code_indent_left
    )

    tool_call_result = next(chunks)
    assert len(programs) == 2
    code_rewriter.set_program_id(tool_call_result['program_id'])

    chunks: ToolOutput = assistant_manager.stream_thread(
        content="Rewrite to use memoization.",
        tool_choice=code_rewriter
    )

    program_id = add_chunks_to_cache(chunks, programs)['program_id']
    assert len(programs) == 3
    print(program_id)
