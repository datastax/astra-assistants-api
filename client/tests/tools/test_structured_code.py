import pytest
from lsprotocol import types
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput

from astra_assistants.astra_assistants_event_handler import AstraEventHandler
import logging
from astra_assistants.astra_assistants_manager import AssistantManager
from astra_assistants.tools.structured_code.program_cache import ProgramCache
from astra_assistants.tools.structured_code.util import program_str_to_program, add_program_to_cache, \
    add_chunks_to_cache
from astra_assistants.tools.structured_code.indent import StructuredCodeIndentLeft
from astra_assistants.tools.structured_code.replace import StructuredCodeReplace
from astra_assistants.tools.structured_code.delete import StructuredCodeDelete
from astra_assistants.tools.structured_code.insert import StructuredCodeInsert
from astra_assistants.tools.structured_code.rewrite import StructuredCodeRewrite
from astra_assistants.tools.structured_code.write import StructuredCodeFileGenerator

logger = logging.getLogger(__name__)


def test_structured_code_raw(patched_openai_client):
    programs = ProgramCache()
    code_generator = StructuredCodeFileGenerator(programs)
    code_replace = StructuredCodeReplace(programs)

    # Create the assistant
    assistant = patched_openai_client.beta.assistants.create(
        name="Smart bot",
        instructions="""
use the structured code tool to generate code to help the user.
""",
        model="gpt-4o",
        tools=[code_generator.to_function(), code_replace.to_function()],
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
        print(f"tool_output: {event_handler.tool_output}")
        program_desc = event_handler.tool_call_results['program_desc']
        code = ""
        with event_handler.stream as stream:
            for text in stream.text_deltas:
                print(text, end="", flush=True)
                code += text
        print(code)
        program = program_str_to_program(code, program_desc.language, program_desc.filename, program_desc.tags, program_desc.description)
        program_id = add_program_to_cache(program, programs)
        print(program_id)

    event_handler = AstraEventHandler(patched_openai_client)
    event_handler.register_tool(code_replace)
    program_id = programs[0].program_id
    program = programs[0].program
    patched_openai_client.beta.threads.messages.create(thread.id, content=f"nice, now add trigonometric functions to program_id {program_id}: \n{program.to_string()}" , role="user")
    code_replace.set_program_id(program_id)
    with patched_openai_client.beta.threads.runs.create_and_stream(
            thread_id=thread.id,
            assistant_id=assistant.id,
            event_handler=event_handler,
            tool_choice=code_replace.tool_choice_object(),
    ) as stream:
        for text in stream.text_deltas:
            print(text, end="", flush=True)
        print()
        tool_output = event_handler.tool_output
        print(f"tool_output: {tool_output}")
    programs.close()


@pytest.mark.asyncio
async def test_structured_code_with_manager(patched_openai_client):
    programs = ProgramCache()

    code_generator = StructuredCodeFileGenerator(programs)
    code_replace = StructuredCodeReplace(programs)
    code_insert = StructuredCodeInsert(programs)
    code_delete = StructuredCodeDelete(programs)
    tools = [code_generator, code_replace, code_insert, code_delete]

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
    program = program_str_to_program(result['text'], result['program_desc'].language, result['program_desc'].filename, result['program_desc'].tags, result['program_desc'].description)
    program_id = add_program_to_cache(program, programs)
    content = f"nice, now add trigonometric functions to program_id {program_id}: \n{program.to_string()}"
    code_replace.set_program_id(program_id)
    result = await assistant_manager.run_thread(
        content=content,
        tool=code_replace
    )

    content = "make a calculator web app that supports simple arithmetic, stick everything in one file."
    chunks = assistant_manager.stream_thread(
        content=content,
        tool_choice=code_generator
    )
    first_chunk = add_chunks_to_cache(chunks, programs)
    assert first_chunk
    programs.close()

def test_structured_rewrite_with_manager(patched_openai_client):
    programs = ProgramCache()

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
    programs.close()


def test_structured_rewrite_and_edit_with_manager(patched_openai_client):
    programs = ProgramCache()

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
    programs.close()


def test_structured_all_with_manager(patched_openai_client):
    programs = ProgramCache()

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
    code_replace = StructuredCodeReplace(programs)
    code_insert = StructuredCodeInsert(programs)
    code_delete = StructuredCodeDelete(programs)

    tools = [code_rewriter, code_indent_left, code_replace, code_insert, code_delete]

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
        assert False, "stream_thread should have failed because we didn't set the program_id"
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
    assert program_id == programs[len(programs)-1].program_id
    assert len(programs) == 3
    print(program_id)

    code_insert.set_program_id(program_id)
    chunks: ToolOutput = assistant_manager.stream_thread(
        content="Add a comment.",
        tool_choice=code_insert
    )

    program_id = add_chunks_to_cache(chunks, programs)['program_id']
    assert len(programs) == 4
    print(program_id)

    code_delete.set_program_id(program_id)
    chunks: ToolOutput = assistant_manager.stream_thread(
        content="Delete a comments.",
        tool_choice=code_delete
    )

    tool_call_result = next(chunks)
    assert len(programs) == 5
    code_rewriter.set_program_id(tool_call_result['program_id'])

    code_replace.set_program_id(program_id)
    chunks: ToolOutput = assistant_manager.stream_thread(
        content="Make comment text in all caps.",
        tool_choice=code_replace
    )

    program_id = add_chunks_to_cache(chunks, programs)['program_id']
    assert len(programs) == 6
    print(program_id)
    programs.close()


def test_program_parser():
    test_input = '''
    Some introductory text.

    ```python
    def example():
        print("This is a code block")
        print("Here is a ``` inside the code block")
        # Another ``` inside comment
    ```
    Some other text.
    ```python
    def another_example():
        print("This is another code block")
    ```
    Final text.
    '''

    result = program_str_to_program(test_input, language='python', filename='example.py')
    answer = '''    def example():
        print("This is a code block")
        print("Here is a ``` inside the code block")
        # Another ``` inside comment'''

    assert result.to_string(False) == answer
    print(result)

    uri = "file:///path/to/file"
    document_version = 1
    text_change_event = types.TextDocumentContentChangeEvent_Type2(
        text="test",
    )
    did_change_payload_obj = types.DidChangeTextDocumentParams(
        text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=document_version),
        content_changes=[
            text_change_event,
        ],
    )
    print(did_change_payload_obj)