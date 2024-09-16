import re
import traceback
from uuid import uuid1

from astra_assistants.tools.structured_code.program_cache import StructuredProgram, StructuredProgramEntry
from astra_assistants.tools.structured_code.write import get_indentation


def extract_code_blocks(program_str: str):
    backtick_positions = [m.start() for m in re.finditer(r'```', program_str)]

    original_blocks = []
    indent_fixed_blocks = []

    if len(backtick_positions) > 0:
        start = backtick_positions[0]
        # Iterate over positions in pairs (start and end)
        for i in range(1, len(backtick_positions)):
            end = backtick_positions[i]

            # Extract the block content
            block_content = program_str[start + 3:end]

            # Check for and remove the optional "python" language identifier
            if block_content.startswith("python"):
                block_content = block_content[6:]

            original_blocks.append(block_content.lstrip('\n').rstrip())
            indent_fixed_blocks.append(block_content.strip())

        return original_blocks, indent_fixed_blocks
    original_blocks.append(program_str)
    indent_fixed_blocks.append(program_str.strip())
    return original_blocks, indent_fixed_blocks


def sanitize_program_str(program_str: str, language: str) -> str:
    og_matches, matches = extract_code_blocks(program_str)

    # Sort the matches by the length of the captured code block, from largest to smallest
    matches.sort(reverse=True)
    og_matches.sort(reverse=True)

    # Iterate over all sorted matches
    i = 0
    for match in matches:
        code = match
        og_code = og_matches[i]
        i += 1

        # If language is Python or unspecified (default to Python), validate the code
        if language is None or language.lower() == 'python' or language.lower() == 'py':
            if is_valid_python_code(code):
                return og_code
        else:
            # TODO: multi-language ts support
            print(f"Need to add tree-sitter support for language: {language}")
            print(f"defaulting to the biggest block for now")
            return og_matches[0]
    print(f"failed to find good blocks for language {language} in {program_str}")
    print(f"defaulting to the biggest block")
    return og_matches[0]


def program_str_to_program(program_str: str, language: str, filename: str, tags=None,
                           description=None) -> StructuredProgram:
    sanitized_program = sanitize_program_str(program_str, language)
    lines = []
    for line in sanitized_program.splitlines():
        lines.append(line)
    return StructuredProgram(language=language, lines=lines, filename=filename, tags=tags, description=description)


def is_valid_python_code(code: str) -> bool:
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False


def add_program_to_cache(program, program_cache):
    program_id = str(uuid1())
    entry = StructuredProgramEntry(program_id=program_id, program=program)
    program_cache.add(entry)
    return program_id


def process_program_with_tool(program, text, tool, edit):
    if tool == "StructuredCodeReplace":
        if edit.end_line_number is None:
            edit.end_line_number = edit.start_line_number
        edit_indentation = get_indentation(text.split()[0])
        if edit_indentation == "":
            program_indentation = get_indentation(program.lines[edit.start_line_number - 1])
            for line in text.splitlines():
                line = f"{program_indentation}{line}"
        program.lines[edit.start_line_number - 1:edit.end_line_number] = text.splitlines()
        return program
    elif tool == "StructuredCodeInsert":
        i = -1
        edit_indentation = get_indentation(text.split()[0])
        if edit_indentation == "":
            program_indentation = get_indentation(program.lines[edit.start_line_number - 1])
            for line in text.splitlines():
                line = f"{program_indentation}{line}"
        for line in text.splitlines():
            program.lines.insert(edit.start_line_number + i, line)
            i += 1
        return program
    elif tool == "StructuredCodeFileGenerator":
        program = program_str_to_program(text, program.language, program.filename, program.tags, program.description)
        program.filename = program.filename.split('.')[0] + '/app.py'
        return program
    else:
        print(f"no changes for tool {tool}")
        program = program_str_to_program(text, program.language, program.filename, program.tags, program.description)
        return program


def add_chunks_to_cache(chunks, cache, function=None):
    try:
        first_chunk = next(chunks)
        assert not isinstance(first_chunk, str)
        if "program_id" in first_chunk:
            program_id = first_chunk["program_id"]
            last_program = cache.get(program_id).program
        else:
            last_program = first_chunk['program_desc']
        # If the tool expects code output in chunks output will be a string
        if isinstance(first_chunk["output"], str):
            text = ""
            for chunk in chunks:
                text += chunk
            program = None
            # tools like file generator don't have edits
            if 'tool' in first_chunk:
                edit = None
                if 'edit' in first_chunk:
                    edit = first_chunk['edit']
                tool = first_chunk['tool']
                text = sanitize_program_str(text, last_program.language)
                program = process_program_with_tool(last_program, text, tool, edit)
                print(f"edit: \n{edit}\ntext: \n{text}")
            else:
                program = program_str_to_program(text, last_program.language, last_program.filename, last_program.tags,
                                                 last_program.description)
            print(f"program after edit: \n{program.to_string()}")
            program_id = add_program_to_cache(program, cache)
            if function is not None:
                function(chunks, text)
            return {'program_id': program_id, 'output': program}
        else:
            if function is not None:
                function(chunks, first_chunk)
                return first_chunk
            else:
                raise Exception(f"No function provided to handle chunks, function required for first_chunk {first_chunk}")
    except Exception as e:
        print(f"Error: {e}")
        trace = traceback.format_exc()
        print(trace)
        return None
