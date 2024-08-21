from typing import List, Optional, Literal, Dict
from uuid import uuid1

from pydantic import BaseModel, Field

from astra_assistants.tools.tool_interface import ToolInterface

class StructuredRewrite(BaseModel):
    thoughts: str = Field(..., description="The message to be described to the user explaining how the edit will work, think step by step.")
    class Config:
        schema_extra = {
            "example": {
                "thoughts": "let's refactor the code in the function `more_puppies` to use a list comprehension instead of a for loop",
            }
        }


class StructuredEdit(BaseModel):
    thoughts: str = Field(..., description="The message to be described to the user explaining how the edit will work, think step by step.")
    lines: Optional[List[str]] = Field(
       None,
        description="List of strings representing each line of code for the modification (not the entire file). Required for insert and replace edits. ALWAYS PRESERVE INDENTATION, i.e. ['    print('puppies')'] instead of ['print('puppies')'] when replacing inside an indented block."
    )
    start_line_number: int = Field(..., description="Line number where the edit starts (first line is line 1). ALWAYS requried")
    end_line_number: Optional[int] = Field(None, description="Line number where the edit ends (line numbers are inclusive, i.e. start_line_number 1 end_line_number 1 will delete/replace 1 line, start_line_number 1 end_line_number 2 will delete/replace two lines), end_line_number is always required for replace and delete, not required for insert")
    mode: Optional[Literal['insert', 'delete', 'replace']] = Field(
        None,
        description="Type of edit being made (must be insert, delete, or replace)"
    )

class IndentLeftEdit(BaseModel):
    thoughts: str = Field(..., description="The message to be described to the user explaining how the indent left edit will work, think step by step.")
    start_line_number: int = Field(..., description="Line number where the indent left edit starts (first line is line 1). ALWAYS requried")
    end_line_number: Optional[int] = Field(None, description="Line number where the indent left edit ends (line numbers are inclusive, i.e. start_line_number 1 end_line_number 1 will indent 1 line, start_line_number 1 end_line_number 2 will indent two lines)")
    class Config:
        schema_extra = {
            "examples": [
                {
                    "thoughts": "let's move lines 55 through 57 to the left by one indentation unit",
                    "start_line_number": 55,
                    "end_line_number": 57,
                },
                {
                    "thoughts": "let's move line 12 to the left by one indentation unit",
                    "start_line_number": 12,
                },
                {
                    "thoughts": "let's move lines 100 through 101 to the left by one indentation unit",
                    "start_line_number": 100,
                    "end_line_number": 101,
                },
            ]
        }



class IndentRightEdit(BaseModel):
    thoughts: str = Field(..., description="The message to be described to the user explaining how the indent right edit will work, think step by step.")
    start_line_number: int = Field(..., description="Line number where the indent right edit starts (first line is line 1). ALWAYS requried")
    end_line_number: Optional[int] = Field(None, description="Line number where the indent right edit ends (line numbers are inclusive, i.e. start_line_number 1 end_line_number 1 will indent 1 line, start_line_number 1 end_line_number 2 will indent two lines)")
    class Config:
        schema_extra = {
            "examples": [
                {
                    "thoughts": "let's move lines 55 through 57 to the right by one indentation unit",
                    "start_line_number": 55,
                    "end_line_number": 57,
                },
                {
                    "thoughts": "let's move line 12 to the right by one indentation unit",
                    "start_line_number": 12,
                },
                {
                    "thoughts": "let's move lines 100 through 101 to the right by one indentation unit",
                    "start_line_number": 100,
                    "end_line_number": 101,
                },
            ]
        }



class StructuredProgram(BaseModel):
    language: str = Field(..., description="Programming language of the code snippet")
    lines: List[str] = Field(..., description="List of strings representing each line of code. Remember to escape any double quotes in the code with a backslash (e.g. lines= \"var = \\\"Hello, world\\\"\"")
    description: Optional[str] = Field(None, description="Brief description of the code snippet")
    filename: str = Field(..., description="Name of the file containing the code snippet")
    tags: Optional[List[str]] = Field(None, description="Tags or keywords related to the code snippet")

    class Config:
        schema_extra = {
            "example": {
                "language": "Python",
                "lines": [
                    "print('Hello, world!')",
                    "print('This is another line of code')"
                ],
                "description": "A simple Hello World program with multiple lines",
                "filename": "hello_world.py",
                "tags": ["example", "hello world", "beginner"]
            }
        }

    def to_string(self, with_line_numbers: bool = True) -> str:
        if with_line_numbers:
            lines = [f"{i+1}: {line}" for i, line in enumerate(self.lines)]
            return "\n".join(lines)
        else:
            return "\n".join(self.lines)



class StructuredCodeGenerator(ToolInterface):

    def __init__(self, program_cache: List[Dict[str, StructuredProgram]]):
        self.program_cache = program_cache

        print("initialized")

    def call(self, program: StructuredProgram) -> Dict[str, any]:
        program_id = str(uuid1())
        program_info = {'program_id': program_id, 'output': program}
        self.program_cache.append(program_info)
        return program_info


def get_indentation(line: str) -> str:
    """Helper function to get the indentation of a line."""
    return line[:len(line) - len(line.lstrip())]


class StructuredCodeEditor(ToolInterface):

    def __init__(self, program_cache: List[Dict[str, StructuredProgram]]):
        self.program_cache = program_cache
        self.program_id = None

        print("initialized")

    def set_program_id(self, program_id):
        self.program_id = program_id


    def call(self, edit: StructuredEdit):
        try:
            program : StructuredProgram = None
            for pair in self.program_cache:
                if pair['program_id'] == self.program_id:
                    program = pair['output'].copy()
                    break
            if not program:
                raise Exception(f"Program id {self.program_id} not found, did you forget to call set_program_id()?")
            print(f"program before edit: \n{program.to_string()}")
            print(f"edit: {edit}")
            if edit.mode == 'insert':
                i = 0
                for line in edit.lines:
                    program.lines.insert(edit.start_line_number + i, line)
                    i += 1
            if edit.mode == 'delete':
                if edit.end_line_number:
                    del program.lines[edit.start_line_number-1:edit.end_line_number]
                else:
                    del program.lines[edit.start_line_number]
            if edit.mode == 'replace':
                edit_indentation = get_indentation(edit.lines[0])
                if edit_indentation == "":
                    program_indentation = get_indentation(program.lines[edit.start_line_number-1])
                    for line in edit.lines:
                        line = f"{edit_indentation}{line}"
                program.lines[edit.start_line_number-1:edit.end_line_number] = edit.lines
            new_program_id = str(uuid1())
            self.program_cache.append({'program_id': new_program_id, 'output': program})
            print(f"program after edit: \n{program.to_string()}")
            return {'program_id': new_program_id, 'output': program}
        except Exception as e:
            print(f"Error: {e}")
            raise e


import tree_sitter_python as tspython
from tree_sitter import Language, Parser

language = Language(tspython.language())
parser = Parser(language)

def get_line_text(line_number, source_code):
    """Helper function to get the text of a specific line."""
    lines = source_code.splitlines()
    if 0 <= line_number < len(lines):
        return lines[line_number]
    return ""


def get_indentation_unit(source_code, target_line):
    """Determine the minimum indentation string around a specific line of code using Tree-sitter."""
    import tree_sitter

    def get_line_text(line_number, source_code):
        """Retrieve the text of a specific line from the source code."""
        lines = source_code.splitlines()
        if 0 <= line_number < len(lines):
            return lines[line_number]
        return ""

    def find_node_at_line(node, line_number):
        """Find the AST node corresponding to the specific line number."""
        if node.start_point[0] <= line_number <= node.end_point[0]:
            for child in node.children:
                found = find_node_at_line(child, line_number)
                if found:
                    return found
            return node
        return None

    def get_indentation_from_node(node, source_code):
        """Calculate the minimum indentation level from the given node."""
        min_indentation = None

        for line_number in range(node.start_point[0], node.end_point[0] + 1):
            line_text = get_line_text(line_number, source_code)
            leading_whitespace = line_text[:len(line_text) - len(line_text.lstrip())]

            if leading_whitespace:
                if min_indentation is None or len(leading_whitespace) < len(min_indentation):
                    min_indentation = leading_whitespace

        return min_indentation

    # Parse the source code
    tree = parser.parse(bytes(source_code, "utf8"))

    # Find the AST node that corresponds to the target line
    target_node = find_node_at_line(tree.root_node, target_line)

    if target_node:
        # Retrieve the minimum indentation of the parent node
        parent_node = target_node.parent
        if parent_node:
            min_indentation = get_indentation_from_node(parent_node, source_code)
        else:
            min_indentation = get_indentation_from_node(target_node, source_code)

        if min_indentation is not None:
            print(f"Minimum indentation: '{min_indentation}'")
            return min_indentation
        else:
            print("No indentation found around the specified line.")
            if target_line > 0:
                return get_indentation_unit(source_code, target_line-1)
    else:
        print(f"No node found at line {target_line + 1}.")
        if target_line > 0:
            return get_indentation_unit(source_code, target_line - 1)

    return ""


class StructuredCodeIndentRight(ToolInterface):

    def __init__(self, program_cache: List[Dict[str, StructuredProgram]]):
        self.program_cache = program_cache
        self.program_id = None

        print("initialized")

    def set_program_id(self, program_id):
        self.program_id = program_id


    def call(self, edit: IndentRightEdit):
        try:
            program : StructuredProgram = None
            for pair in self.program_cache:
                if pair['program_id'] == self.program_id:
                    program = pair['output'].copy()
                    break
            if not program:
                raise Exception(f"Program id {self.program_id} not found, did you forget to call set_program_id()?")
            print(f"program before edit: \n{program.to_string()}")
            print(f"edit: {edit}")
            
            indentation_unit = get_indentation_unit(program.to_string(with_line_numbers=False), edit.start_line_number-1)
            
            i = edit.start_line_number-1
            if edit.end_line_number is not None:
                while i < edit.end_line_number:
                    program.lines[i] = f"{indentation_unit}{program.lines[i]}"
                    i += 1
            else:
                program.lines[i] = f"{indentation_unit}{program.lines[i]}"

            new_program_id = str(uuid1())
            self.program_cache.append({'program_id': new_program_id, 'output': program})
            print(f"program after edit: \n{program.to_string()}")
            return {'program_id': new_program_id, 'output': program}
        except Exception as e:
            print(f"Error: {e}")
            raise e

class StructuredCodeIndentLeft(ToolInterface):

    def __init__(self, program_cache: List[Dict[str, StructuredProgram]]):
        self.program_cache = program_cache
        self.program_id = None

        print("initialized")

    def set_program_id(self, program_id):
        self.program_id = program_id


    def call(self, edit: IndentLeftEdit):
        try:
            program : StructuredProgram = None
            for pair in self.program_cache:
                if pair['program_id'] == self.program_id:
                    program = pair['output'].copy()
                    break
            if not program:
                raise Exception(f"Program id {self.program_id} not found, did you forget to call set_program_id()?")
            print(f"program before edit: \n{program.to_string()}")
            print(f"edit: {edit}")

            indentation_unit = get_indentation_unit(program.to_string(with_line_numbers=False), edit.start_line_number-1)
            i = edit.start_line_number-1
            if edit.end_line_number is not None:
                while i < edit.end_line_number and i < len(program.lines):
                    program.lines[i] = program.lines[i].replace(indentation_unit, "", 1)
                    i += 1
            else:
                program.lines[i] = program.lines[i].replace(indentation_unit, "", 1)

            new_program_id = str(uuid1())
            self.program_cache.append({'program_id': new_program_id, 'output': program})
            print(f"program after edit: \n{program.to_string()}")
            return {'program_id': new_program_id, 'output': program}
        except Exception as e:
            print(f"Error: {e}")
            raise e


class StructuredCodeRewrite(ToolInterface):

    def __init__(self, program_cache: List[Dict[str, StructuredProgram]]):
        self.program_cache = program_cache
        self.program_id = None

        print("initialized")

    def set_program_id(self, program_id):
        self.program_id = program_id


    def call(self, edit: StructuredRewrite):
        try:
            program : StructuredProgram = None
            for pair in self.program_cache:
                if pair['program_id'] == self.program_id:
                    program = pair['output'].copy()
                    break
            if not program:
                raise Exception(f"Program id {self.program_id} not found, did you forget to call set_program_id()?")

            instructions = (f"Rewrite the code snippet based on the instructions provided.\n"
                            f"## Instructions:\n"
                            f"Only return the code wrapped in block quotes:\n"
                            f"for example:\n"
                            f"```\n"
                            f"code goes here\n"
                            f"```\n"
                            f"do not return anything else\n"
                            f"{edit.thoughts}\n"
                            f"## Code Snippet:\n"
                            f"{program.to_string()}")
            print(f"providing instructions: \n{instructions}")

            return {'program_id': self.program_id, 'output': instructions}
        except Exception as e:
            print(f"Error: {e}")
            raise e


def program_str_to_program(program_str: str, language, filename, tags = None, description = None) -> StructuredProgram:
    program_str = program_str.lstrip('\n').rstrip('\n')
    if program_str.startswith("```") and program_str.endswith("```"):
        program_str = program_str[3:-3]
        program_str = program_str.lstrip('\n').rstrip('\n')
    if tags is None:
        tags = []
    lines = []
    for line in program_str.split("\n"):
        lines.append(line)

    return StructuredProgram(
        filename=filename,
        language=language,
        lines=lines,
        description=description,
        tags=tags
    )

def add_program_to_cache(program, program_cache):
    program_id = str(uuid1())
    program_cache.append({
        "program_id": program_id,
        "output": program
    })
    return program_id

def add_chunks_to_cache(chunks, cache, function = None):
    first_chunk = next(chunks)
    assert not isinstance(first_chunk, str)
    program_id = first_chunk["program_id"]
    last_program = None
    for cached_program in cache:
        if cached_program["program_id"] == program_id:
            last_program = cached_program['output']
            break
    assert last_program is not None
    if isinstance(first_chunk["output"], str):
        text = ""
        for chunk in chunks:
            text += chunk
        program = program_str_to_program(text, last_program.language, last_program.filename, last_program.tags, last_program.description)
        program_id = add_program_to_cache(program, cache)
        return first_chunk
    else:
        if function is not None:
            function(chunks, first_chunk)
            return first_chunk
        else:
            raise Exception(f"No function provided to handle chunks, function required for first_chunk {first_chunk}")