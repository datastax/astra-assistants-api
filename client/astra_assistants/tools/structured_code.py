from typing import List, Optional, Literal, Dict
from uuid import uuid1

from pydantic import BaseModel, Field
from tree_sitter_languages import get_language, get_parser

from astra_assistants.tools.tool_interface import ToolInterface


class StructuredEdit(BaseModel):
    thoughts: str = Field(..., description="The message to be described to the user explaining how the edit will work, think step by step.")
    lines: Optional[List[str]] = Field(
        ...,
        description="List of strings representing each line of code for the modification (not the entire file). Required for insert and replace edits. ALWAYS PRESERVE INDENTATION, i.e. ['    print('puppies')'] instead of ['print('puppies')'] when replacing inside an indented block."
    )
    start_line_number: int = Field(None, description="Line number where the edit starts (first line is line 1). ALWAYS requried")
    end_line_number: Optional[int] = Field(None, description="Line number where the edit ends (line numbers are inclusive, i.e. start_line_number 1 end_line_number 1 will delete/replace 1 line, start_line_number 1 end_line_number 2 will delete/replace two lines), end_line_number is always required for replace and delete, not required for insert")
    mode: Optional[Literal['insert', 'delete', 'replace']] = Field(
        None,
        description="Type of edit being made (must be insert, delete, or replace)"
    )

class IndentLeftEdit(BaseModel):
    thoughts: str = Field(..., description="The message to be described to the user explaining how the indent left edit will work, think step by step.")
    start_line_number: int = Field(None, description="Line number where the indent left edit starts (first line is line 1). ALWAYS requried")
    end_line_number: Optional[int] = Field(None, description="Line number where the indent left edit ends (line numbers are inclusive, i.e. start_line_number 1 end_line_number 1 will indent 1 line, start_line_number 1 end_line_number 2 will indent two lines)")


class IndentRightEdit(BaseModel):
    thoughts: str = Field(..., description="The message to be described to the user explaining how the indent right edit will work, think step by step.")
    start_line_number: int = Field(None, description="Line number where the indent right edit starts (first line is line 1). ALWAYS requried")
    end_line_number: Optional[int] = Field(None, description="Line number where the indent right edit ends (line numbers are inclusive, i.e. start_line_number 1 end_line_number 1 will indent 1 line, start_line_number 1 end_line_number 2 will indent two lines)")


class StructuredProgram(BaseModel):
    language: str = Field(..., description="Programming language of the code snippet")
    lines: List[str] = Field(..., description="List of strings representing each line of code. Remember to escape any double quotes in the code with a backslash (e.g. lines= \"var = \\\"Hello, world\\\"\"")
    description: Optional[str] = Field(None, description="Brief description of the code snippet")
    filename: str = Field(None, description="Name of the file containing the code snippet")
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
                return f"Program id {self.program_id} not found"
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
                edit_indentation = get_indentation(edit.lines[edit.start_line_number-1])
                if edit_indentation == "":
                    program_indentation = get_indentation(program.lines[edit.start_line_number-1])
                    for line in edit.lines:
                        program.lines.insert(edit.start_line_number, f"{program_indentation}{line}")
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
    """Determine the minimum indentation string around a specific line of code."""
    def find_node_at_line(node, line_number):
        if node.start_point[0] <= line_number <= node.end_point[0]:
            for child in node.children:
                found = find_node_at_line(child, line_number)
                if found:
                    return found
            return node
        return None

    tree = parser.parse(bytes(source_code, "utf8"))
    # Find the AST node that corresponds to the target line
    target_node = find_node_at_line(tree.root_node, target_line)

    if target_node:
        # Initialize minimum indentation
        min_indentation = None

        # Check the node's own line
        start_line_text = get_line_text(target_node.start_point[0], source_code)
        start_line_indentation = start_line_text[:len(start_line_text) - len(start_line_text.lstrip())]

        # Check the line containing the node's start
        if start_line_indentation:
            min_indentation = start_line_indentation

        # Check the surrounding lines within the node's range
        for line_number in range(target_node.start_point[0], target_node.end_point[0] + 1):
            line_text = get_line_text(line_number, source_code)
            leading_whitespace = line_text[:len(line_text) - len(line_text.lstrip())]

            if leading_whitespace:
                if min_indentation is None or len(leading_whitespace) < len(min_indentation):
                    min_indentation = leading_whitespace

        if min_indentation is not None:
            print(f"Minimum indentation: '{min_indentation}'")
            return min_indentation
        else:
            print("No indentation found around the specified line.")
            return ""
    else:
        print(f"No node found at line {target_line + 1}.")
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
                return f"Program id {self.program_id} not found"
            print(f"program before edit: \n{program.to_string()}")
            print(f"edit: {edit}")
            
            indentation_unit = get_indentation_unit(program.to_string(with_line_numbers=False), edit.start_line_number-1)
            
            i = edit.start_line_number-1
            while i < edit.end_line_number:
                program.lines[i] = f"{indentation_unit}{program.lines[i]}"
                i += 1

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
                return f"Program id {self.program_id} not found"
            print(f"program before edit: \n{program.to_string()}")
            print(f"edit: {edit}")

            indentation_unit = get_indentation_unit(program.to_string(with_line_numbers=False), edit.start_line_number-1)
            i = edit.start_line_number-1
            while i < edit.end_line_number:
                program.lines[i] = program.lines[i].replace(indentation_unit, "", 1)
                i += 1

            new_program_id = str(uuid1())
            self.program_cache.append({'program_id': new_program_id, 'output': program})
            print(f"program after edit: \n{program.to_string()}")
            return {'program_id': new_program_id, 'output': program}
        except Exception as e:
            print(f"Error: {e}")
            raise e
