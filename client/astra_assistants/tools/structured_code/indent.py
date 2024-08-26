from typing import Optional, List, Dict
from uuid import uuid1

import tree_sitter_python as tspython
from pydantic import BaseModel, Field
from tree_sitter import Language, Parser

from astra_assistants.tools.structured_code.program_cache import ProgramCache, StructuredProgramEntry, StructuredProgram
from astra_assistants.tools.tool_interface import ToolInterface

ts_language = Language(tspython.language())
parser = Parser(ts_language)


def get_line_text(line_number, source_code):
    """Helper function to get the text of a specific line."""
    lines = source_code.splitlines()
    if 0 <= line_number < len(lines):
        return lines[line_number]
    return ""


def get_indentation_unit(source_code, target_line):
    """Determine the minimum indentation string around a specific line of code using Tree-sitter."""

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


class StructuredCodeIndentRight(ToolInterface):

    def __init__(self, program_cache: ProgramCache):
        self.program_cache = program_cache
        self.program_id = None

        print("initialized")

    def set_program_id(self, program_id):
        self.program_id = program_id


    def call(self, edit: IndentRightEdit):
        try:
            program = None
            for entry in self.program_cache:
                if entry.program_id == self.program_id:
                    program = entry.program.copy()
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
            entry = StructuredProgramEntry(program_id=new_program_id, program=program)
            self.program_cache.append(entry)
            print(f"program after edit: \n{program.to_string()}")
            return {'program_id': new_program_id, 'output': program}
        except Exception as e:
            print(f"Error: {e}")
            raise e


class StructuredCodeIndentLeft(ToolInterface):

    def __init__(self, program_cache: ProgramCache):
        self.program_cache = program_cache
        self.program_id = None

        print("initialized")

    def set_program_id(self, program_id):
        self.program_id = program_id


    def call(self, edit: IndentLeftEdit):
        try:
            program = None
            for entry in self.program_cache:
                if entry.program_id == self.program_id:
                    program = entry.program.copy()
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
            entry = StructuredProgramEntry(program_id=new_program_id, program=program)
            self.program_cache.append(entry)
            print(f"program after edit: \n{program.to_string()}")
            return {'program_id': new_program_id, 'output': program}
        except Exception as e:
            print(f"Error: {e}")
            raise e
