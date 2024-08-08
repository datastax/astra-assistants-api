from typing import List, Optional, Literal, Dict
from uuid import uuid1

from pydantic import BaseModel, Field

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


class StructuredProgram(BaseModel):
    language: str = Field(..., description="Programming language of the code snippet")
    lines_of_code: List[str] = Field(..., description="List of strings representing each line of code. Remember to escape any double quotes in the code with a backslash (e.g. lines_of_code = \"var = \\\"Hello, world\\\"\"")
    description: Optional[str] = Field(None, description="Brief description of the code snippet")
    filename: str = Field(None, description="Name of the file containing the code snippet")
    tags: Optional[List[str]] = Field(None, description="Tags or keywords related to the code snippet")

    class Config:
        schema_extra = {
            "example": {
                "language": "Python",
                "lines_of_code": [
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
            lines = [f"{i+1}: {line}" for i, line in enumerate(self.lines_of_code)]
            return "\n".join(lines)
        else:
            return "\n".join(self.lines_of_code)



class StructuredCodeGenerator(ToolInterface):

    def __init__(self, program_cache: List[Dict[str, StructuredProgram]]):
        self.program_cache = program_cache

        print("initialized")

    def call(self, program: StructuredProgram) -> Dict[str, any]:
        program_id = str(uuid1())
        program_info = {'program_id': program_id, 'output': program}
        self.program_cache.append(program_info)
        return program_info


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
                    program.lines_of_code.insert(edit.start_line_number + i, line)
                    i += 1
            if edit.mode == 'delete':
                if edit.end_line_number:
                    del program.lines_of_code[edit.start_line_number-1:edit.end_line_number]
                else:
                    del program.lines_of_code[edit.start_line_number]
            if edit.mode == 'replace':
                program.lines_of_code[edit.start_line_number-1:edit.end_line_number] = edit.lines
            print(f"program after edit: \n{program.to_string()}")
            return {'program_id': self.program_id, 'output': program}
        except Exception as e:
            print(f"Error: {e}")
            raise e
