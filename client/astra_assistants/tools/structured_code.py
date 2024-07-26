from typing import List, Optional, Literal, Dict
from uuid import uuid1

from pydantic import BaseModel, Field

from astra_assistants.tools.tool_interface import ToolInterface


class StructuredEdit(BaseModel):
    program_id: str = Field(..., description="ID of the program being edited")
    lines: Optional[List[str]] = Field(
        ...,
        description="List of strings representing each line of code for the modification (not the entire file). Required for insert and replace edits. ALWAYS PRESERVE INDENTATION, i.e. ['    print('puppies')'] instead of ['print('puppies')'] when replacing inside an indented block."
    )
    start_index: int = Field(None, description="Index of the line where the edit starts. ALWAYS requried")
    end_index: Optional[int] = Field(None, description="Index of the line where the edit ends (indexes are inclusive, i.e. start_index 1 end_index 1 will delete/replace 1 line, start_index 1 end_index 2 will delete/replace two lines), always required for replace and delete, not required for insert")
    mode: Optional[Literal['insert', 'delete', 'replace']] = Field(
        None,
        description="Type of edit being made (must be insert, delete, or replace)"
    )


class StructuredProgram(BaseModel):
    language: str = Field(..., description="Programming language of the code snippet")
    lines_of_code: List[str] = Field(..., description="List of strings representing each line of code. Remember to escape any double quotes in the code with a backslash (e.g. lines_of_code = \"var = \\\"Hello, world\\\"\"")
    description: Optional[str] = Field(None, description="Brief description of the code snippet")
    filename: Optional[str] = Field(None, description="Name of the file containing the code snippet")
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

        print("initialized")

    def call(self, edit: StructuredEdit):
        try:
            program : StructuredProgram = None
            for pair in self.program_cache:
                if pair['program_id'] == edit.program_id:
                    program = pair['output'].copy()
                    break
            if not program:
                return f"Program id {edit.program_id} not found"
            print(f"program before edit: \n{program.to_string()}")
            print(f"edit: {edit}")
            if edit.mode == 'insert':
                i = 0
                for line in edit.lines:
                    program.lines_of_code.insert(edit.start_index+ i, line)
                    i += 1
            if edit.mode == 'delete':
                if edit.end_index:
                    del program.lines_of_code[edit.start_index-1:edit.end_index]
                else:
                    del program.lines_of_code[edit.start_index]
            if edit.mode == 'replace':
                program.lines_of_code[edit.start_index-1:edit.end_index] = edit.lines
            print(f"program after edit: \n{program.to_string()}")
            return {'program_id': edit.program_id, 'output': program}
        except Exception as e:
            print(f"Error: {e}")
            raise e
