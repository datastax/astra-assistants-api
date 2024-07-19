from typing import List, Optional, Literal, Dict
from uuid import uuid1

from pydantic import BaseModel, Field

from astra_assistants.tools.tool_interface import ToolInterface


class StructuredEdit(BaseModel):
    program_id: str = Field(..., description="ID of the program being edited")
    lines: Optional[List[str]] = Field(
        ...,
        description="List of strings representing each line of code. Required for insert and replace edits"
    )
    location_start: int = Field(None, description="Index of the line where the edit starts")
    location_end: Optional[int] = Field(None, description="Index of the line where the edit ends, always required for replace and delete, not required for insert")
    mode: Optional[Literal['insert', 'delete', 'replace']] = Field(
        None,
        description="Type of edit being made (must be insert, delete, or replace)"
    )


class StructuredProgram(BaseModel):
    language: str = Field(..., description="Programming language of the code snippet")
    lines_of_code: List[str] = Field(..., description="List of strings representing each line of code")
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
            program = None
            for pair in self.program_cache:
                if pair['program_id'] == edit.program_id:
                    program = pair['output'].copy()
                    break
            if not program:
                return f"Program id {edit.program_id} not found"
            if edit.mode == 'insert':
                i = 0
                for line in edit.lines:
                    program.lines_of_code.insert(edit.location_start + i, line)
                    i += 1
            if edit.mode == 'delete':
                if edit.location_end:
                    del program.lines_of_code[edit.location_start:edit.location_end]
                else:
                    del program.lines_of_code[edit.location_start]
            if edit.mode == 'replace':
                program.lines_of_code[edit.location_start:edit.location_end] = edit.lines
            program_info = {'program_id': str(uuid1()), 'output': program}
            self.program_cache.append(program_info)
            return program_info
        except Exception as e:
            print(f"Error: {e}")
            raise e
