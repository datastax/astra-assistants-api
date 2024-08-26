from typing import List, Dict
from uuid import uuid1

from pydantic import BaseModel, Field

from astra_assistants.tools.structured_code.program_cache import StructuredProgramEntry, ProgramCache, StructuredProgram
from astra_assistants.tools.tool_interface import ToolInterface


class StructuredEditDelete(BaseModel):
    start_line_number: int = Field(..., description="Line number where the delete starts (first line is line 1). ALWAYS requried")
    end_line_number: int = Field(..., description="Line number where the edit ends (line numbers are inclusive, i.e. start_line_number 1 end_line_number 1 will delete 1 line, start_line_number 1 end_line_number 2 will delete two lines), end_line_number is always required for delete")
    class Config:
        schema_extra = {
            "examples": [
                {
                    "start_line_number": 2,
                    "end_line_number": 4,
                },
                {
                    "start_line_number": 36,
                    "end_line_number": 36,
                },
            ]
        }


class StructuredCodeDelete(ToolInterface):

    def __init__(self, program_cache: ProgramCache):
        self.program_cache = program_cache
        self.program_id = None

        print("initialized")

    def set_program_id(self, program_id):
        self.program_id = program_id


    def call(self, edit: StructuredEditDelete):
        try:
            program = None
            for entry in self.program_cache:
                if entry.program_id == self.program_id:
                    program = entry.program.copy()
                    break
            if not program:
                raise Exception(f"Program id {self.program_id} not found, did you forget to call set_program_id()?")

            if edit.end_line_number:
                del program.lines[edit.start_line_number-1:edit.end_line_number]
            else:
                del program.lines[edit.start_line_number]

            new_program_id = str(uuid1())
            entry = StructuredProgramEntry(program_id=new_program_id, program=program)
            self.program_cache.append(entry)
            print(f"program after edit: \n{program.to_string()}")
            return {'program_id': new_program_id, 'output': program}

        except Exception as e:
            print(f"Error: {e}")
            raise e
