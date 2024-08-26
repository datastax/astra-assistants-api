from typing import List, Dict

from pydantic import BaseModel, Field

from astra_assistants.tools.structured_code.program_cache import ProgramCache, StructuredProgram
from astra_assistants.tools.tool_interface import ToolInterface


class StructuredEditReplace(BaseModel):
    thoughts: str = Field(...,
                          description="The message to be described to the user explaining how the replace will work, think step by step.")
    start_line_number: int = Field(...,
                                   description="Line number where the replace starts (first line is line 1). ALWAYS requried")
    end_line_number: int = Field(...,
                                 description="Line number where the replace ends (line numbers are inclusive, i.e. start_line_number 1 end_line_number 1 will replace 1 line, start_line_number 1 end_line_number 2 will replace two lines), end_line_number is always required for replace")

    class Config:
        schema_extra = {
            "examples": [
                {
                    "thoughts": "replace lines 2, 3, and 4 with the new code",
                    "start_line_number": 2,
                    "end_line_number": 4,
                },
                {
                    "thoughts": "replace lines 7 with the new code",
                    "start_line_number": 7,
                    "end_line_number": 7,
                },
            ]
        }


class StructuredCodeReplace(ToolInterface):

    def __init__(self, program_cache: ProgramCache):
        self.program_cache = program_cache
        self.program_id = None

        print("initialized")

    def set_program_id(self, program_id):
        self.program_id = program_id

    def call(self, edit: StructuredEditReplace):
        try:
            program = None
            for entry in self.program_cache:
                if entry.program_id == self.program_id:
                    program = entry.program.copy()
                    break
            if not program:
                raise Exception(f"Program id {self.program_id} not found, did you forget to call set_program_id()?")

            instructions = (f"Write some code based on the instructions provided.\n"
                            f"## Instructions:\n"
                            f"Only return the code wrapped in block quotes:\n"
                            f"for example:\n"
                            f"```\n"
                            f"code goes here\n"
                            f"```\n"
                            f"do not return anything else\n"
                            f"{edit.thoughts}\n"
                            f"your code will replace the lines {edit.start_line_number} through {edit.end_line_number} in the code below\n"
                            f"## Code Snippet:\n"
                            f"{program.to_string()}")
            print(f"providing instructions: \n{instructions}")

            return {'program_id': self.program_id, 'output': instructions, 'tool': self.__class__.__name__,
                    'edit': edit}
        except Exception as e:
            print(f"Error: {e}")
            raise e
