from typing import List, Dict

from pydantic import BaseModel, Field

from astra_assistants.tools.structured_code.program_cache import ProgramCache, StructuredProgram
from astra_assistants.tools.tool_interface import ToolInterface
from astra_assistants.utils import copy_program_from_cache


class StructuredEditInsert(BaseModel):
    thoughts: str = Field(..., description="The message to be described to the user explaining how the insert will work, think step by step.")
    start_line_number: int = Field(..., description="Line number where the insert starts, the new code will get inserted at this line pushing whatever is in this line down (first line is line 1). ALWAYS requried")
    class Config:
        schema_extra = {
            "example": {
                "thoughts": "let's add a the new \"/search\" endpoint definition below \"/\" make sure to name the function get (since it's an http get method) and return the Div() to swap with HTMX",
                "start_line_number": 20,
            }
        }


class StructuredCodeInsert(ToolInterface):

    def __init__(self, program_cache: ProgramCache):
        self.program_cache = program_cache
        self.program_id = None

        print("initialized")

    def set_program_id(self, program_id):
        self.program_id = program_id


    def call(self, edit: StructuredEditInsert):
        try:
            program = copy_program_from_cache(self.program_id, self.program_cache)

            instructions = (f"Write some code based on the instructions provided.\n"
                            f"## Instructions:\n"
                            f"Only return the code wrapped in block quotes:\n"
                            f"for example:\n"
                            f"```\n"
                            f"code goes here\n"
                            f"```\n"
                            f"do not return anything else\n"
                            f"{edit.thoughts}\n"
                            f"your code will be inserted at line {edit.start_line_number} in the code below. "
                            f"Existing code will shift down\n"
                            f"## Code Snippet:\n"
                            f"{program.to_string()}")
            print(f"providing instructions: \n{instructions}")

            return {'program_id': self.program_id, 'output': instructions, 'tool': self.__class__.__name__, 'edit': edit}
        except Exception as e:
            print(f"Error: {e}")
            raise e
