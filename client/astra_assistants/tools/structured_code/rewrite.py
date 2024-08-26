from typing import List, Dict

from pydantic import BaseModel, Field

from astra_assistants.tools.structured_code.program_cache import ProgramCache, StructuredProgram
from astra_assistants.tools.tool_interface import ToolInterface


class StructuredRewrite(BaseModel):
    thoughts: str = Field(..., description="The message to be described to the user explaining how the edit will work, think step by step.")
    class Config:
        schema_extra = {
            "example": {
                "thoughts": "let's refactor the code in the function `more_puppies` to use a list comprehension instead of a for loop",
            }
        }


class StructuredCodeRewrite(ToolInterface):

    def __init__(self, program_cache: ProgramCache):
        self.program_cache = program_cache
        self.program_id = None

        print("initialized")

    def set_program_id(self, program_id):
        self.program_id = program_id


    def call(self, edit: StructuredRewrite):
        try:
            program = None
            for pair in self.program_cache:
                if pair.program_id == self.program_id:
                    program = pair.program.copy()
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

            return {'program_id': self.program_id, 'output': instructions, 'tool': self.__class__.__name__, 'edit': edit}
        except Exception as e:
            print(f"Error: {e}")
            raise e
