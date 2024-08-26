from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from astra_assistants.tools.structured_code.program_cache import ProgramCache
from astra_assistants.tools.tool_interface import ToolInterface




class StructuredProgramDescription(BaseModel):
    thoughts: str = Field(...,
                          description="The message to be described to the user explaining how to create the file, "
                                      "think step by step.")
    language: str = Field(..., description="Programming language of the code snippet")
    description: Optional[str] = Field(None, description="Brief description of the code snippet")
    filename: str = Field(..., description="Name of the file containing the code snippet")
    tags: Optional[List[str]] = Field(None, description="Tags or keywords related to the code snippet")

    class Config:
        schema_extra = {
            "example": {
                "thoughts": ("Write a simple hello world program using python, fasthtml, and htmx. "
                             "We always keep everything in one file."),
                "language": "Python",
                "description": "A simple Hello World program",
                "filename": "hello_world.py",
                "tags": ["example", "hello world", "beginner"]
            }
        }


class StructuredCodeFileGenerator(ToolInterface):

    def __init__(self, program_cache: ProgramCache):
        self.program_cache = program_cache

        print("initialized")

    def call(self, program: StructuredProgramDescription) -> Dict[str, any]:
        instructions = (f"Write one file of code based on the instructions provided.\n"
                        f"## Instructions:\n"
                        f"Only return the code wrapped in a single set of block quotes:\n"
                        f"for example:\n"
                        f"```\n"
                        f"code goes here\n"
                        f"```\n"
                        f"do not return anything else\n"
                        f"{program.thoughts}\n")
        print(f"providing instructions: \n{instructions}")
        return {'output': instructions, 'program_desc': program, 'tool': self.__class__.__name__}


def get_indentation(line: str) -> str:
    """Helper function to get the indentation of a line."""
    return line[:len(line) - len(line.lstrip())]
