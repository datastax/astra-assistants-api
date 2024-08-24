from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from astra_assistants.tools.tool_interface import ToolInterface


class StructuredProgram(BaseModel):
    language: str = Field(..., description="Programming language of the code snippet")
    lines: List[str] = Field(...,
                             description="List of strings representing each line of code. Remember to escape any "
                                         "double quotes in the code with a backslash (e.g. lines= \"var = \\\"Hello, "
                                         "world\\\"\"")
    description: Optional[str] = Field(None, description="Brief description of the code snippet")
    filename: str = Field(..., description="Name of the file containing the code snippet")
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
            lines = [f"{i + 1}: {line}" for i, line in enumerate(self.lines)]
            return "\n".join(lines)
        else:
            return "\n".join(self.lines)


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

    def __init__(self, program_cache: List[Dict[str, StructuredProgram]]):
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
