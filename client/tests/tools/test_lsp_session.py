from astra_assistants.tools.structured_code.lsp.session_manager import LspSessionManager
from astra_assistants.tools.structured_code.program_cache import ProgramCache, StructuredProgramEntry
from astra_assistants.tools.structured_code.structured_code import StructuredProgram


CONTENTS = """import sys

print(x)
"""


def test_publish_diagnostics():
    sender = LspSessionManager()

    program = StructuredProgramEntry(
        program_id="1",
        program=StructuredProgram(
            lines=CONTENTS.splitlines(),
            language="python",
            filename="test.py",
        ),
    )

    programs = ProgramCache()
    programs.append(program)

    programs[0].program.to_string(with_line_numbers=False)