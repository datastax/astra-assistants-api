from astra_assistants.tools.structured_code.lsp.session_manager import LspSessionManager
from astra_assistants.tools.structured_code.program_cache import ProgramCache, StructuredProgramEntry, StructuredProgram

CONTENTS = """import sys

print(x)
"""


def test_publish_diagnostics():
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

    programs.get_latest().program.to_string(with_line_numbers=False)
    programs.close()