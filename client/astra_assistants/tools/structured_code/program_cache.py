import tempfile
import traceback
from typing import List, Optional

from lsprotocol import types, converters
from openai import BaseModel
from pydantic import Field

from astra_assistants.tools.structured_code.lsp.constants import as_uri
from astra_assistants.tools.structured_code.lsp.lsp_session import CODE_ACTION
from astra_assistants.tools.structured_code.lsp.session_manager import LspSessionManager
from astra_assistants.tools.structured_code.lsp.util import convert_keys_to_camel_case
from astra_assistants.tools.structured_code.lsp.workspace_edits import apply_workspace_edit

converter = converters.get_converter()


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


class StructuredProgramEntry(BaseModel):
    program_id: str = Field(..., description="Unique identifier for the program")
    program: StructuredProgram = Field(..., description="Structured program object")
    diagnostics: Optional[List[types.Diagnostic]] = Field(None, description="List of diagnostics for the program")
    code_action_message: Optional[str] = Field(None,
                                               description="Message describing any code action(s) applied to the program")

    class Config:
        arbitrary_types_allowed = True


class ProgramCache(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.session_manager = LspSessionManager()

    def append(self, item: StructuredProgramEntry) -> None:
        self.process(item)
        super().append(item)

    def extend(self, iterable: List[StructuredProgramEntry]) -> None:
        for item in iterable:
            self.process(item)
        super().extend(iterable)

    def insert(self, index: int, item: StructuredProgramEntry) -> None:
        self.process(item)
        super().insert(index, item)

    def close(self) -> None:
        self.session_manager.close()

    def process(self, item: StructuredProgramEntry) -> None:
        program_str = item.program.to_string(with_line_numbers=False)
        with tempfile.NamedTemporaryFile(suffix=".py") as fp:
            try:
                fp.write(program_str.encode("utf-8"))
                fp.flush()
                uri = as_uri(fp.name)

                message, diags, program_str = self.apply_code_actions(uri, program_str)

                item.program.lines = program_str.splitlines()

                item.diagnostics = diags
                item.code_action_message = message
                print(message)
            except Exception as e:
                trace = traceback.format_exc()
                print(e)

    def apply_code_actions(self, uri, program_str, document_version=1, applied=None):
        if applied is None:
            applied = []
        diags = self.get_diagnostics(uri, program_str, document_version)

        message = ""
        if len(diags) == 0:
            return message, diags, program_str

        start = None
        end = None
        for diag in diags:
            end, start = compare_and_set_ends(diag, end, start)

        diag_range = types.Range(start=start, end=end)

        code_action_params = types.CodeActionParams(
            text_document=types.TextDocumentIdentifier(uri=uri),
            range=diag_range,
            context=types.CodeActionContext(
                diagnostics=diags,
            )
        )

        code_action_params_dict = convert_keys_to_camel_case(converter.unstructure(code_action_params))
        code_actions_dict = self.session_manager.send_request(CODE_ACTION, code_action_params_dict)

        linebreak = "\n"
        if len(applied) > 0:
            message = f"applied code actions: \n{linebreak.join(applied)}"
        if len(code_actions_dict) == 0:
            return message, diags, program_str

        code_actions = []
        for code_action in code_actions_dict:
            code_action_obj = converter.structure(code_action, types.CodeAction)
            code_actions.append(code_action_obj)

        print(code_actions)
        for code_action in code_actions:
            if code_action.title not in applied:
                try:
                    result_str = apply_workspace_edit(code_actions[0].edit)
                    if result_str is not None:
                        program_str = result_str
                        applied.append(code_actions[0].title)
                        return self.apply_code_actions(uri, program_str, document_version + 1, applied)
                except Exception as e:
                    print(e)
                    raise e
        return message, diags, program_str

    def get_diagnostics(self, uri, program_str, document_version=1):
        payload = {
            "textDocument": {
                "uri": uri,
                "languageId": "python",
                "version": document_version,
                "text": program_str,
            }
        }
        notification = None
        if document_version == 1:
            notification = self.session_manager.send_notification("textDocument/didOpen", payload)
        else:

            text_change_event = types.TextDocumentContentChangeEvent_Type2(
                text=program_str,
            )
            did_change_payload_obj = types.DidChangeTextDocumentParams(
                text_document=types.VersionedTextDocumentIdentifier(uri=uri, version=document_version),
                content_changes=[
                    text_change_event,
                ],
            )
            did_change_payload_dict = convert_keys_to_camel_case(converter.unstructure(did_change_payload_obj))
            notification = self.session_manager.send_notification("textDocument/didChange", did_change_payload_dict)
        diagnostics = notification["diagnostics"]
        diags = []
        for diagnostic in diagnostics:
            diag_obj = converter.structure(diagnostic, types.Diagnostic)
            diags.append(diag_obj)
        print(diags)
        return diags


def compare_and_set_ends(diag, end, start):
    if start is None:
        start = diag.range.start
        end = diag.range.end
    else:
        if diag.range.start.line < start.line:
            start = diag.range.start
        if diag.range.end.line > end.line:
            end = diag.range.end
        if diag.range.start.line == start.line and diag.range.start.character < start.character:
            start = diag.range.start
        if diag.range.end.line == end.line and diag.range.end.character > end.character:
            end = diag.range.end
    return end, start
