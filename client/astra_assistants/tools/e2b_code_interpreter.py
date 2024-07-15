from e2b import Sandbox

from e2b_code_interpreter import CodeInterpreter

from astra_assistants.tools.tool_interface import ToolInterface


class E2BCodeInterpreter(ToolInterface):

    def __init__(self):
        print("initializing code interpreter")

        running_sandboxes = Sandbox.list()
        # Find the sandbox by metadata
        for running_sandbox in running_sandboxes:
            sandbox = Sandbox.reconnect(running_sandbox.sandbox_id)
            sandbox.close()
        else:
            # Sandbox not found
            pass
        self.code_interpreter = CodeInterpreter()

        print("initialized")

    def call(self, arguments):
        code = arguments['arguments']
        exec = self.code_interpreter.notebook.exec_cell(
            code,
        )
        return exec.text