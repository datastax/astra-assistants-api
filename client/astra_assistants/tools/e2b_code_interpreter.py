from e2b_code_interpreter import Sandbox

from astra_assistants.tools.tool_interface import ToolInterface


class E2BCodeInterpreter(ToolInterface):

    def __init__(self):
        print("initializing code interpreter")

        running_sandboxes = Sandbox.list()
        # Find the sandbox by metadata
        for running_sandbox in running_sandboxes:
            sandbox = Sandbox.connect(running_sandbox.sandbox_id)
            sandbox.kill()
        else:
            # Sandbox not found
            pass
        self.code_interpreter = Sandbox()

        print("initialized")

    def call(self, arguments):
        code = arguments['arguments']
        execution = self.code_interpreter.run_code(code)
        return execution.text
