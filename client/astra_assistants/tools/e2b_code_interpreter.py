import time
from astra_assistants.tools.tool_interface import ToolInterface
from e2b_code_interpreter import CodeInterpreter
class E2BCodeInterpreter(ToolInterface):

    def __init__(self):
        print("+ Initializing code interpreter")

        running_sandboxes = CodeInterpreter.list()
        # Find the sandbox by metadata
        # for running_sandbox in running_sandboxes:
        # Note: You might have multiple running sandboxes so it's probably better to just take the first one (for dev purposes only)
        # or use metadata to find the right one
        #
        # The medata would for example be a user ID that you pass during the creation of the sandbox
        # CodeIntepreter(metadata=...)
        #
        # For now, we'll just take the first running sandbox
        if running_sandboxes:
            running_sandbox = running_sandboxes[0]

            print("+ Running sandbox found, will connect...")
            sandbox = CodeInterpreter.connect(running_sandbox.sandbox_id)
            print("+ Connected to the running sandbox")

            # Reset the timer and keep sandbox alive for 300 seconds
            sandbox.set_timeout(300)
            self.code_interpreter = sandbox
        else:
            print("+ No running sandbox, will create a new one...")
            start_time = time.time()
            # By default the sandbox stays alive for 300 seconds
            self.code_interpreter = CodeInterpreter()
            end_time = time.time()
            print(f"+ Time taken to create the code interpreter sandbox: {end_time - start_time} seconds")

        print("+ Code interpreter sandbox initialized")

    def call(self, arguments):
        code = arguments['arguments']
        print("+ Executing code inside the code interpreter sandbox...")
        start_time = time.time()
        exec = self.code_interpreter.notebook.exec_cell(
            code,
        )
        end_time = time.time()
        print(f"+ Code executed, it took: {end_time - start_time} seconds")
        return exec.text


