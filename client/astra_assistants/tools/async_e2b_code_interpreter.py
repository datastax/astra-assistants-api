
import time
from astra_assistants.tools.tool_interface import ToolInterface
from e2b_code_interpreter import AsyncCodeInterpreter

class AsyncE2BCodeInterpreter(ToolInterface):
    @staticmethod
    async def create():
        print("+ Initializing async code interpreter")
        running_sandboxes = await AsyncCodeInterpreter.list()

        if running_sandboxes:
            print("+ Running sandbox found, will connect...")
            sandbox_id = running_sandboxes[0].sandbox_id
            async_code_interpreter = await AsyncCodeInterpreter.connect(sandbox_id)
            print("+ Connected to the running sandbox")

            return AsyncE2BCodeInterpreter(async_code_interpreter)
        else:
            async_code_interpreter = await AsyncCodeInterpreter.create()
            return AsyncE2BCodeInterpreter(async_code_interpreter)

    def __init__(self, async_code_interpreter: AsyncCodeInterpreter) -> None:
        self.code_interpreter = async_code_interpreter

    def call(self, arguments):
        raise NotImplementedError("Sync call isn't supported for this async tool")

    # TODO: Async version of the call
    # async def acall(self, arguments):
    #     code = arguments['arguments']
    #     print("+ Executing code inside the code interpreter sandbox...")
    #     start_time = time.time()
    #     exec = await self.code_interpreter.notebook.exec_cell(
    #         code,
    #     )
    #     end_time = time.time()
    #     print(f"+ Code executed, it took: {end_time - start_time} seconds")
    #     return exec.text