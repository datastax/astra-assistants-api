import json
import logging
from openai.lib.streaming import AssistantEventHandler
from typing_extensions import override
from pydantic import BaseModel


class AstraEventHandler(AssistantEventHandler):
    def __init__(self, client):
        super().__init__()  # Initialize the base class
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.tools = {}
        self.tool_outputs = []
        self.stream = None

    def register_tool(self, tool):
        self.tools[tool.to_function()['function']['name']] = tool

    @override
    def on_tool_call_done(self, tool_call):
        self.logger.info(tool_call)
        self.logger.info(f'arguments: {tool_call.function.arguments}')
        results = self.run_tool(tool_call)
        self.tool_outputs.append({
            'tool_call_id': tool_call.id,
            'output': results
        })

        self.stream = self.client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self._AssistantEventHandler__current_run.thread_id,
            run_id=self._AssistantEventHandler__current_run.id,
            tool_outputs=self.tool_outputs,
            event_handler=AssistantEventHandler()
        )
        print("got the stream")

    def run_tool(self, tool_call):
        tool_name = tool_call.function.name
        if tool_name in self.tools:
            try:
                tool = self.tools[tool_name]
                arguments = json.loads(tool_call.function.arguments)
                model: BaseModel = tool.get_model()
                if issubclass(model , BaseModel):
                    arguments = model(**arguments)
                results = tool.call(arguments)
                return results
            except Exception as e:
                self.logger.error(f"Error running tool {tool_name}: {e}")
                raise e
        else:
            self.logger.error(f"Tool {tool_name} not found.")
            return None