import logging
from typing import List, Dict

from openai import OpenAI
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput

from astra_assistants import patch
from astra_assistants.astra_assistants_event_handler import AstraEventHandler
from astra_assistants.tools.tool_interface import ToolInterface

logger = logging.getLogger(__name__)

class AssistantManager:
    def __init__(self, instructions: str, tools: List[ToolInterface], model: str = "gpt-4o", name: str = "managed_assistant", thread_id: str = None, thread: str = None):

        self.client = patch(OpenAI())
        self.model = model
        self.instructions = instructions
        self.tools = tools
        self.name = name
        self.additional_instructions = None

        self.assistant = self.create_assistant()

        if thread_id is None and thread is None:
            self.thread = self.create_thread()
        elif thread is not None:
            self.thread = thread
        elif thread_id is not None:
            self.thread = self.client.beta.threads.retrieve(thread_id)

        print(f'assistant {self.assistant}')
        print(f'thread {self.thread}')

    def get_client(self):
        return self.client

    def get_assistant(self):
        return self.assistant

    def get_tool(self):
        return self.tool

    def create_assistant(self):
        tool_functions = []
        for tool in self.tools:
            tool_functions.append(tool.to_function())

        # Create and return the assistant
        self.assistant = self.client.beta.assistants.create(
            name=self.name,
            instructions=self.instructions,
            model=self.model,
            tools=tool_functions
        )
        print("Assistant created:", self.assistant)
        return self.assistant

    def create_thread(self):
        # Create and return a new thread
        thread = self.client.beta.threads.create()
        print("Thread generated:", thread)
        return thread

    async def run_thread(self, tool, content, thread_id: str = None, thread = None) -> ToolOutput:
        if thread_id is not None:
            thread = self.client.beta.threads.retrieve(thread_id)
        elif thread is None:
            thread = self.thread

        assistant = self.assistant
        event_handler = AstraEventHandler(self.client)
        event_handler.register_tool(tool)
        try:
            self.client.beta.threads.messages.create(
                thread_id=thread.id, role="user", content=content
            )
            with self.client.beta.threads.runs.create_and_stream(
                    thread_id=thread.id,
                    assistant_id=assistant.id,
                    event_handler=event_handler,
                    tool_choice=tool.tool_choice_object(),
                    additional_instructions=self.additional_instructions
            ) as stream:
                for part in stream:
                    pass
            text = ""
            with event_handler.stream as stream:
                for part in stream.text_deltas:
                    text += part

                tool_call_results = event_handler.tool_call_results

                if not isinstance(tool_call_results, str) and tool_call_results is not None:
                    tool_call_results['text'] = text
                    tool_call_results['error'] = event_handler.error

                print(tool_call_results)
                return tool_call_results
        except Exception as e:
            print(e)
            raise e