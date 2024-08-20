import logging
from typing import List, Dict

from openai import OpenAI
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput

from astra_assistants import patch
from astra_assistants.astra_assistants_event_handler import AstraEventHandler
from astra_assistants.tools.tool_interface import ToolInterface

logger = logging.getLogger(__name__)

class AssistantManager:
    def __init__(self, instructions: str, model: str = "gpt-4o", name: str = "managed_assistant", tools: List[ToolInterface] = None, thread_id: str = None, thread: str = None, assistant_id: str = None):
        if tools is None:
            tools = []
        self.client = patch(OpenAI())
        self.model = model
        self.instructions = instructions
        self.tools = tools
        self.name = name
        self.tool_call_arguments = None

        if assistant_id is not None:
            self.assistant = self.client.beta.assistants.retrieve(assistant_id)
        else:
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

    def stream_thread(self, content, tool_choice = None, thread_id: str = None, thread = None, additional_instructions = None):
        if thread_id is not None:
            thread = self.client.beta.threads.retrieve(thread_id)
        elif thread is None:
            thread = self.thread

        assistant = self.assistant
        event_handler = AstraEventHandler(self.client)
        if self.tools is not None:
            for tool in self.tools:
                event_handler.register_tool(tool)
        if tool_choice is not None:
            if hasattr(tool_choice, 'tool_choice_object'):
                tool_choice = tool_choice.tool_choice_object()
            else:
                tool_choice = tool_choice
        try:
            self.client.beta.threads.messages.create(
                thread_id=thread.id, role="user", content=content
            )
            args = {
                "thread_id": thread.id,
                "assistant_id": assistant.id,
                "event_handler": event_handler,
                "additional_instructions": additional_instructions
            }
            # Conditionally add 'tool_choice' if it's not None
            if tool_choice is not None:
                args["tool_choice"] = tool_choice

            text = ""
            with self.client.beta.threads.runs.create_and_stream(**args) as stream:
                for text in stream.text_deltas:
                    yield text

            tool_call_results = None
            tool_call_arguments = None
            self.tool_call_arguments = event_handler.arguments
            if event_handler.stream is not None:
                if event_handler.tool_call_results is not None:
                    yield event_handler.tool_call_results
                with event_handler.stream as stream:
                    for text in stream.text_deltas:
                        yield text
        except Exception as e:
            print(e)
            raise e
        
    async def run_thread(self, content, tool = None, thread_id: str = None, thread = None, additional_instructions = None) -> ToolOutput:
        if thread_id is not None:
            thread = self.client.beta.threads.retrieve(thread_id)
        elif thread is None:
            thread = self.thread

        assistant = self.assistant
        event_handler = AstraEventHandler(self.client)
        tool_choice = None
        if tool is not None:
            event_handler.register_tool(tool)
            tool_choice = tool.tool_choice_object()
        try:
            self.client.beta.threads.messages.create(
                thread_id=thread.id, role="user", content=content
            )
            args = {
                "thread_id": thread.id,
                "assistant_id": assistant.id,
                "event_handler": event_handler,
                "additional_instructions": additional_instructions
            }
            # Conditionally add 'tool_choice' if it's not None
            if tool_choice is not None:
                args["tool_choice"] = tool_choice

            text = ""
            with self.client.beta.threads.runs.create_and_stream(**args) as stream:
                for part in stream.text_deltas:
                    text += part
                    
            tool_call_results = None
            if event_handler.stream is not None:
                with event_handler.stream as stream:
                    for part in stream.text_deltas:
                        text += part

                    tool_call_results = event_handler.tool_call_results

                    if not isinstance(tool_call_results, str) and tool_call_results is not None:
                        tool_call_results['text'] = text
                        tool_call_results['error'] = event_handler.error

                    print(tool_call_results)
                    tool_call_results
            if tool_call_results is not None:
                return tool_call_results
            return {"text": text}
        except Exception as e:
            print(e)
            raise e