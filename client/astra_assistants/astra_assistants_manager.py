import logging
import os
import uuid
from typing import List, Any

from litellm import get_llm_provider

from astra_assistants import patch, OpenAIWithDefaultKey
from astra_assistants.astra_assistants_event_handler import AstraEventHandler
from astra_assistants.tools.tool_interface import ToolInterface
from astra_assistants.utils import env_var_is_missing, get_env_vars_for_provider
from astra_assistants.mcp_openai_adapter import MCPOpenAIAAdapter, MCPRepresentation

logger = logging.getLogger(__name__)

class AssistantManager:
    def __init__(self,
                 instructions: str = None,
                 model: str = "gpt-4o",
                 name: str = "managed_assistant",
                 tools: List[ToolInterface] = None,
                 thread_id: str = None,
                 thread: Any = None,
                 assistant_id: str = None,
                 client = None,
                 tool_resources = None,
                 mcp_represenations: MCPRepresentation = None
                 ):

        if instructions is None and assistant_id is None:
            raise Exception("Instructions must be provided if assistant_id is not provided")
        if tools is None:
            tools = []

        self.tools = tools

        # Initialize client using the provided client or the default based on environment tokens.
        if client is not None:
            self.client = client
        else:
            if os.getenv("ASTRA_DB_APPLICATION_TOKEN") is not None:
                self.client = patch(OpenAIWithDefaultKey())
            else:
                provider = get_llm_provider(model)[1]
                env_vars = get_env_vars_for_provider(provider)
                if env_var_is_missing(provider, env_vars):
                    raise Exception(f"Missing environment variables {env_vars}")
                self.client = OpenAIWithDefaultKey()
        self.model = model
        self.instructions = instructions
        self.tool_resources = tool_resources
        self.name = name
        self.tool_call_arguments = None

        if assistant_id is not None:
            self.assistant = self.client.beta.assistants.retrieve(assistant_id)
        else:
            self.assistant = self.create_assistant()

        # Instead of auto-creating the thread, we assign _thread based on input.
        if thread is not None:
            self._thread = thread
        elif thread_id is not None:
            self._thread = self.client.beta.threads.retrieve(thread_id)
        else:
            self._thread = None  # Lazy initialization; will be created on first access.

        self.mcp_adapter = None
        self.register_mcp(mcp_represenations)

        logger.info(f'assistant {self.assistant}')
        logger.info(f'thread {self._thread}')

    @property
    def thread(self):
        # Lazy creation of thread when accessed directly.
        if self._thread is None:
            self._thread = self.create_thread()
        return self._thread

    def register_mcp(self, mcp_representations):
        # If MCP representations are provided, convert them to tools using the adapter.
        if mcp_representations is not None:
            self.mcp_adapter = MCPOpenAIAAdapter(mcp_representations)

            mcp_tools = self.mcp_adapter.get_tools()
            self.tools.extend(mcp_tools)

            schemas = self.mcp_adapter.get_json_schema_for_tools()
            assistant = self.client.beta.assistants.update(assistant_id=self.assistant.id, tools=schemas)
            self.assistant = assistant

    def get_client(self):
        return self.client

    def get_assistant(self):
        return self.assistant

    def get_tool(self):
        return self.tool

    def create_assistant(self):
        tool_holder = []
        for tool in self.tools:
            if hasattr(tool, 'to_function'):
                tool_holder.append(tool.to_function())
        if len(tool_holder) == 0:
            tool_holder = self.tools

        # Create and return the assistant with the combined tool definitions.
        self.assistant = self.client.beta.assistants.create(
            name=self.name,
            instructions=self.instructions,
            model=self.model,
            tools=tool_holder,
            tool_resources=self.tool_resources
        )
        logger.debug("Assistant created: %s", self.assistant)
        return self.assistant

    def create_thread(self):
        # Create and return a new thread.
        thread = self.client.beta.threads.create()
        logger.debug("Thread generated: %s", thread)
        return thread

    def stream_thread(self, content, tool_choice = None, thread_id: str = None, thread: Any = None, additional_instructions = None):
        if thread_id is not None:
            thread = self.client.beta.threads.retrieve(thread_id)
        elif thread is None:
            # Use lazy property access; if not set, it will create one.
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
            if tool_choice is not None:
                args["tool_choice"] = tool_choice

            text = ""
            with self.client.beta.threads.runs.create_and_stream(**args) as stream:
                for text in stream.text_deltas:
                    yield text

            self.tool_call_arguments = event_handler.arguments
            if event_handler.stream is not None:
                if event_handler.tool_call_results is not None:
                    yield event_handler.tool_call_results
                with event_handler.stream as stream:
                    for text in stream.text_deltas:
                        yield text
        except Exception as e:
            logger.error(e)
            raise e

    async def run_thread(self, content, tool = None, thread_id: str = None, thread: Any = None, additional_instructions = None):
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

        for tool in self.tools:
            event_handler.register_tool(tool)

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
            if tool_choice is not None:
                args["tool_choice"] = tool_choice

            text = ""
            with self.client.beta.threads.runs.stream(**args) as stream:
                for part in stream.text_deltas:
                    text += part

            tool_call_results = None
            if event_handler.stream is not None:
                with event_handler.stream as stream:
                    for part in stream.text_deltas:
                        text += part

                    tool_call_results = event_handler.tool_call_results
                    if tool_call_results is not None:
                        file_search = event_handler.file_search

                        tool_call_results['file_search'] = file_search
                        tool_call_results['text'] = text
                        tool_call_results['arguments'] = event_handler.arguments
                    else:
                        print("event_handler.stream is not None but tool_call_results is None, bug?")

                    logger.info(tool_call_results)
            if tool_call_results is not None:
                return tool_call_results
            return {"text": text, "file_search": event_handler.file_search}
        except Exception as e:
            logger.error(e)
            raise e

    def shutdown(self):
        if self.mcp_adapter:
            self.mcp_adapter.shutdown()
