import asyncio
import os
import threading
import re
from abc import ABC
from contextlib import AsyncExitStack
from typing import List, Union, Optional, Literal, Dict, Any, Type

from mcp.types import CallToolResult
from pydantic import BaseModel, Field, create_model

# Import the high‐level MCP client interfaces from the official SDK.
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from astra_assistants.tools.tool_interface import ToolInterface, ToolResult

# --- MCP Representation Models ---

class MCPRepresentationBase(BaseModel):
    type: str

class MCPRepresentationStdio(MCPRepresentationBase):
    type: str = Literal["stdio"]
    command: str
    arguments: Optional[List[str]] = None
    env_vars: Optional[List[str]] = None

class MCPRepresentationSSE(MCPRepresentationBase):
    type: str = Literal["sse"]
    sse_url: str

MCPRepresentation = Union[MCPRepresentationStdio, MCPRepresentationSSE]

# --- Helper functions ---

def generate_pydantic_model_from_schema(schema: Dict[str, Any], model_name: str = "DynamicModel") -> Type[BaseModel]:
    fields = {}
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict
    }
    for field_name, field_schema in properties.items():
        field_type = type_mapping.get(field_schema.get("type"), Any)
        if field_name in required_fields:
            fields[field_name] = (field_type, ...)
        else:
            fields[field_name] = (field_type, None)
    return create_model(model_name, **fields)

def to_camel_case(s: str) -> str:
    return ''.join(word.capitalize() for word in re.split(r'[_-]', s))

# --- MCP Tool Adapter (implements ToolInterface) ---

class MCPToolAdapter(ToolInterface, ABC):
    def __init__(self, representation: MCPRepresentation, mcp_session: ClientSession, mcp_tool):
        self.representation = representation
        self.mcp_session = mcp_session
        self.mcp_tool = mcp_tool

    def get_model(self):
        return generate_pydantic_model_from_schema(
            self.mcp_tool.inputSchema,
            to_camel_case(self.mcp_tool.name)
        )

    def to_function(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.mcp_tool.name,
                "description": self.mcp_tool.description,
                "parameters": self.mcp_tool.inputSchema
            }
        }

    def call(self, arguments: BaseModel) -> CallToolResult:
        # Use the background loop to run the async call synchronously.
        future = asyncio.run_coroutine_threadsafe(
            self.mcp_session.call_tool(
                self.mcp_tool.name,
                arguments=arguments.model_dump()
            ),
            self.mcp_session_loop  # set below when session is created
        )
        return {"output": future.result().content[0].text}

# --- MCP OpenAI Adapter ---

class MCPOpenAIAAdapter:
    """
    This adapter connects to an MCP server using the official Python SDK (via stdio transport)
    on a dedicated background thread. This allows synchronous methods (like call) to schedule
    async work via asyncio.run_coroutine_threadsafe.
    """
    def __init__(
            self,
            mcp_representations: List[MCPRepresentation] = None,
    ):
        self.exit_stack = AsyncExitStack()
        self.mcp_representations = mcp_representations or []
        self.server_params = []
        for rep in self.mcp_representations:
            if rep.type == 'stdio':
                env_vars = {"PATH": os.environ["PATH"]}
                if rep.env_vars is not None:
                    # Assume env_vars are provided as a dict-like mapping or as "KEY=VALUE" strings.
                    for var in rep.env_vars:
                        if "=" in var:
                            key, value = var.split("=", 1)
                            env_vars[key] = value
                # Split command into executable and arguments.
                parts = rep.command.split()
                executable = parts[0]
                initial_args = parts[1:]
                combined_args = initial_args + (rep.arguments or [])
                server_param = StdioServerParameters(
                    command=executable,
                    args=combined_args,
                    env=env_vars,
                )
                self.server_params.append(server_param)
            elif rep.type == 'sse':
                self.server_params.append(rep.sse_url)
        self.session: Optional[ClientSession] = None
        self.tools: List[MCPToolAdapter] = []
        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(target=self._run_bg_loop, daemon=True)
        self._bg_thread.start()

    def _run_bg_loop(self):
        asyncio.set_event_loop(self._bg_loop)
        self._bg_loop.run_forever()

    def sync_connect(self):
        """
        Synchronously connect to the MCP server using the background loop.
        This schedules the async connect() coroutine on the background loop.
        """
        for server_param in self.server_params:
            asyncio.run_coroutine_threadsafe(self._connect(server_param), self._bg_loop).result()

    async def _connect(self, server_param):
        transport = await self.exit_stack.enter_async_context(stdio_client(server_param))
        self.stdio, self.write = transport
        # Create the session on the background loop.
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        # Attach the background loop reference to each tool adapter.
        result = await self.session.list_tools()
        for rep in self.mcp_representations:
            for tool in result.tools:
                adapter = MCPToolAdapter(representation=rep, mcp_session=self.session, mcp_tool=tool)
                # Set the event loop used by this session (i.e. the background loop)
                adapter.mcp_session_loop = self._bg_loop
                self.tools.append(adapter)

    def get_tools(self) -> List[MCPToolAdapter]:
        if self.session is None:
            self.sync_connect()
        return self.tools

    def get_json_schema_for_tools(self) -> List[dict]:
        # Since to_function() is synchronous, simply return the schemas.
        return [tool_adapter.to_function() for tool_adapter in self.tools]

    def shutdown(self):
        """
        Cleanly shuts down the background loop and thread.
        """
        # First, if session exists, schedule exit of the exit stack.
        if self.session is not None:
            future = asyncio.run_coroutine_threadsafe(self.exit_stack.aclose(), self._bg_loop)
            try:
                future.result(timeout=5)
            except Exception as e:
                print("Error during exit_stack.aclose():", e)
            self.session = None
        # Signal the background loop to stop.
        self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        # Wait for the background thread to finish.
        self._bg_thread.join(timeout=5)
        # Close the loop.
        self._bg_loop.close()