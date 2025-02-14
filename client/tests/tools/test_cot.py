import json
import os

import pytest
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput

from astra_assistants.astra_assistants_event_handler import AstraEventHandler
import logging
from astra_assistants.astra_assistants_manager import AssistantManager
from astra_assistants.tools.cot import ChainOfThoughtTool, ChainOfThought

logger = logging.getLogger(__name__)


def test_cot():
    # use OpenAI client instead of patched OpenAI
    del os.environ["ASTRA_DB_APPLICATION_TOKEN"]

    cot = ChainOfThoughtTool()

    assistant_manager = AssistantManager(
        name="Thoughtful reasoner",
        instructions="""
use the chain of thought tool to answer questions.
""",
        model="gpt-4o-2024-08-06",
        tools=[cot],
    )

    text = ""
    is_not_complete = True
    while is_not_complete:
        chunks: ToolOutput = assistant_manager.stream_thread(
            content="How many r's in the word strawberry",
            tool_choice=cot
        )
        chunk = next(chunks)
        assert not isinstance(chunk, str), "Expected a dict"
        tool_call_results_dict = chunk["output"]
        print(f"tool_call_results_dict: {tool_call_results_dict}")
        is_not_complete = not chunk["cot"].is_complete
        text = ""
        for chunk in chunks:
            print(chunk, end="", flush=True)
            text += chunk
        print("WIP:\n" + text)

    print("Answer:\n" + text)


def test_get_string():
    test = {"thoughts":"I need to count the number of 'r' characters in the word \"strawberry\".","doubts":["What is the word we are analyzing?","How many 'r's are there in \"strawberry\"?","Is every instance of 'r' included in the count?","Are there any variations of 'r' that need to be considered?","Is the spelling of \"strawberry\" confirmed and correct?"],"potential_answers":["2"],"is_complete":False}
    string = ChainOfThought(**test).to_string()
    print(string)