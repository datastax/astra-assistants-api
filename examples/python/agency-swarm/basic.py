from openai import OpenAI
from astra_assistants import patch
from agency_swarm import Agent, Agency, set_openai_client
from dotenv import load_dotenv

load_dotenv("./.env")
load_dotenv("../../../.env")

client = patch(OpenAI())

set_openai_client(client)

ceo = Agent(name="CEO",
            description="Responsible for client communication, task planning, and management.",
            instructions="Please communicate with users and other agents.",
            model="anthropic/claude-3-haiku-20240307",
            files_folder="./examples/python/agency-swarm/files",
            tools=[])

agency = Agency([ceo])