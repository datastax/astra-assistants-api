from openai import OpenAI
from astra_assistants import patch
from agency_swarm import Agent, Agency, set_openai_client
from dotenv import load_dotenv

load_dotenv("./.env")
load_dotenv("../../../.env")

client = patch(OpenAI())
#client = OpenAI()

set_openai_client(client)

ceo = Agent(name="CEO",
            description="Responsible for client communication, task planning, and management.",
            instructions="Please communicate with users and other agents.",
            model="anthropic/claude-3-haiku-20240307",
            #model="gpt-3.5-turbo",
            files_folder="./examples/python/agency-swarm/files",
            tools=[])

agency = Agency([ceo])

assistant = client.beta.assistants.retrieve(ceo.id)
print(assistant)

completion = agency.get_completion("What's something interesting about language models?")
print(completion)