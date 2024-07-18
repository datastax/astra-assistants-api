from openai import OpenAI
from astra_assistants import patch
from agency_swarm import Agent, Agency, set_openai_client
from dotenv import load_dotenv

load_dotenv("./.env")
load_dotenv("../../../.env")

# remember to set OLLAMA_API_BASE_URL="http://ollama:11434" and base_url="http://localhost:8000/v1" in your env
client = patch(OpenAI())

set_openai_client(client)

ceo = Agent(name="CEO",
            description="Responsible for client communication, task planning, and management.",
            instructions="Please communicate with users and other agents.",
            model="ollama_chat/deepseek-coder-v2", # ensure that the model has been pulled in ollama
            files_folder="./examples/python/agency-swarm/files",
            tools=[])

agency = Agency([ceo])

assistant = client.beta.assistants.retrieve(ceo.id)
print(assistant)

completion = agency.get_completion("What's something interesting about language models?")
print(completion)
