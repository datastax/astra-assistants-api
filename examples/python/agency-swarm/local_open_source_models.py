from openai import OpenAI
from astra_assistants import patch
from agency_swarm import Agent, Agency, set_openai_client
from dotenv import load_dotenv

load_dotenv("./.env")
load_dotenv("../../../.env")

client = patch(OpenAI(default_headers={"LLM-PARAM-base-url": "http://localhost:11434"}))
# if using docker-compose, pass custom header to point to the ollama container instead of localhost
# client = patch(OpenAI(default_headers={"LLM-PARAM-base-url": "http://ollama:11434"}))

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