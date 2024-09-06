# we now provide OpenAI and AsyncOpenAI classes from the astra_assistants package
# to get around having to provide OPENAI_API_KEY
from astra_assistants import OpenAI, patch, AsyncOpenAI

print(OpenAI.__name__)
client = patch(OpenAI())
async_client = patch(AsyncOpenAI())
