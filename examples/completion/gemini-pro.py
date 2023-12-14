import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("./.env")


# you do have to pass a key because the client requires it but it doesn't have to be valid since we're using a third party LLM
OPENAI_API_KEY="Fake key"
ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")

GOOGLE_PROJECT_ID=os.getenv("GOOGLE_PROJECT_ID")

# get a vertexai api key here https://ai.google.dev/tutorials/setup
VERTEXAI_API_KEY=os.getenv("VERTEXAI_API_KEY")

base_url=os.getenv("base_url", "https://open-assistant-ai.astra.datastax.com/v1")
print(base_url)

client = OpenAI(
    base_url=base_url,
    api_key=OPENAI_API_KEY,
    default_headers={
        "astra-api-token": ASTRA_DB_APPLICATION_TOKEN,
        "VERTEXAI-PROJECT": GOOGLE_PROJECT_ID,
        "api-key" : VERTEXAI_API_KEY,
    }
)

model="palm/gemini-pro"

prompt = "Hi what is your favorite ice cream"
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a friendly assistant."},
        {"role": "user", "content": prompt}
    ]
)

print(f'prompt> {prompt}')
print(f'artist-{model}>\n{response.choices[0].message.content}')

