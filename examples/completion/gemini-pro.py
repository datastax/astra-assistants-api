import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("./.env")


# you do have to pass a key because the client requires it but it doesn't have to be valid since we're using a third party LLM
OPENAI_API_KEY="Fake key"
ASTRA_DB_TOKEN=os.getenv("ASTRA_DB_TOKEN")

GOOGLE_JSON_PATH=os.getenv("GOOGLE_JSON_PATH")
GOOGLE_PROJECT_ID=os.getenv("GOOGLE_PROJECT_ID")

base_url=os.getenv("base_url", "https://open-assistant-ai.astra.datastax.com/v1")
print(base_url)

client = OpenAI(
    base_url=base_url,
    api_key=OPENAI_API_KEY,
    default_headers={
        "astra-api-token": ASTRA_DB_TOKEN,
        "embedding-model": "textembedding-gecko@002",
        "VERTEXAI-PROJECT": GOOGLE_PROJECT_ID,
    }
)

print("Uploading file:")

# Upload the JSON auth file
# this will get stored in your astradb as plain text, make sure your db adequately secured.
# you only need to do this once, then you can save your file.id and use it to connect your openai client
file = client.files.create(
    file=open(
        GOOGLE_JSON_PATH,
        "rb",
    ),
    purpose="auth",
)

print(file.id)

del client

client = OpenAI(
    base_url=base_url,
    api_key=OPENAI_API_KEY,
    default_headers={
        "astra-api-token": ASTRA_DB_TOKEN,
        "embedding-model": "textembedding-gecko@002",
        "VERTEXAI-PROJECT": GOOGLE_PROJECT_ID,
        "google-application-credentials-file-id": file.id
    }
)

model="gemini-pro"

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

