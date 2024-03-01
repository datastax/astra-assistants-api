import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../../.env")

logger = logging.getLogger(__name__)

# you do have to pass a key because the client requires it but it doesn't have to be valid since we're using a third party LLM
OPENAI_API_KEY="Fake key"
ASTRA_DB_TOKEN=os.getenv("ASTRA_DB_TOKEN")
GOOGLE_JSON_PATH=os.getenv("GOOGLE_JSON_PATH")
GOOGLE_PROJECT_ID=os.getenv("GOOGLE_PROJECT_ID")

#base_url="http://127.0.0.1:8000/v1"
#base_url="https://open-assistant-ai.astra.datastax.com/v1"
base_url="https://open-assistant-ai.dev.cloud.datastax.com/v1"

client = OpenAI(
    base_url=base_url,
    api_key=OPENAI_API_KEY,
    default_headers={
        "astra-api-token": ASTRA_DB_TOKEN,
        "embedding-model": "textembedding-gecko@002",
        "VERTEXAI-PROJECT": GOOGLE_PROJECT_ID,
    }
)

logger.info("Uploading file:")

# Upload the file
file = client.files.create(
    file=open(
        GOOGLE_JSON_PATH,
        "rb",
    ),
    purpose="auth",
)

logger.info(file.id)

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

model="chat-bison"

prompt = "Draw an ASCII art kitten eating icecream"
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an amazing ascii art generator bot, no text just art."},
        {"role": "user", "content": prompt}
    ]
)

logger.info(f'prompt> {prompt}')
logger.info(f'artist-{model}>\n{response.choices[0].message.content}')
