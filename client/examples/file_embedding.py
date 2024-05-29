from dotenv import load_dotenv
from openai import OpenAI

from streaming_assistants import patch

load_dotenv("./.env")

client = patch(OpenAI())

file = client.files.create(
    file=open(
        "./tests/language_models_are_unsupervised_multitask_learners.pdf",
        "rb",
    ),
    purpose="assistants",
    embedding_model="text-embedding-3-large",
)