from openai import OpenAI
from astra_assistants import patch
from dotenv import load_dotenv

load_dotenv('./.env')


def print_embedding(model):
    text="Draw your favorite animal."
    response = client.embeddings.create(
        model=model,
        input=[text]
    )
    print(f'text> {text}')
    print(f'embedding-model-{model}>\n{response.data[0].embedding}')
    print(f'size{len(response.data[0].embedding)}')


client = patch(OpenAI())


model = "cohere/embed-english-v3.0"
print_embedding(model)

model = "amazon.titan-embed-text-v1"
print_embedding(model)

model="text-embedding-ada-002"
print_embedding(model)

model="text-embedding-3-small"
print_embedding(model)

model="text-embedding-3-large"
print_embedding(model)