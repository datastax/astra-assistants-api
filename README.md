# Astra Assistant API Service

A drop-in compatible service for the OpenAI beta Assistants API with support for persistent threads, files, assistants, messages, and more using AstraDB (DataStax's db as a service offering powered by Apache Cassandra and jvector).

Compatible with existing OpenAI apps via the OpenAI SDKs by changing a single line of code.

## Getting Started

Simply [create an Astra DB Vector database](https://astra.datastax.com/signup) and replace:
```python
client = OpenAI(
    api_key=OPENAI_API_KEY,
)
```
with:
```python
client = OpenAI(
    base_url="https://open-assistant-ai.astra.datastax.com/v1", 
    api_key=OPENAI_API_KEY,
    default_headers={
        "astra-api-token": ASTRA_DB_APPLICATION_TOKEN,
    }
)
```

Optionally if you have an existing astra db you can pass your db_id in a second header. Otherwise the system will create one on your behalf and name it `assistant_api_db` using your token. Note, this means that the first request will hang until your db is ready (could be a couple of minutes). This will only happen once.

```python
client = OpenAI(
    base_url="https://open-assistant-ai.astra.datastax.com/v1", 
    api_key=OPENAI_API_KEY,
    default_headers={
        "astra-api-token": ASTRA_DB_APPLICATION_TOKEN,
        "astra-db-id": ASTRA_DB_ID
    }
)
```

Now you're ready to create an assistant

```
assistant = client.beta.assistants.create(
  instructions="You are a personal math tutor. When asked a math question, write and run code to answer the question.",
  model="gpt-4-1106-preview",
  tools=[{"type": "retrieval"}]
)
```

By default, the service uses [AstraDB](https://astra.datastax.com/signup) as the database/vector store and OpenAI for embeddings and chat completion.

## Third party LLM Support

We now support [many third party models](https://docs.litellm.ai/docs/providers) for both embeddings and completion thanks to [litellm](https://github.com/BerriAI/litellm). Pass the api key of your service using `api-key` and `embedding-model` headers.

```
client = OpenAI(
    base_url="https://open-assistant-ai.astra.datastax.com/v1", 
    api_key="NONE",
    default_headers={
        "astra-api-token": ASTRA_DB_APPLICATION_TOKEN,
        "api-key": COHERE_API_KEY,
        "embedding-model": "cohere/embed-english-v3.0",
    }
)
```


Note: remember to also pass your third party model to the assistant on create:

```
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
    model="cohere/command",
)
```

For AWS Bedrock you can pass additional custom headers:

```
client = OpenAI(
    base_url="https://open-assistant-ai.astra.datastax.com/v1", 
    api_key="NONE",
    default_headers={
        "astra-api-token": ASTRA_DB_APPLICATION_TOKEN,
        "embedding-model": "amazon.titan-embed-text-v1",
        "LLM-PARAM-aws-access-key-id": BEDROCK_AWS_ACCESS_KEY_ID,
        "LLM-PARAM-aws-secret-access-key": BEDROCK_AWS_SECRET_ACCESS_KEY,
        "LLM-PARAM-aws-region-name": BEDROCK_AWS_REGION,
    }
)
```

and again, specify the custom model for the assistant.

```
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
    model="meta.llama2-13b-chat-v1",
)
```

Additional examples including third party LLMs (bedrock, cohere, perplexity, etc.) can be found under `tests/examples`


To run the examples using poetry create a .env file in this directory with your secrets and run:

    poetry install

and 

    poetry run python examples/python/retreival/basic.py

## Coverage

See our coverage report [here](./coverage.md)

## Roadmap:
 - [X] Support for other embedding models and LLMs
 - [X] Function support
 - [ ] Streaming support
 - [ ] Pluggable RAG strategies

