# Astra Assistant API Service

A drop-in compatible service for the OpenAI beta Assistants API with support for persistent threads, files, assistants, messages, retreival, function calling and more using AstraDB (DataStax's db as a service offering powered by Apache Cassandra and jvector).

Compatible with existing OpenAI apps via the OpenAI SDKs by changing a single line of code.

## Getting Started

Install the [streaming-assistants](https://github.com/phact/streaming-assistants) dependency with your favorite package manager:

```
poetry add streaming_assistants
```

[Signup for Astra and get an Admin API token](https://astra.datastax.com/signup):

Set your environment variables (depending on what LLMs you want to use), see the [.env.bkp](./.env.bkp) file for an example:

```
#!/bin/bash

# astra has a generous free tier, no cc required 
# https://astra.datastax.com/ --> tokens --> administrator user --> generate
export ASTRA_DB_APPLICATION_TOKEN=
# https://platform.openai.com/api-keys --> create new secret key
export OPENAI_API_KEY=

# https://www.perplexity.ai/settings/api  --> generate
export PERPLEXITYAI_API_KEY=

# https://dashboard.cohere.com/api-keys
export COHERE_API_KEY=

#bedrock models https://docs.aws.amazon.com/bedrock/latest/userguide/setting-up.html
export AWS_REGION_NAME=
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=

#vertexai models https://console.cloud.google.com/vertex-ai
export GOOGLE_JSON_PATH=
export GOOGLE_PROJECT_ID=

#gemini api https://makersuite.google.com/app/apikey
export GEMINI_API_KEY=
```

Then import and patch your client:

```python
from openai import OpenAI
from streaming_assistants import patch
client = patch(OpenAI())
```
The system will create a db on your behalf and name it `assistant_api_db` using your token. Note, this means that the first request will hang until your db is ready (could be a couple of minutes). This will only happen once.

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

You can pass different models, just make sure you have the right corresponding api key in your environment.

```
model="gpt-4-1106-preview"
#model="gpt-3.5-turbo"
#model="cohere/command"
#model="perplexity/mixtral-8x7b-instruct"
#model="perplexity/pplx-70b-online"
#model="anthropic.claude-v2"
#model="gemini/gemini-pro"
#model = "meta.llama2-13b-chat-v1"

assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
    model=model,
)
```

for third party embedding models we support `embedding_model` in `client.files.create`:
```
file = client.files.create(
    file=open(
        "./test/language_models_are_unsupervised_multitask_learners.pdf",
        "rb",
    ),
    purpose="assistants",
    embedding_model="text-embedding-3-large",
)
```

To run the examples using poetry create a .env file in this directory with your secrets and run:

    poetry install

Create your .env file and add your keys to it:

    cp .env.bkp .env

and 

    poetry run python examples/python/chat_completion/basic.py

    poetry run python examples/python/retrieval/basic.py

    poetry run python examples/python/streaming_retrieval/basic.py

    poetry run python examples/python/function_calling/basic.py


## Coverage

See our coverage report [here](./coverage.md)

## Roadmap:
 - [X] Support for other embedding models and LLMs
 - [X] function calling
 - [X] Streaming support
 - [ ] Pluggable RAG strategies
