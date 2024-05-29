# astra-assistants

Astra Assistants is a drop in replacement for OpenAI's assistant API that supports the full Assistants v2 API (including streaming and vector_stores). This python library wraps the OpenAI SDK with additional capabilities and provides syntactic sugar for passing credentials for third party LLMs.

# How to use    

Install astra_assistants using your python package manager of choice:

```
poetry add astra_assistants
```


import and patch your client:

```
from openai import OpenAI
from astra_assistants import patch

client = patch(OpenAI())
```

# Server

The astra-assistants server code is now open source (Apache2)! 

Check it out here https://github.com/datastax/astra-assistants-api

# Authentication

Provide api keys for third party LLMs via environment variables. We support LLM completions through [litellm](https://github.com/BerriAI/litellm) and support litellm environmental variables and models.

Rename the [.env.bkp](./.env.bkp) to `.env` and fill in the appropriate values for the LLMs you want to use.

```
#!/bin/bash

# AstraDB -> https://astra.datastax.com/ --> tokens --> administrator user --> generate
export ASTRA_DB_APPLICATION_TOKEN=""

# OpenAI Models - https://platform.openai.com/api-keys --> create new secret key
export OPENAI_API_KEY=""

# Groq Models - https://console.groq.com/keys
export GROQ_API_KEY=""

# Anthropic claude models - https://console.anthropic.com/settings/keys
export ANTHROPIC_API_KEY=""

# Gemini models -> https://makersuite.google.com/app/apikey
export GEMINI_API_KEY=""

# Perplexity models -> https://www.perplexity.ai/settings/api  --> generate
export PERPLEXITYAI_API_KEY=""

# Cohere models -> https://dashboard.cohere.com/api-keys
export COHERE_API_KEY=""

# Bedrock models -> https://docs.aws.amazon.com/bedrock/latest/userguide/setting-up.html
export AWS_REGION_NAME=""
export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""

# vertexai models https://console.cloud.google.com/vertex-ai
export GOOGLE_JSON_PATH=""
export GOOGLE_PROJECT_ID=""

# AI21 models
export AI21_API_KEY=""

# Aleph Alpha models
export ALEPHALPHA_API_KEY=""

# Anyscale models
export ANYSCALE_API_KEY=""

# Azure models
export AZURE_API_KEY=""
export AZURE_API_BASE=""
export AZURE_API_VERSION=""
export AZURE_AD_TOKEN=""
export AZURE_API_TYPE=""

# Baseten models
export BASETEN_API_KEY=""

# Cloudflare Workers models
export CLOUDFLARE_API_KEY=""
export CLOUDFLARE_ACCOUNT_ID=""

# DeepInfra models
export DEEPINFRA_API_KEY=""

# DeepSeek models
export DEEPSEEK_API_KEY=""

# Fireworks AI models
export FIREWORKS_AI_API_KEY=""

# Hugging Face models
export HUGGINGFACE_API_KEY=""
export HUGGINGFACE_API_BASE=""

# Mistral models
export MISTRAL_API_KEY=""

# NLP Cloud models
export NLP_CLOUD_API_KEY=""

# OpenAI models
export OPENAI_API_KEY=""
export OPENAI_ORGANIZATION=""
export OPENAI_API_BASE=""

# OpenRouter models
export OPENROUTER_API_KEY=""
export OR_SITE_URL=""
export OR_APP_NAME=""

# PaLM models
export PALM_API_KEY=""

# Replicate models
export REPLICATE_API_KEY=""

# TogetherAI models
export TOGETHERAI_API_KEY=""

# Vertex AI models
export VERTEXAI_PROJECT=""
export VERTEXAI_LOCATION=""
export GOOGLE_APPLICATION_CREDENTIALS=""

# Voyage models
export VOYAGE_API_KEY=""

# WatsonX models
export WATSONX_URL=""
export WATSONX_APIKEY=""
export WATSONX_TOKEN=""
export WATSONX_PROJECT_ID=""
export WATSONX_DEPLOYMENT_SPACE_ID=""

# XInference models
export XINFERENCE_API_BASE=""
export XINFERENCE_API_KEY=""
```
