silicon -o environment_vars.png -l bash -f 'Hack; DejaVu Sans Mono=31' --background '#4f2a83' --shadow-color '#555' --shadow-blur-radius 30 <<EOF
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

# ... for all models see the .env.bkp file
EOF
