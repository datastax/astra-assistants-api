import os

provider_env_var_map = {
    "openai": {
        "api_key": "OPENAI_API_KEY"
    },
    "groq": {
        "api_key": "GROQ_API_KEY",
        "astra_token": "ASTRA_DB_APPLICATION_TOKEN"
    },
    "anthropic": {
        "api_key": "ANTHROPIC_API_KEY",
        "astra_token": "ASTRA_DB_APPLICATION_TOKEN"
    },
    "gemini": {
        "api_key": "GEMINI_API_KEY",
        "astra_token": "ASTRA_DB_APPLICATION_TOKEN"
    },
    "perplexity": {
        "api_key": "PERPLEXITYAI_API_KEY",
        "astra_token": "ASTRA_DB_APPLICATION_TOKEN"
    },
    "cohere": {
        "api_key": "COHERE_API_KEY",
        "astra_token": "ASTRA_DB_APPLICATION_TOKEN"
    },
    "bedrock": {
        "region": "AWS_REGION_NAME",
        "access_key_id": "AWS_ACCESS_KEY_ID",
        "secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "astra_token": "ASTRA_DB_APPLICATION_TOKEN"
    },
    "other": {
        "model":"model",
        "api_key": "api_key",
        "astra_token": "ASTRA_DB_APPLICATION_TOKEN"
    }
}

def get_env_vars_for_provider(provider: str) -> dict:
    """
    Returns the correct environment variable names for the specified provider.
    If multiple environment variables are required, it returns them in a dictionary.
    """
    return provider_env_var_map.get(provider.lower(), {})

def env_var_is_missing(provider: str, env_vars: dict) -> bool:
    for _, v in env_vars.items():
        secret = os.getenv(v)
        if secret is None:
            return True
    return False


def copy_program_from_cache(program_id, program_cache):
    if program_id is None:
        raise Exception("You must call set_program_id() before calling call()")
    try:
        program = program_cache.get(program_id).program.copy()
        return program
    except Exception as e:
        print(f"program_id {program_id} not found in cache: Error: {e}")
        raise e