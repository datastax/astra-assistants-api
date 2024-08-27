import logging

import pytest

logger = logging.getLogger(__name__)

def print_chat_completion(model, client):
    prompt="provide the weather in ny today in json format, it's 75 degrees F and sunny"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an amazing json generator."},
            {"role": "user", "content": prompt}
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }],
        tool_choice={'type': 'function', 'function': {'name': 'get_current_weather'}},
    )
    assert len(response.choices[0].message.tool_calls) > 0
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an amazing json generator."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
    )

    logger.info(f'prompt> {prompt}')
    logger.info(f'artist-{model}>\n{response.choices[0].message.content}')


    logger.info('now streaming')
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an amazing ascii art generator bot, no text just art."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    logger.info(f"prompt> {prompt}")
    logger.info(f"artist-{model}>")
    i = 0
    for part in response:
        i += 1
        if part.choices[0].finish_reason is not None:
            break
        logger.info(part.choices[0].delta.content)

    assert i > 0


def test_chat_completion_gpt4(patched_openai_client):
    model="gpt-4-1106-preview"
    print_chat_completion(model, patched_openai_client)

def test_chat_completion_gpt_4o_mini(patched_openai_client):
    model="gpt-4o-mini"
    print_chat_completion(model, patched_openai_client)

def test_chat_completion_groq_llama3(patched_openai_client):
    model="groq/llama3-8b-8192"
    print_chat_completion(model, patched_openai_client)

def test_chat_completion_cohere(patched_openai_client):
    model="cohere_chat/command-r"
    print_chat_completion(model, patched_openai_client)

@pytest.mark.skip(reason="Tool choice not supporeted / working consistently")
def test_chat_completion_perp_mixtral(patched_openai_client):
    model="perplexity/llama-3.1-70b-instruct"
    print_chat_completion(model, patched_openai_client)

def test_chat_completion_claude(patched_openai_client):
    model="claude-3-haiku-20240307"
    print_chat_completion(model, patched_openai_client)

@pytest.mark.skip(reason="Tool choice not supporeted / working consistently")
def test_chat_completion_gemini_pro(patched_openai_client):
    model="gemini/gemini-1.5-pro-latest"
    print_chat_completion(model, patched_openai_client)
