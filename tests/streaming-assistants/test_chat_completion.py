import logging

logger = logging.getLogger(__name__)

def print_chat_completion(model, client):
    prompt="Draw your favorite animal."
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an amazing ascii art generator bot, no text just art."},
            {"role": "user", "content": prompt}
        ]
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

def test_chat_completion_gpt3_5(patched_openai_client):
    model="gpt-3.5-turbo"
    print_chat_completion(model, patched_openai_client)

def test_chat_completion_cohere(patched_openai_client):
    model="cohere/command"
    print_chat_completion(model, patched_openai_client)

def test_chat_completion_perp_mixtral(patched_openai_client):
    model="perplexity/mixtral-8x7b-instruct"
    print_chat_completion(model, patched_openai_client)

def test_chat_completion_claude(patched_openai_client):
    model="claude-3-haiku-20240307"
    print_chat_completion(model, patched_openai_client)

def test_chat_completion_gemini_pro(patched_openai_client):
    model="gemini/gemini-pro"
    print_chat_completion(model, patched_openai_client)
