from openai import OpenAI
from astra_assistants import patch
from dotenv import load_dotenv

load_dotenv('./.env')


def print_chat_completion(model):
    prompt="Draw your favorite animal."
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an amazing ascii art generator bot, no text just art."},
            {"role": "user", "content": prompt}
        ]
    )
    print(f'prompt> {prompt}')
    print(f'artist-{model}>\n{response.choices[0].message.content}')

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an amazing ascii art generator bot, no text just art."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    print(f"prompt> {prompt}")
    print(f"artist-{model}>")
    for part in response:
        if part.choices[0].finish_reason is not None:
            break
        print(part.choices[0].delta.content, end="")


client = patch(OpenAI(max_retries=1))


model="gpt-4-1106-preview"
print_chat_completion(model)

model="gpt-3.5-turbo"
print_chat_completion(model)

model="cohere/command"
print_chat_completion(model)

model="perplexity/llama-3.1-70b-instruct"
print_chat_completion(model)

model="anthropic.claude-v2"
print_chat_completion(model)

model="gemini/gemini-1.5-flash"
print_chat_completion(model)
