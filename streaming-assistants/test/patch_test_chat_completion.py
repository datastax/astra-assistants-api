from openai import OpenAI
from streaming_assistants import patch
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


client = patch(OpenAI())


model="gpt-4-1106-preview"
print_chat_completion(model)

model="gpt-3.5-turbo"
print_chat_completion(model)

model="cohere/command"
print_chat_completion(model)

model="perplexity/mixtral-8x7b-instruct"
print_chat_completion(model)

model="anthropic.claude-v2"
print_chat_completion(model)

model="gemini/gemini-pro"
print_chat_completion(model)
