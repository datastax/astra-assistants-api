import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("./.env")


# you do have to pass a key because the client requires it but it doesn't have to be valid since we're using a third party LLM
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
ASTRA_DB_TOKEN=os.getenv("ASTRA_DB_TOKEN")

base_url=os.getenv("base_url", "https://open-assistant-ai.astra.datastax.com/v1")

client = OpenAI(
    base_url=base_url,
    api_key=OPENAI_API_KEY,
    default_headers={
        "astra-api-token": ASTRA_DB_TOKEN,
    }
)

model="gpt-4-1106-preview"

prompt = "Draw an ASCII art kitten eating icecream"
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an amazing ascii art generator bot, no text just art."},
        {"role": "user", "content": prompt}
    ]
)

print(f'prompt> {prompt}')
print(f'artist-{model}>\n{response.choices[0].message.content}')

prompt = "Draw a more complex ASCII art image of a kitten eating ice cream"
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an amazing ascii art generator bot, no text just art."},
        {"role": "user", "content": prompt}
    ]
)

print(f'prompt> {prompt}')
print(f'artist-{model}>\n{response.choices[0].message.content}')

prompt="Draw an even more complex ASCII art image of cats eating ice cream"
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an amazing ascii art generator bot, no text just art."},
        {"role": "user", "content": prompt}
    ]
)

print(f'prompt> {prompt}')
print(f'artist-{model}>\n{response.choices[0].message.content}')

prompt="Make an ASCII art masterpiece featuring cats eating ice cream, with unbelievable detail"
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an amazing ascii art generator bot, no text just art."},
        {"role": "user", "content": prompt}
    ]
)

print(f'prompt> {prompt}')
print(f'artist-{model}>\n{response.choices[0].message.content}')