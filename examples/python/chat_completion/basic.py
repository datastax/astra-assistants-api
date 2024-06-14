from openai import OpenAI
from dotenv import load_dotenv
from astra_assistants import patch

load_dotenv("./.env")
load_dotenv("../../../.env")

def print_chat_completion(client, model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an amazing ascii art generator bot, no text just art."},
            {"role": "user", "content": prompt}
        ]
    )
    print(f"prompt> {prompt}")
    print(f"artist-{model}>\n{response.choices[0].message.content}")
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

# Ensure the right environment variables are configured for the model you are using
#model="anthropic/claude-3-opus-20240229"
model="anthropic/claude-3-sonnet-20240229"
#model="gpt-4-1106-preview"
#model="gpt-3.5-turbo"
#model="cohere_chat/command-r"
#model="perplexity/mixtral-8x7b-instruct"
#model="perplexity/pplx-70b-online"
#model="anthropic.claude-v2"
#model="groq/llama3-8b-8192"
#model="gemini/gemini-1.5-pro-latest"
#model = "meta.llama2-13b-chat-v1"


client = patch(OpenAI())

prompt = "Draw an ASCII art kitten eating icecream"
print_chat_completion(client, model, prompt)

prompt = "Draw a more complex ASCII art image of a kitten eating ice cream"
print_chat_completion(client, model, prompt)

prompt="Draw an even more complex ASCII art image of cats eating ice cream"
print_chat_completion(client, model, prompt)

prompt="Make an ASCII art masterpiece featuring cats eating ice cream, with unbelievable detail"
print_chat_completion(client, model, prompt)