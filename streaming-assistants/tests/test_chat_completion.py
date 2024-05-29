def print_chat_completion(model, client):
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



def test_chat_completion_gpt4(openai_client):
    model="gpt-4-1106-preview"
    print_chat_completion(model, openai_client)

def test_chat_completion_gpt3_5(openai_client):
    model="gpt-3.5-turbo"
    print_chat_completion(model, openai_client)

def test_chat_completion_groq(openai_client):
    model="groq/llama3-8b-8192"
    print_chat_completion(model, openai_client)

def test_chat_completion_cohere(openai_client):
    model="command-r"
    print_chat_completion(model, openai_client)

def test_chat_completion_perp_mixtral(openai_client):
    model="perplexity/mixtral-8x7b-instruct"
    print_chat_completion(model, openai_client)

def test_chat_completion_claude(openai_client):
    model="anthropic.claude-v2"
    print_chat_completion(model, openai_client)

def test_chat_completion_gemini_pro(openai_client):
    model="gemini/gemini-pro"
    print_chat_completion(model, openai_client)
