import time

def run_with_assistant(assistant, client):
    user_message = "What's your favorite animal."

    thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        stream=True
    )

    i=0
    print(f"{assistant.model} - streaming=>")
    for part in run:
        i+=1
        print(part)
    assert i > 0



instructions="You're an animal expert who gives very long winded answers with flowery prose."
def test_run_gpt3_5(openai_client):
    gpt3_assistant = openai_client.beta.assistants.create(
        name="GPT3 Animal Tutor",
        instructions=instructions,
        model="gpt-3.5-turbo",
    )

    assistant = openai_client.beta.assistants.retrieve(gpt3_assistant.id)
    print(assistant)

    run_with_assistant(gpt3_assistant, openai_client)

def test_run_cohere(openai_client):
    cohere_assistant = openai_client.beta.assistants.create(
        name="Cohere Animal Tutor",
        instructions=instructions,
        model="cohere/command",
    )
    run_with_assistant(cohere_assistant, openai_client)

def test_run_perp(openai_client):
    perplexity_assistant = openai_client.beta.assistants.create(
        name="Perplexity/Mixtral Animal Tutor",
        instructions=instructions,
        model="perplexity/mixtral-8x7b-instruct",
    )
    run_with_assistant(perplexity_assistant, openai_client)

def test_run_claude(openai_client):
    claude_assistant = openai_client.beta.assistants.create(
        name="Claude Animal Tutor",
        instructions=instructions,
        model="anthropic.claude-v2",
    )
    run_with_assistant(claude_assistant, openai_client)

def test_run_gemini(openai_client):
    gemini_assistant = openai_client.beta.assistants.create(
        name="Gemini Animal Tutor",
        instructions=instructions,
        model="gemini/gemini-pro",
    )
    run_with_assistant(gemini_assistant, openai_client)