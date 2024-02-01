from openai import OpenAI
from streaming_assistants import patch
from dotenv import load_dotenv
import time

load_dotenv('./.env')


def test_run_with_assistant(assistant, client):
    user_message = "What's your favorite animal."

    thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    # Waiting in a loop
    while True:
        if run.status == 'failed':
            raise ValueError("Run is in failed state")
        if run.status == 'completed' or run.status == 'generating':
            print(f"run status: {run.status}")
            break
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)


    print(f"thread.id {thread.id}")
    response = client.beta.threads.messages.list(thread_id=thread.id, stream=True)

    print(f"{assistant.model} =>")
    for part in response:
        print(part.data[0].content[0].text.value)

    response = client.beta.threads.messages.list(thread_id=thread.id)
    print(response.data[0].content[0].text.value)



client = patch(OpenAI())

gpt3_assistant = client.beta.assistants.create(
    name="GPT3 Math Tutor",
    instructions="You're an animal expert. Answer questions briefly, in a sentence or less.",
    model="gpt-3.5-turbo",
)
test_run_with_assistant(gpt3_assistant, client)

cohere_assistant = client.beta.assistants.create(
    name="Cohere Math Tutor",
    instructions="You are an animal expert. Answer questions briefly, in a sentence or less.",
    model="cohere/command",
)
test_run_with_assistant(cohere_assistant, client)

perplexity_assistant = client.beta.assistants.create(
    name="Perplexity/Mixtral Math Tutor",
    instructions="You are an animal expert. Answer questions briefly, in a sentence or less.",
    model="perplexity/mixtral-8x7b-instruct",
)
test_run_with_assistant(perplexity_assistant, client)

claude_assistant = client.beta.assistants.create(
    name="Claude Math Tutor",
    instructions="You are an animal expert. Answer questions briefly, in a sentence or less.",
    model="anthropic.claude-v2",
)
test_run_with_assistant(claude_assistant, client)

gemini_assistant = client.beta.assistants.create(
    name="Gemini Math Tutor",
    instructions="You are an animal expert. Answer questions briefly, in a sentence or less.",
    model="gemini/gemini-pro",
)
test_run_with_assistant(gemini_assistant, client)