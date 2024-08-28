from openai import OpenAI
from astra_assistants import patch

from dotenv import load_dotenv
import time

load_dotenv('./.env')

client = patch(OpenAI())

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
    response = client.beta.threads.messages.list(
        thread_id=thread.id,
        stream=True
    )

    print(f"{assistant.model} - streaming=>")
    for part in response:
        print(part.data[0].content[0].delta.value, end="")


    #Note, we can list now that the run is completed, we know the run is completed because we finished streaming
    print(f"{assistant.model} no streaming=>")
    response = client.beta.threads.messages.list(thread_id=thread.id)
    print(response.data[0].content[0].text.value)




instructions="You're an animal expert who gives very long winded answers with flowery prose."

gpt3_assistant = client.beta.assistants.create(
    name="GPT3 Animal Tutor",
    instructions=instructions,
    model="gpt-3.5-turbo",
)

assistant = client.beta.assistants.retrieve(gpt3_assistant.id)
print(assistant)
test_run_with_assistant(gpt3_assistant, client)

cohere_assistant = client.beta.assistants.create(
    name="Cohere Animal Tutor",
    instructions=instructions,
    model="cohere/command",
)
test_run_with_assistant(cohere_assistant, client)

perplexity_assistant = client.beta.assistants.create(
    name="Perplexity/Mixtral Animal Tutor",
    instructions=instructions,
    model="perplexity/llama-3.1-70b-instruct",
)
test_run_with_assistant(perplexity_assistant, client)

claude_assistant = client.beta.assistants.create(
    name="Claude Animal Tutor",
    instructions=instructions,
    model="anthropic.claude-v2",
)
test_run_with_assistant(claude_assistant, client)

gemini_assistant = client.beta.assistants.create(
    name="Gemini Animal Tutor",
    instructions=instructions,
    model=gemini/gemini-1.5-flash,
)
test_run_with_assistant(gemini_assistant, client)