user_message = "puppies"
def test_threads(patched_openai_client):
    client = patched_openai_client

    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    client.beta.threads.messages.update(thread_id=thread.id, message_id=message.id, content="kittens")
    client.beta.threads.messages.delete(thread_id=thread.id, message_id=message.id)