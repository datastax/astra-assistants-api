user_message = "puppies"
def test_threads(patched_openai_client):
    client = patched_openai_client

    thread = client.beta.threads.create()
    the_same_thread = client.beta.threads.retrieve(thread_id=thread.id)

    assert the_same_thread.id == thread.id

    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    modified_message = client.beta.threads.messages.update(thread_id=thread.id, message_id=message.id, content="kittens")
    assert modified_message.content[0].text.value == "kittens"
    assert modified_message.role == "user"
    assert modified_message.id == message.id
    assert modified_message.thread_id == thread.id

    assert message.content[0].text.value == user_message
    assert message.role == "user"
    assert message.id == modified_message.id
    assert message.thread_id == thread.id

    client.beta.threads.messages.delete(thread_id=thread.id, message_id=message.id)