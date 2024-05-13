import pytest



def test_assistants_crud(openai_client):
    assistants = openai_client.beta.assistants.list().data

    try:
        openai_client.beta.assistants.retrieve("invalid-assistant-id")
        assert False, "Expected an error"
    except Exception as e:
        assert e.status_code == 404

    asst = openai_client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
        model="gpt-4-1106-preview",
    )
    assert asst.name == "Math Tutor"
    assert asst.instructions == "You are a personal math tutor. Answer questions briefly, in a sentence or less."
    assert asst.model == "gpt-4-1106-preview"

    asst = openai_client.beta.assistants.retrieve(asst.id)
    assert asst.name == "Math Tutor"

    asst = openai_client.beta.assistants.update(asst.id, name="Math Tutor 2")
    assert asst.name == "Math Tutor 2"
    assert asst.instructions == "You are a personal math tutor. Answer questions briefly, in a sentence or less."
    assert asst.model == "gpt-4-1106-preview"

    asst = openai_client.beta.assistants.retrieve(asst.id)
    assert asst.name == "Math Tutor 2"

    openai_client.beta.assistants.delete(asst.id)

    assistants = openai_client.beta.assistants.list().data


def test_no_collisions(openai_client):
    """Tests that assistants with the same name do not collide."""
    asst1 = openai_client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
        model="gpt-4-1106-preview",
    )
    asst2 = openai_client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
        model="gpt-4-1106-preview",
    )
    assert asst1.id != asst2.id
    assert asst1.name == asst2.name
    assert asst1.instructions == asst2.instructions
    assert asst1.model == asst2.model

    assistants = openai_client.beta.assistants.list().data
