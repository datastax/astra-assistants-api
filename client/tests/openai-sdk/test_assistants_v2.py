import pytest

def test_assistants_crud(openai_client):

    try:
        openai_client.beta.assistants.retrieve("invalid-assistant-id")
        assert False, "Expected an error"
    except Exception as e:
        assert e.status_code == 404

    with open("tests/fixtures/fake_content.txt", "rb") as f:
        file = openai_client.files.create(
            purpose="assistants",
            file=f,
        )

        tools=[{'type': 'file_search'}]
        tool_resources={'file_search': {'vector_stores': [{'file_ids': [file.id]}]}}
        metadata={}
        temperature=0.3
        top_p=1.0
        response_format="auto"
        asst = openai_client.beta.assistants.create(
            name="Math Tutor",
            instructions="You are a personal math tutor. Answer questions briefly, in a sentence or less.",
            model="gpt-4-1106-preview",
            description="a nice assistant",
            tools=tools,
            tool_resources=tool_resources,
            metadata=metadata,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
        )

        assistant_list = openai_client.beta.assistants.list()
        assistants = assistant_list.data
        assert len(assistants) > 0

        assert asst.tools[0].type == tools[0]['type']
        assert asst.metadata == metadata
        assert asst.temperature == temperature
        assert asst.top_p == top_p
        assert asst.response_format == response_format
        assert len(asst.tool_resources.file_search.vector_store_ids[0]) > 0

        vs = openai_client.beta.vector_stores.retrieve(asst.tool_resources.file_search.vector_store_ids[0])
        assert vs.id == asst.tool_resources.file_search.vector_store_ids[0]

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

        try:
            asst = openai_client.beta.assistants.retrieve(asst.id)
        except Exception as e:
            assert e.status_code == 404

        assistants = openai_client.beta.assistants.list().data
        for assistant in assistants:
            assert asst.id != assistant.id

        print(assistants)


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
