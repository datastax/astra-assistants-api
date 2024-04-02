def test_files_crud(openai_client):
    files = openai_client.files.list().data

    with open("tests/fixtures/fake_content.txt", "rb") as f:
        file = openai_client.files.create(
            purpose="assistants",
            file=f,
        )
    assert file.purpose == "assistants"
    assert file.filename == "fake_content.txt"
    assert file.embedding_model is not None

    file = openai_client.files.retrieve(file.id)
    assert file.purpose == "assistants"
    assert file.filename == "fake_content.txt"

    openai_client.files.delete(file.id)

    files = openai_client.files.list().data


def test_no_collisions(openai_client):
    """Tests that files with the same name and purpose do not collide."""
    with open("tests/fixtures/fake_content.txt", "rb") as f:
        file1 = openai_client.files.create(
            purpose="assistants",
            file=f,
        )
    with open("tests/fixtures/fake_content.txt", "rb") as f:
        file2 = openai_client.files.create(
            purpose="assistants",
            file=f,
        )
    assert file1.id != file2.id
    assert file1.purpose == file2.purpose
    assert file1.filename == file2.filename

    files = openai_client.files.list().data


def test_finetuning_files_are_forwarded(openai_client):
    """Tests that files with purpose 'fine-tune' are forwarded to the OpenAI API."""
    # TODO
