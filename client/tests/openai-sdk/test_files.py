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


def test_same_file_same_id(openai_client):
    """Tests that files with the same name and purpose do not collide."""
    # original file
    with open("tests/fixtures/fake_content.txt", "rb") as f:
        file1 = openai_client.files.create(
            purpose="assistants",
            file=f,
        )
    # same name different content
    with open("tests/fixtures/v2/fake_content.txt", "rb") as f:
        file2 = openai_client.files.create(
            purpose="assistants",
            file=f,
        )
    # different name same content
    with open("tests/fixtures/same_fake_content.txt", "rb") as f:
        file3 = openai_client.files.create(
            purpose="assistants",
            file=f,
        )
    # same name same content
    with open("tests/fixtures/duplicate/fake_content.txt", "rb") as f:
        file4 = openai_client.files.create(
            purpose="assistants",
            file=f,
        )


    # same name different content
    assert file1.id != file2.id
    assert file1.purpose == file2.purpose
    assert file1.filename == file2.filename


    # different name same content
    assert file1.id != file3.id
    assert file1.purpose == file3.purpose
    assert file1.filename != file3.filename

    # same name same content
    assert file1.id == file4.id
    assert file1.purpose == file4.purpose
    assert file1.filename == file4.filename


    files = openai_client.files.list().data


def test_finetuning_files_are_forwarded(openai_client):
    """Tests that files with purpose 'fine-tune' are forwarded to the OpenAI API."""
    # TODO
