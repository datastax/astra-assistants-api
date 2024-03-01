import os

import pytest

#from impl.services import chunks

#FAKE_EMB_LIST = [1] + [0] * 1535


#@pytest.fixture(scope="function", autouse=True)
#def cleanup_asst_table(db_client):
#    """Cleans up the assistants table after each test."""
#    db_client.truncate_table("files")
#    db_client.truncate_table("file_chunks")


#@pytest.fixture(scope="function", autouse=True)
#def mock_embeddings(monkeypatch):
#    """Mocks the get_embeddings util function"""
#    def mock_get_embeddings(documents, model):
#        return [FAKE_EMB_LIST for _ in documents]
#    monkeypatch.setattr(
#        chunks,
#        "get_embeddings",
#        mock_get_embeddings,
#    )


def test_files_crud(openai_client):
    files = openai_client.files.list().data

    with open("tests/fixtures/fake_content.txt", "rb") as f:
        file = openai_client.files.create(
            purpose="assistants",
            file=f,
        )
    assert file.purpose == "assistants"
    assert file.filename == "fake_content.txt"

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
