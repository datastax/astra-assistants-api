# coding: utf-8
import pytest
from fastapi.testclient import TestClient


from openapi_server.models.delete_file_response import DeleteFileResponse  # noqa: F401
from openapi_server.models.list_files_response import ListFilesResponse  # noqa: F401


@pytest.mark.skip(reason="use client")
def test_create_file(client: TestClient):
    """Test case for create_file

    Upload a file that can be used across various endpoints. The size of all the files uploaded by one organization can be up to 100 GB.  The size of individual files can be a maximum of 512 MB or 2 million tokens for Assistants. See the [Assistants Tools guide](/docs/assistants/tools) to learn more about the types of files supported. The Fine-tuning API only supports `.jsonl` files.  Please [contact us](https://help.openai.com/) if you need to increase these storage limits. 
    """

    headers = {
        "Authorization": "Bearer special-key",
    }
    data = {
        "file": '/path/to/file',
        "purpose": 'purpose_example'
    }
    response = client.request(
        "POST",
        "/files",
        headers=headers,
        data=data,
    )

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


@pytest.mark.skip(reason="use client")
def test_delete_file(client: TestClient):
    """Test case for delete_file

    Delete a file.
    """

    headers = {
        "Authorization": "Bearer special-key",
    }
    response = client.request(
        "DELETE",
        "/files/{file_id}".format(file_id='file_id_example'),
        headers=headers,
    )

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


@pytest.mark.skip(reason="use client")
def test_download_file(client: TestClient):
    """Test case for download_file

    Returns the contents of the specified file.
    """

    headers = {
        "Authorization": "Bearer special-key",
    }
    response = client.request(
        "GET",
        "/files/{file_id}/content".format(file_id='file_id_example'),
        headers=headers,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


@pytest.mark.skip(reason="use client")
def test_list_files(client: TestClient):
    """Test case for list_files

    Returns a list of files that belong to the user's organization.
    """
    params = [("purpose", 'purpose_example')]
    headers = {
        "Authorization": "Bearer special-key",
    }
    response = client.request(
        "GET",
        "/files",
        headers=headers,
        params=params,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


@pytest.mark.skip(reason="use client")
def test_retrieve_file(client: TestClient):
    """Test case for retrieve_file

    Returns information about a specific file.
    """

    headers = {
        "Authorization": "Bearer special-key",
    }
    response = client.request(
        "GET",
        "/files/{file_id}".format(file_id='file_id_example'),
        headers=headers,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200

