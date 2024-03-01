# coding: utf-8
import pytest
from fastapi.testclient import TestClient


from openapi_server.models.create_image_edit_request_model import CreateImageEditRequestModel  # noqa: F401
from openapi_server.models.create_image_request import CreateImageRequest  # noqa: F401
from openapi_server.models.images_response import ImagesResponse  # noqa: F401


def test_create_image(client: TestClient):
    """Test case for create_image

    Creates an image given a prompt.
    """
    create_image_request = {"response_format":"url","size":"1024x1024","model":"dall-e-3","style":"vivid","prompt":"A cute baby sea otter","n":1,"quality":"standard"}

    headers = {
        "Authorization": "Bearer special-key",
    }
    response = client.request(
        "POST",
        "/images/generations",
        headers=headers,
        json=create_image_request,
    )

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


@pytest.mark.skip(reason="use client")
def test_create_image_edit(client: TestClient):
    """Test case for create_image_edit

    Creates an edited or extended image given an original image and a prompt.
    """

    headers = {
        "Authorization": "Bearer special-key",
    }
    data = {
        "image": '/path/to/file',
        "prompt": 'prompt_example',
        "mask": '/path/to/file',
        "model": 'dall-e-3',
        "n": 1,
        "size": '1024x1024'
    }
    response = client.request(
        "POST",
        "/images/edits",
        headers=headers,
        data=data,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


@pytest.mark.skip(reason="use client")
def test_create_image_variation(client: TestClient):
    """Test case for create_image_variation

    Creates a variation of a given image.
    """

    headers = {
        "Authorization": "Bearer special-key",
    }
    data = {
        "image": '/path/to/file',
        "model": 'dall-e-3',
        "n": 1,
        "response_format": 'url',
        "size": '1024x1024',
        "user": 'user_example'
    }
    response = client.request(
        "POST",
        "/images/variations",
        headers=headers,
        data=data,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200

