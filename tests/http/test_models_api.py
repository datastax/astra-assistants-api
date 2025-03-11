# coding: utf-8
import pytest
from fastapi.testclient import TestClient


from openapi_server.models.delete_model_response import DeleteModelResponse  # noqa: F401
from openapi_server.models.list_models_response import ListModelsResponse  # noqa: F401
from openapi_server.models.model import Model  # noqa: F401
from conftest import MODEL, get_headers


@pytest.mark.skip(reason="use client")
def test_delete_model(client: TestClient):
    """Test case for delete_model

    Delete a fine-tuned model. You must have the Owner role in your organization to delete a model.
    """

    headers = get_headers(MODEL)
    response = client.request(
        "DELETE",
        "/models/{model}".format(model='ft:gpt-3.5-turbo:acemeco:suffix:abc123'),
        headers=headers,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


def test_list_models(client: TestClient):
    """Test case for list_models

    Lists the currently available models, and provides basic information about each one such as the owner and availability.
    """

    headers = get_headers(MODEL)
    response = client.request(
        "GET",
        "/models",
        headers=headers,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


def test_retrieve_model(client: TestClient):
    """Test case for retrieve_model

    Retrieves a model instance, providing basic information about the model such as the owner and permissioning.
    """

    headers = get_headers(MODEL)
    response = client.request(
        "GET",
        "/models/{model}".format(model='gpt-3.5-turbo'),
        headers=headers,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200

