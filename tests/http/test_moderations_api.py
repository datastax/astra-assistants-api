# coding: utf-8

from fastapi.testclient import TestClient


from openapi_server.models.create_moderation_request import CreateModerationRequest  # noqa: F401
from openapi_server.models.create_moderation_response import CreateModerationResponse  # noqa: F401
from tests.http.conftest import get_headers, MODEL


def test_create_moderation(client: TestClient):
    """Test case for create_moderation

    Classifies if text violates OpenAI's Content Policy
    """
    create_moderation_request = {"model":"text-moderation-stable", "input": "I really love puppies"}

    headers = get_headers(MODEL)
    response = client.request(
        "POST",
        "/moderations",
        headers=headers,
        json=create_moderation_request,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200

