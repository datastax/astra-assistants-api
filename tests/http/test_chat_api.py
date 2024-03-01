# coding: utf-8

from fastapi.testclient import TestClient


from openapi_server.models.create_chat_completion_request import CreateChatCompletionRequest  # noqa: F401
from openapi_server.models.create_chat_completion_response import CreateChatCompletionResponse  # noqa: F401
from tests.http.conftest import get_headers, MODEL


def test_create_chat_completion(client: TestClient):
    """Test case for create_chat_completion

    Creates a model response for the given chat conversation.
    """
    create_chat_completion_request = {
        "logit_bias":{"11":-6},
        "seed":2147483647,
        "functions":[],
        "max_tokens":1,
        "presence_penalty":0.38485356667327286,
        "n":2,
        "top_p":1,
        "frequency_penalty":-1.6796687238155954,
        "response_format":{"type":"json_object"},
        "stream":0,
        "temperature":1,
        "messages":[
            {
                "role": "system",
                "content": "You are a helpful assistant that speaks json."
            },
            {
                "role": "user",
                "content": "Hello!"
            }
        ],
        "model":"gpt-3.5-turbo",
        "user":"user-1234"
    }

    headers = get_headers(MODEL)
    response = client.request(
        "POST",
        "/chat/completions",
        headers=headers,
        json=create_chat_completion_request,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200

