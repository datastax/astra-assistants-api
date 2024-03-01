# coding: utf-8
import logging
from fastapi.testclient import TestClient

from openapi_server.models.create_completion_request import CreateCompletionRequest  # noqa: F401
from openapi_server.models.create_completion_response import CreateCompletionResponse  # noqa: F401
from tests.http.conftest import get_headers, MODEL

logger = logging.getLogger(__name__)

def test_create_completion(client: TestClient):
    """Test case for create_completion

    Creates a completion for the provided prompt and parameters.
    """
    create_completion_request = {
        "model": "gpt-3.5-turbo-instruct",
        "prompt": "puppies are the best",
        "logit_bias":{11:-1},
        "seed":-2147483648,
        "max_tokens":16,
        "presence_penalty":0.25495066265333133,
        "echo": False,
        "suffix":"test.",
        "n":1,
        "logprobs":2,
        "top_p":1,
        "frequency_penalty":0.4109824732281613,
        "best_of":1,
        "stream": False,
        "temperature":1,
        "user":"user-1234"
    }

    parsed_create_completion_request = CreateCompletionRequest.parse_obj(create_completion_request)
    logger.info(parsed_create_completion_request)

    headers = get_headers(MODEL)
    response = client.post(
        "/completions",
        headers=headers,
        json=create_completion_request,
    )

    logger.info(response)
    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200

