# coding: utf-8

from fastapi.testclient import TestClient


from openapi_server.models.create_embedding_request import CreateEmbeddingRequest  # noqa: F401
from openapi_server.models.create_embedding_response import CreateEmbeddingResponse  # noqa: F401
from tests.http.conftest import MODEL, get_headers


def test_create_embedding(client: TestClient):
    """Test case for create_embedding

    Creates an embedding vector representing the input text.
    """
    create_embedding_request = {"input":"The quick brown fox jumped over the lazy dog","encoding_format":"float","model":"text-embedding-3-small","dimensions":1}

    headers = get_headers(MODEL)
    response = client.request(
        "POST",
        "/embeddings",
        headers=headers,
        json=create_embedding_request,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200

