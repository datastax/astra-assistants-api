import os
import pytest
from impl.routes_v2.vector_stores import create_vector_store, create_vector_store_file
from openapi_server_v2.models.create_vector_store_request import CreateVectorStoreRequest
from openapi_server_v2.models.create_vector_store_file_request import CreateVectorStoreFileRequest
from openapi_server_v2.models.vector_store_object import VectorStoreObject
from openapi_server_v2.models.vector_store_file_object import VectorStoreFileObject
from impl.astra_vector import CassandraClient

@pytest.fixture(scope="module")
def astradb():
    # Setup Cassandra client
    client = CassandraClient()
    yield client
    client.close()

def test_vector_store_usage_bytes(astradb):
    # Create a vector store
    vector_store_request = CreateVectorStoreRequest(name="Test Vector Store", file_ids=[])
    vector_store: VectorStoreObject = create_vector_store(vector_store_request, astradb)

    # Attach files to the vector store
    file_paths = ["./tests/fixtures/sample1.txt", "./tests/fixtures/sample2.txt"]
    total_usage_bytes = 0

    for file_path in file_paths:
        file_size = os.path.getsize(file_path)
        total_usage_bytes += file_size

        file_request = CreateVectorStoreFileRequest(file_id=file_path)
        vector_store_file: VectorStoreFileObject = create_vector_store_file(vector_store.id, file_request, astradb)
        assert vector_store_file.usage_bytes == file_size

    # Verify the usage_bytes attribute of the vector store
    updated_vector_store: VectorStoreObject = create_vector_store(vector_store_request, astradb)
    assert updated_vector_store.usage_bytes == total_usage_bytes
