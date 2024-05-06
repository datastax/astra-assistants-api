# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from openapi_server_v2.models.create_vector_store_file_batch_request import CreateVectorStoreFileBatchRequest
from openapi_server_v2.models.create_vector_store_file_request import CreateVectorStoreFileRequest
from openapi_server_v2.models.create_vector_store_request import CreateVectorStoreRequest
from openapi_server_v2.models.delete_vector_store_file_response import DeleteVectorStoreFileResponse
from openapi_server_v2.models.delete_vector_store_response import DeleteVectorStoreResponse
from openapi_server_v2.models.list_vector_store_files_response import ListVectorStoreFilesResponse
from openapi_server_v2.models.list_vector_stores_response import ListVectorStoresResponse
from openapi_server_v2.models.update_vector_store_request import UpdateVectorStoreRequest
from openapi_server_v2.models.vector_store_file_batch_object import VectorStoreFileBatchObject
from openapi_server_v2.models.vector_store_file_object import VectorStoreFileObject
from openapi_server_v2.models.vector_store_object import VectorStoreObject
from openapi_server_v2.security_api import get_token_ApiKeyAuth

class BaseVectorStoresApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseVectorStoresApi.subclasses = BaseVectorStoresApi.subclasses + (cls,)
    def cancel_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
    ) -> VectorStoreFileBatchObject:
        ...


    def create_vector_store(
        self,
        create_vector_store_request: CreateVectorStoreRequest,
    ) -> VectorStoreObject:
        ...


    def create_vector_store_file(
        self,
        vector_store_id: str,
        create_vector_store_file_request: CreateVectorStoreFileRequest,
    ) -> VectorStoreFileObject:
        ...


    def create_vector_store_file_batch(
        self,
        vector_store_id: str,
        create_vector_store_file_batch_request: CreateVectorStoreFileBatchRequest,
    ) -> VectorStoreFileBatchObject:
        ...


    def delete_vector_store(
        self,
        vector_store_id: str,
    ) -> DeleteVectorStoreResponse:
        ...


    def delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> DeleteVectorStoreFileResponse:
        ...


    def get_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreObject:
        ...


    def get_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileObject:
        ...


    def get_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
    ) -> VectorStoreFileBatchObject:
        ...


    def list_files_in_vector_store_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        limit: int,
        order: str,
        after: str,
        before: str,
        filter: str,
    ) -> ListVectorStoreFilesResponse:
        ...


    def list_vector_store_files(
        self,
        vector_store_id: str,
        limit: int,
        order: str,
        after: str,
        before: str,
        filter: str,
    ) -> ListVectorStoreFilesResponse:
        ...


    def list_vector_stores(
        self,
        limit: int,
        order: str,
        after: str,
        before: str,
    ) -> ListVectorStoresResponse:
        ...


    def modify_vector_store(
        self,
        vector_store_id: str,
        update_vector_store_request: UpdateVectorStoreRequest,
    ) -> VectorStoreObject:
        ...
