# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from openapi_server.models.batch import Batch
from openapi_server.models.create_batch_request import CreateBatchRequest
from openapi_server.models.list_batches_response import ListBatchesResponse
from openapi_server.security_api import get_token_ApiKeyAuth

class BaseBatchApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseBatchApi.subclasses = BaseBatchApi.subclasses + (cls,)
    def cancel_batch(
        self,
        batch_id: str,
    ) -> Batch:
        ...


    def create_batch(
        self,
        create_batch_request: CreateBatchRequest,
    ) -> Batch:
        ...


    def list_batches(
        self,
        after: str,
        limit: int,
    ) -> ListBatchesResponse:
        ...


    def retrieve_batch(
        self,
        batch_id: str,
    ) -> Batch:
        ...
