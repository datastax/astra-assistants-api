# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from openapi_server.models.delete_file_response import DeleteFileResponse
from openapi_server.models.list_files_response import ListFilesResponse
from openapi_server.models.open_ai_file import OpenAIFile
from openapi_server.security_api import get_token_ApiKeyAuth

class BaseFilesApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseFilesApi.subclasses = BaseFilesApi.subclasses + (cls,)
    def create_file(
        self,
        file: str,
        purpose: str,
    ) -> OpenAIFile:
        ...


    def delete_file(
        self,
        file_id: str,
    ) -> DeleteFileResponse:
        ...


    def download_file(
        self,
        file_id: str,
    ) -> str:
        ...


    def list_files(
        self,
        purpose: str,
    ) -> ListFilesResponse:
        ...


    def retrieve_file(
        self,
        file_id: str,
    ) -> OpenAIFile:
        ...
