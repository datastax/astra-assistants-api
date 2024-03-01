# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from openapi_server.models.create_image_edit_request_model import CreateImageEditRequestModel
from openapi_server.models.create_image_request import CreateImageRequest
from openapi_server.models.images_response import ImagesResponse
from openapi_server.security_api import get_token_ApiKeyAuth

class BaseImagesApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseImagesApi.subclasses = BaseImagesApi.subclasses + (cls,)
    def create_image(
        self,
        create_image_request: CreateImageRequest,
    ) -> ImagesResponse:
        ...


    def create_image_edit(
        self,
        image: str,
        prompt: str,
        mask: str,
        model: CreateImageEditRequestModel,
        n: int,
        size: str,
    ) -> ImagesResponse:
        ...


    def create_image_variation(
        self,
        image: str,
        model: CreateImageEditRequestModel,
    ) -> ImagesResponse:
        ...
