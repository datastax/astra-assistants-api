# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from openapi_server.models.create_moderation_request import CreateModerationRequest
from openapi_server.models.create_moderation_response import CreateModerationResponse
from openapi_server.security_api import get_token_ApiKeyAuth

class BaseModerationsApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseModerationsApi.subclasses = BaseModerationsApi.subclasses + (cls,)
    def create_moderation(
        self,
        create_moderation_request: CreateModerationRequest,
    ) -> CreateModerationResponse:
        ...
