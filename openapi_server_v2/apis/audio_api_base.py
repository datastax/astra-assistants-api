# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from openapi_server_v2.models.create_speech_request import CreateSpeechRequest
from openapi_server_v2.models.create_transcription200_response import CreateTranscription200Response
from openapi_server_v2.models.create_transcription_request_model import CreateTranscriptionRequestModel
from openapi_server_v2.models.create_translation200_response import CreateTranslation200Response
from openapi_server_v2.security_api import get_token_ApiKeyAuth

class BaseAudioApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseAudioApi.subclasses = BaseAudioApi.subclasses + (cls,)
    def create_speech(
        self,
        create_speech_request: CreateSpeechRequest,
    ) -> file:
        ...


    def create_transcription(
        self,
        file: str,
        model: CreateTranscriptionRequestModel,
        language: str,
        prompt: str,
        response_format: str,
        temperature: ,
        timestamp_granularities: List[str],
    ) -> CreateTranscription200Response:
        ...


    def create_translation(
        self,
        file: str,
        model: CreateTranscriptionRequestModel,
        prompt: str,
        response_format: str,
        temperature: ,
    ) -> CreateTranslation200Response:
        ...
