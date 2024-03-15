# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from openapi_server.models.create_speech_request import CreateSpeechRequest
from openapi_server.models.create_transcription_request_model import CreateTranscriptionRequestModel
from openapi_server.models.create_transcription_response import CreateTranscriptionResponse
from openapi_server.models.create_translation_response import CreateTranslationResponse
from openapi_server.security_api import get_token_ApiKeyAuth

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
    ) -> CreateTranscriptionResponse:
        ...


    def create_translation(
        self,
        file: str,
        model: CreateTranscriptionRequestModel,
        prompt: str,
        response_format: str,
        temperature: ,
    ) -> CreateTranslationResponse:
        ...
