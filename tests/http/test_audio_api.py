# coding: utf-8
import pytest
from fastapi.testclient import TestClient


from openapi_server.models.create_speech_request import CreateSpeechRequest  # noqa: F401
from openapi_server.models.create_transcription_request_model import CreateTranscriptionRequestModel  # noqa: F401
from openapi_server.models.create_transcription_response import CreateTranscriptionResponse  # noqa: F401
from openapi_server.models.create_translation_response import CreateTranslationResponse  # noqa: F401
from conftest import get_headers, MODEL


def test_create_speech(client: TestClient):
    """Test case for create_speech

    Generates audio from the input text.
    """
    create_speech_request = {"model": "tts-1","voice":"alloy","input":"input","response_format":"mp3","speed":0.5503105714228793}

    headers = get_headers(MODEL)
    response = client.request(
        "POST",
        "/audio/speech",
        headers=headers,
        json=create_speech_request,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


@pytest.mark.skip(reason="use client")
def test_create_transcription(client: TestClient):
    """Test case for create_transcription

    Transcribes audio into the input language.
    """

    headers = get_headers(MODEL)
    response = client.request(
        "POST",
        "/audio/transcriptions",
        headers=headers,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


@pytest.mark.skip(reason="use client")
def test_create_translation(client: TestClient):
    """Test case for create_translation

    Translates audio into English.
    """

    headers = {
        "Authorization": "Bearer special-key",
    }
    data = {
        "file": '/path/to/file',
        "model": 'tts-1',
        "prompt": 'prompt_example',
        "response_format": 'json',
        "temperature": 0
    }
    response = client.request(
        "POST",
        "/audio/translations",
        headers=headers,
        data=data,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200
