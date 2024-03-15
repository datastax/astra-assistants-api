import contextlib
import io
import logging
import os

import httpx
import pytest
from dotenv import load_dotenv
from httpx import AsyncClient
from litellm import utils

from impl.main import app
from impl.model.create_run_request import CreateRunRequest
from openapi_server.models.assistant_object import AssistantObject
from openapi_server.models.create_assistant_request import CreateAssistantRequest
from openapi_server.models.create_thread_request import CreateThreadRequest
from openapi_server.models.thread_object import ThreadObject

logger = logging.getLogger(__name__)


LLM_PARAM_AWS_REGION_NAME = "LLM-PARAM-aws-region-name"
LLM_PARAM_AWS_SECRET_ACCESS_KEY = "LLM-PARAM-aws-secret-access-key"
LLM_PARAM_AWS_ACCESS_KEY_ID = "LLM-PARAM-aws-access-key-id"

MODEL = "gpt-3.5-turbo"

def get_headers(model):
    load_dotenv("../../.env")
    load_dotenv("./.env")

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "astra-api-token": os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    }

    with contextlib.redirect_stdout(io.StringIO()):
        key = None
        triple = utils.get_llm_provider(model)
        provider = triple[1]
        dynamic_key = triple[2]
        if provider == "bedrock":
            headers[LLM_PARAM_AWS_ACCESS_KEY_ID] = os.getenv("AWS_ACCESS_KEY_ID")
            headers[LLM_PARAM_AWS_SECRET_ACCESS_KEY] = os.getenv("AWS_SECRET_ACCESS_KEY")
            headers[LLM_PARAM_AWS_REGION_NAME] = os.getenv("AWS_REGION_NAME")
        if provider != "openai":
            key = utils.get_api_key(provider, dynamic_key)
        if provider == "gemini":
            key = os.getenv("GEMINI_API_KEY")
        if key is not None:
            headers["api-key"] = key

    return headers


@pytest.mark.asyncio
async def test_streaming_run():
    async with AsyncClient(app=app, base_url='http://testserver/v1') as client:
        create_thread_request = {"metadata":{},"messages":[{"metadata":{},"role":"user","file_ids":[],"content":"content"},{"metadata":{},"role":"user","file_ids":[],"content":"content"}]}

        parsed_create_thread_request = CreateThreadRequest.parse_obj(create_thread_request)

        logger.info(parsed_create_thread_request)

        headers = get_headers(MODEL)
        logger.info(headers)
        response= await client.post(
            "/threads",
            headers=headers,
            json=create_thread_request,
        )

        thread = ThreadObject.parse_raw(response.content)

        # uncomment below to assert the status code of the HTTP response


        create_assistant_request = {"instructions":"instructions","metadata":{},"name":"name","file_ids":[],"description":"description","model":MODEL,"tools":[]}

        parsed_create_assistant_request = CreateAssistantRequest.parse_obj(create_assistant_request)
        logger.info(parsed_create_assistant_request)
        headers = get_headers(MODEL)

        response = await client.post(
            "/assistants",
            headers=headers,
            json=create_assistant_request,
        )

        assistant = AssistantObject.parse_raw(response.content)

        create_run_request = {"instructions":"instructions","metadata":{},"assistant_id":assistant.id,"model":MODEL,"tools":[], "stream": True}

        parsed_create_run_request = CreateRunRequest.parse_obj(create_run_request)

        logger.info(parsed_create_run_request)

        headers = get_headers(MODEL)

        response = await client.post(
            "/threads/{thread_id}/runs".format(thread_id=thread.id),
            headers=headers,
            json=create_run_request,
        )

        # uncomment below to assert the status code of the HTTP response
        assert response.status_code == 200
        for line in response.iter_lines():
            if line:
                print(f"Received: {line}")
