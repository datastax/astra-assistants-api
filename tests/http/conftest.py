import contextlib
import io
import os
import logging
import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient

from impl.main import app as application, startup_event
from litellm import utils


@pytest.fixture(autouse=True)
def configure_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class PrefixedTestClient(TestClient):
    def __init__(self, app: FastAPI, prefix: str = "/v1"):
        super().__init__(app)
        self.prefix = prefix

    def request(self, method: str, url: str, **kwargs):
        # Prepend the prefix to the URL path
        prefixed_url = self.prefix + url
        return super().request(method, prefixed_url, **kwargs)


@pytest.fixture
def app() -> FastAPI:
    application.dependency_overrides = {}
    return application

@pytest.fixture
def client(app) -> TestClient:
    with PrefixedTestClient(app) as client:
        yield client

#this just runs startup
@pytest.fixture
async def async_app() -> FastAPI:
    await application.router.startup()


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
