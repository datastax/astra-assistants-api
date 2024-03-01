import os
import logging
from typing import Any, Optional, Union

import pytest
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import USE_CLIENT_DEFAULT, Response
from httpx._client import UseClientDefault
from openai import OpenAI
from streaming_assistants import patch
from impl.main import app as application

load_dotenv("./../../.env")
load_dotenv("./.env")

os.environ["OPENAI_LOG"] = "WARN"


@pytest.fixture(autouse=True)
def configure_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class UpdatedTestClient(TestClient):
    """Need to mask the `send` method for it to work with OpenAI's client."""

    def send(
        self,
        *args: Any,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        allow_redirects: Optional[bool] = None,
        **kwargs: Any,
    ) -> Response:
        redirect = self._choose_redirect_arg(follow_redirects, allow_redirects)
        return super().send(*args, follow_redirects=redirect, **kwargs)


@pytest.fixture
def app() -> FastAPI:
    application.dependency_overrides = {}

    return application


@pytest.fixture
def client(app) -> TestClient:
    return UpdatedTestClient(app, root_path="v1")


@pytest.fixture(scope="function")
def openai_client(client) -> OpenAI:
    oai = patch(OpenAI(
        http_client=client,
    ))
    return oai