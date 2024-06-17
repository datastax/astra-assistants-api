import os
from typing import Any, Optional, Tuple, Union

import pytest
import logging
from dotenv import load_dotenv
from httpx import USE_CLIENT_DEFAULT, Response, Client
from httpx._client import UseClientDefault
from openai import OpenAI

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def configure_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class UpdatedTestClient(Client):
    """Need to mask the `send` method for it to work with OpenAI's client."""

    def send(
        self,
        *args: Any,
        follow_redirects: Union[bool, UseClientDefault] = USE_CLIENT_DEFAULT,
        allow_redirects: Optional[bool] = None,
        **kwargs: Any,
    ) -> Response:
        if follow_redirects is USE_CLIENT_DEFAULT:
            follow_redirects = True if allow_redirects is None else allow_redirects
        return super().send(*args, follow_redirects=follow_redirects, **kwargs)

        #redirect = self._choose_redirect_arg(follow_redirects, allow_redirects)
        #return super().send(*args, follow_redirects=redirect, **kwargs)



@pytest.fixture
def client() -> Client:
    return UpdatedTestClient()


load_dotenv("./../../.env")
load_dotenv("./.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

if not OPENAI_API_KEY or not ASTRA_DB_TOKEN:
    pytest.skip("No Astra DB credentials found in environment.")


@pytest.fixture(scope="function")
def openai_client(client) -> OpenAI:
    return OpenAI(
        api_key=OPENAI_API_KEY,
        http_client=client,
        base_url="http://127.0.0.1:8000/v1",

    default_headers={
            "astra-api-token": ASTRA_DB_TOKEN,
        }
    )