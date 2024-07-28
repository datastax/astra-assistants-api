import os
import logging
import time

import pytest
import requests
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from astra_assistants import patch

load_dotenv("./../../.env")
load_dotenv("./.env")

os.environ["OPENAI_LOG"] = "WARN"


@pytest.fixture(autouse=True)
def configure_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class SafeLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            super().emit(record)
        except ValueError:
            pass  # Ignore ValueError raised due to logging after file handles are closed


@pytest.fixture(scope="module")
def wait_for_server():
    # Wait for the server to be up
    timeout = 20  # seconds
    start_time = time.time()
    url = "http://127.0.0.1:8000/v1/health"
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                break  # Server is ready
        except requests.exceptions.ConnectionError:
            pass  # Server not ready yet

        if time.time() - start_time > timeout:
            raise RuntimeError(f"Server did not start within {timeout} seconds")

        time.sleep(0.5)  # Wait a bit before trying again

    yield  # Tests run here


@pytest.fixture(scope="function")
def patched_openai_client(wait_for_server) -> OpenAI:
    oai = patch(OpenAI())
    #oai = OpenAI()
    return oai

@pytest.fixture(scope="function")
def async_patched_openai_client(wait_for_server) -> OpenAI:
    oai = patch(AsyncOpenAI())
    #oai = AsyncOpenAI()
    return oai
