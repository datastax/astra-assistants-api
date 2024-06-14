import pytest
from dotenv import load_dotenv
from openai import OpenAI

from astra_assistants import patch


load_dotenv("./../.env")
load_dotenv("./.env")
@pytest.fixture(scope="function")
def openai_client() -> OpenAI:
    oai = patch(OpenAI())
    return oai
