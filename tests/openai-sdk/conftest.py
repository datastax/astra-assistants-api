import os
from typing import Any, Optional, Tuple, Union

import pytest
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import USE_CLIENT_DEFAULT, Response
from httpx._client import UseClientDefault
from openai import OpenAI

from impl.astra_vector import CassandraClient
from impl.main import app as application
from impl.routes.utils import datastore_cache

logger = logging.getLogger(__name__)


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


TESTING_KEYSPACE = "assistant_api"

load_dotenv("./../../.env")
load_dotenv("./.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

if not OPENAI_API_KEY or not ASTRA_DB_TOKEN:
    pytest.skip("No Astra DB credentials found in environment.")

@pytest.fixture(scope="session")
async def db_client() -> CassandraClient:
    """Gets a DB client for testing."""
    # import pdb; pdb.set_trace()
    return await datastore_cache(token=ASTRA_DB_TOKEN, dbid=None)


#@pytest.fixture(scope="session", autouse=True)
#def db_cleanup(db_client) -> None:
#    """Sets up the DB for testing."""
#    # Run tests
#    yield
#
#    # Cleanup
#    if astra_vector.CASSANDRA_KEYSPACE != TESTING_KEYSPACE:
#        raise ValueError("CASSANDRA_KEYSPACE was changed during testing.")
#
#    try:
#        for table in db_client.get_tables(TESTING_KEYSPACE):
#            db_client.truncate_table(table)
#    except Exception as e:
#        logger.error("Could not clean up DB:")
#        logger.error(e)



@pytest.fixture(scope="function")
def openai_client(client) -> OpenAI:
    return OpenAI(
        api_key=OPENAI_API_KEY,
        http_client=client,
        base_url="/v1",
        default_headers={
            "astra-api-token": ASTRA_DB_TOKEN,
        }
    )
