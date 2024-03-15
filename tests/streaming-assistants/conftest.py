import asyncio
import os
import logging
import threading
import time

import pytest
import requests
import uvicorn
from dotenv import load_dotenv
from openai import OpenAI
from streaming_assistants import patch

from impl.main import app

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
def start_application():
    stop_event = threading.Event()

    def run_server(this_stop_event):
        config = uvicorn.Config(app=app, host="127.0.0.1", port=8000, log_level="debug")
        server = uvicorn.Server(config=config)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def serve_app():
            await server.serve()

        async def check_for_stop():
            while not this_stop_event.is_set():
                await asyncio.sleep(1)  # Check for the stop event every second
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.CRITICAL)
            await server.shutdown()  # Initiate server shutdown
            if loop.is_running():
                loop.stop()  # Stop the event loop if it's still running

        loop.create_task(serve_app())
        loop.create_task(check_for_stop())
        loop.run_forever()

    server_thread = threading.Thread(target=run_server, args=(stop_event,))
    server_thread.start()

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

    # After the tests, signal the server thread to stop and wait for it to finish
    stop_event.set()
    server_thread.join(timeout=100)
    if server_thread.is_alive():
        print("Server thread did not exit cleanly.")


@pytest.fixture(scope="function")
def openai_client(start_application) -> OpenAI:
    oai = patch(OpenAI())
    return oai
