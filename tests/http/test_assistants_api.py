# coding: utf-8
import pytest
import logging

from fastapi.testclient import TestClient

from impl.model.create_assistant_request import CreateAssistantRequest
from impl.model.create_run_request import CreateRunRequest
from impl.model.message_object import MessageObject
from impl.model.modify_message_request import ModifyMessageRequest
from openapi_server.models.assistant_file_object import AssistantFileObject  # noqa: F401
from openapi_server.models.assistant_object import AssistantObject  # noqa: F401
from openapi_server.models.create_assistant_file_request import CreateAssistantFileRequest  # noqa: F401
from openapi_server.models.create_message_request import CreateMessageRequest  # noqa: F401
from openapi_server.models.create_thread_and_run_request import CreateThreadAndRunRequest  # noqa: F401
from openapi_server.models.create_thread_request import CreateThreadRequest  # noqa: F401
from openapi_server.models.delete_assistant_file_response import DeleteAssistantFileResponse  # noqa: F401
from openapi_server.models.delete_assistant_response import DeleteAssistantResponse  # noqa: F401
from openapi_server.models.delete_thread_response import DeleteThreadResponse  # noqa: F401
from openapi_server.models.list_assistant_files_response import ListAssistantFilesResponse  # noqa: F401
from openapi_server.models.list_assistants_response import ListAssistantsResponse  # noqa: F401
from openapi_server.models.list_message_files_response import ListMessageFilesResponse  # noqa: F401
from openapi_server.models.list_run_steps_response import ListRunStepsResponse  # noqa: F401
from openapi_server.models.list_runs_response import ListRunsResponse  # noqa: F401
from openapi_server.models.message_file_object import MessageFileObject  # noqa: F401
from openapi_server.models.modify_run_request import ModifyRunRequest  # noqa: F401
from openapi_server.models.modify_thread_request import ModifyThreadRequest  # noqa: F401
from openapi_server.models.run_object import RunObject  # noqa: F401
from openapi_server.models.run_step_object import RunStepObject  # noqa: F401
from openapi_server.models.thread_object import ThreadObject  # noqa: F401
from tests.http.conftest import get_headers, MODEL

logger = logging.getLogger(__name__)

model="gpt-3.5-turbo"


@pytest.mark.skip(reason="Other tests use this function")
def test_create_assistant(client: TestClient):
    """Test case for create_assistant

    Create an assistant with a model and instructions.
    """
    create_assistant_request = {"instructions":"instructions","metadata":{},"name":"name","file_ids":[],"description":"description","model":MODEL,"tools":[]}

    parsed_create_assistant_request = CreateAssistantRequest.parse_obj(create_assistant_request)
    logger.info(parsed_create_assistant_request)
    headers = get_headers(MODEL)

    response = client.request(
        "POST",
        "/assistants",
        headers=headers,
        json=create_assistant_request,
    )

    assert response.status_code == 200
    if response.status_code == 200:
        logger.info(response)
        assistant = AssistantObject.parse_raw(response.content)
        return assistant

    return None


@pytest.mark.skip(reason="Not implemented")
def test_create_assistant_file(client: TestClient):
    """Test case for create_assistant_file

    Create an assistant file by attaching a [File](/docs/api-reference/files) to an [assistant](/docs/api-reference/assistants).
    """

    assistant = test_create_assistant(client)

    create_assistant_file_request = {"file_id":"file_id"}

    parsed_create_assistant_file_request = CreateAssistantFileRequest.parse_obj(create_assistant_file_request)

    logger.info(parsed_create_assistant_file_request)

    headers = get_headers(MODEL)

    response = client.request(
        "POST",
        "/assistants/{assistant_id}/files".format(assistant_id=assistant.id),
        headers=headers,
        json=create_assistant_file_request,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


def test_create_message(client: TestClient):
    """Test case for create_message

    Create a message.
    """

    thread = test_create_thread(client)

    create_message_request = {"metadata":{},"role":"user","file_ids":[],"content":"content"}

    parsed_create_message_request = CreateMessageRequest.parse_obj(create_message_request)

    logger.info(parsed_create_message_request)

    headers = get_headers(MODEL)

    response = client.request(
        "POST",
        "/threads/{thread_id}/messages".format(thread_id=thread.id),
        headers=headers,
        json=create_message_request,
    )

    logger.info(response)
    assert response.status_code == 200

    message = MessageObject.parse_raw(response.content)
    return message




def test_create_thread(client: TestClient):
    """Test case for create_thread

    Create a thread.
    """
    create_thread_request = {"metadata":{},"messages":[{"metadata":{},"role":"user","file_ids":[],"content":"content"},{"metadata":{},"role":"user","file_ids":[],"content":"content"}]}

    parsed_create_thread_request = CreateThreadRequest.parse_obj(create_thread_request)

    logger.info(parsed_create_thread_request)

    headers = get_headers(MODEL)
    logger.info(headers)
    response = client.request(
        "POST",
        "/threads",
        headers=headers,
        json=create_thread_request,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200

    thread = ThreadObject.parse_raw(response.content)
    return thread


def test_delete_assistant(client: TestClient):
    """Test case for delete_assistant

    Delete an assistant.
    """

    assistant = test_create_assistant(client)
    headers = get_headers(MODEL)

    response = client.request(
        "DELETE",
        "/assistants/{assistant_id}".format(assistant_id=assistant.id),
        headers=headers,
    )

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


@pytest.mark.skip(reason="Not implemented")
def test_delete_assistant_file(client: TestClient):
    """Test case for delete_assistant_file

    Delete an assistant file.
    """

    headers = {
        "Authorization": "Bearer special-key",
    }
    response = client.request(
        "DELETE",
        "/assistants/{assistant_id}/files/{file_id}".format(assistant_id='assistant_id_example', file_id='file_id_example'),
        headers=headers,
    )

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_delete_thread(client: TestClient):
    """Test case for delete_thread

    Delete a thread.
    """

    thread = test_create_thread(client)
    headers = get_headers(MODEL)
    response = client.request(
        "DELETE",
        "/threads/{thread_id}".format(thread_id='thread_id_example'),
        headers=headers,
    )
    logger.info(response)

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


def test_get_assistant(client: TestClient):
    """Test case for get_assistant

    Retrieves an assistant.
    """

    assistant = test_create_assistant(client)
    headers = get_headers(MODEL)
    response = client.request(
        "GET",
        "/assistants/{assistant_id}".format(assistant_id=assistant.id),
        headers=headers,
    )
    logger.info(response)

    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


@pytest.mark.skip(reason="Not implemented")
def test_get_assistant_file(client: TestClient):
    """Test case for get_assistant_file

    Retrieves an AssistantFile.
    """

    headers = {
        "Authorization": "Bearer special-key",
    }
    response = client.request(
        "GET",
        "/assistants/{assistant_id}/files/{file_id}".format(assistant_id='assistant_id_example', file_id='file_id_example'),
        headers=headers,
    )

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_get_message(client: TestClient):
    """Test case for get_message

    Retrieve a message.
    """

    message = test_create_message(client)
    headers = get_headers(MODEL)
    response = client.request(
        "GET",
        "/threads/{thread_id}/messages/{message_id}".format(thread_id=message.thread_id, message_id=message.id),
        headers=headers,
    )

    logger.info(response)
    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


@pytest.mark.skip(reason="Not implemented")
def test_get_message_file(client: TestClient):
    """Test case for get_message_file

    Retrieves a message file.
    """

    headers = {
        "Authorization": "Bearer special-key",
    }
    response = client.request(
        "GET",
        "/threads/{thread_id}/messages/{message_id}/files/{file_id}".format(thread_id='thread_abc123', message_id='msg_abc123', file_id='file-abc123'),
        headers=headers,
    )

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200



@pytest.mark.skip(reason="Not implemented")
def test_get_run_step(client: TestClient):
    """Test case for get_run_step

    Retrieves a run step.
    """

    headers = {
        "Authorization": "Bearer special-key",
    }
    response = client.request(
        "GET",
        "/threads/{thread_id}/runs/{run_id}/steps/{step_id}".format(thread_id='thread_id_example', run_id='run_id_example', step_id='step_id_example'),
        headers=headers,
    )

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_get_thread(client: TestClient):
    """Test case for get_thread

    Retrieves a thread.
    """

    thread = test_create_thread(client)
    headers = get_headers(MODEL)
    response = client.request(
        "GET",
        "/threads/{thread_id}".format(thread_id=thread.id),
        headers=headers,
    )

    logger.info(response)
    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


@pytest.mark.skip(reason="Not implemented")
def test_list_assistant_files(client: TestClient):
    """Test case for list_assistant_files

    Returns a list of assistant files.
    """
    params = [("limit", 20),     ("order", 'desc'),     ("after", 'after_example'),     ("before", 'before_example')]
    headers = {
        "Authorization": "Bearer special-key",
    }
    response = client.request(
        "GET",
        "/assistants/{assistant_id}/files".format(assistant_id='assistant_id_example'),
        headers=headers,
        params=params,
    )

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_list_assistants(client: TestClient):
    """Test case for list_assistants

    Returns a list of assistants.
    """
    params = [("limit", 20),     ("order", 'desc') ]
    headers = get_headers(MODEL)
    response = client.request(
        "GET",
        "/assistants",
        headers=headers,
        params=params,
    )

    logger.info(response)
    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


@pytest.mark.skip(reason="Not implemented")
def test_list_message_files(client: TestClient):
    """Test case for list_message_files

    Returns a list of message files.
    """
    params = [("limit", 20),     ("order", 'desc'),     ("after", 'after_example'),     ("before", 'before_example')]
    headers = {
        "Authorization": "Bearer special-key",
    }
    response = client.request(
        "GET",
        "/threads/{thread_id}/messages/{message_id}/files".format(thread_id='thread_id_example', message_id='message_id_example'),
        headers=headers,
        params=params,
    )

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_list_messages(client: TestClient):
    """Test case for list_messages

    Returns a list of messages for a given thread.
    """
    params = [("limit", 20),     ("order", 'desc')]
    headers = get_headers(MODEL)
    response = client.request(
        "GET",
        "/threads/{thread_id}/messages".format(thread_id='thread_id_example'),
        headers=headers,
        params=params,
    )

    logger.info(response)
    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


@pytest.mark.skip(reason="Not implemented")
def test_list_run_steps(client: TestClient):
    """Test case for list_run_steps

    Returns a list of run steps belonging to a run.
    """
    params = [("limit", 20),     ("order", 'desc'),     ("after", 'after_example'),     ("before", 'before_example')]
    headers = {
        "Authorization": "Bearer special-key",
    }
    response = client.request(
        "GET",
        "/threads/{thread_id}/runs/{run_id}/steps".format(thread_id='thread_id_example', run_id='run_id_example'),
        headers=headers,
        params=params,
    )

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_modify_assistant(client: TestClient):
    """Test case for modify_assistant

    Modifies an assistant.
    """
    modify_assistant_request = {"instructions":"instructions","metadata":"{}","name":"name","file_ids":["file_ids","file_ids","file_ids","file_ids","file_ids"],"description":"description","model":"model","tools":[{"type":"code_interpreter"},{"type":"code_interpreter"},{"type":"code_interpreter"},{"type":"code_interpreter"},{"type":"code_interpreter"}]}

    headers = {
        "Authorization": "Bearer special-key",
    }
    response = client.request(
        "POST",
        "/assistants/{assistant_id}".format(assistant_id='assistant_id_example'),
        headers=headers,
        json=modify_assistant_request,
    )

    # uncomment below to assert the status code of the HTTP response
    #assert response.status_code == 200


def test_modify_message(client: TestClient):
    """Test case for modify_message

    Modifies a message.
    """

    message = test_create_message(client)
    modify_message_request = {"metadata":{}}

    headers = get_headers(MODEL)
    response = client.request(
        "POST",
        "/threads/{thread_id}/messages/{message_id}".format(thread_id=message.thread_id, message_id=message.id),
        headers=headers,
        json=modify_message_request,
    )

    logger.info(response)
    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200

def test_modify_message_content(client: TestClient):
    """Test case for modify_message

    Modifies a message.
    """

    message = test_create_message(client)
    modify_message_request = {"metadata":{}, "content": "puppies"}

    headers = get_headers(MODEL)
    response = client.request(
        "POST",
        "/threads/{thread_id}/messages/{message_id}".format(thread_id=message.thread_id, message_id=message.id),
        headers=headers,
        json=modify_message_request,
    )

    logger.info(response)
    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200

def test_delete_message(client: TestClient):
    """Test case for delete_message

    Deletes a message.
    """

    message = test_create_message(client)

    headers = get_headers(MODEL)
    response = client.request(
        "DELETE",
        "/threads/{thread_id}/messages/{message_id}".format(thread_id=message.thread_id, message_id=message.id),
        headers=headers,
    )

    logger.info(response)
    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


def test_modify_thread(client: TestClient):
    """Test case for modify_thread

    Modifies a thread.
    """

    thread = test_create_thread(client)

    modify_thread_request = {"metadata":{}}

    headers = get_headers(MODEL)
    response = client.request(
        "POST",
        "/threads/{thread_id}".format(thread_id=thread.id),
        headers=headers,
        json=modify_thread_request,
    )

    logger.info(response)
    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200


def test_modify_assistant(client: TestClient):
    """Test case for modify_assistant

    Modifies an assistant.
    """
    assistant = test_create_assistant(client)
    modify_assistant_request = {"instructions":"instructions","metadata":{},"name":"name","file_ids":[],"description":"description","model":"model","tools":[]}

    headers = get_headers(MODEL)
    response = client.request(
        "POST",
        "/assistants/{assistant_id}".format(assistant_id=assistant.id),
        headers=headers,
        json=modify_assistant_request,
    )

    logger.info(response)
    # uncomment below to assert the status code of the HTTP response
    assert response.status_code == 200
