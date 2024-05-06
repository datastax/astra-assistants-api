# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from openapi_server.models.assistant_object import AssistantObject
from openapi_server.models.create_assistant_request import CreateAssistantRequest
from openapi_server.models.create_message_request import CreateMessageRequest
from openapi_server.models.create_run_request import CreateRunRequest
from openapi_server.models.create_thread_and_run_request import CreateThreadAndRunRequest
from openapi_server.models.create_thread_request import CreateThreadRequest
from openapi_server.models.delete_assistant_response import DeleteAssistantResponse
from openapi_server.models.delete_message_response import DeleteMessageResponse
from openapi_server.models.delete_thread_response import DeleteThreadResponse
from openapi_server.models.list_assistants_response import ListAssistantsResponse
from openapi_server.models.list_messages_response import ListMessagesResponse
from openapi_server.models.list_run_steps_response import ListRunStepsResponse
from openapi_server.models.list_runs_response import ListRunsResponse
from openapi_server.models.message_object import MessageObject
from openapi_server.models.modify_assistant_request import ModifyAssistantRequest
from openapi_server.models.modify_message_request import ModifyMessageRequest
from openapi_server.models.modify_run_request import ModifyRunRequest
from openapi_server.models.modify_thread_request import ModifyThreadRequest
from openapi_server.models.run_object import RunObject
from openapi_server.models.run_step_object import RunStepObject
from openapi_server.models.submit_tool_outputs_run_request import SubmitToolOutputsRunRequest
from openapi_server.models.thread_object import ThreadObject
from openapi_server.security_api import get_token_ApiKeyAuth

class BaseAssistantsApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseAssistantsApi.subclasses = BaseAssistantsApi.subclasses + (cls,)
    def cancel_run(
        self,
        thread_id: str,
        run_id: str,
    ) -> RunObject:
        ...


    def create_assistant(
        self,
        create_assistant_request: CreateAssistantRequest,
    ) -> AssistantObject:
        ...


    def create_message(
        self,
        thread_id: str,
        create_message_request: CreateMessageRequest,
    ) -> MessageObject:
        ...


    def create_run(
        self,
        thread_id: str,
        create_run_request: CreateRunRequest,
    ) -> RunObject:
        ...


    def create_thread(
        self,
        create_thread_request: CreateThreadRequest,
    ) -> ThreadObject:
        ...


    def create_thread_and_run(
        self,
        create_thread_and_run_request: CreateThreadAndRunRequest,
    ) -> RunObject:
        ...


    def delete_assistant(
        self,
        assistant_id: str,
    ) -> DeleteAssistantResponse:
        ...


    def delete_message(
        self,
        thread_id: str,
        message_id: str,
    ) -> DeleteMessageResponse:
        ...


    def delete_thread(
        self,
        thread_id: str,
    ) -> DeleteThreadResponse:
        ...


    def get_assistant(
        self,
        assistant_id: str,
    ) -> AssistantObject:
        ...


    def get_message(
        self,
        thread_id: str,
        message_id: str,
    ) -> MessageObject:
        ...


    def get_run(
        self,
        thread_id: str,
        run_id: str,
    ) -> RunObject:
        ...


    def get_run_step(
        self,
        thread_id: str,
        run_id: str,
        step_id: str,
    ) -> RunStepObject:
        ...


    def get_thread(
        self,
        thread_id: str,
    ) -> ThreadObject:
        ...


    def list_assistants(
        self,
        limit: int,
        order: str,
        after: str,
        before: str,
    ) -> ListAssistantsResponse:
        ...


    def list_messages(
        self,
        thread_id: str,
        limit: int,
        order: str,
        after: str,
        before: str,
        run_id: str,
    ) -> ListMessagesResponse:
        ...


    def list_run_steps(
        self,
        thread_id: str,
        run_id: str,
        limit: int,
        order: str,
        after: str,
        before: str,
    ) -> ListRunStepsResponse:
        ...


    def list_runs(
        self,
        thread_id: str,
        limit: int,
        order: str,
        after: str,
        before: str,
    ) -> ListRunsResponse:
        ...


    def modify_assistant(
        self,
        assistant_id: str,
        modify_assistant_request: ModifyAssistantRequest,
    ) -> AssistantObject:
        ...


    def modify_message(
        self,
        thread_id: str,
        message_id: str,
        modify_message_request: ModifyMessageRequest,
    ) -> MessageObject:
        ...


    def modify_run(
        self,
        thread_id: str,
        run_id: str,
        modify_run_request: ModifyRunRequest,
    ) -> RunObject:
        ...


    def modify_thread(
        self,
        thread_id: str,
        modify_thread_request: ModifyThreadRequest,
    ) -> ThreadObject:
        ...


    def submit_tool_ouputs_to_run(
        self,
        thread_id: str,
        run_id: str,
        submit_tool_outputs_run_request: SubmitToolOutputsRunRequest,
    ) -> RunObject:
        ...
