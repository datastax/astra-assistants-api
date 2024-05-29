import inspect
import os
import io
from functools import wraps
from types import MethodType, FunctionType
from typing import Callable, Literal, Union, List, Dict, Any, TypedDict, cast, Optional
import contextlib

import httpx
from openai import Stream, OpenAI, AsyncOpenAI
from openai._base_client import make_request_options
from openai._models import BaseModel
from openai._types import NOT_GIVEN, Headers, Query, Body, NotGiven
from openai._utils import maybe_transform
from openai.pagination import SyncCursorPage
from openai.types.beta.thread_create_and_run_params import ThreadMessage

from litellm import utils
from openai.types.beta.threads import message_create_params, Message

from dotenv import load_dotenv

load_dotenv("./.env")



LLM_PARAM_AWS_REGION_NAME = "LLM-PARAM-aws-region-name"
LLM_PARAM_AWS_SECRET_ACCESS_KEY = "LLM-PARAM-aws-secret-access-key"
LLM_PARAM_AWS_ACCESS_KEY_ID = "LLM-PARAM-aws-access-key-id"

DOCS_URL="https://docs.datastax.com/en/astra-db-serverless/tutorials/astra-assistants-api.html"


BETA_HEADER = {"OpenAI-Beta": "assistants=v2"}

def is_async(func: Callable) -> bool:
    """Returns true if the callable is async, accounting for wrapped callables"""
    return inspect.iscoroutinefunction(func) or (
            hasattr(func, "__wrapped__") and inspect.iscoroutinefunction(func.__wrapped__)
    )

def wrap_update_messages(original_update):
    @wraps(original_update)
    def sync_update(self, *args, **kwargs):
        thread_id = kwargs.get("thread_id")
        message_id = kwargs.get("message_id")
        content = kwargs.get("content", NOT_GIVEN)
        role = kwargs.get("role", NOT_GIVEN)
        attachments = kwargs.get("attachments", NOT_GIVEN)
        metadata = kwargs.get("metadata", NOT_GIVEN)
        extra_headers = kwargs.get("extra_headers", None)
        extra_headers = {**BETA_HEADER, **(extra_headers or {})}
        extra_query = kwargs.get("extra_query", None)
        extra_body = kwargs.get("extra_body", None)
        timeout = kwargs.get("timeout", NOT_GIVEN)


        return self._post(
            f"/threads/{thread_id}/messages/{message_id}",
            body=maybe_transform(
                {
                    "content": content,
                    "role": role,
                    "attachments": attachments,
                    "metadata": metadata,
                },
                message_create_params.MessageCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Message,
        )


    @wraps(original_update)
    async def async_update(self, *args, **kwargs):
        thread_id = kwargs.get("thread_id")
        message_id = kwargs.get("message_id")
        content = kwargs.get("content", NOT_GIVEN)
        role = kwargs.get("role", NOT_GIVEN)
        attachments = kwargs.get("attachments", NOT_GIVEN)
        metadata = kwargs.get("metadata", NOT_GIVEN)
        extra_headers = kwargs.get("extra_headers", None)
        extra_headers = {**BETA_HEADER, **(extra_headers or {})}
        extra_query = kwargs.get("extra_query", None)
        extra_body = kwargs.get("extra_body", None)
        timeout = kwargs.get("timeout", NOT_GIVEN)

        return await self._post(
            f"/threads/{thread_id}/messages/{message_id}",
            body=maybe_transform(
                {
                    "content": content,
                    "role": role,
                    "attachments": attachments,
                    "metadata": metadata,
                },
                message_create_params.MessageCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Message,
        )


    # Check if the original function is async and choose the appropriate wrapper
    func_is_async = is_async(original_update)
    wrapper_function = async_update if func_is_async else sync_update

    # Set documentation for the wrapper function
    wrapper_function.__doc__ = original_update.__doc__

    return wrapper_function

class MessageDeleted(BaseModel):
    id: str

    deleted: bool

    object: Literal["thread.message.deleted"]


def sync_delete(
            self,
            thread_id: str,
            message_id: str,
            *,
            # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
            # The extra values given here take precedence over values defined on the client or passed to this method.
            extra_headers: Headers | None = None,
            extra_query: Query | None = None,
            extra_body: Body | None = None,
            timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> MessageDeleted:
    """Synchronous version of delete"""
    if not thread_id:
        raise ValueError(f"Expected a non-empty value for `thread_id` but received {thread_id!r}")
    if not message_id:
        raise ValueError(f"Expected a non-empty value for `thread_id` but received {thread_id!r}")

    extra_headers = {**BETA_HEADER, **(extra_headers or {})}
    url = f"/threads/{thread_id}/messages/{message_id}"
    return self._delete(
        url,
        options=make_request_options(
            extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
        ),
        cast_to=MessageDeleted,
    )


async def async_delete(
        self,
        thread_id: str,
        message_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
) -> MessageDeleted:
    """Synchronous version of delete"""
    if not thread_id:
        raise ValueError(f"Expected a non-empty value for `thread_id` but received {thread_id!r}")
    if not message_id:
        raise ValueError(f"Expected a non-empty value for `thread_id` but received {thread_id!r}")
    extra_headers = {**BETA_HEADER, **(extra_headers or {})}
    url = f"/threads/{thread_id}/messages/{message_id}"
    return await self._delete(
        url,
        options=make_request_options(
            extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
        ),
        cast_to=MessageDeleted,
    )


def delete_message(self, thread_id: str, message_id: str, **kwargs):
    """
    Delete a message from a thread.

    Args:
        thread_id (str): The ID of the thread.
        message_id (str): The ID of the message to delete.
        **kwargs: Additional keyword arguments to pass to the underlying API call.

    Returns:
        A response object containing the result of the API call.
    """
    url = f"/threads/{thread_id}/messages/{message_id}"
    if is_async(self.list):
        return async_delete(self, thread_id, message_id, **kwargs)
    else:
        return sync_delete(self, thread_id, message_id, **kwargs)


def wrap_create(original_create, client):
    @wraps(original_create)
    def patched_create(self, *args, **kwargs):
        # Assuming the argument we"re interested in is named "special_argument"
        model = kwargs.get("model")

        assistant_id = kwargs.get("assistant_id")
        if assistant_id is not None and "beta.threads.runs" in str(type(self)):
            print(assistant_id)
            assistant = client.beta.assistants.retrieve(assistant_id)
            model = assistant.model
            if(
                getattr(assistant, 'tool_resources', None) and
                getattr(assistant.tool_resources, 'file_search', None) and
                getattr(assistant.tool_resources.file_search, 'vector_store_ids', None)
            ):
                # TODO figure out how to get the model from the tool resources
                vector_store_id = assistant.tool_resources.file_search.vector_store_ids[0]
                vector_store = client.beta.vector_stores.retrieve(vector_store_id)
                #file_id = assistant.file_ids[0]
                #file = client.files.retrieve(file_id)
                #if file.embedding_model is not None:
                #    extra_headers = kwargs.get("extra_headers", None)
                #    extra_headers = {**BETA_HEADER, "embedding-model": file.embedding_model, **(extra_headers or {})}
                #    kwargs["extra_headers"] = extra_headers

        if model is not None:
            try:
                assign_key_based_on_model(model, client)
            except Exception as e:
                raise RuntimeError(f"Invalid model {model} or key. Make sure you set the right environment variable.") from None

        # Call the original "create" method
        result = original_create(*args, **kwargs)

        return result
    return patched_create

def wrap_file_create(original_create, client):
    @wraps(original_create)
    def patched_create(self, *args, **kwargs):
        # Assuming the argument we"re interested in is named "special_argument"
        model = kwargs.get("embedding_model")
        if model is not None:
            extra_headers = kwargs.get("extra_headers", None)
            extra_headers = {**BETA_HEADER, "embedding-model": model, **(extra_headers or {})}
            kwargs["extra_headers"] = extra_headers
            kwargs.pop("embedding_model")
            try:
                assign_key_based_on_model(model, client)
            except Exception as e:
                raise RuntimeError(f"Invalid model {model} or key. Make sure you set the right environment variable.") from None
        else:
            if kwargs.get("extra_headers") is not None:
                if "embedding_model" in kwargs.get("extra_headers"):
                    kwargs.get("extra_headers").pop("embedding-model")
            if "api-key" in client._custom_headers:
                client._custom_headers.pop("api-key")
        # Call the original "create" method
        result = original_create(*args, **kwargs)

        return result
    return patched_create

def assign_key_based_on_model(model, client):
    with contextlib.redirect_stdout(io.StringIO()):
        key = None
        triple = utils.get_llm_provider(model)
        provider = triple[1]
        dynamic_key = triple[2]
        if provider == "cohere_chat":
            provider = "cohere"
        if provider == "bedrock":
            if os.getenv("AWS_ACCESS_KEY_ID") is None or os.getenv("AWS_SECRET_ACCESS_KEY") is None or os.getenv("AWS_REGION_NAME") is None:
                raise Exception("For bedrock models you must set the AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_REGION_NAME environment variables")
            client._custom_headers[LLM_PARAM_AWS_ACCESS_KEY_ID] = os.getenv("AWS_ACCESS_KEY_ID")
            client._custom_headers[LLM_PARAM_AWS_SECRET_ACCESS_KEY] = os.getenv("AWS_SECRET_ACCESS_KEY")
            client._custom_headers[LLM_PARAM_AWS_REGION_NAME] = os.getenv("AWS_REGION_NAME")
        else:
            if LLM_PARAM_AWS_ACCESS_KEY_ID in client._custom_headers:
                client._custom_headers.pop(LLM_PARAM_AWS_ACCESS_KEY_ID)
            if LLM_PARAM_AWS_SECRET_ACCESS_KEY in client._custom_headers:
                client._custom_headers.pop(LLM_PARAM_AWS_SECRET_ACCESS_KEY)
            if LLM_PARAM_AWS_REGION_NAME in client._custom_headers:
                client._custom_headers.pop(LLM_PARAM_AWS_REGION_NAME)
        if provider != "openai":
            key = utils.get_api_key(provider, dynamic_key)
        if provider == "gemini":
            key = os.getenv("GEMINI_API_KEY")
        if key is not None:
            client._custom_headers["api-key"] = key
        else:
            if "api-key" in client._custom_headers:
                client._custom_headers.pop("api-key")
    return client


def add_astra_header(client):
    ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    if "astra-api-token" in client.default_headers:
        return client
    if ASTRA_DB_APPLICATION_TOKEN is None:
        raise Exception("ASTRA_DB_APPLICATION_TOKEN is not set, this setting is required to use stateful endpoints with the Astra Assistants API\nGet your token from https://astra.datastax.com/")
    client._custom_headers["astra-api-token"] = ASTRA_DB_APPLICATION_TOKEN


def wrap_method(original_method, client):
    @wraps(original_method)
    def patched_method(self, *args, **kwargs):

        add_astra_header(client)

        result = original_method(self, *args, **kwargs)
        return result

    return patched_method

def wrap_file_create_method(original_method, client):
    @wraps(original_method)
    def patched_method(self, *args, **kwargs):

        add_astra_header(client)

        result = original_method(*args, **kwargs)
        return result

    return patched_method


def patch_methods(obj, client, visited=None):
    """
    Recursively patch methods of an object to modify `client.default_headers` before calling.
    """
    if visited is None:
        visited = set()

    if obj in visited:
        return
    visited.add(obj)

    for attr_name in dir(obj):
        # Avoid patching special methods and prevent infinite recursion
        if attr_name.startswith("_"):
            continue

        attr = getattr(obj, attr_name)

        # Patch methods
        if isinstance(attr, MethodType):
            original_method = attr.__func__
            patched_method = wrap_method(original_method, client)
            setattr(obj, attr_name, MethodType(patched_method, obj))
            visited.add(obj)
        # Recursively patch nested objects, excluding known non-object types
        elif hasattr(attr, "__dict__") and not isinstance(attr, (str, int, float, list, dict, set, tuple)):
            patch_methods(attr, client, visited)


def enhance_copy_method(original_copy, client):
    def enhanced_copy(self, *args, **kwargs):
        copied_instance = original_copy(*args, **kwargs)
        patch(copied_instance)
        return copied_instance
    return enhanced_copy

def patch(client: Union[OpenAI, AsyncOpenAI]):


    if client.base_url == "https://api.openai.com/v1/":
        base_url = os.getenv("base_url") or os.getenv("BASE_URL", "https://open-assistant-ai.astra.datastax.com/v1")
        client.base_url=base_url

    print(f"Patching OpenAI client, it will now communicate to Astra Assistants API: {client.base_url}\nLearn more about Astra at: {DOCS_URL}")

    # for astra headers (all beta endpoints)
    patch_methods(client.beta,client)
    # for astra headers (file endpoints (not beta))
    patch_methods(client.files,client)

    # for model api_key derivation
    methods_to_wrap_with_model_arg = [
        client.beta.assistants.create,
        client.chat.completions.create,
        client.embeddings.create,
        client.beta.threads.runs.create,
        client.beta.threads.runs.create_and_stream,
    ]
    for original_method in methods_to_wrap_with_model_arg:
        bound_instance = original_method.__self__
        method_name = original_method.__name__

        setattr(bound_instance, method_name, MethodType(wrap_create(original_method, client), bound_instance))

    # fancy model / embedding_model derivation for files
    client.files.create = MethodType(wrap_file_create(client.files.create, client), client.files.create)

    # support message deletion
    client.beta.threads.messages.delete = MethodType(delete_message, client.beta.threads.messages)

    # Wrap client.beta.threads.messages.update to support modifying content
    client.beta.threads.messages.update = MethodType(wrap_update_messages(client.beta.threads.messages.update), client.beta.threads.messages)

    # patch the copy method so that the copied instance is also patched
    client.copy = MethodType(enhance_copy_method(client.copy, client), client.copy)
    client.with_options = client.copy

    return client