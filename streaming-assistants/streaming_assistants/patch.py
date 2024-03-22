import inspect
import os
import io
from functools import wraps
from types import MethodType, FunctionType
from typing import Callable, Literal, Union, List, Dict, Any, TypedDict, cast, Optional
import contextlib
from openai import Stream, OpenAI, AsyncOpenAI
from openai._base_client import make_request_options
from openai._models import BaseModel
from openai._types import NOT_GIVEN
from openai._utils import maybe_transform
from openai.pagination import SyncCursorPage
from openai.types.beta.thread_create_and_run_params import ThreadMessage

from litellm import utils

LLM_PARAM_AWS_REGION_NAME = "LLM-PARAM-aws-region-name"
LLM_PARAM_AWS_SECRET_ACCESS_KEY = "LLM-PARAM-aws-secret-access-key"
LLM_PARAM_AWS_ACCESS_KEY_ID = "LLM-PARAM-aws-access-key-id"


def is_async(func: Callable) -> bool:
    """Returns true if the callable is async, accounting for wrapped callables"""
    return inspect.iscoroutinefunction(func) or (
            hasattr(func, "__wrapped__") and inspect.iscoroutinefunction(func.__wrapped__)
    )


def wrap_list(original_list):
    @wraps(original_list)
    def sync_list(
            self,
            *args,
            **kwargs,
    ) -> Union[SyncCursorPage[ThreadMessage], Stream[MessageChunk]]:
        thread_id = kwargs.get("thread_id")
        after = kwargs.get("after", NOT_GIVEN)
        before = kwargs.get("before", NOT_GIVEN)
        limit = kwargs.get("limit", NOT_GIVEN)
        order = kwargs.get("order", NOT_GIVEN)
        extra_headers = kwargs.get("extra_headers", None)
        extra_query = kwargs.get("extra_query", None)
        extra_body = kwargs.get("extra_body", None)
        timeout = kwargs.get("timeout", NOT_GIVEN)
        stream = kwargs.get("stream", False)
        if stream:
            if limit is not NOT_GIVEN:
                if limit != 1:
                    raise ValueError("Streaming requests require that the limit parameter is set to 1")
            else:
                limit = 1
            if after is not NOT_GIVEN or before is not NOT_GIVEN:
                raise ValueError("Streaming requests cannot use the after or before parameters")
            if order is not NOT_GIVEN and order != "desc":
                raise ValueError("Streaming requests always use desc order, order asc is invalid")
            return self._get(
                f"/threads/{thread_id}/messages",
                stream=True,
                stream_cls=Stream[MessageChunk],
                cast_to=ThreadMessage,
                options=make_request_options(
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    timeout=timeout,
                    query=maybe_transform(
                        {
                            "after": after,
                            "before": before,
                            "limit": limit,
                            "order": order,
                            "stream": stream,
                        },
                        MessageListWithStreamingParams,
                    ),
                ),
            )
        else:
            # Call the original "list" method for non-streaming requests
            return original_list(*args, **kwargs)

    @wraps(original_list)
    async def async_list(
            self,
            *args,
            **kwargs
    ) -> Union[SyncCursorPage[ThreadMessage], Stream[MessageChunk]]:
        thread_id = kwargs.get("thread_id")
        after = kwargs.get("after", NOT_GIVEN)
        before = kwargs.get("before", NOT_GIVEN)
        limit = kwargs.get("limit", NOT_GIVEN)
        order = kwargs.get("order", NOT_GIVEN)
        extra_headers = kwargs.get("extra_headers", None)
        extra_query = kwargs.get("extra_query", None)
        extra_body = kwargs.get("extra_body", None)
        timeout = kwargs.get("timeout", NOT_GIVEN)
        stream = kwargs.get("stream", False)

        if stream:
            response = await self._get(
                f"/threads/{thread_id}/messages",
                stream=True,
                stream_cls=Stream[MessageChunk],
                cast_to=ThreadMessage,
                options=make_request_options(
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    timeout=timeout,
                    query=maybe_transform(
                        {
                            "after": after,
                            "before": before,
                            "limit": limit,
                            "order": order,
                            "stream": stream,
                        },
                        MessageListWithStreamingParams,
                    ),
                ),
            )
            return response
        else:
            # Call the original "list" method for non-streaming requests
            return await original_list(*args, **kwargs)

    # Check if the original function is async and choose the appropriate wrapper
    func_is_async = is_async(original_list)
    wrapper_function = async_list if func_is_async else sync_list

    # Set documentation for the wrapper function
    wrapper_function.__doc__ = original_list.__doc__

    return wrapper_function



class Delta(BaseModel):
    value: str

class Content(BaseModel):
    delta: Delta
    type: str

class DataMessageChunk(BaseModel):
    id: str
    """message id"""
    object: Literal["thread.message.chunk"]
    """The object type, which is always `list`."""
    content: List[Content]
    """List of content deltas, always use content[0] because n cannot be > 1 for gpt-3.5 and newer"""
    created_at: int
    """The object type, which is always `list`."""
    thread_id: str
    """id for the thread"""
    role: str
    """Role: user or assistant"""
    assistant_id: str
    """assistant id used to generate message, if applicable"""
    run_id: str
    """run id used to generate message, if applicable"""
    file_ids: List[str]
    """files used in RAG for this message, if any"""
    metadata: Dict[str, Any]
    """metadata"""



class MessageChunk(BaseModel):
    object: Literal["list"]
    """The object type, which is always `list`."""

    data: List[DataMessageChunk]
    """A list of messages for the thread.
    """

    first_id: str
    """message id of the first message in the stream
    """

    last_id: str
    """message id of the last message in the stream
    """


class MessageListWithStreamingParams(TypedDict, total=False):
    after: str
    """A cursor for use in pagination.

    `after` is an object ID that defines your place in the list. For instance, if
    you make a list request and receive 100 objects, ending with obj_foo, your
    subsequent call can include after=obj_foo in order to fetch the next page of the
    list.
    """

    before: str
    """A cursor for use in pagination.

    `before` is an object ID that defines your place in the list. For instance, if
    you make a list request and receive 100 objects, ending with obj_foo, your
    subsequent call can include before=obj_foo in order to fetch the previous page
    of the list.
    """

    limit: int
    """A limit on the number of objects to be returned.

    Limit can range between 1 and 100, and the default is 20.
    """

    order: Literal["asc", "desc"]
    """Sort order by the `created_at` timestamp of the objects.

    `asc` for ascending order and `desc` for descending order.
    """
    streaming: bool


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
            kwargs["extra_headers"] = { "embedding-model": model}
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

def patch(client: Union[OpenAI, AsyncOpenAI]):


    if client.base_url == "https://api.openai.com/v1/":
        base_url = os.getenv("base_url") or os.getenv("BASE_URL", "https://open-assistant-ai.astra.datastax.com/v1")
        client.base_url=base_url

    print(f"Patching OpenAI client, it will now communicate to Astra Assistants API: {client.base_url}\nLearn more about Astra at: https://docs.datastax.com/en/astra/astra-db-vector/integrations/astra-assistants-api.html")

    # for astra headers
    patch_methods(client.beta,client)
    client.files.create = MethodType(wrap_file_create_method(client.files.create, client), client.files.create)

    # for stream
    client.beta.threads.messages.list = MethodType(wrap_list(client.beta.threads.messages.list), client.beta.threads.messages)

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

    return client
