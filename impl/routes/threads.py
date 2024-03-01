import asyncio
import json
import logging
import os
import time
import re
from datetime import datetime
from typing import Any, Dict, Union
from uuid import uuid1

from cassandra.query import UNSET_VALUE
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    HTTPException,
    Path,
    Query,
)
from fastapi.encoders import jsonable_encoder
from starlette.responses import StreamingResponse

from impl.astra_vector import CassandraClient
from impl.routes.utils import verify_db_client, get_litellm_kwargs, infer_embedding_model, infer_embedding_api_key
from impl.services.inference_utils import get_chat_completion, get_async_chat_completion_response
from openapi_server.models.create_message_request import CreateMessageRequest
from openapi_server.models.create_run_request import CreateRunRequest
from openapi_server.models.create_thread_request import CreateThreadRequest
from openapi_server.models.delete_thread_response import DeleteThreadResponse
from openapi_server.models.list_messages_response import ListMessagesResponse
from openapi_server.models.list_messages_stream_response import ListMessagesStreamResponse
from openapi_server.models.list_runs_response import ListRunsResponse
from openapi_server.models.message_content_delta_object import MessageContentDeltaObject
from openapi_server.models.message_content_delta_object_text import MessageContentDeltaObjectText
from openapi_server.models.message_content_text_object_text import (
    MessageContentTextObjectText,
)
from openapi_server.models.message_object import MessageObject
from openapi_server.models.message_object_content_inner import MessageObjectContentInner
from openapi_server.models.message_stream_response_object import MessageStreamResponseObject
from openapi_server.models.modify_message_request import ModifyMessageRequest
from openapi_server.models.modify_thread_request import ModifyThreadRequest
from openapi_server.models.run_object import RunObject
from openapi_server.models.run_object_required_action import RunObjectRequiredAction
from openapi_server.models.run_object_required_action_submit_tool_outputs import RunObjectRequiredActionSubmitToolOutputs
from openapi_server.models.run_tool_call_object import RunToolCallObject
from openapi_server.models.run_tool_call_object_function import RunToolCallObjectFunction
from openapi_server.models.submit_tool_outputs_run_request import SubmitToolOutputsRunRequest
from openapi_server.models.thread_object import ThreadObject

router = APIRouter()

logger = logging.getLogger(__name__)

@router.post(
    "/threads",
    responses={
        200: {"model": ThreadObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Create a thread.",
    response_model_by_alias=True,
)
async def create_thread(
    create_thread_request: CreateThreadRequest = Body(None, description=""),
    astradb: CassandraClient = Depends(verify_db_client),
) -> ThreadObject:
    created_at = int(time.mktime(datetime.now().timetuple()))
    thread_id = str(uuid1())

    metadata = {}
    if create_thread_request.metadata is not None:
        metadata = create_thread_request.metadata

    if create_thread_request.messages is not None:
        for message in create_thread_request.messages:
            message_id = str(uuid1())
            created_at = int(time.mktime(datetime.now().timetuple()))
            file_ids = []
            if message.file_ids is not None:
                file_ids = message.file_ids
            metadata = {}
            if message.metadata is not None:
                metadata = message.metadata

            astradb.upsert_message(
                id=message_id,
                object="thread.message",
                created_at=created_at,
                thread_id=thread_id,
                role=message.role,
                content=[message.content],
                assistant_id="TODO",
                run_id="None",
                file_ids=file_ids,
                metadata=metadata,
            )


    return astradb.upsert_thread(
        id=thread_id, object="thread", created_at=created_at, metadata=metadata
    )


@router.post(
    "/threads/{thread_id}/messages",
    responses={
        200: {"model": MessageObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Create a message.",
    response_model_by_alias=True,
)
async def create_message(
    thread_id: str = Path(
        ...,
        description="The ID of the [thread](/docs/api-reference/threads) to create a message for.",
    ),
    create_message_request: CreateMessageRequest = Body(None, description=""),
    astradb: CassandraClient = Depends(verify_db_client),
) -> MessageObject:
    created_at = int(time.mktime(datetime.now().timetuple()))
    id = str(uuid1())

    return astradb.upsert_message(
        id=id,
        object="thread.message",
        created_at=created_at,
        thread_id=thread_id,
        role=create_message_request.role,
        content=[create_message_request.content],
        assistant_id="TODO",
        run_id="None",
        file_ids=create_message_request.file_ids,
        metadata=create_message_request.metadata,
    )


@router.get(
    "/threads/{thread_id}/messages/{message_id}",
    responses={
        200: {"model": MessageObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Retrieve a message.",
    response_model_by_alias=True,
)
async def get_message(
        thread_id: str = Path(..., description="The ID of the [thread](/docs/api-reference/threads) to which this message belongs."),
        message_id: str = Path(..., description="The ID of the message to retrieve."),
        astradb: CassandraClient = Depends(verify_db_client),
) -> MessageObject:
    return astradb.get_message(thread_id=thread_id, message_id=message_id)


@router.post(
    "/threads/{thread_id}/messages/{message_id}",
    responses={
        200: {"model": MessageObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Modifies a message.",
    response_model_by_alias=True,
)
async def modify_message(
        thread_id: str = Path(..., description="The ID of the thread to which this message belongs."),
        message_id: str = Path(..., description="The ID of the message to modify."),
        modify_message_request: ModifyMessageRequest = Body(None, description=""),
        astradb: CassandraClient = Depends(verify_db_client),
) -> MessageObject:
    return astradb.upsert_message(
        id=message_id,
        object="thread.message",
        created_at=None,
        thread_id=thread_id,
        role=None,
        content=None,
        assistant_id=None,
        run_id=None,
        file_ids=None,
        metadata=modify_message_request.metadata,
    )


def extractFunctionArguments(content):
    pattern = r"\`\`\`.*({.*})\n\`\`\`"
    match = re.search(pattern, content, re.S)
    if match:
        extracted_text = match.group(1).strip()
        return extracted_text
    else:
        try:
            json.loads(content)
            return content
        except Exception as e:
            logger.error(e)
            raise ValueError("Could not extract function arguments from LLM response, may have not been properly formatted. Consider retrying or use a different model.")

def extractFunctionName(content: str, candidates: [str]):
    candidates = "|".join(candidates)
    pattern = fr"({candidates})"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()
        return extracted_text
    else:
        raise ValueError("Could not extract function name from LLM response, may not have been properly formatted")



@router.post(
    "/threads/{thread_id}/runs",
    responses={
        200: {"model": RunObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Create a run.",
    response_model_by_alias=True,
)
async def create_run(
    background_tasks: BackgroundTasks,
    thread_id: str = Path(..., description="The ID of the thread to run."),
    create_run_request: CreateRunRequest = Body(None, description=""),
    litellm_kwargs: Dict[str, Any] = Depends(get_litellm_kwargs),
    astradb: CassandraClient = Depends(verify_db_client),
    embedding_model: str = Depends(infer_embedding_model),
    embedding_api_key: str = Depends(infer_embedding_api_key),
) -> RunObject:

    # TODO: implement thread locking for in-progress runs
    # New Messages cannot be added to the Thread.
    # New Runs cannot be created on the Thread.
    created_at = int(time.mktime(datetime.now().timetuple()))
    run_id = str(uuid1())
    status = "queued"

    tools = create_run_request.tools
    if tools is None:
        tools = []

    metadata = create_run_request.metadata
    if metadata is None:
        metadata = {}

    file_ids = []

    model = create_run_request.model
    # TODO: implement support for ChatCompletionToolChoiceOption
    assistant = astradb.get_assistant(id=create_run_request.assistant_id)
    if model is None:
        if assistant is None:
            raise HTTPException(status_code=404, detail="Assistant not found")
        model = assistant.model
        file_ids = assistant.file_ids
        tools = assistant.tools

    messages = get_messages_by_thread(astradb, thread_id, order="asc")



    instructions = create_run_request.instructions
    if instructions is None:
        instructions = assistant.instructions

    toolsJson = []
    if len(tools) == 0:
        message_id = str(uuid1())
        created_at = int(time.mktime(datetime.now().timetuple()))

        # initialize message
        astradb.upsert_message(
            id=message_id,
            object="thread.message",
            created_at=created_at,
            thread_id=thread_id,
            role="assistant",
            content=[],
            assistant_id=assistant.id,
            run_id=run_id,
            file_ids=file_ids,
            metadata={},
        )
        background_tasks.add_task(
            process_rag,
            run_id,
            thread_id,
            file_ids,
            messages.data,
            model,
            instructions,
            astradb,
            litellm_kwargs,
            embedding_model,
            assistant.id,
            message_id,
            embedding_api_key
        )
        status = "generating"


    for tool in tools:
       if tool.type == "retrieval":
           message_id = str(uuid1())
           created_at = int(time.mktime(datetime.now().timetuple()))

           # initialize message
           astradb.upsert_message(
               id=message_id,
               object="thread.message",
               created_at=created_at,
               thread_id=thread_id,
               role="assistant",
               content=[],
               assistant_id=assistant.id,
               run_id=run_id,
               file_ids=file_ids,
               metadata={},
           )
           # async calls to rag
           background_tasks.add_task(
               process_rag,
               run_id,
               thread_id,
               file_ids,
               messages.data,
               model,
               instructions,
               astradb,
               litellm_kwargs,
               embedding_model,
               assistant.id,
               message_id,
               embedding_api_key
           )
           status = "generating"
       if tool.type == "function":
            toolsJson.append(tool.function.dict())


    required_action=None

    if len(toolsJson) > 0:
        litellm_kwargs["functions"] = toolsJson
        message_string, message_content = summarize_message_content(instructions, messages.data)
        message = await get_chat_completion(messages=message_content, model=model, **litellm_kwargs)

        tool_call_object_id = str(uuid1())
        run_tool_calls = []
        if 'function_call' in message.model_fields_set:
            function_call = RunToolCallObjectFunction(name=message.function_call.name, arguments=message.function_call.arguments)
        else:
            arguments = extractFunctionArguments(message.content)

            candidates = [tool['name'] for tool in toolsJson]
            name = extractFunctionName(message.content, candidates)

            function_call = RunToolCallObjectFunction(name=name, arguments=arguments)
        run_tool_calls.append(RunToolCallObject(id=tool_call_object_id, type='function', function=function_call))
        tool_outputs = RunObjectRequiredActionSubmitToolOutputs(tool_calls=run_tool_calls)
        required_action = RunObjectRequiredAction(type='submit_tool_outputs', submit_tool_outputs=tool_outputs).json()
        status = "requires_action"

        message_id = str(uuid1())
        created_at = int(time.mktime(datetime.now().timetuple()))

        # persist message
        astradb.upsert_message(
            id=message_id,
            object="thread.message",
            created_at=created_at,
            thread_id=thread_id,
            role=message.role,
            content=[message.content],
            assistant_id=assistant.id,
            run_id=run_id,
            file_ids=file_ids,
            metadata={},
        )

    run = astradb.upsert_run(
        id=run_id,
        object="thread.run",
        created_at=created_at,
        thread_id=thread_id,
        assistant_id=create_run_request.assistant_id,
        status=status,
        required_action=required_action,
        last_error=None,
        expires_at=0,
        started_at=0,
        cancelled_at=0,
        failed_at=0,
        completed_at=0,
        model=model,
        instructions=instructions,
        tools=tools,
        file_ids=file_ids,
        metadata=metadata,
    )
    logger.info(f"created run {run.id} for thread {run.thread_id}")
    return run

def summarize_message_content(instructions, messages):
    message_content = []
    if instructions is None:
        instructions = ""
    message_content.append({"role": "system", "content": instructions})
    for message in messages:
        role = message.role
        contentList = message.content
        for content in contentList:
            message_content.append({"role": role, "content": content.text.value})

    # filter messages to only include user messages
    userContent = [message for message in message_content if message["role"] == "user"]

    message_string = ""
    if len(userContent) > 0:
        message_string = userContent[0]["content"]
    return message_string, message_content # maybe trim message history?

async def process_rag(
    run_id, thread_id, file_ids, messages, model, instructions, astradb, litellm_kwargs, embedding_model, assistant_id, message_id, embedding_api_key
):
    try:
        # TODO: Deal with run status better
        message_string, message_content = summarize_message_content(instructions, messages)
        # TODO incorporate file_ids into the search using where in
        if len(file_ids) > 0:
            context_json = astradb.annSearch(
                table="file_chunks",
                vector_index_column="embedding",
                search_string=message_string,
                partitions=file_ids,
                litellm_kwargs=litellm_kwargs,
                embedding_model=embedding_model,
                embedding_api_key=embedding_api_key,
            )

            user_message = message_content.pop()
            message_content.append({"role": "system", "content": "Important, ALWAYS use the following information to craft your responses. Include the relevant file_id and chunk_id references in parenthesis as part of your response i.e. (chunk_id dbd94b44-cb13-11ee-a868-fd1abaa6ff88_18)."})
            for context in context_json:
                # TODO improve citations https://platform.openai.com/docs/guides/prompt-engineering/six-strategies-for-getting-better-results
                content = (
                    "the information below comes from file_id - "
                    + context["file_id"]
                    + " and chunk_id - "
                    + context["chunk_id"]
                    + ":\n"
                    + context["content"]
                    + ":\n"
                )
                message_content.append({"role": "system", "content": content})
            message_content.append(user_message)

        litellm_kwargs["stream"] = True

        logger.info(f"generating for message_content: {message_content}")
        response = await get_async_chat_completion_response(
            messages=message_content,
            model=model,
            **litellm_kwargs,
        )
        text = ""
        created_at = UNSET_VALUE
        start_time = time.time()
        frequency_in_seconds = 1

        if 'gemini' in model:
            for part in response:
                text += part.candidates[0].content.parts[0].text
                start_time = await maybe_checkpoint(assistant_id, astradb, created_at, file_ids, frequency_in_seconds, message_id,
                                       run_id, start_time, text, thread_id)
        else:
            done = False
            while not done:
                async for part in response:
                    if part.choices[0].finish_reason is not None:
                        done = True
                    delta = part.choices[0].delta.content
                    if delta is not None and isinstance(delta, str):
                        text += delta
                    start_time = await maybe_checkpoint(assistant_id, astradb, created_at, file_ids, frequency_in_seconds, message_id,
                                           run_id, start_time, text, thread_id)

        # final message upsert
        astradb.upsert_message(
            id=message_id,
            object="thread.message",
            created_at=created_at,
            thread_id=thread_id,
            role="assistant",
            content=[text],
            assistant_id=assistant_id,
            run_id=run_id,
            file_ids=file_ids,
            metadata={},
        )

        astradb.update_run_status(thread_id=thread_id, id=run_id, status="completed")
        logger.info(f"run completed run_id {run_id} thread_id {thread_id}")
    except Exception as e:
        astradb.update_run_status(thread_id=thread_id, id=run_id, status="failed")
        logger.error(e)
        raise


async def maybe_checkpoint(assistant_id, astradb, created_at, file_ids, frequency_in_seconds, message_id, run_id,
                           start_time, text, thread_id):
    current_time = time.time()
    if current_time - start_time >= frequency_in_seconds:
        logger.info("Checkpointing message")
        logger.info(f"text: {text}")

        astradb.upsert_message(
            id=message_id,
            object="thread.message",
            created_at=created_at,
            thread_id=thread_id,
            role="assistant",
            content=[text],
            assistant_id=assistant_id,
            run_id=run_id,
            file_ids=file_ids,
            metadata={},
        )
        start_time = time.time()
        return start_time
    return start_time


@router.get(
    "/threads/{thread_id}/runs",
    responses={
        200: {"model": ListRunsResponse, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Returns a list of runs belonging to a thread.",
    response_model_by_alias=True,
)
async def list_runs(
    thread_id: str = Path(..., description="The ID of the thread the run belongs to."),
    limit: int = Query(
        20,
        description="A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20. ",
    ),
    order: str = Query(
        "desc",
        description="Sort order by the &#x60;created_at&#x60; timestamp of the objects. &#x60;asc&#x60; for ascending order and &#x60;desc&#x60; for descending order. ",
    ),
    after: str = Query(
        None,
        description="A cursor for use in pagination. &#x60;after&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include after&#x3D;obj_foo in order to fetch the next page of the list. ",
    ),
    before: str = Query(
        None,
        description="A cursor for use in pagination. &#x60;before&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include before&#x3D;obj_foo in order to fetch the previous page of the list. ",
    ),
    astradb: CassandraClient = Depends(verify_db_client),
) -> ListRunsResponse:
    # TODO fix data model to support limit and sort
    raw_runs = astradb.selectFromTableByPK(
        table="runs", partitionKeys=["thread_id"], args={"thread_id": thread_id}
    )
    if order is None or order == "desc":
        # sort by created_at
        raw_runs = sorted(raw_runs, key=lambda k: k["created_at"], reverse=True)
    else:
        raw_runs = sorted(raw_runs, key=lambda k: k["created_at"], reverse=False)

    if limit is not None:
        raw_runs = raw_runs[:limit]

    runs = []
    for run in raw_runs:
        created_at = int(run["created_at"].timestamp() * 1000)
        expires_at = int(run["expires_at"].timestamp() * 1000)
        started_at = int(run["started_at"].timestamp() * 1000)
        cancelled_at = int(run["cancelled_at"].timestamp() * 1000)
        completed_at = int(run["completed_at"].timestamp() * 1000)
        failed_at = int(run["failed_at"].timestamp() * 1000)
        required_action = run["required_action"]
        required_action_object = None
        if required_action is not None:
            required_action_object = RunObjectRequiredAction.parse_raw(required_action)


        tools = []
        if run["tools"]:
            tools = run["tools"]

        metadata = run["metadata"]
        if metadata is None:
            metadata = {}

        file_ids = run["file_ids"]
        if file_ids is None:
            file_ids = []

        tools = run["tools"]
        if tools is None:
            tools = []

        runs.append(
            RunObject(
                id=run["id"],
                object="run",
                created_at=created_at,
                thread_id=run["thread_id"],
                assistant_id=run["assistant_id"],
                status=run["status"],
                required_action=required_action_object,
                last_error=run["last_error"],
                expires_at=expires_at,
                started_at=started_at,
                cancelled_at=cancelled_at,
                failed_at=failed_at,
                completed_at=completed_at,
                model=run["model"],
                instructions=run["instructions"],
                tools=tools,
                file_ids=file_ids,
                metadata=metadata,
            )
        )
    if len(raw_runs) == 0:
        return ListRunsResponse(
            data=runs, object="runs", first_id="none", last_id="none", has_more=False
        )
    first_id = raw_runs[0]["id"]
    last_id = raw_runs[len(raw_runs) - 1]["id"]
    return ListRunsResponse(
        data=runs, object="runs", first_id=first_id, last_id=last_id, has_more=False
    )


@router.get(
    "/threads/{thread_id}/runs/{run_id}",
    responses={
        200: {"model": RunObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Retrieves a run.",
    response_model_by_alias=True,
)
async def get_run(
    thread_id: str = Path(
        ...,
        description="The ID of the [thread](/docs/api-reference/threads) that was run.",
    ),
    run_id: str = Path(..., description="The ID of the run to retrieve."),
    astradb: CassandraClient = Depends(verify_db_client),
) -> RunObject:
    run = astradb.get_run(id=run_id, thread_id=thread_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found for thread {thread_id}")
    return run


@router.get(
    "/threads/{thread_id}/messages",
    tags=["Assistants"],
    summary="Returns a list of messages for a given thread.",
    response_model=None
)
async def list_messages(
    thread_id: str = Path(
        description="The ID of the [thread](/docs/api-reference/threads) the messages belong to.",
    ),
    limit: int = Query(
        None,
        description="A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20. ",
    ),
    order: str = Query(
        "desc",
        description="Sort order by the &#x60;created_at&#x60; timestamp of the objects. &#x60;asc&#x60; for ascending order and &#x60;desc&#x60; for descending order. ",
    ),
    after: str = Query(
        None,
        description="A cursor for use in pagination. &#x60;after&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include after&#x3D;obj_foo in order to fetch the next page of the list. ",
    ),
    before: str = Query(
        None,
        description="A cursor for use in pagination. &#x60;before&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include before&#x3D;obj_foo in order to fetch the previous page of the list. ",
    ),
    stream: bool = Query(
        False,
        description="Whether to stream messages. If set to true, events will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available. The stream will terminate with a &#x60;data: [DONE]&#x60; message when the job is finished (succeeded, cancelled, or failed).  If set to false, only messages generated so far will be returned. "
    ),
    astradb: CassandraClient = Depends(verify_db_client),
) -> Union[ListMessagesResponse, StreamingResponse]:
    if stream:
        if limit is not None:
            if (limit != 1):
                raise HTTPException(
                    status_code=500,
                    detail="Limit must be 1 when using streaming",
                )
        else:
            limit = 1
        return StreamingResponse(stream_messages_by_thread(astradb, thread_id, limit, order, after, before), media_type="text/event-stream")
    if limit is None:
        limit = 20
    # default is desc
    return get_messages_by_thread(astradb, thread_id, limit, order, after, before)


def get_and_process_messages(astradb, thread_id, limit, order, after, before):
    # TODO: implement pagination
    if after is not None or before is not None:
        raise HTTPException(
            status_code=500,
            detail="Pagination is not yet implemented for this endpoint, do not pass after or before.",
        )
    raw_messages = None
    # TODO fix datamodel to support sorting and limit pushdown
    raw_messages = astradb.selectFromTableByPK(
        table="messages", partitionKeys=["thread_id"], args={"thread_id": thread_id}
    )


    # sort raw_messages by created_at desc
    if order is None or order == "desc":
        # First sort by 'role' in ascending order, this is a tie-breaker and assistant comes before user
        raw_messages.sort(key=lambda x: x["role"], reverse=False)
        # Then sort by 'created_at' in descending order, which maintains the previous ordering for ties
        raw_messages.sort(key=lambda x: x["created_at"], reverse=True)
    else:
        # viceversa
        raw_messages.sort(key=lambda x: x["role"], reverse=True)
        raw_messages.sort(key=lambda x: x["created_at"], reverse=False)

    if limit is not None:
        raw_messages = raw_messages[:limit]

    messages = []
    for message in raw_messages:
        created_at = int(message["created_at"].timestamp() * 1000)

        metadata = message["metadata"]
        if metadata is None:
            metadata = {}

        file_ids = message["file_ids"]
        if file_ids is None:
            file_ids = []

        # take content from list and turn into MessageObjectContentInner
        contentList = []
        content = message["content"]
        if content is not None:
            for text in message["content"]:
                contentObjectText = MessageContentTextObjectText(value=text, annotations=[])
                contentList.append(
                    MessageObjectContentInner(type="text", text=contentObjectText)
                )

        messages.append(
            MessageObject(
                id=message["id"],
                object="thread.message",
                created_at=created_at,
                thread_id=message["thread_id"],
                role=message["role"],
                content=contentList,
                assistant_id=message["assistant_id"],
                run_id=message["run_id"],
                file_ids=file_ids,
                metadata=metadata,
            )
        )
    return messages
def get_messages_by_thread(astradb, thread_id, limit=None, order=None, after=None, before=None):
    messages = get_and_process_messages(astradb, thread_id, limit, order, after, before)

    if len(messages) == 0:
        return ListMessagesResponse(data=[], object="runs", first_id="none", last_id="none", has_more=False)

    first_id = messages[0].id
    last_id = messages[len(messages) - 1].id

    return ListMessagesResponse(
        data=messages, object="runs", first_id=first_id, last_id=last_id, has_more=False
    )

async def get_and_process_assistant_messages(astradb, thread_id, limit, order, after, before):
    messages = get_and_process_messages(astradb, thread_id, limit, order, after, before)

    if messages[0].run_id=="None":
        await asyncio.sleep(1)
        return await get_and_process_assistant_messages(astradb, thread_id, limit, order, after, before)
    return messages


async def stream_messages_by_thread(astradb, thread_id, limit, order, after, before):
    messages = await get_and_process_assistant_messages(astradb, thread_id, limit, order, after, before)

    first_id = messages[0].id
    last_id = messages[len(messages) - 1].id

    current_message = None
    last_message_length=0
    for message in messages:
        current_message = message
        if len(message.content) == 0:
            break
        json_data, last_message_length = await package_message(first_id, last_id, message, thread_id, last_message_length)
        yield f"data: {json_data}\n\n"

    last_message = current_message
    run_id = last_message.run_id
    if run_id == "None":
        run_id = messages[0].run_id
    try:
        while True:
            message = await get_message(thread_id, last_message.id, astradb)
            if message.content != last_message.content:
                json_data, last_message_length = await package_message(first_id, last_id, message, thread_id, last_message_length)
                yield f"data: {json_data}\n\n"
                last_message.content = message.content
            await asyncio.sleep(1)
            run = await get_run(thread_id, run_id, astradb)
            if (run.status != "generating"):
                # do a final pass
                message = await get_message(thread_id, last_message.id, astradb)
                if message.content != last_message.content:
                    json_data, last_message_length = await package_message(first_id, last_id, message, thread_id, last_message_length)
                    yield f"data: {json_data}\n\n"
                    last_message.content = message.content
                break
    except Exception as e:
        logger.error(e)
        yield f"data: []"



async def package_message(first_id, last_id, message, thread_id, last_message_length):
    created_at = message.created_at
    role = message.role
    assistant_id = message.assistant_id
    # message.content here is a MessageObjectContentInner
    if message.content is not None and len(message.content) > 0:
        text_object = MessageContentDeltaObjectText(value=f"{message.content[0].text.value[last_message_length:]}")
        this_message_length = len(message.content[0].text.value)
    else:
        text_object = MessageContentDeltaObjectText(value=f"")
    content = [MessageContentDeltaObject(delta=text_object, type="text")]
    message_id = message.id
    object_text = "thread.message"
    run_id = message.run_id
    file_ids = []
    if message.file_ids is not None:
        file_ids = message.file_ids
    metadata = {}
    if message.metadata is not None:
        metadata = message.metadata
    data = MessageStreamResponseObject(
        id=message_id,
        object=object_text,
        created_at=created_at,
        thread_id=thread_id,
        role=role,
        content=content,
        assistant_id=assistant_id,
        run_id=run_id,
        file_ids=file_ids,
        metadata=metadata
    )
    response_obj = ListMessagesStreamResponse(
        object="list",
        data=[data],
        first_id=first_id,
        last_id=last_id
    )
    json_data = json.dumps(jsonable_encoder(response_obj))
    return json_data, this_message_length


@router.get(
    "/threads/{thread_id}",
    responses={
        200: {"model": ThreadObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Retrieves a thread.",
    response_model_by_alias=True,
)

async def get_thread(
        thread_id: str = Path(..., description="The ID of the thread to retrieve."),
        astradb: CassandraClient = Depends(verify_db_client),
        ) -> ThreadObject:
    return astradb.get_thread(thread_id)


@router.post(
    "/threads/{thread_id}",
    responses={
        200: {"model": ThreadObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Modifies a thread.",
    response_model_by_alias=True,
)
async def modify_thread(
        thread_id: str = Path(..., description="The ID of the thread to modify. Only the &#x60;metadata&#x60; can be modified."),
        modify_thread_request: ModifyThreadRequest = Body(None, description=""),
        astradb: CassandraClient = Depends(verify_db_client),
) -> ThreadObject:
    metadata = modify_thread_request.metadata
    return astradb.upsert_thread(
        id=thread_id,
        object="thread",
        created_at=None,
        metadata=metadata
    )

@router.delete(
    "/threads/{thread_id}",
    responses={
        200: {"model": DeleteThreadResponse, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Delete a thread.",
    response_model_by_alias=True,
)
async def delete_thread(
        thread_id: str = Path(..., description="The ID of the thread to delete."),
        astradb: CassandraClient = Depends(verify_db_client),
) -> DeleteThreadResponse:
    astradb.delete_by_pk(table="threads", key="id", value=thread_id)
    return DeleteThreadResponse(
        id=thread_id,
        object="thread",
        deleted=True
    )

@router.post(
    "/threads/{thread_id}/runs/{run_id}/submit_tool_outputs",
    responses={
        200: {"model": RunObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="When a run has the &#x60;status: \&quot;requires_action\&quot;&#x60; and &#x60;required_action.type&#x60; is &#x60;submit_tool_outputs&#x60;, this endpoint can be used to submit the outputs from the tool calls once they&#39;re all completed. All outputs must be submitted in a single request. ",
    response_model_by_alias=True,
)
async def submit_tool_ouputs_to_run(
        thread_id: str = Path(..., description="The ID of the [thread](/docs/api-reference/threads) to which this run belongs."),
        run_id: str = Path(..., description="The ID of the run that requires the tool output submission."),
        submit_tool_outputs_run_request: SubmitToolOutputsRunRequest = Body(None, description=""),
        litellm_kwargs: Dict[str, Any] = Depends(get_litellm_kwargs),
        astradb: CassandraClient = Depends(verify_db_client),
) -> RunObject:
    try:
        logger.info(submit_tool_outputs_run_request)

        run = astradb.get_run(id=run_id, thread_id=thread_id)
        assistant = astradb.get_assistant(id=run.assistant_id)
        if assistant is None:
            raise HTTPException(status_code=404, detail="Assistant not found")
        model = assistant.model
        file_ids = assistant.file_ids

        messages = get_messages_by_thread(astradb, thread_id, order="asc")
        message_string, message_content = summarize_message_content(assistant.instructions, messages.data)
        for tool_output in submit_tool_outputs_run_request.tool_outputs:
            # some models do not allow system messages in the middle, maybe this should be model specific?
            # message_content.append({"role": "system", "content": f"tool response for {tool_output.tool_call_id} is {tool_output.output}"})
            message_content.append({"role": "user", "content": f"tool response for {tool_output.tool_call_id} is {tool_output.output}"})
        # TODO MAKE THIS BIT DRY
        message = await get_chat_completion(
            messages=message_content,
            model=model,
            **litellm_kwargs,
        )
        completion = message.content

        id = str(uuid1())
        created_at = int(time.mktime(datetime.now().timetuple()))

        astradb.upsert_message(
            id=id,
            object="thread.message",
            created_at=created_at,
            thread_id=thread_id,
            role="assistant",
            content=[completion],
            assistant_id="TODO",
            run_id=run_id,
            file_ids=file_ids,
            metadata={},
        )
        astradb.update_run_status(thread_id=thread_id, id=run_id, status="completed")
        run = astradb.get_run(id=run_id, thread_id=thread_id)
        return run
    except Exception as e:
        logger.info(e)
        astradb.update_run_status(thread_id=thread_id, id=run_id, status="failed")
        raise

