import asyncio
import traceback
from datetime import datetime
import json
import logging
import re
import time
from typing import Dict, Any, Union, get_origin, Type, List, Optional


from fastapi import APIRouter, Body, Depends, Path, HTTPException, Query
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from impl.astra_vector import CassandraClient
from impl.background import background_task_set, add_background_task
from impl.model_v2.create_run_request import CreateRunRequest
from impl.model_v2.message_object import MessageObject
from impl.model_v2.modify_message_request import ModifyMessageRequest
from impl.model_v2.run_object import RunObject
from impl.routes.files import retrieve_file
from impl.routes.utils import verify_db_client, get_litellm_kwargs, infer_embedding_model, infer_embedding_api_key
from impl.routes_v2.assistants_v2 import get_assistant_obj
from impl.routes_v2.vector_stores import read_vsf
from impl.services.inference_utils import get_chat_completion, get_async_chat_completion_response
from impl.utils import map_model, store_object, read_object, read_objects, generate_id
from openapi_server_v2.models.assistants_api_response_format_option import AssistantsApiResponseFormatOption
from openapi_server_v2.models.assistants_api_tool_choice_option import AssistantsApiToolChoiceOption
from openapi_server_v2.models.message_delta_object_delta_content_inner import MessageDeltaObjectDeltaContentInner
from openapi_server_v2.models.message_stream_event import MessageStreamEvent
from openapi_server_v2.models.run_step_delta_object_delta_step_details import RunStepDeltaObjectDeltaStepDetails
from openapi_server_v2.models.run_step_delta_step_details_tool_calls_object_tool_calls_inner import \
    RunStepDeltaStepDetailsToolCallsObjectToolCallsInner
from openapi_server_v2.models.run_step_stream_event import RunStepStreamEvent
from openapi_server_v2.models.run_stream_event import RunStreamEvent

from openapi_server_v2.models.truncation_object import TruncationObject
from openapi_server_v2.models.assistant_stream_event import AssistantStreamEvent
from openapi_server_v2.models.create_message_request import CreateMessageRequest
from openapi_server_v2.models.create_thread_and_run_request import CreateThreadAndRunRequest
from openapi_server_v2.models.create_thread_request import CreateThreadRequest
from openapi_server_v2.models.delete_message_response import DeleteMessageResponse
from openapi_server_v2.models.delete_thread_response import DeleteThreadResponse
from openapi_server_v2.models.list_messages_response import ListMessagesResponse
from openapi_server_v2.models.list_runs_response import ListRunsResponse
from openapi_server_v2.models.message_content_text_object import MessageContentTextObject
from openapi_server_v2.models.message_content_text_object_text import MessageContentTextObjectText
from openapi_server_v2.models.message_delta_content_text_object import MessageDeltaContentTextObject
from openapi_server_v2.models.message_delta_content_text_object_text import MessageDeltaContentTextObjectText
from openapi_server_v2.models.message_delta_object import MessageDeltaObject
from openapi_server_v2.models.message_delta_object_delta import MessageDeltaObjectDelta
from openapi_server_v2.models.modify_thread_request import ModifyThreadRequest
from openapi_server_v2.models.open_ai_file import OpenAIFile
from openapi_server_v2.models.run_object_required_action import RunObjectRequiredAction
from openapi_server_v2.models.run_object_required_action_submit_tool_outputs import \
    RunObjectRequiredActionSubmitToolOutputs
from openapi_server_v2.models.run_step_delta_object import RunStepDeltaObject
from openapi_server_v2.models.run_step_delta_object_delta import RunStepDeltaObjectDelta
from openapi_server_v2.models.run_step_delta_step_details_tool_calls_function_object import \
    RunStepDeltaStepDetailsToolCallsFunctionObject
from openapi_server_v2.models.run_step_delta_step_details_tool_calls_object import \
    RunStepDeltaStepDetailsToolCallsObject
from openapi_server_v2.models.run_step_details_message_creation_object import RunStepDetailsMessageCreationObject
from openapi_server_v2.models.run_step_details_message_creation_object_message_creation import \
    RunStepDetailsMessageCreationObjectMessageCreation
from openapi_server_v2.models.run_step_details_tool_calls_file_search_object import \
    RunStepDetailsToolCallsFileSearchObject
from openapi_server_v2.models.run_step_details_tool_calls_object import RunStepDetailsToolCallsObject
from openapi_server_v2.models.run_step_details_tool_calls_object_tool_calls_inner import \
    RunStepDetailsToolCallsObjectToolCallsInner
from openapi_server_v2.models.run_step_object import RunStepObject
from openapi_server_v2.models.run_step_object_step_details import RunStepObjectStepDetails
from openapi_server_v2.models.run_tool_call_object import RunToolCallObject
from openapi_server_v2.models.run_tool_call_object_function import RunToolCallObjectFunction
from openapi_server_v2.models.submit_tool_outputs_run_request import SubmitToolOutputsRunRequest
from openapi_server_v2.models.thread_object import ThreadObject

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
    created_at = int(time.mktime(datetime.now().timetuple()) * 1000)
    thread_id = generate_id("thread")

    messages = []
    if create_thread_request.messages is not None:
        for raw_message in create_thread_request.messages:
            message = MessageObject.from_dict(raw_message)
            messages.append(message)

            astradb.upsert_table_from_base_model("messages_v2", message)

    thread = map_model(
        source_instance=create_thread_request,
        target_model_class=ThreadObject,
        extra_fields={"object": "thread", "id": thread_id, "created_at": created_at}
    )
    return astradb.upsert_table_from_base_model("threads", thread)

@router.get(
    "/threads/{thread_id}",
    responses={
        200: {"model": ThreadObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Retrieves a thread.",
    response_model_by_alias=True,
    response_model=False,
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
        thread_id: str = Path(...,
                              description="The ID of the thread to modify. Only the &#x60;metadata&#x60; can be modified."),
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
    "/threads/{thread_id}/messages",
    responses={
        # TODO - impl
        200: {"model": MessageObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Create a message.",
    response_model=MessageObject
)
async def create_message(
        thread_id: str = Path(
            ...,
            description="The ID of the [thread](/docs/api-reference/threads) to create a message for.",
        ),
        create_message_request: CreateMessageRequest = Body(None, description=""),
        astradb: CassandraClient = Depends(verify_db_client),
) -> MessageObject:
    created_at = int(time.mktime(datetime.now().timetuple()) * 1000)
    message_id = generate_id("msg")

    content = MessageContentTextObject(
        text=MessageContentTextObjectText(
            value=create_message_request.content,
            annotations=[],
        ),
        type="text"
    )

    extra_fields = {
        "id": message_id,
        "status": "completed",
        "thread_id": thread_id,
        "created_at": created_at,
        "object": "thread.message",
        "content": [content]
    }
    return await store_object(astradb=astradb, obj=create_message_request, target_class=MessageObject, table_name="messages_v2", extra_fields=extra_fields)




@router.get(
    "/threads/{thread_id}/messages/{message_id}",
    responses={
        200: {"model": MessageObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Retrieve a message.",
    response_model=MessageObject
)
async def get_message(
        thread_id: str = Path(...,
                              description="The ID of the [thread](/docs/api-reference/threads) to which this message belongs."),
        message_id: str = Path(..., description="The ID of the message to retrieve."),
        astradb: CassandraClient = Depends(verify_db_client),
) -> MessageObject:
    messages = astradb.select_from_table_by_pk(
        table="messages_v2",
        partition_keys=["id", "thread_id"],
        args={"id": message_id, "thread_id": thread_id},
        allow_filtering=True
    )
    if len(messages) == 0:
        raise HTTPException(status_code=404, detail="Message not found.")
    message = messages_json_to_objects(messages)[0]
    return message

def messages_json_to_objects(raw_messages):
    messages = []
    for raw_message in raw_messages:
        if 'content' in raw_message and raw_message['content'] is not None:
            content_array = raw_message['content'].copy()
            i=0
            for raw_content in content_array:
                content = MessageContentTextObject.from_json(raw_content)
                raw_message['content'][i] = content
                i+=1
            message = MessageObject(**raw_message)
            messages.append(message)
    return messages


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
    content = MessageContentTextObject(
        text=MessageContentTextObjectText(
            value=modify_message_request.content,
            annotations=[],
        ),
        type="text"
    )
    message = await get_message(thread_id, message_id, astradb)
    extra_fields={
        "id": message_id,
        "object": "thread.message",
        "content": [content],
        "thread_id": thread_id,
        "created_at": message.created_at
    }
    await store_object(astradb=astradb, obj=modify_message_request, target_class=MessageObject, table_name="messages_v2", extra_fields=extra_fields)
    message = await get_message(thread_id, message_id, astradb)

    logger.info(f'message upserted: {message}')
    return message


@router.delete(
    "/threads/{thread_id}/messages/{message_id}",
    responses={
        200: {"model": DeleteMessageResponse, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Delete a message.",
    response_model_by_alias=True,
)
async def delete_message(
        thread_id: str = Path(..., description="The ID of the thread to delete."),
        message_id: str = Path(..., description="The ID of the message to delete."),
        astradb: CassandraClient = Depends(verify_db_client),
) -> DeleteMessageResponse:
    astradb.delete_by_pks(table="messages", keys=["id", "thread_id"], values=[message_id, thread_id])
    return DeleteMessageResponse(
        id=message_id,
        object="thread.message.deleted",
        deleted=True
    )


def extractFunctionArguments(content):
    pattern = r"\`\`\`.*({.*})\n\`\`\`"
    match = re.search(pattern, content, re.S)
    if match:
        extracted_text = match.group(1).strip()
        return extracted_text
    else:
        try:
            content_obj = json.loads(content)
            return content_obj['arguments']
            #json.loads(content)
            #return content
        except Exception as e:
            logger.error(e)
            raise ValueError(
                "Could not extract function arguments from LLM response, may have not been properly formatted. Consider retrying or use a different model.")


def extractFunctionName(content: str, candidates: [str]):
    candidates = "|".join(candidates)
    pattern = fr"({candidates})"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()
        return extracted_text
    else:
        raise ValueError("Could not extract function name from LLM response, may not have been properly formatted")


def make_event(target_class: Type[BaseModel],data: BaseModel, event: str) -> AssistantStreamEvent:
    try:
        event = target_class.from_dict(
            {
                "event":event,
                "data": data.to_dict()
            }
        )
        return event
    except Exception as e:
        logger.error(e)
        raise e


async def yield_event_from_object(obj: BaseModel, target_class: Type[BaseModel], event: str, obj_status: str = None):
    try:
        holder = obj.copy()
        if obj_status is not None:
            holder.status = obj_status
        event = make_event(target_class=target_class, data=holder, event=event)
        event_json = event.to_json()
        yield f"data: {event_json}\n\n"
    except Exception as e:
        logger.error(e)
        raise e


async def yield_events_from_object(
        obj: BaseModel,
        target_class: Type[BaseModel],
        obj_statuses: List[Optional[str]],
        events: List[str],
        extra_fields: Dict[str, Any] = None
):
    try:
        if extra_fields is None:
            extra_fields = {}
        holder = obj.copy()
        for key, value in extra_fields.items():
            if hasattr(holder, key):
                setattr(holder, key, value)
            else:
                logger.warn(f"key {key} not found in {holder}, could be a bug.")
        assert len(obj_statuses) == len(events), "obj_statuses and events must be the same length"
        for i in range(len(obj_statuses)):
            async for event in yield_event_from_object(
                obj=holder,
                target_class=target_class,
                obj_status=obj_statuses[i],
                event=events[i],
            ):
                yield event
    except Exception as e:
        logger.error(e)
        raise e

async def run_event_stream(run, message_id, astradb):
    try:
        # this kicks all three required events
        async for event in yield_events_from_object(
            obj=run,
            target_class=RunStreamEvent,
            obj_statuses=["queued", "queued", "in_progress"],
            events=["thread.run.created", "thread.run.queued", "thread.run.in_progress"],
            extra_fields={
                "required_action": None,
            }
        ):
            yield event

        if run.status == "requires_action":
            # annoyingly the sdk looks for a run step even though the data we need is in the RunRequiresAction
            # data.delta.step_details_tool_calls
            step_details = RunStepObjectStepDetails(
                actual_instance=RunStepDetailsToolCallsObject(type="tool_calls", tool_calls=[])

            )

            run_step_id = message_id.replace("msg_", "step_")
            run_step = RunStepObject(
                type="tool_calls",
                thread_id=run.thread_id,
                run_id=run.id,
                # TODO: maybe change this ID.
                id=run_step_id,
                status="in_progress",
                created_at=run.created_at,
                assistant_id=run.assistant_id,
                step_details=step_details,
                object="thread.run.step",
                last_error=None,
                expired_at=None,
                cancelled_at=None,
                completed_at=None,
                failed_at=None,
                metadata=None,
                usage=None,
            )
            async for event in yield_event_from_object(
                obj=run_step,
                target_class=RunStepStreamEvent,
                obj_status="in_progress",
                event="thread.run.step.created",
            ):
                yield event

            async for event in yield_event_from_object(
                obj=run,
                target_class=RunStreamEvent,
                obj_status="in_progress",
                event="thread.run.in_progress",
            ):
                yield event


            tool_calls = []
            index = 0
            for run_tool_call in run.required_action.submit_tool_outputs.tool_calls:
                func_tool_call = RunStepDeltaStepDetailsToolCallsFunctionObject(**run_tool_call.dict(), index=index)
                tool_call = RunStepDeltaStepDetailsToolCallsObjectToolCallsInner(actual_instance=func_tool_call)
                index += 1
                tool_calls.append(tool_call)

            tool_call_delta_object = RunStepDeltaStepDetailsToolCallsObject(type="tool_calls", tool_calls=tool_calls)
            step_details = RunStepDeltaObjectDeltaStepDetails(actual_instance=tool_call_delta_object)
            step_delta = RunStepDeltaObjectDelta(step_details=step_details)
            # TODO: maybe change this ID.
            run_step_delta = RunStepDeltaObject(id=run_step.id, delta=step_delta, object="thread.run.step.delta")

            async for event in yield_event_from_object(obj=run_step_delta, target_class=RunStepStreamEvent, obj_status=None, event="thread.run.step.delta"):
                yield event

            #event = make_event(data=run_step_delta, event="thread.run.step.delta")
            #event_json = event.json()
            #yield f"data: {event_json}\n\n"

            # persist run step
            astradb.upsert_run_step(run_step)

            async for event in yield_event_from_object(obj=run, target_class=RunStreamEvent, obj_status=run.status, event=f"thread.run.{run.status}"):
                yield event
            #run_holder = RunObject(**run.dict())
            #event = make_event(data=run_holder, event=f"thread.run.{run_holder.status}")
            #event_json = event.json()
            #yield f"data: {event_json}\n\n"
            return

        # this works because we make the run_step id the same as the message_id
        run_step_id = message_id.replace("msg_", "step_")
        try:
            run_step = read_object(
                astradb=astradb,
                target_class=RunStepObject,
                table_name="run_steps",
                partition_keys = ["run_id", "id"],
                args={"id": run_step_id, "run_id": run.id}
            )
        except Exception as e:
            if e.status_code == 404:
                run_step = None
            else:
                logger.error(e)
                raise e

        #run_step = astradb.get_run_step(run_id=run.id, id=run_step_id)
        if run_step is not None:
            async for event in yield_events_from_object(
                obj=run_step,
                target_class=RunStepStreamEvent,
                obj_statuses=["in_progress", "in_progress"],
                events=["thread.run.step.created", "thread.run.step.in_progress"],
                extra_fields={"required_action": None}
            ):
                yield event

            #event = make_event(data=run_step, event=f"thread.run.step.created")
            #event_json = event.json()
            #yield f"data: {event_json}\n\n"
            #event = make_event(data=run_step, event=f"thread.run.step.in_progress")
            #event_json = event.json()
            #yield f"data: {event_json}\n\n"

            # retrieval_tool_call_deltas = []
            # index = 0
            # for run_tool_call in run_step.step_details.tool_calls:
            #    tool_call = RetrievalToolCallDelta(**run_tool_call.dict(), index=index)
            #    index += 1
            #    retrieval_tool_call_deltas.append(tool_call)
            # tool_call_delta_object = ToolCallDeltaObject(type="tool_calls", tool_calls=retrieval_tool_call_deltas)

            while run_step.status != "completed":
                run_step = read_object(
                    astradb=astradb,
                    target_class=RunStepObject,
                    table_name="run_steps",
                    partition_keys = ["run_id", "id"],
                    args={"id": run_step_id, "run_id": run.id}
                )
                await asyncio.sleep(1)
            tool_call_delta_object = RunStepDeltaStepDetailsToolCallsObject(type="tool_calls", tool_calls=None)
            step_details = RunStepDeltaObjectDeltaStepDetails(actual_instance=tool_call_delta_object)
            step_delta = RunStepDeltaObjectDelta(step_details=step_details)
            run_step_delta = RunStepDeltaObject(id=run_step.id, delta=step_delta, object="thread.run.step.delta")
            async for event in yield_event_from_object(
                obj=run_step_delta,
                target_class=RunStepStreamEvent,
                obj_status=None,
                event="thread.run.step.delta",
            ):
                yield event
            #event = make_event(data=run_step_delta, event="thread.run.step.delta")
            #event_json = event.json()
            #yield f"data: {event_json}\n\n"
            async for event in yield_event_from_object(
                    obj=run_step,
                    target_class=RunStepStreamEvent,
                    obj_status=None,
                    event="thread.run.step.completed",
            ):
                yield event
            #event = make_event(data=run_step, event=f"thread.run.step.completed")
            #event_json = event.json()
            #yield f"data: {event_json}\n\n"

        async for event in stream_message_events(astradb, run.thread_id, 1, "desc", None, None, run):
            yield event
    except Exception as e:
        # This usually means the client is broken
        # TODO: cancel the run.
        logger.error(e)


async def stream_message_events(astradb, thread_id, limit, order, after, before, run):
    try:
        logger.debug(background_task_set)
        logger.info(f"fetching messages for thread {thread_id}")
        messages = await get_and_process_assistant_messages(astradb, thread_id, limit, order, after, before)

        current_message = None
        last_message_length = 0

        if len(messages) > 0:
            # if the message already has content, clear it for the created and in progress event. It will flow in the deltas.
            #message = messages[0].dict().copy()
            #message['content'] = []
            #message_holder = MessageObject(**message, status="in_progress")
            async for event in yield_events_from_object(
                obj=messages[0],
                target_class=MessageStreamEvent,
                obj_statuses=["in_progress", "in_progress"],
                events=["thread.message.created", "thread.message.in_progress"],
                extra_fields={"content": []}
            ):
                yield event
            #event = make_event(data=message_holder, event="thread.message.created")
            #event_json = event.json()
            #yield f"data: {event_json}\n\n"
            #event = make_event(data=message_holder, event="thread.message.in_progress")
            #event_json = event.json()
            #yield f"data: {event_json}\n\n"
        i = 0
        for message in messages:
            current_message = message
            if len(message.content) == 0:
                break
            message_delta, last_message_length = await extract_message_delta(message, last_message_length, i)

            async for event in yield_event_from_object(
                obj=message_delta,
                target_class=MessageStreamEvent,
                event="thread.message.delta"
            ):
                yield event
            #json_data, last_message_length = await package_message(first_id, last_id, message, thread_id,
            #                                                       last_message_length)
            #event_json = await make_text_delta_event(i, json_data, message, run)
            #yield f"data: {event_json}\n\n"

        last_message = current_message
        run_id = last_message.run_id
        if run_id is None:
            run_id = messages[0].run_id
        while True:
            message = await get_message(thread_id, last_message.id, astradb)
            if message.content != last_message.content:

                message_delta, last_message_length = await extract_message_delta(message, last_message_length, i)
                async for event in yield_event_from_object(
                        obj=message_delta,
                        target_class=MessageStreamEvent,
                        event="thread.message.delta"
                ):
                    yield event
                #json_data, last_message_length = await package_message(first_id, last_id, message, thread_id,
                #                                                       last_message_length)
                #event_json = await make_text_delta_event(i, json_data, message, run)
                #yield f"data: {event_json}\n\n"
                last_message.content = message.content
            await asyncio.sleep(1)
            assert run_id is not None, "run_id missing from message"
            run = await read_run(thread_id, run_id, astradb)
            if (run.status == "completed"):
                # do a final pass
                message = await get_message(thread_id, last_message.id, astradb)
                if message.content != last_message.content:

                    message_delta, last_message_length = await extract_message_delta(message, last_message_length, i)

                    async for event in yield_event_from_object(
                            obj=message_delta,
                            target_class=MessageStreamEvent,
                            event="thread.message.delta"
                    ):
                        yield event
                    #json_data, last_message_length = await package_message(first_id, last_id, message, thread_id,
                    #                                                       last_message_length)
                    #event_json = await make_text_delta_event(i, json_data, message, run)
                    #yield f"data: {event_json}\n\n"
                    last_message.content = message.content
                break
    except Exception as e:
        logger.error(e)
        # TODO - cancel run, mark message incomplete
        # yield f"data: []"



# TODO - add attachments?
async def init_message(thread_id, assistant_id, run_id, astradb, created_at, content=None):
    if content is None:
        content = []
    message_id = generate_id("msg")
    message_obj = MessageObject(
        id=message_id,
        object="thread.message",
        created_at=created_at,
        thread_id=thread_id,
        status="in_progress",
        role="assistant",
        content=content,
        assistant_id=assistant_id,
        run_id=run_id,
        incomplete_details=None,
        completed_at=None,
        incomplete_at=None,
        metadata=None,
        attachments=None
    )
    message = await store_object(astradb=astradb, obj=message_obj, target_class=MessageObject, table_name="messages_v2", extra_fields={})
    return message.id



@router.post(
    "/threads/{thread_id}/runs",
    responses={
        200: {"model": RunObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Create a run.",
    response_model_by_alias=True,
    response_model=None
)
async def create_run(
        thread_id: str = Path(..., description="The ID of the thread to run."),
        create_run_request: CreateRunRequest = Body(None, description=""),
        litellm_kwargs: tuple[Dict[str, Any]] = Depends(get_litellm_kwargs),
        astradb: CassandraClient = Depends(verify_db_client),
        embedding_model: str = Depends(infer_embedding_model),
        embedding_api_key: str = Depends(infer_embedding_api_key),
) -> RunObject | StreamingResponse:
    # TODO: implement thread locking for in-progress runs
    # New Messages cannot be added to the Thread.
    # New Runs cannot be created on the Thread.
    created_at = int(time.mktime(datetime.now().timetuple()) * 1000)
    run_id = generate_id("run")
    status = "queued"

    message_id = None

    tools = create_run_request.tools

    metadata = create_run_request.metadata
    if metadata is None:
        metadata = {}


    model = create_run_request.model
    # TODO: implement support for ChatCompletionToolChoiceOption
    assistant = await get_assistant_obj(assistant_id=create_run_request.assistant_id, astradb=astradb)

    if tools is None:
        tools = []
        if assistant.tools is not None:
            tools = assistant.tools

    tool_resources = []
    if model is None:
        if assistant is None:
            raise HTTPException(status_code=404, detail="Assistant not found")
        model = assistant.model
        tool_resources = assistant.tool_resources

    messages = get_messages_by_thread(astradb, thread_id, order="asc")

    instructions = create_run_request.instructions
    if instructions is None:
        instructions = assistant.instructions

    if create_run_request.additional_instructions is not None:
        instructions = instructions + "\n Additional Instructions:\n" + create_run_request.additional_instructions

    toolsJson = []
    if len(tools) == 0:

        # we use the same created_at as the run
        message_id = await init_message(thread_id=thread_id, assistant_id=assistant.id, run_id=run_id, astradb=astradb, created_at=created_at)

        bkd_task = process_rag(
            run_id,
            thread_id,
            tool_resources,
            messages.data,
            model,
            instructions,
            astradb,
            litellm_kwargs,
            embedding_model,
            assistant.id,
            message_id,
            embedding_api_key,
            created_at
        )
        await add_background_task(function=bkd_task, run_id=run_id, thread_id=thread_id, astradb=astradb)

        status = "in_progress"

    for tool_obj in tools:
        tool = tool_obj.actual_instance
        if tool.type == "file_search":
            created_at = int(time.mktime(datetime.now().timetuple()) * 1000)

            # initialize message
            message_id = await init_message(thread_id=thread_id, assistant_id=assistant.id, run_id=run_id, astradb=astradb, created_at=created_at)

            # create run_step
            # Note the run_step id is the same as the message_id
            run_step_id = message_id.replace("msg_", "step_")
            run_step = RunStepObject(
                id=run_step_id,
                assistant_id=assistant.id,
                created_at=created_at,
                object="thread.run.step",
                run_id=run_id,
                status="in_progress",
                thread_id=thread_id,
                type="tool_calls",
                step_details=RunStepObjectStepDetails(
                    actual_instance=RunStepDetailsToolCallsObject(
                        type="tool_calls",
                        tool_calls=[
                            RunStepDetailsToolCallsObjectToolCallsInner(
                                actual_instance=RunStepDetailsToolCallsFileSearchObject(
                                    id=run_step_id,
                                    type="file_search",
                                    file_search={},
                                )
                            ),
                        ],
                    )
                ),
                last_error=None,
                expired_at=None,
                cancelled_at=None,
                failed_at=None,
                completed_at=None,
                metadata=None,
                usage=None,
            )
            logger.info(f"creating run_step {run_step}")
            astradb.upsert_run_step(run_step)

            # async calls to rag
            bkd_task = process_rag(
                run_id,
                thread_id,
                tool_resources,
                messages.data,
                model,
                instructions,
                astradb,
                litellm_kwargs,
                embedding_model,
                assistant.id,
                message_id,
                embedding_api_key,
                created_at,
                run_step.id
            )
            await add_background_task(function=bkd_task, run_id=run_id, thread_id=thread_id, astradb=astradb)

            status = "in_progress"
        if tool.type == "function":
            toolsJson.append(tool.dict())

    required_action = None

    if len(toolsJson) > 0:
        litellm_kwargs[0]["tools"] = toolsJson
        if create_run_request.tool_choice is not None and hasattr(create_run_request.tool_choice, "to_dict"):
            litellm_kwargs[0]["tool_choice"] = create_run_request.tool_choice.to_dict()
        else:
            litellm_kwargs[0]["tool_choice"] = "auto"
        message_content = summarize_message_content(instructions, messages.data, False)
        try:
            message = await get_chat_completion(messages=message_content, model=model, **litellm_kwargs[0])
        except Exception as e:
            logger.error(f"error: {e}, tenant {astradb.dbid}, model {model}, messages.data {messages.data}, create_run_request {create_run_request}")
            raise HTTPException(status_code=500, detail=f"Error processing message, {e}")

        tool_call_object_id = generate_id("call")
        run_tool_calls = []
        # TODO: fix this, we can't hang off message.content because it turns out you can have both a message and a tool call.
        #if message.content is None:
        if hasattr(message, "tool_calls"):
            for tool_call in message.tool_calls:
                tool_call_object_function = RunToolCallObjectFunction(name=tool_call.function.name, arguments=tool_call.function.arguments)
                run_tool_calls.append(RunToolCallObject(id=tool_call_object_id, type='function', function=tool_call_object_function))

        else:
            #TODO: most models formally support tools now, maybe remove this logic
            try:
                arguments = extractFunctionArguments(message.content)
                candidates = [tool['function']['name'] for tool in toolsJson]
                name = extractFunctionName(message.content, candidates)

                tool_call_object_function = RunToolCallObjectFunction(name=name, arguments=str(arguments))
                run_tool_calls.append(RunToolCallObject(id=tool_call_object_id, type='function', function=tool_call_object_function))
            except Exception as e:
                logger.info("did not find function call in message content")
                status = "completed"
                message_id = generate_id("msg")
                created_at = int(time.mktime(datetime.now().timetuple()) * 1000)

                content = MessageContentTextObject(
                    text=MessageContentTextObjectText(
                        value=message.content,
                        annotations=[],
                    ),
                    type="text"
                )
                # persist message
                message_obj = MessageObject(
                    id=message_id,
                    object="thread.message",
                    created_at=created_at,
                    thread_id=thread_id,
                    role=message.role,
                    content=[content],
                    assistant_id=assistant.id,
                    run_id=run_id,
                    metadata={},
                    status=status,
                    incomplete_details=None,
                    completed_at=created_at,
                    incomplete_at=None,
                    attachments=None,
                )
                await store_object(astradb=astradb, obj=message_obj, target_class=MessageObject, table_name="messages_v2", extra_fields={})

        if len(run_tool_calls) > 0:
            tool_outputs = RunObjectRequiredActionSubmitToolOutputs(tool_calls=run_tool_calls)
            required_action = RunObjectRequiredAction(type='submit_tool_outputs', submit_tool_outputs=tool_outputs)
            status = "requires_action"

            message_id = generate_id("msg")
            created_at = int(time.mktime(datetime.now().timetuple()) * 1000)

            # groq can't handle an assistant call with no content and perplexity can't handle non-alternating user/assistant messages
            if message.content is None:
                message.content = MessageContentTextObject(type='text', text=MessageContentTextObjectText(value="tool call placeholder", annotations=[]))
            else:
                message.content = MessageContentTextObject(type='text', text=MessageContentTextObjectText(value=message.content, annotations=[]))
            # persist message
            message_obj = MessageObject(
                id=message_id,
                object="thread.message",
                created_at=created_at,
                thread_id=thread_id,
                role=message.role,
                content=[message.content],
                assistant_id=assistant.id,
                run_id=run_id,
                metadata={},
                status="in_progress",
                incomplete_details=None,
                completed_at=created_at,
                incomplete_at=None,
                attachments=None,
            )
            await store_object(astradb=astradb, obj=message_obj, target_class=MessageObject, table_name="messages_v2", extra_fields={})


    run = await store_run(
        id=run_id,
        created_at=created_at,
        thread_id=thread_id,
        assistant_id=create_run_request.assistant_id,
        status=status,
        required_action=required_action,
        model=model,
        tools=tools,
        instructions=instructions,
        create_run_request=create_run_request,
        astradb=astradb,
    )
    logger.info(f"created run {run.id} for thread {run.thread_id}")

    if message_id is None:
        logger.error(f"message_id is None this should not happen, thread_id {thread_id}, create_run_request {create_run_request}, dbid {astradb.dbid}")

    if create_run_request.stream:
        return StreamingResponse(run_event_stream(run=run, message_id=message_id, astradb=astradb),
                                 media_type="text/event-stream")
    else:
        return run.to_dict()


async def update_run_status(thread_id, id, status, astradb):
    obj = RunObject.construct(
        id=id,
        object=None,
        created_at=None,
        thread_id=thread_id,
        assistant_id=None,
        status=status,
        required_action=None,
        last_error=None,
        expires_at=None,
        started_at=None,
        cancelled_at=None,
        failed_at=None,
        completed_at=None,
        incomplete_details=None,
        model=None,
        instructions=None,
        tools=None,
        metadata=None,
        usage=None,
        temperature=None,
        top_p=None,
        max_prompt_tokens=None,
        max_completion_tokens=None,
        truncation_strategy=None,
        tool_choice=None,
        response_format=None,
    )
    run = await store_object(astradb=astradb, obj=obj, target_class=RunObject, table_name="runs_v2", extra_fields={})
    return run


async def store_run(id, created_at, thread_id, assistant_id, status, required_action, model, tools, instructions, create_run_request, astradb):
    truncation_strategy = create_run_request.truncation_strategy
    if truncation_strategy is None:
        truncation_strategy = TruncationObject(type="auto")

    tool_choice = create_run_request.tool_choice
    if tool_choice is None:
        tool_choice = AssistantsApiToolChoiceOption(actual_instance="auto")

    response_format = create_run_request.response_format
    if response_format is None:
        response_format = AssistantsApiResponseFormatOption(actual_instance="auto")

    tools_dict = []
    for tool in tools:
        tools_dict.append(tool.to_dict())

    # TODO - support expiration
    extra_fields = {
        "id": id,
        "object": "thread.run",
        "created_at": created_at,
        "thread_id": thread_id,
        "assistant_id": assistant_id,
        "status": status,
        "required_action": required_action,
        "model": model,
        "instructions": instructions,
        "tools": tools,
        "truncation_strategy": truncation_strategy,
        "tool_choice": tool_choice,
        "response_format": response_format,

    }
    run = await store_object(astradb=astradb, obj=create_run_request, target_class=RunObject, table_name="runs_v2", extra_fields=extra_fields)
    return run


def summarize_message_content(instructions, messages, filter_user_messages=False):
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
    user_content = [message for message in message_content if message["role"] == "user"]

    if filter_user_messages:
        return user_content
    else:
        return message_content  # maybe trim message history?




# https://platform.openai.com/docs/assistants/tools/file-search/how-it-works
async def process_rag(
        run_id, thread_id, tool_resources, messages, model, instructions, astradb, litellm_kwargs, embedding_model,
        assistant_id, message_id, embedding_api_key, created_at, run_step_id=None
):
    try:
        logger.info(f"Processing RAG {run_id}")
        # TODO: Deal with run status better
        message_content = []
        if run_step_id is not None:
            message_content = summarize_message_content(instructions, messages, True)
            search_string_messages = message_content.copy()

            # TODO: enforce this with instructor?
            search_string_prompt = "There's a corpus of files that are relevant to your task. You can search these with semantic search. Based on the conversation so far what search string would you search for to better inform your next response (REPLY ONLY WITH THE SEARCH STRING)?"

            # dummy assistant message because some models don't allow two user messages back to back
            search_string_messages.append(
                {"role": "assistant", "content": "I need more information to generate a good response."})
            search_string_messages.append({"role": "user", "content": search_string_prompt})

            search_completion_response = await get_chat_completion(
                messages=search_string_messages,
                model=model,
                **litellm_kwargs[0],
            )
            search_string = search_completion_response.content
            logger.debug(f"ANN search_string {search_string}")

            file_ids = []
            if tool_resources.file_search is not None:
                if tool_resources.file_search.vector_store_ids is not None:
                    for vector_store_id in tool_resources.file_search.vector_store_ids:
                        vector_store_files = await read_vsf(vector_store_id=vector_store_id, astradb=astradb)
                        for vector_store_file in vector_store_files:
                            file_ids.append(vector_store_file.id)
            if len(file_ids) > 0:
                created_at = int(time.mktime(datetime.now().timetuple())*1000)
                context_json = astradb.annSearch(
                    table="file_chunks",
                    vector_index_column="embedding",
                    search_string=search_string,
                    partitions=file_ids,
                    litellm_kwargs=litellm_kwargs[1],
                    embedding_model=embedding_model,
                    embedding_api_key=embedding_api_key,
                )

                # get the unique file_ids from the context_json
                file_ids = list(set([chunk["file_id"] for chunk in context_json]))

                file_meta = {}
                # TODO fix
                for file_id in file_ids:
                    # TODO - IMPL
                    file: OpenAIFile = await retrieve_file(file_id, astradb)
                    file_object = {
                        "file_name": file.filename,
                        "file_id": file.id,
                        "bytes": file.bytes,
                        "search_string": search_string,
                    }
                    file_meta[file_id] = file_object

                # add file metadata from file_meta to context_json
                context_json_meta = [{**chunk, **file_meta[chunk['file_id']]} for chunk in context_json if
                                     chunk['file_id'] in file_meta]

                completed_at = int(time.mktime(datetime.now().timetuple()))

                # TODO: consider [optionally?] excluding the content payload because it can be big
                run_step_id = message_id.replace("msg_", "step_")
                details = RunStepObjectStepDetails(
                    actual_instance=RunStepDetailsToolCallsObject(
                        type="tool_calls",
                        tool_calls=[
                            RunStepDetailsToolCallsObjectToolCallsInner(
                                actual_instance=RunStepDetailsToolCallsFileSearchObject(
                                    id=run_step_id,
                                    type="file_search",
                                    file_search={"chunks": context_json_meta},
                                )
                            ),
                        ],
                    )
                )

                run_step = RunStepObject(
                    id=run_step_id,
                    assistant_id=assistant_id,
                    completed_at=completed_at,
                    created_at=created_at,
                    object="thread.run.step",
                    run_id=run_id,
                    status="completed",
                    step_details=details,
                    thread_id=thread_id,
                    type="tool_calls",
                    last_error=None,
                    expired_at=None,
                    cancelled_at=None,
                    failed_at=None,
                    metadata=None,
                    usage=None,
                )
                logger.info(f"creating run_step {run_step}")
                astradb.upsert_run_step(run_step)

                user_message = message_content.pop()
                message_content.append({"role": "system",
                                        "content": "Important, ALWAYS use the following information to craft your responses. Include the relevant file_name and chunk_id references in parenthesis as part of your response i.e. (chunk_id dbd94b44-cb13-11ee-a868-fd1abaa6ff88_18)."})
                for context in context_json_meta:
                    # TODO improve citations https://platform.openai.com/docs/guides/prompt-engineering/six-strategies-for-getting-better-results
                    content = (
                            "the information below comes from file_name - "
                            + context["file_name"]
                            + " and chunk_id - "
                            + context["chunk_id"]
                            + ":\n"
                            + context["content"]
                            + ":\n"
                    )
                    message_content.append({"role": "system", "content": content})
                message_content.append(user_message)

        else:
            message_content = summarize_message_content(instructions, messages, False)

        litellm_kwargs[0]["stream"] = True

        logger.info(f"generating for message_content: {message_content}")

        for message in message_content:
            if message["content"] == "":
                message_content.remove(message)
        response = await get_async_chat_completion_response(
            messages=message_content,
            model=model,
            **litellm_kwargs[0],
        )
    except asyncio.CancelledError as e:
        logger.error(e)
        # TODO maybe do a cancelled run step with more details?
        await update_run_status(thread_id=thread_id, id=run_id, status="failed", astradb=astradb)
        logger.error("process_rag cancelled")
        raise RuntimeError("process_rag cancelled")
    except Exception as e:
        logger.error(e)
        # TODO maybe do a cancelled run step with more details?
        await update_run_status(thread_id=thread_id, id=run_id, status="failed", astradb=astradb)
        logger.error("process_rag cancelled")
        raise RuntimeError("process_rag cancelled")

    try:
        text = ""
        start_time = time.time()
        frequency_in_seconds = 1

        if 'gemini' in model:
            async for part in response:
                if part.choices[0].delta.content is not None:
                    text += part.choices[0].delta.content
                    start_time = await maybe_checkpoint(assistant_id, astradb,
                                                        frequency_in_seconds, message_id,
                                                        run_id, start_time, text, thread_id, created_at)
        else:
            done = False
            while not done:
                async for part in response:
                    if part.choices[0].finish_reason is not None:
                        done = True
                    delta = part.choices[0].delta.content
                    if delta is not None and isinstance(delta, str):
                        text += delta
                    start_time = await maybe_checkpoint(assistant_id, astradb,
                                                        frequency_in_seconds, message_id,
                                                        run_id, start_time, text, thread_id, created_at)


        # final message upsert
        await complete_message_with_text(assistant_id, astradb, message_id, run_id, text, thread_id, created_at)

        await update_run_status(thread_id=thread_id, id=run_id, status="completed", astradb=astradb)
        logger.info(f"processed rag for run_id {run_id} thread_id {thread_id}")
    except Exception as e:
        await update_run_status(thread_id=thread_id, id=run_id, status="failed", astradb=astradb)
        logger.error(traceback.format_exc())
        logger.error(e)
        raise e
    except asyncio.CancelledError:
        logger.error("process_rag cancelled")
        raise RuntimeError("process_rag cancelled")


async def complete_message_with_text(assistant_id, astradb, message_id, run_id, text, thread_id, created_at):
    content = MessageContentTextObject(
        text=MessageContentTextObjectText(
            value=text,
            annotations=[],
        ),
        type="text"
    )
    await complete_message_with_content(
        assistant_id=assistant_id,
        astradb=astradb,
        message_id=message_id,
        run_id=run_id,
        content=content,
        thread_id=thread_id,
        created_at=created_at
    )

async def complete_message_with_content(assistant_id, astradb, message_id, run_id, content, thread_id, created_at):
    completed_at = int(time.mktime(datetime.now().timetuple()) * 1000)
    message = MessageObject.construct(
        id=message_id,
        object="thread.message",
        created_at=created_at,
        thread_id=thread_id,
        status="completed",
        role="assistant",
        content=[content],
        assistant_id=assistant_id,
        run_id=run_id,
        incomplete_details=None,
        completed_at=completed_at,
        incomplete_at=None,
        metadata={},
        attachments=None
    )
    await store_object(astradb=astradb, obj=message, target_class=MessageObject, table_name="messages_v2", extra_fields={})


async def maybe_checkpoint(assistant_id, astradb, frequency_in_seconds, message_id, run_id,
                           start_time, text, thread_id, created_at):
    current_time = time.time()
    if current_time - start_time >= frequency_in_seconds:
        logger.info("Checkpointing message")
        logger.debug(f"text: {text}")

        content = MessageContentTextObject(
            text=MessageContentTextObjectText(
                value=text,
                annotations=[],
            ),
            type="text"
        )
        message = MessageObject.construct(
            id=message_id,
            object="thread.message",
            created_at=created_at,
            thread_id=thread_id,
            status="in_progress",
            role="assistant",
            content=[content],
            assistant_id=assistant_id,
            run_id=run_id,
            incomplete_details=None,
            completed_at=None,
            incomplete_at=None,
            metadata={},
            attachments=None
        )
        await store_object(astradb=astradb, obj=message, target_class=MessageObject, table_name="messages_v2", extra_fields={})
        start_time = time.time()
        return start_time
    return start_time


@router.get(
    "/threads/{thread_id}/runs",
    responses={
        # TODO - impl
        200: {"model": ListRunsResponse, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Returns a list of runs belonging to a thread.",
    response_model_by_alias=True,
    response_model=None,
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
    runs: [RunObject] = read_objects(
        astradb=astradb,
        target_class=RunObject,
        table_name="runs_v2",
        partition_keys=[],
        args={}
    )
    first_id = runs[0].id
    last_id = runs[len(runs) - 1].id
    runs_response = ListRunsResponse(
        data=runs,
        object="assistants",
        first_id=first_id,
        last_id=last_id,
        has_more=False,
    )
    return runs_response.to_dict()


@router.get(
    "/threads/{thread_id}/runs/{run_id}",
    responses={
        200: {"model": RunObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Retrieves a run.",
    response_model_by_alias=True,
    response_model=None
)
async def get_run(
        thread_id: str = Path(
            ...,
            description="The ID of the [thread](/docs/api-reference/threads) that was run.",
        ),
        run_id: str = Path(..., description="The ID of the run to retrieve."),
        astradb: CassandraClient = Depends(verify_db_client),
) -> RunObject:
    run = await read_run(thread_id=thread_id, run_id=run_id, astradb=astradb)
    return run.to_dict()


async def read_run(thread_id, run_id, astradb):
    run: RunObject = read_object(
        astradb=astradb,
        target_class=RunObject,
        table_name="runs_v2",
        partition_keys=["id", "thread_id"],
        args={"id": run_id, "thread_id": thread_id}
    )
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
        astradb: CassandraClient = Depends(verify_db_client),
        # TODO - impl
) -> Union[ListMessagesResponse]:
    logger.info(f"Listing messages for thread {thread_id}")
    if limit is None:
        limit = 20
    # default is desc
    messages = get_messages_by_thread(astradb, thread_id, limit, order, after, before)
    return messages.to_dict()


def get_and_process_messages(astradb, thread_id, limit, order, after, before):
    # TODO: implement pagination
    if after is not None or before is not None:
        raise HTTPException(
            status_code=500,
            detail="Pagination is not yet implemented for this endpoint, do not pass after or before.",
        )
    raw_messages = None
    # TODO fix datamodel to support sorting and limit pushdown
    raw_messages = astradb.select_from_table_by_pk(
        table="messages_v2", partition_keys=["thread_id"], args={"thread_id": thread_id}
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

    messages = messages_json_to_objects(raw_messages)
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

    if len(messages) == 0 or messages[0].run_id is None:
        await asyncio.sleep(1)
        return await get_and_process_assistant_messages(astradb, thread_id, limit, order, after, before)
    return messages



async def extract_message_delta(message, last_message_length, index):
    text_delta= ""
    this_message_length = 0
    if message.content is not None and len(message.content) > 0:
        text_delta = message.content[0].text.value[last_message_length:]
        this_message_length = len(message.content[0].text.value)
    # TODO maybe support annotations here?
    text_object_text = MessageDeltaContentTextObjectText(
        value=text_delta,
    )
    text_object = MessageDeltaContentTextObject(
        index=index,
        type="text",
        text=text_object_text,
    )
    message_delta = MessageDeltaObject(
        id=message.id,
        object='thread.message.delta',
        delta=MessageDeltaObjectDelta(
            content=[MessageDeltaObjectDeltaContentInner(actual_instance=text_object)],
            role=message.role,
        )
    )
    return message_delta, this_message_length


@router.post(
    "/threads/{thread_id}/runs/{run_id}/submit_tool_outputs",
    responses={
        200: {"model": RunObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="When a run has the &#x60;status: \&quot;requires_action\&quot;&#x60; and &#x60;required_action.type&#x60; is &#x60;submit_tool_outputs&#x60;, this endpoint can be used to submit the outputs from the tool calls once they&#39;re all completed. All outputs must be submitted in a single request. ",
    response_model_by_alias=True,
    response_model=None
)
async def submit_tool_ouputs_to_run(
        thread_id: str = Path(...,
                              description="The ID of the [thread](/docs/api-reference/threads) to which this run belongs."),
        run_id: str = Path(..., description="The ID of the run that requires the tool output submission."),
        # TODO - impl
        submit_tool_outputs_run_request: SubmitToolOutputsRunRequest = Body(None, description=""),
        litellm_kwargs: tuple[Dict[str, Any]] = Depends(get_litellm_kwargs),
        astradb: CassandraClient = Depends(verify_db_client),
) -> RunObject | StreamingResponse:
    try:
        logger.info(submit_tool_outputs_run_request)

        run = await read_run(thread_id=thread_id, run_id=run_id, astradb=astradb)
        assistant = await get_assistant_obj(assistant_id=run.assistant_id, astradb=astradb)
        if assistant is None:
            raise HTTPException(status_code=404, detail="Assistant not found")
        model = assistant.model

        messages = get_messages_by_thread(astradb, thread_id, order="asc")
        message_content = summarize_message_content(assistant.instructions, messages.data)
        for tool_output in submit_tool_outputs_run_request.tool_outputs:
            # some models do not allow system messages in the middle, maybe this should be model specific?
            # message_content.append({"role": "system", "content": f"tool response for {tool_output.tool_call_id} is {tool_output.output}"})
            message_content.append(
                {"role": "user", "content": f"tool response for {tool_output.tool_call_id} is {tool_output.output}"})
        # TODO MAKE THIS BIT DRY
        if not submit_tool_outputs_run_request.stream:
            message = await get_chat_completion(
                messages=message_content,
                model=model,
                **litellm_kwargs[0],
            )
            text = message.content

            id = generate_id("msg")
            created_at = int(time.mktime(datetime.now().timetuple())*1000)


            await complete_message_with_text(
                assistant_id=run.assistant_id,
                astradb=astradb,
                message_id=id,
                run_id=run_id,
                text=text,
                thread_id=thread_id,
                created_at=created_at
            )
            await update_run_status(thread_id=thread_id, id=run_id, status="completed", astradb=astradb)
            run = await read_run(thread_id=thread_id, run_id=run_id, astradb=astradb)
            return run.to_dict()
        else:

            created_at = int(time.mktime(datetime.now().timetuple())*1000)
            # initialize message
            message_id = await init_message(
                thread_id=thread_id,
                assistant_id=run.assistant_id,
                run_id=run_id,
                astradb=astradb,
                created_at=created_at
            )
            try:
                response = await get_async_chat_completion_response(
                    messages=message_content,
                    model=model,
                    stream=True,
                    **litellm_kwargs[0],
                )
                return StreamingResponse(message_delta_streamer(message_id, created_at, response, run, astradb),
                                         media_type="text/event-stream")
            except asyncio.CancelledError:
                logger.error("process_rag cancelled")
                raise RuntimeError("process_rag cancelled")
    except Exception as e:
        logger.info(e)
        await update_run_status(thread_id=thread_id, id=run_id, status="failed", astradb=astradb)
        raise


async def message_delta_streamer(message_id, created_at, response, run, astradb):
    try:
        async for event in yield_events_from_object(
                obj=run,
                target_class=RunStreamEvent,
                obj_statuses=["queued", "in_progress"],
                events=["thread.run.queued", "thread.run.in_progress"],
                extra_fields={"required_action": None}
        ):
            yield event
        #run_holder = RunObject(**run.dict())
        #run_holder.required_action = None
        #run_holder.status = "queued"
        #event = make_event(data=run_holder.to_dict(), event=f"thread.run.{run_holder.status}")
        #event_json = event.json()
        #yield f"data: {event_json}\n\n"
        #run_holder.status = "in_progress"
        #event = make_event(data=run_holder.to_dict(), event=f"thread.run.{run_holder.status}")
        #event_json = event.json()
        #yield f"data: {event_json}\n\n"

        message_creation = RunStepDetailsMessageCreationObjectMessageCreation(message_id=message_id)
        message_step_details = RunStepDetailsMessageCreationObject(type="message_creation", message_creation=message_creation)
        step_details = RunStepObjectStepDetails(actual_instance=message_step_details)
        run_step_id = message_id.replace("msg_", "step_")
        run_step = RunStepObject(
            type="message_creation",
            thread_id=run.thread_id,
            run_id=run.id,
            id=run_step_id,
            status="in_progress",
            created_at=run.created_at,
            assistant_id=run.assistant_id,
            step_details=step_details,
            object="thread.run.step",
            last_error=None,
            expired_at=None,
            cancelled_at=None,
            failed_at=None,
            completed_at=None,
            metadata=None,
            usage=None,
        )
        async for event in yield_events_from_object(
            obj=run_step,
            target_class=RunStepStreamEvent,
            obj_statuses=[run_step.status, run_step.status],
            events=["thread.run.step.created","thread.run.step.in_progress"],
        ):
            yield event

        #event = make_event(data=run_step, event=f"thread.run.step.created")
        #event_json = event.json()
        #yield f"data: {event_json}\n\n"
        #event = make_event(data=run_step, event=f"thread.run.step.in_progress")
        #event_json = event.json()
        #yield f"data: {event_json}\n\n"

        # TODO support attatchments?
        message = MessageObject(
            id=message_id,
            object="thread.message",
            created_at=created_at,
            thread_id=run.thread_id,
            status="in_progress",
            role="assistant",
            content=[],
            assistant_id=run.assistant_id,
            run_id=run.id,
            incomplete_details=None,
            completed_at=None,
            incomplete_at=None,
            metadata=None,
            attachments=None
        )

        async for event in yield_events_from_object(
            obj=message,
            target_class=MessageStreamEvent,
            obj_statuses=[message.status, message.status],
            events=["thread.message.created","thread.message.in_progress"],
        ):
            yield event

        #event = make_event(data=message_holder, event="thread.message.created")
        #event_json = event.json()
        #yield f"data: {event_json}\n\n"
        #event = make_event(data=message_holder, event="thread.message.in_progress")
        #event_json = event.json()
        #yield f"data: {event_json}\n\n"

        text = ""
        start_time = time.time()
        frequency_in_seconds = 1
        i = 0

        if 'gemini' in run.model:
            async for part in response:
                if part.choices[0].delta.content is not None:
                    delta = part.choices[0].delta.content
                    message_delta = await make_text_delta_obj_from_chunk(delta, i, run, message_id)
                    async for event in yield_event_from_object(
                            obj=message_delta,
                            target_class=MessageStreamEvent,
                            event="thread.message.delta"
                    ):
                        yield event
                    i += 1
                    #yield f"data: {event_json}\n\n"
                    text += delta
                    start_time = await maybe_checkpoint(run.assistant_id, astradb,
                                                        frequency_in_seconds, message_id,
                                                        run.id, start_time, text, run.thread_id, created_at)
        else:
            done = False
            while not done:
                async for part in response:
                    if part.choices[0].finish_reason is not None:
                        done = True
                    delta = part.choices[0].delta.content
                    if delta is not None and isinstance(delta, str):
                        message_delta = await make_text_delta_obj_from_chunk(delta, i, run, message_id)
                        async for event in yield_event_from_object(
                                obj=message_delta,
                                target_class=MessageStreamEvent,
                                event="thread.message.delta"
                        ):
                            yield event
                        i += 1
                        text += delta
                    start_time = await maybe_checkpoint(run.assistant_id, astradb,
                                                        frequency_in_seconds, message_id,
                                                        run.id, start_time, text, run.thread_id, created_at)

        # final message upsert
        # TODO - support annotations
        content_text = MessageContentTextObject(
            text=MessageContentTextObjectText(
                value=text,
                annotations=[],
            ),
            type="text"
        )
        #content = MessageObjectContentInner(actual_instance=content_text)
        message.content = [content_text]
        await store_object(astradb=astradb, obj=message, target_class=MessageObject, table_name="messages_v2", extra_fields={})
        await update_run_status(thread_id=run.thread_id, id=run.id, status="completed", astradb=astradb)
        logger.info(f"completed run_id {run.id} thread_id {run.thread_id} with tool submission")

    except Exception as e:
        logger.info(e)
        await update_run_status(thread_id=run.thread_id, id=run.id, status="failed", astradb=astradb)
        raise


async def make_text_delta_obj_from_chunk(chunk, i, run, message_id):
    # TODO maybe support annotations here?
    text_object_text = MessageDeltaContentTextObjectText(
        value=chunk,
    )
    text_object = MessageDeltaContentTextObject(
        index=i,
        type="text",
        text=text_object_text,
    )
    message_delta = MessageDeltaObject(
        id=message_id,
        object='thread.message.delta',
        delta=MessageDeltaObjectDelta(
            content=[MessageDeltaObjectDeltaContentInner(actual_instance=text_object)],
            role="assistant",
        )
    )
    return message_delta


@router.post(
    "/threads/runs",
    responses={
        200: {"model": RunObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Create a thread and run it in one request.",
    response_model_by_alias=True,
)
async def create_thread_and_run(
        # TODO - make copy of CreateThreadAndRunRequest to handle LiteralGenericAlias issue with Tools
        # also do it for  create run
        create_thread_and_run_request: CreateThreadAndRunRequest = Body(None, description=""),
        astradb: CassandraClient = Depends(verify_db_client),
        embedding_model: str = Depends(infer_embedding_model),
        embedding_api_key: str = Depends(infer_embedding_api_key),
        litellm_kwargs: tuple[Dict[str, Any]] = Depends(get_litellm_kwargs),
) -> RunObject:
    create_thread_request = create_thread_and_run_request.thread
    if create_thread_request is None:
        raise HTTPException(status_code=400, detail="thread is required.")

    thread = await create_thread(create_thread_request, astradb)

    create_run_request = CreateRunRequest(
        assistant_id=create_thread_and_run_request.assistant_id,
        model=create_thread_and_run_request.model,
        instructions=create_thread_and_run_request.instructions,
        tools=create_thread_and_run_request.tools,
        metadata=create_thread_and_run_request.metadata
    )
    return await create_run(
        thread_id=thread.id,
        create_run_request=create_run_request,
        astradb=astradb,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        litellm_kwargs=litellm_kwargs,
    )
