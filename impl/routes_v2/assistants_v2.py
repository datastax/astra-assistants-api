import asyncio
from datetime import datetime
import logging
import time
from uuid import uuid1

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Path

from impl.astra_vector import CassandraClient
from impl.model_v2.create_assistant_request import CreateAssistantRequest
from impl.model_v2.modify_assistant_request import ModifyAssistantRequest
from impl.routes.utils import verify_db_client
from impl.utils import store_object, read_object, read_objects
from openapi_server_v2.models.assistant_object import AssistantObject
from openapi_server_v2.models.assistants_api_response_format_option import AssistantsApiResponseFormatOption
from openapi_server_v2.models.delete_assistant_response import DeleteAssistantResponse
from openapi_server_v2.models.list_assistants_response import ListAssistantsResponse

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get(
    "/assistants",
    responses={
        200: {"model": ListAssistantsResponse, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Returns a list of assistants.",
    response_model_by_alias=True,
    response_model=None
)
async def list_assistants(
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
) -> ListAssistantsResponse:
    assistants: [AssistantObject] = read_objects(
        astradb=astradb,
        target_class=AssistantObject,
        table_name="assistants_v2",
        partition_keys=[],
        args={}
    )
    first_id = assistants[0].id
    last_id = assistants[len(assistants) - 1].id
    assistants_response = ListAssistantsResponse(
        data=assistants,
        object="assistants",
        first_id=first_id,
        last_id=last_id,
        has_more=False,
    )
    return assistants_response.to_dict()


@router.post(
    "/assistants",
    responses={
        200: {"model": AssistantObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Create an assistant with a model and instructions.",
    response_model_by_alias=True,
    response_model=None
)
async def create_assistant(
        create_assistant_request: CreateAssistantRequest = Body(None, description=""),
        astradb: CassandraClient = Depends(verify_db_client),
) -> AssistantObject:
    assistant_id = str(uuid1())
    created_at = int(time.mktime(datetime.now().timetuple()) * 1000)
    logging.info(f"going to create assistant with id: {assistant_id} and details {create_assistant_request}")

    extra_fields = {
        "id": assistant_id,
        "created_at": created_at,
        "object": "assistant",
    }
    if create_assistant_request.response_format is not None:
        extra_fields["response_format"] = AssistantsApiResponseFormatOption.from_dict(create_assistant_request.response_format)
    assistant: AssistantObject = await store_object(astradb=astradb, obj=create_assistant_request,
                                                    target_class=AssistantObject, table_name="assistants_v2",
                                                    extra_fields=extra_fields)

    # assistant : AssistantObject = map_model(source_instance=create_assistant_request, target_model_class=AssistantObject, extra_fields=extra_fields)
    # astradb.upsert_table_from_base_model(table_name="assistants_v2", obj=assistant)

    logging.info(f"created assistant with id: {assistant.id}")

    logging.info(f"with these details: {assistant}")
    assistant = assistant.to_dict()
    return assistant


@router.post(
    "/assistants/{assistant_id}",
    responses={
        200: {"model": AssistantObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Modify an assistant.",
    response_model_by_alias=True,
    response_model=None
)
async def modify_assistant(
        assistant_id: str = Path(..., description="The ID of the assistant to modify."),
        modify_assistant_request: ModifyAssistantRequest = Body(None, description=""),
        astradb: CassandraClient = Depends(verify_db_client),
) -> AssistantObject:
    extra_fields = {
        "id": assistant_id,
        "object": "assistant"
    }
    if modify_assistant_request.response_format is not None:
        extra_fields['response_format'] = AssistantsApiResponseFormatOption(actual_instance=modify_assistant_request.response_format)
    await store_object(astradb=astradb, obj=modify_assistant_request, target_class=AssistantObject,
                       table_name="assistants_v2", extra_fields=extra_fields)
    # combined_fields = combine_fields(extra_fields, modify_assistant_request, AssistantObject)
    # astradb.upsert_table_from_dict(table_name="assistants_v2", obj=combined_fields)

    assistant = await get_assistant_obj(astradb=astradb, assistant_id=assistant_id)
    logger.info(f'assistant upserted: {assistant}')
    return assistant.to_dict()


@router.delete(
    "/assistants/{assistant_id}",
    responses={
        200: {"model": DeleteAssistantResponse, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Delete an assistant.",
    response_model_by_alias=True,
)
async def delete_assistant(
        assistant_id: str,
        astradb: CassandraClient = Depends(verify_db_client),
) -> DeleteAssistantResponse:
    astradb.delete_assistant(id=assistant_id)
    return DeleteAssistantResponse(
        id=str(assistant_id), deleted=True, object="assistant"
    )


@router.get(
    "/assistants/{assistant_id}",
    responses={
        200: {"model": AssistantObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Retrieves an assistant.",
    response_model_by_alias=True,
    response_model=None,
)
async def get_assistant(
        assistant_id: str,
        astradb: CassandraClient = Depends(verify_db_client),
) -> AssistantObject:
    assistant = await get_assistant_obj(astradb=astradb, assistant_id=assistant_id)
    return assistant.to_dict()

async def get_assistant_obj(astradb, assistant_id):
    assistant = read_object(
        astradb=astradb,
        target_class=AssistantObject,
        table_name="assistants_v2",
        partition_keys=["id"],
        args={"id": assistant_id}
    )
    return assistant