import asyncio
from datetime import datetime
import logging
import time
from uuid import uuid1

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Path

from impl.astra_vector import CassandraClient
from impl.model_v2.create_assistant_request import CreateAssistantRequest
from impl.routes.utils import verify_openai_token, verify_db_client
from impl.utils import map_model, combine_fields
from openapi_server_v2.models.assistant_object import AssistantObject
from openapi_server_v2.models.assistant_object_tools_inner import AssistantObjectToolsInner
from openapi_server_v2.models.delete_assistant_response import DeleteAssistantResponse
from openapi_server_v2.models.list_assistants_response import ListAssistantsResponse
from openapi_server_v2.models.modify_assistant_request import ModifyAssistantRequest

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
        openai_token: str = Depends(verify_openai_token),
        astradb: CassandraClient = Depends(verify_db_client),
) -> ListAssistantsResponse:
    raw_assistants = astradb.selectAllFromTable(table="assistants_v2")

    assistants = []
    if len(raw_assistants) == 0:
        return ListAssistantsResponse(
            data=assistants,
            object="list",
            first_id="none",
            last_id="none",
            has_more=False,
        )
    for raw_assistant in raw_assistants:
        # TODO is there a better way to handle this?
        if raw_assistant['tools'] is None:
            raw_assistant['tools'] = []
        assistant = AssistantObject.from_dict(raw_assistant)
        assistants.append(assistant)

    first_id = raw_assistants[0]["id"]
    last_id = raw_assistants[len(raw_assistants) - 1]["id"]
    assistants_response = ListAssistantsResponse(
        data=assistants,
        object="assistants",
        first_id=first_id,
        last_id=last_id,
        has_more=False,
    )
    return assistants_response


@router.post(
    "/assistants",
    responses={
        200: {"model": AssistantObject, "description": "OK"},
    },
    tags=["Assistants"],
    summary="Create an assistant with a model and instructions.",
    response_model_by_alias=True,
)
async def create_assistant(
        create_assistant_request: CreateAssistantRequest = Body(None, description=""),
        openai_token: str = Depends(verify_openai_token),
        astradb: CassandraClient = Depends(verify_db_client),
) -> AssistantObject:
    assistant_id = str(uuid1())
    logging.info(f"going to create assistant with id: {assistant_id} and details {create_assistant_request}")

    # validate
    tool_resources = create_assistant_request.tool_resources
    if tool_resources is None:
        tool_resources = []

    tools = create_assistant_request.tools
    if tools is None:
        tools = []

    retrieval_tool = AssistantObjectToolsInner(type='retrieval', function=None)
    if tool_resources is not None and retrieval_tool not in tools and tool_resources != []:
        # raise http error
        raise HTTPException(status_code=400, detail="Retrieval tool is required when tool_resources is not [].")


    description = create_assistant_request.description
    if description is None:
        description = ""

    created_at = int(time.mktime(datetime.now().timetuple())*1000)

    extra_fields = {"id": assistant_id, "created_at": created_at, "object": "assistant"}
    assistant : AssistantObject = map_model(source_instance=create_assistant_request, target_model_class=AssistantObject, extra_fields=extra_fields)

    astradb.upsert_table_from_base_model(table_name="assistants_v2", obj=assistant)

    logging.info(f"created assistant with id: {assistant.id}")

    logging.info(f"with these details: {assistant}")
    return assistant


@router.post("/assistants/{assistant_id}", response_model=AssistantObject)
async def modify_assistant(
        assistant_id: str = Path(..., description="The ID of the assistant to modify."),
        modify_assistant_request: ModifyAssistantRequest = Body(None, description=""),
        openai_token: str = Depends(verify_openai_token),
        astradb: CassandraClient = Depends(verify_db_client),
) -> AssistantObject:
    metadata = modify_assistant_request.metadata

    tools = modify_assistant_request.tools
    if tools is None:
        tools = []

    #assistant = astradb.get_assistant(id=assistant_id)
    #if assistant is None:
    #    logger.warn(f"this should not happen")
    #    asyncio.sleep(1)
    #    return modify_assistant(assistant_id, modify_assistant_request, openai_token, astradb)
    #logger.info(f'assistant before upsert: {assistant}')
    extra_fields={"id": assistant_id, "object": "assistant"}
    combined_fields = combine_fields(extra_fields, modify_assistant_request, AssistantObject)
    astradb.upsert_table_from_dict(table_name="assistants_v2", obj=combined_fields)

    assistant = await get_assistant(assistant_id, openai_token, astradb)
    logger.info(f'assistant upserted: {assistant}')
    return assistant


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
        openai_token: str = Depends(verify_openai_token),
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
)
async def get_assistant(
        assistant_id: str,
        openai_token: str = Depends(verify_openai_token),
        astradb: CassandraClient = Depends(verify_db_client),
) -> AssistantObject:
    assistants = astradb.select_from_table_by_pk(
        table="assistants_v2",
        partitionKeys=["id"],
        args={"id": assistant_id}
    )
    if len(assistants) == 0:
        raise HTTPException(status_code=404, detail="Assistant not found.")
    # TODO is there a better way to handle this?
    if assistants[0]['tools'] is None:
        assistants[0]['tools'] = []
    assistant = AssistantObject.from_dict(assistants[0])
    return assistant