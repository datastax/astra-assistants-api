from datetime import datetime
import logging
import time

from fastapi import APIRouter, Path, Depends, Body, Query

from impl.astra_vector import CassandraClient
from impl.model_v2.vector_store_object import VectorStoreObject
from impl.routes.utils import verify_db_client
from impl.utils import read_object, store_object, read_objects, generate_id
from openapi_server_v2.models.create_vector_store_file_request import CreateVectorStoreFileRequest
from openapi_server_v2.models.create_vector_store_request import CreateVectorStoreRequest
from openapi_server_v2.models.list_vector_store_files_response import ListVectorStoreFilesResponse
from openapi_server_v2.models.vector_store_file_object import VectorStoreFileObject
from openapi_server_v2.models.vector_store_object_file_counts import VectorStoreObjectFileCounts

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get(
    "/vector_stores/{vector_store_id}",
    responses={
        200: {"model": VectorStoreObject, "description": "OK"},
    },
    tags=["Vector Stores"],
    summary="Retrieves a vector store.",
    response_model_by_alias=True,
)
async def get_vector_store(
        vector_store_id: str = Path(..., description="The ID of the vector store to retrieve."),
        astradb: CassandraClient = Depends(verify_db_client),
) -> VectorStoreObject:
    partition_keys = ["id"]
    args = {"id": vector_store_id}

    vector_store: VectorStoreObject = read_object(
        astradb=astradb,
        target_class=VectorStoreObject,
        table_name="vector_stores",
        partition_keys=partition_keys,
        args=args
    )
    return vector_store


@router.post(
    "/vector_stores",
    responses={
        200: {"model": VectorStoreObject, "description": "OK"},
    },
    tags=["Vector Stores"],
    summary="Create a vector store.",
    response_model_by_alias=True,
)
async def create_vector_store(
        create_vector_store_request: CreateVectorStoreRequest = Body(None, description=""),
        astradb: CassandraClient = Depends(verify_db_client),
) -> VectorStoreObject:
    vector_store_id = generate_id("vs")
    created_at = int(time.mktime(datetime.now().timetuple()) * 1000)

    usage_bytes = 0
    for file_id in create_vector_store_request.file_ids:
        request = CreateVectorStoreFileRequest(file_id=file_id)
        await create_vector_store_file(
            vector_store_id=vector_store_id,
            create_vector_store_file_request=request,
            astradb=astradb
        )
        #TODO - compute usage_bytes

    file_id_count = len(create_vector_store_request.file_ids)
    file_counts = VectorStoreObjectFileCounts(
        in_progress=0,
        completed=file_id_count,
        failed=0,
        cancelled=0,
        total=file_id_count
    )
    extra_fields = {
        "object": "vector_store",
        "usage_bytes": usage_bytes,
        "file_counts": file_counts,
        "status": "completed",
        "id": vector_store_id,
        "created_at": created_at
    }

    vector_store = await store_object(
        astradb=astradb,
        obj=create_vector_store_request,
        target_class=VectorStoreObject,
        table_name="vector_stores",
        extra_fields=extra_fields
    )
    return vector_store



@router.post(
    "/vector_stores/{vector_store_id}/files",
    responses={
        200: {"model": VectorStoreFileObject, "description": "OK"},
    },
    tags=["Vector Stores"],
    summary="Create a vector store file by attaching a [File](/docs/api-reference/files) to a [vector store](/docs/api-reference/vector-stores/object).",
    response_model_by_alias=True,
)
async def create_vector_store_file(
        vector_store_id: str = Path(..., description="The ID of the vector store for which to create a File. "),
        create_vector_store_file_request: CreateVectorStoreFileRequest = Body(None, description=""),
        astradb: CassandraClient = Depends(verify_db_client),
) -> VectorStoreFileObject:
    created_at = int(time.mktime(datetime.now().timetuple()) * 1000)

    extra_fields = {
        "id": create_vector_store_file_request.file_id,
        "vector_store_id": vector_store_id,
        "object": "vector_store.file",
        "created_at": created_at,
        # TODO - grab from file
        "usage_bytes": -1,
        "status": "completed"
    }
    vector_store_file: VectorStoreFileObject = await store_object(
        astradb=astradb,
        obj=create_vector_store_file_request,
        target_class=VectorStoreFileObject,
        table_name="vector_store_files",
        extra_fields=extra_fields
    )
    return vector_store_file


@router.get(
    "/vector_stores/{vector_store_id}/files",
    responses={
        200: {"model": ListVectorStoreFilesResponse, "description": "OK"},
    },
    tags=["Vector Stores"],
    summary="Returns a list of vector store files.",
    response_model_by_alias=True,
    response_model=None
)
async def list_vector_store_files(
        vector_store_id: str = Path(..., description="The ID of the vector store that the files belong to."),
        limit: int = Query(20, description="A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20. "),
        order: str = Query('desc', description="Sort order by the &#x60;created_at&#x60; timestamp of the objects. &#x60;asc&#x60; for ascending order and &#x60;desc&#x60; for descending order. "),
        after: str = Query(None, description="A cursor for use in pagination. &#x60;after&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include after&#x3D;obj_foo in order to fetch the next page of the list. "),
        before: str = Query(None, description="A cursor for use in pagination. &#x60;before&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include before&#x3D;obj_foo in order to fetch the previous page of the list. "),
        filter: str = Query(None, description="Filter by file status. One of &#x60;in_progress&#x60;, &#x60;completed&#x60;, &#x60;failed&#x60;, &#x60;cancelled&#x60;."),
        astradb: CassandraClient = Depends(verify_db_client),
) -> ListVectorStoreFilesResponse:
    # TODO - support limit and paging
    vector_store_files = await read_vsf(vector_store_id, astradb)
    first = vector_store_files[0].id
    last = vector_store_files[len(vector_store_files)-1].id
    vsf_response = ListVectorStoreFilesResponse(
        data=vector_store_files,
        object="vector_store_files",
        first_id=first,
        last_id=last,
        has_more=False
    )
    return vsf_response


async def read_vsf(vector_store_id, astradb):
    partition_keys = ["vector_store_id"]
    args = {"vector_store_id": vector_store_id}

    vector_store_files: [VectorStoreFileObject] = read_objects(
        astradb=astradb,
        target_class=VectorStoreFileObject,
        table_name="vector_store_files",
        partition_keys=partition_keys,
        args=args
    )
    return vector_store_files