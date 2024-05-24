import hashlib
import time
import logging
import uuid
from datetime import datetime
from typing import Any, Dict
from uuid import uuid1

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Path,
    Request,
    Query,
    UploadFile,
    HTTPException,
)
from litellm import utils
from slowapi import Limiter

from impl.astra_vector import CassandraClient
from impl.services.chunks import get_document_chunks
from impl.services.file import get_document_from_file
from openapi_server_v2.models.delete_file_response import DeleteFileResponse
from openapi_server_v2.models.list_files_response import ListFilesResponse

from .utils import (
    verify_db_client,
    get_litellm_kwargs,
    check_if_using_openai,
    forward_request,
    infer_embedding_model,
)
from ..model.open_ai_file import OpenAIFile
from ..rate_limiter import limiter

router = APIRouter()

logger = logging.getLogger(__name__)


def upload_file_to_uuid_str(upload_file):
    spooled_file = upload_file.file
    spooled_file.seek(0)
    file_data = upload_file.filename.encode('utf-8') + spooled_file.read()
    sha256_hash = hashlib.sha256(file_data).hexdigest()
    hash_as_uuid = uuid.UUID(sha256_hash[:32])
    spooled_file.seek(0)

    return str(hash_as_uuid)

@router.post(
    "/files",
    responses={
        200: {"model": OpenAIFile, "description": "OK"},
    },
    tags=["Files"],
    summary="Upload a file that can be used across various endpoints/features. The size of all the files uploaded by one organization can be up to 100 GB.  The size of individual files for can be a maximum of 512MB. See the [Assistants Tools guide](/docs/assistants/tools) to learn more about the types of files supported. The Fine-tuning API only supports &#x60;.jsonl&#x60; files.  Please [contact us](https://help.openai.com/) if you need to increase these storage limits. ",
    response_model_by_alias=True,
)
@limiter.limit("1/second")
async def create_file(
    request: Request,
    file: UploadFile = File(...),
    purpose: str = Form(
        None,
        description="The intended purpose of the uploaded file.  Use \\\&quot;fine-tune\\\&quot; for [Fine-tuning](/docs/api-reference/fine-tuning) and \\\&quot;assistants\\\&quot; for [Assistants](/docs/api-reference/assistants) and [Messages](/docs/api-reference/messages). This allows us to validate the format of the uploaded file is correct for fine-tuning. ",
    ),
    litellm_kwargs: Dict[str, Any] = Depends(get_litellm_kwargs),
    embedding_model: str = Depends(infer_embedding_model),
    astradb: CassandraClient = Depends(verify_db_client),
    using_openai: bool = Depends(check_if_using_openai),
) -> OpenAIFile:
    # Supported purposes from: https://platform.openai.com/docs/api-reference/files/object
    if purpose in ["fine-tune", "fine-tune-results"]:
        # TODO: Potentially support other models
        if using_openai:
            return await forward_request(request)
        else:
            raise NotImplementedError("File upload is currently only supported for OpenAI")

    file_id = upload_file_to_uuid_str(file)
    if purpose in ["auth"]:
        created_at = int(time.mktime(datetime.now().timetuple()))
        obj = "file"
        filename = file.filename
        fmt = file.content_type
        bytes = len(file.file.read())
        file.file.seek(0)

        content = file.file.read().decode("utf-8")

        openAIFile = astradb.upsert_content_only_file(
            id=file_id,
            object=obj,
            purpose=purpose,
            created_at=created_at,
            filename=filename,
            format=fmt,
            bytes=bytes,
            content=content,
        )
        return openAIFile

    created_at = int(time.mktime(datetime.now().timetuple()))
    obj = "file"
    filename = file.filename
    fmt = file.content_type
    bytes = len(file.file.read())
    file.file.seek(0)

    try:
        existing_file = await retrieve_file(file_id, astradb)
        return existing_file

    except HTTPException as e:
        if e.status_code != 404:
            raise e
        document = await get_document_from_file(file, file_id)

        litellm_kwargs_embedding = litellm_kwargs.copy()
        triple = utils.get_llm_provider(embedding_model)
        provider = triple[1]
        if provider != "bedrock":
            if litellm_kwargs_embedding.get("aws_access_key_id") is not None:
                litellm_kwargs_embedding.pop("aws_access_key_id")
            if litellm_kwargs_embedding.get("aws_secret_access_key") is not None:
                litellm_kwargs_embedding.pop("aws_secret_access_key")
            if litellm_kwargs_embedding.get("aws_region_name") is not None:
                litellm_kwargs_embedding.pop("aws_region_name")
        logger.info("getting chunks")
        format = file.filename.format()
        chunks = get_document_chunks(
            documents=[document],
            chunk_token_size=None,
            embedding_model=embedding_model,
            format=format,
            **litellm_kwargs_embedding,
        )
        # TODO: make this a background task
        logger.info("upserting file and chunks")
        openAIFile = astradb.upsert_file(
            id=file_id,
            object=obj,
            purpose=purpose,
            created_at=created_at,
            filename=filename,
            format=fmt,
            bytes=bytes,
            chunks=chunks,
            embedding_model=embedding_model,
            **litellm_kwargs,
        )
        logger.info(f"File created {openAIFile}")
        return openAIFile


@router.delete(
    "/files/{file_id}",
    responses={
        200: {"model": DeleteFileResponse, "description": "OK"},
    },
    tags=["Files"],
    summary="Delete a file.",
    response_model_by_alias=True,
)
async def delete_file(
    file_id: str,
    astradb: CassandraClient = Depends(verify_db_client),
) -> DeleteFileResponse:
    astradb.delete_by_pk(key="id", value=file_id, table="files")
    return DeleteFileResponse(id=file_id, deleted=True, object="file")


@router.get(
    "/files/{file_id}/content",
    responses={
        200: {"model": str, "description": "OK"},
    },
    tags=["Files"],
    summary="Returns the contents of the specified file.",
    response_model_by_alias=True,
)
async def download_file(
    file_id: str = Path(..., description="The ID of the file to use for this request."),
    astradb: CassandraClient = Depends(verify_db_client),
) -> str:
    ...


@router.get(
    "/files",
    responses={
        200: {"model": ListFilesResponse, "description": "OK"},
    },
    tags=["Files"],
    summary="Returns a list of files that belong to the user&#39;s organization.",
    response_model_by_alias=True,
)
async def list_files(
    purpose: str = Query(None, description="Only return files with the given purpose."),
    astradb: CassandraClient = Depends(verify_db_client),
) -> ListFilesResponse:
    raw_files = astradb.selectAllFromTable(table="files")

    files = []
    for file in raw_files:
        created_at = int(file["created_at"].timestamp() * 1000)
        status = file["status"]
        if status == "success":
            if file["purpose"] == "auth":
                status = "uploaded"
            else:
                status = "processed"
        files.append(
            OpenAIFile(
                id=file["id"],
                bytes=file["bytes"],
                created_at=created_at,
                filename=file["filename"],
                object="file",
                purpose=file["purpose"],
                status=status,
            )
        )

    file_list = ListFilesResponse(data=files, object="list")
    return file_list


@router.get(
    "/files/{file_id}",
    responses={
        200: {"model": OpenAIFile, "description": "OK"},
    },
    tags=["Files"],
    summary="Returns information about a specific file.",
    response_model_by_alias=True,
)
async def retrieve_file(
    file_id: str = Path(..., description="The ID of the file to use for this request."),
    astradb: CassandraClient = Depends(verify_db_client),
) -> OpenAIFile:
    response = astradb.select_from_table_by_pk(
        table="files", partition_keys=["id"], args={"id":file_id}
    )
    if len(response) > 0:
        raw_file = response[0]
        status_details = None
        if "status_details" in raw_file:
            status_details= raw_file["status_details"]
        embedding_model = None
        if "embedding_model" in raw_file:
            embedding_model= raw_file["embedding_model"]

        return OpenAIFile(
            id=raw_file["id"],
            bytes=raw_file["bytes"],
            created_at=int(raw_file["created_at"].timestamp() * 1000),
            filename=raw_file["filename"],
            object="file",
            purpose=raw_file["purpose"],
            status=raw_file["status"],
            status_details=status_details,
            embedding_model=embedding_model,
        )
    raise HTTPException(status_code=404, detail="File not found")
