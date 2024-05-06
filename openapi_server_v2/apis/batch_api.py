# coding: utf-8

from typing import Dict, List  # noqa: F401
import importlib
import pkgutil

from openapi_server.apis.batch_api_base import BaseBatchApi
import impl

from fastapi import (  # noqa: F401
    APIRouter,
    Body,
    Cookie,
    Depends,
    Form,
    Header,
    Path,
    Query,
    Response,
    Security,
    status,
)

from openapi_server.models.extra_models import TokenModel  # noqa: F401
from openapi_server.models.batch import Batch
from openapi_server.models.create_batch_request import CreateBatchRequest
from openapi_server.models.list_batches_response import ListBatchesResponse
from openapi_server.security_api import get_token_ApiKeyAuth

router = APIRouter()

ns_pkg = impl
for _, name, _ in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + "."):
    importlib.import_module(name)


@router.post(
    "/batches/{batch_id}/cancel",
    responses={
        200: {"model": Batch, "description": "Batch is cancelling. Returns the cancelling batch&#39;s details."},
    },
    tags=["Batch"],
    summary="Cancels an in-progress batch.",
    response_model_by_alias=True,
)
async def cancel_batch(
    batch_id: str = Path(..., description="The ID of the batch to cancel.")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> Batch:
    ...


@router.post(
    "/batches",
    responses={
        200: {"model": Batch, "description": "Batch created successfully."},
    },
    tags=["Batch"],
    summary="Creates and executes a batch from an uploaded file of requests",
    response_model_by_alias=True,
)
async def create_batch(
    create_batch_request: CreateBatchRequest = Body(None, description="")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> Batch:
    ...


@router.get(
    "/batches",
    responses={
        200: {"model": ListBatchesResponse, "description": "Batch listed successfully."},
    },
    tags=["Batch"],
    summary="List your organization&#39;s batches.",
    response_model_by_alias=True,
)
async def list_batches(
    after: str = Query(None, description="A cursor for use in pagination. &#x60;after&#x60; is an object ID that defines your place in the list. For instance, if you make a list request and receive 100 objects, ending with obj_foo, your subsequent call can include after&#x3D;obj_foo in order to fetch the next page of the list. ")
,
    limit: int = Query(20, description="A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20. ")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> ListBatchesResponse:
    ...


@router.get(
    "/batches/{batch_id}",
    responses={
        200: {"model": Batch, "description": "Batch retrieved successfully."},
    },
    tags=["Batch"],
    summary="Retrieves a batch.",
    response_model_by_alias=True,
)
async def retrieve_batch(
    batch_id: str = Path(..., description="The ID of the batch to retrieve.")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> Batch:
    ...
