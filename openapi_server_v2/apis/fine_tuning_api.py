# coding: utf-8

from typing import Dict, List  # noqa: F401
import importlib
import pkgutil

from openapi_server_v2.apis.fine_tuning_api_base import BaseFineTuningApi
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

from openapi_server_v2.models.extra_models import TokenModel  # noqa: F401
from openapi_server_v2.models.create_fine_tuning_job_request import CreateFineTuningJobRequest
from openapi_server_v2.models.fine_tuning_job import FineTuningJob
from openapi_server_v2.models.list_fine_tuning_job_checkpoints_response import ListFineTuningJobCheckpointsResponse
from openapi_server_v2.models.list_fine_tuning_job_events_response import ListFineTuningJobEventsResponse
from openapi_server_v2.models.list_paginated_fine_tuning_jobs_response import ListPaginatedFineTuningJobsResponse
from openapi_server_v2.security_api import get_token_ApiKeyAuth

router = APIRouter()

ns_pkg = impl
for _, name, _ in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + "."):
    importlib.import_module(name)


@router.post(
    "/fine_tuning/jobs/{fine_tuning_job_id}/cancel",
    responses={
        200: {"model": FineTuningJob, "description": "OK"},
    },
    tags=["Fine-tuning"],
    summary="Immediately cancel a fine-tune job. ",
    response_model_by_alias=True,
)
async def cancel_fine_tuning_job(
    fine_tuning_job_id: str = Path(..., description="The ID of the fine-tuning job to cancel. ")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> FineTuningJob:
    ...


@router.post(
    "/fine_tuning/jobs",
    responses={
        200: {"model": FineTuningJob, "description": "OK"},
    },
    tags=["Fine-tuning"],
    summary="Creates a fine-tuning job which begins the process of creating a new model from a given dataset.  Response includes details of the enqueued job including job status and the name of the fine-tuned models once complete.  [Learn more about fine-tuning](/docs/guides/fine-tuning) ",
    response_model_by_alias=True,
)
async def create_fine_tuning_job(
    create_fine_tuning_job_request: CreateFineTuningJobRequest = Body(None, description="")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> FineTuningJob:
    ...


@router.get(
    "/fine_tuning/jobs/{fine_tuning_job_id}/events",
    responses={
        200: {"model": ListFineTuningJobEventsResponse, "description": "OK"},
    },
    tags=["Fine-tuning"],
    summary="Get status updates for a fine-tuning job. ",
    response_model_by_alias=True,
)
async def list_fine_tuning_events(
    fine_tuning_job_id: str = Path(..., description="The ID of the fine-tuning job to get events for. ")
,
    after: str = Query(None, description="Identifier for the last event from the previous pagination request.")
,
    limit: int = Query(20, description="Number of events to retrieve.")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> ListFineTuningJobEventsResponse:
    ...


@router.get(
    "/fine_tuning/jobs/{fine_tuning_job_id}/checkpoints",
    responses={
        200: {"model": ListFineTuningJobCheckpointsResponse, "description": "OK"},
    },
    tags=["Fine-tuning"],
    summary="List checkpoints for a fine-tuning job. ",
    response_model_by_alias=True,
)
async def list_fine_tuning_job_checkpoints(
    fine_tuning_job_id: str = Path(..., description="The ID of the fine-tuning job to get checkpoints for. ")
,
    after: str = Query(None, description="Identifier for the last checkpoint ID from the previous pagination request.")
,
    limit: int = Query(10, description="Number of checkpoints to retrieve.")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> ListFineTuningJobCheckpointsResponse:
    ...


@router.get(
    "/fine_tuning/jobs",
    responses={
        200: {"model": ListPaginatedFineTuningJobsResponse, "description": "OK"},
    },
    tags=["Fine-tuning"],
    summary="List your organization&#39;s fine-tuning jobs ",
    response_model_by_alias=True,
)
async def list_paginated_fine_tuning_jobs(
    after: str = Query(None, description="Identifier for the last job from the previous pagination request.")
,
    limit: int = Query(20, description="Number of fine-tuning jobs to retrieve.")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> ListPaginatedFineTuningJobsResponse:
    ...


@router.get(
    "/fine_tuning/jobs/{fine_tuning_job_id}",
    responses={
        200: {"model": FineTuningJob, "description": "OK"},
    },
    tags=["Fine-tuning"],
    summary="Get info about a fine-tuning job.  [Learn more about fine-tuning](/docs/guides/fine-tuning) ",
    response_model_by_alias=True,
)
async def retrieve_fine_tuning_job(
    fine_tuning_job_id: str = Path(..., description="The ID of the fine-tuning job. ")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> FineTuningJob:
    ...
