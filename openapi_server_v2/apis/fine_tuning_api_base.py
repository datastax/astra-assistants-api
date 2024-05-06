# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from openapi_server.models.create_fine_tuning_job_request import CreateFineTuningJobRequest
from openapi_server.models.fine_tuning_job import FineTuningJob
from openapi_server.models.list_fine_tuning_job_checkpoints_response import ListFineTuningJobCheckpointsResponse
from openapi_server.models.list_fine_tuning_job_events_response import ListFineTuningJobEventsResponse
from openapi_server.models.list_paginated_fine_tuning_jobs_response import ListPaginatedFineTuningJobsResponse
from openapi_server.security_api import get_token_ApiKeyAuth

class BaseFineTuningApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseFineTuningApi.subclasses = BaseFineTuningApi.subclasses + (cls,)
    def cancel_fine_tuning_job(
        self,
        fine_tuning_job_id: str,
    ) -> FineTuningJob:
        ...


    def create_fine_tuning_job(
        self,
        create_fine_tuning_job_request: CreateFineTuningJobRequest,
    ) -> FineTuningJob:
        ...


    def list_fine_tuning_events(
        self,
        fine_tuning_job_id: str,
        after: str,
        limit: int,
    ) -> ListFineTuningJobEventsResponse:
        ...


    def list_fine_tuning_job_checkpoints(
        self,
        fine_tuning_job_id: str,
        after: str,
        limit: int,
    ) -> ListFineTuningJobCheckpointsResponse:
        ...


    def list_paginated_fine_tuning_jobs(
        self,
        after: str,
        limit: int,
    ) -> ListPaginatedFineTuningJobsResponse:
        ...


    def retrieve_fine_tuning_job(
        self,
        fine_tuning_job_id: str,
    ) -> FineTuningJob:
        ...
