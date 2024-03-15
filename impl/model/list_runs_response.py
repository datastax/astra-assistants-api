from typing import List

from impl.model.run_object import RunObject
from openapi_server.models.list_runs_response import ListRunsResponse as ListRunsResponseGenerated


class ListRunsResponse(ListRunsResponseGenerated):
    data: List[RunObject]
