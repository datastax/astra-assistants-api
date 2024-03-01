# coding: utf-8

from __future__ import annotations
from datetime import date, datetime  # noqa: F401

import re  # noqa: F401
from typing import Any, Dict, List, Optional  # noqa: F401

from pydantic import AnyUrl, BaseModel, EmailStr, Field, validator  # noqa: F401
from openapi_server.models.run_step_details_tool_calls_code_output_image_object import RunStepDetailsToolCallsCodeOutputImageObject
from openapi_server.models.run_step_details_tool_calls_code_output_image_object_image import RunStepDetailsToolCallsCodeOutputImageObjectImage
from openapi_server.models.run_step_details_tool_calls_code_output_logs_object import RunStepDetailsToolCallsCodeOutputLogsObject


class RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner(BaseModel):
    """NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).

    Do not edit the class manually.

    RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner - a model defined in OpenAPI

        type: The type of this RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner.
        logs: The logs of this RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner.
        image: The image of this RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner.
    """

    type: str = Field(alias="type")
    logs: str = Field(alias="logs")
    image: RunStepDetailsToolCallsCodeOutputImageObjectImage = Field(alias="image")

RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner.update_forward_refs()