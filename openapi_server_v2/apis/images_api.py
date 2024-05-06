# coding: utf-8

from typing import Dict, List  # noqa: F401
import importlib
import pkgutil

from openapi_server.apis.images_api_base import BaseImagesApi
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
from openapi_server.models.create_image_edit_request_model import CreateImageEditRequestModel
from openapi_server.models.create_image_request import CreateImageRequest
from openapi_server.models.images_response import ImagesResponse
from openapi_server.security_api import get_token_ApiKeyAuth

router = APIRouter()

ns_pkg = impl
for _, name, _ in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + "."):
    importlib.import_module(name)


@router.post(
    "/images/generations",
    responses={
        200: {"model": ImagesResponse, "description": "OK"},
    },
    tags=["Images"],
    summary="Creates an image given a prompt.",
    response_model_by_alias=True,
)
async def create_image(
    create_image_request: CreateImageRequest = Body(None, description="")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> ImagesResponse:
    ...


@router.post(
    "/images/edits",
    responses={
        200: {"model": ImagesResponse, "description": "OK"},
    },
    tags=["Images"],
    summary="Creates an edited or extended image given an original image and a prompt.",
    response_model_by_alias=True,
)
async def create_image_edit(
    image: str = Form(None, description="The image to edit. Must be a valid PNG file, less than 4MB, and square. If mask is not provided, image must have transparency, which will be used as the mask.")
,
    prompt: str = Form(None, description="A text description of the desired image(s). The maximum length is 1000 characters.")
,
    mask: str = Form(None, description="An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where &#x60;image&#x60; should be edited. Must be a valid PNG file, less than 4MB, and have the same dimensions as &#x60;image&#x60;.")
,
    model: CreateImageEditRequestModel = Form(None, description="")
,
    n: int = Form(1, description="The number of images to generate. Must be between 1 and 10.", ge=1, le=10)
,
    size: str = Form('1024x1024', description="The size of the generated images. Must be one of &#x60;256x256&#x60;, &#x60;512x512&#x60;, or &#x60;1024x1024&#x60;.")
,
    response_format: str = Form('url', description="The format in which the generated images are returned. Must be one of &#x60;url&#x60; or &#x60;b64_json&#x60;. URLs are only valid for 60 minutes after the image has been generated.")
,
    user: str = Form(None, description="A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids). ")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> ImagesResponse:
    ...


@router.post(
    "/images/variations",
    responses={
        200: {"model": ImagesResponse, "description": "OK"},
    },
    tags=["Images"],
    summary="Creates a variation of a given image.",
    response_model_by_alias=True,
)
async def create_image_variation(
    image: str = Form(None, description="The image to use as the basis for the variation(s). Must be a valid PNG file, less than 4MB, and square.")
,
    model: CreateImageEditRequestModel = Form(None, description="")
,
    n: int = Form(1, description="The number of images to generate. Must be between 1 and 10. For &#x60;dall-e-3&#x60;, only &#x60;n&#x3D;1&#x60; is supported.", ge=1, le=10)
,
    response_format: str = Form('url', description="The format in which the generated images are returned. Must be one of &#x60;url&#x60; or &#x60;b64_json&#x60;. URLs are only valid for 60 minutes after the image has been generated.")
,
    size: str = Form('1024x1024', description="The size of the generated images. Must be one of &#x60;256x256&#x60;, &#x60;512x512&#x60;, or &#x60;1024x1024&#x60;.")
,
    user: str = Form(None, description="A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids). ")
,
    token_ApiKeyAuth: TokenModel = Security(
        get_token_ApiKeyAuth
    ),
) -> ImagesResponse:
    ...
