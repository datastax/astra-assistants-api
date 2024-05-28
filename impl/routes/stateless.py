"""The stateless endpoints that do not depend on information from DB"""
import logging
import time
import uuid
from typing import Any, Dict

import litellm
from fastapi.encoders import jsonable_encoder
import json


from fastapi import APIRouter, Depends, Request, HTTPException
from litellm import ModelResponse
from starlette.responses import StreamingResponse, JSONResponse

from openapi_server.models.chat_completion_stream_response_delta import ChatCompletionStreamResponseDelta
from openapi_server.models.completion_usage import CompletionUsage
from openapi_server.models.create_chat_completion_stream_response import CreateChatCompletionStreamResponse
from openapi_server.models.create_chat_completion_response import CreateChatCompletionResponse
from openapi_server.models.create_chat_completion_stream_response_choices_inner import \
    CreateChatCompletionStreamResponseChoicesInner
from openapi_server.models.create_embedding_response import CreateEmbeddingResponse
from impl.services.inference_utils import get_embeddings_response, get_async_chat_completion_response
from openapi_server.models.create_embedding_response_usage import CreateEmbeddingResponseUsage

from .utils import get_litellm_kwargs, check_if_using_openai, forward_request
from ..model.create_chat_completion_request import CreateChatCompletionRequest
from ..model.create_chat_completion_response_choices_inner import CreateChatCompletionResponseChoicesInner
from ..model.create_embedding_request import CreateEmbeddingRequest
from ..model.embedding import Embedding

router = APIRouter()


logger = logging.getLogger(__name__)


async def _completion_from_request(
    chat_request: CreateChatCompletionRequest,
    using_openai: bool,
    **litellm_kwargs: Any,
) -> CreateChatCompletionResponse | StreamingResponse:
    # NOTE: litellm_kwargs should contain auth

    messages = []
    for message in chat_request.messages:
        message_dict = {
            "content": message.content,
            "role": message.role
        }

        # Add additional fields only if they are not None
        if message.tool_calls is not None:
            message_dict["tool_calls"] = message.tool_calls

        if message.function_call is not None:
            message_dict["function_call"] = message.function_call

        if message.tool_call_id is not None:
            message_dict["tool_call_id"] = message.tool_call_id

        if message.name is not None:
            message_dict["name"] = message.name

        messages.append(message_dict)

    tools = []
    if chat_request.tools is not None:
        for tool in chat_request.tools:
            tools.append(tool.to_dict())

    functions = []
    if chat_request.functions is not None:
        for function in chat_request.functions:
            function_dict = {
                "name": function.name,
                "description": function.description,
                "params": function.params
            }

            # Add additional fields only if they are not None
            if function.return_type is not None:
                function_dict["return_type"] = function.return_type

            if function.return_description is not None:
                function_dict["return_description"] = function.return_description

            if function.examples is not None:
                function_dict["examples"] = function.examples

            functions.append(function_dict)

    kwargs = {
        "model": chat_request.model,
        "messages": messages,
        "temperature": chat_request.temperature,
        "top_p": chat_request.top_p,
        "n": chat_request.n,
        "stream": chat_request.stream,
        "stop": chat_request.stop,
        "max_tokens": chat_request.max_tokens,
        "presence_penalty": chat_request.presence_penalty,
        "frequency_penalty": chat_request.frequency_penalty,
        "response_format": chat_request.response_format,
        "seed": chat_request.seed,
        "tools": chat_request.tools,
        "tool_choice": chat_request.tool_choice,
        **litellm_kwargs,
    }

    if len(functions) > 0:
        kwargs["functions"] = functions

    if len(tools) > 0:
        kwargs["tools"] = tools


    if chat_request.logit_bias is not None:
        kwargs["logit_bias"] = chat_request.logit_bias

    if chat_request.user is not None:
        kwargs["user"] = chat_request.user

    #litellm.verbose_logger = True
    response = await get_async_chat_completion_response(**kwargs)

    # TODO fix this
    if response is not ModelResponse:
        logger.error("Internal Error calling liteLLM")

    choices = []
    if chat_request.stream is not None and chat_request.stream:
        return StreamingResponse(chat_completion_streamer(response, chat_request.model), media_type="text/event-stream")
    for choice in response.choices:
        finish_reason = choice.finish_reason
        if finish_reason is None:
            finish_reason = "None"
        choice_message = choice.message.dict()
        inner = CreateChatCompletionResponseChoicesInner(
                finish_reason=finish_reason,
                index=choice.index,
                message=choice_message
            )
        choices.append(inner)

    usage = CompletionUsage(prompt_tokens=response.usage['prompt_tokens'],completion_tokens=response.usage['completion_tokens'],total_tokens=response.usage['total_tokens'])

    return CreateChatCompletionResponse(
        id=response.id,
        choices=choices,  # Ensure that this is a list of CreateChatCompletionResponseChoicesInner
        created=response.created,
        model=response.model,
        system_fingerprint=response.system_fingerprint,
        object=response.object,
        usage=usage  # Ensure that this is of type CompletionUsage
    )

async def chat_completion_streamer(response, model):
    # Logic for streaming chat completions
        if (response.__class__.__name__ == "GenerateContentResponse"):
            i = 0
            id = str(uuid.uuid1())
            created_time = int(time.time())
            for part in response:
                choices = []
                #TODO: check function calls here
                delta = ChatCompletionStreamResponseDelta(content=str(part.candidates[0].content.parts[0].text))
                choices.append(CreateChatCompletionStreamResponseChoicesInner(
                    finish_reason=None,
                    index=i,
                    delta=delta.dict()
                ))
                i+=1
                response_obj = CreateChatCompletionStreamResponse(
                    id=id,
                    choices=choices,
                    created=created_time,
                    model=model,
                    system_fingerprint=None,
                    object="chat.completion.chunk",
                )
                json_data = json.dumps(jsonable_encoder(response_obj))
                yield f"data: {json_data}\n\n"
        else:
            async for part in response:
                choices = []
                for choice in part.choices:
                    finish_reason = choice.finish_reason
                    choices.append(CreateChatCompletionStreamResponseChoicesInner(
                        finish_reason=finish_reason,
                        index=choice.index,
                        delta=choice.delta.dict()
                    ))
                response_obj = CreateChatCompletionStreamResponse(
                    id=part.id,
                    choices=choices,
                    created=part.created,
                    model=part.model,
                    system_fingerprint=part.system_fingerprint,
                    object=part.object,
                    )
                json_data = json.dumps(jsonable_encoder(response_obj))
                yield f"data: {json_data}\n\n"


async def maybe_forward_request(request: Request, using_openai: bool):
    """Either forwards the request to OpenAI or raises a NotImplementedError"""
    if using_openai:
        return await forward_request(request)
    else:
        return JSONResponse(
            status_code=501, content={"message": "Only OpenAI is currently supported for this endpoint " + request.method + " " + request.url.path}
        )


@router.post(
    "/moderations",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/CreateModerationResponse'}}}}},
    tags=['Moderations'],
    summary="Classifies if text violates OpenAI's Content Policy",
    response_model_by_alias=True
)
async def create_moderation(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> Any:
    return await maybe_forward_request(request, using_openai)


@router.post(
    "/chat/completions",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/CreateChatCompletionResponse'}}}}},
    tags=['Chat'],
    summary="""Creates a model response for the given chat conversation.""",
    response_model_by_alias=True,
    response_model=None
)
async def create_chat_completion(
    create_chat_completion_request: CreateChatCompletionRequest,
    litellm_kwargs: Dict[str, Any] = Depends(get_litellm_kwargs),
    using_openai: bool = Depends(check_if_using_openai),
) -> Any:
    return await _completion_from_request(create_chat_completion_request, using_openai, **litellm_kwargs)


@router.post(
    "/completions",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/CreateCompletionResponse'}}}}},
    tags=['Completions'],
    summary="""Creates a completion for the provided prompt and parameters.""",
    response_model_by_alias=True
)
async def create_completion(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)



@router.post(
    "/images/generations",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ImagesResponse'}}}}},
    tags=['Images'],
    summary="""Creates an image given a prompt.""",
    response_model_by_alias=True
)
async def create_image(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.post(
    "/images/edits",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ImagesResponse'}}}}},
    tags=['Images'],
    summary="""Creates an edited or extended image given an original image and a prompt.""",
    response_model_by_alias=True
)
async def create_image_edit(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.post(
    "/images/variations",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ImagesResponse'}}}}},
    tags=['Images'],
    summary="""Creates a variation of a given image.""",
    response_model_by_alias=True
)
async def create_image_variation(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.post(
    "/embeddings",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/CreateEmbeddingResponse'}}}}},
    tags=['Embeddings'],
    summary="""Creates an embedding vector representing the input text.""",
    response_model_by_alias=True
)
async def create_embedding(
    create_embedding_request: CreateEmbeddingRequest,
    litellm_kwargs: Dict[str, Any] = Depends(get_litellm_kwargs),
) -> CreateEmbeddingResponse:
    if create_embedding_request.encoding_format is not None:
        litellm_kwargs["encoding_format"] = create_embedding_request.encoding_format
    if create_embedding_request.user is not None:
        litellm_kwargs["user"] = create_embedding_request.user

    embedding_response = get_embeddings_response(
        texts=create_embedding_request.input,
        model=create_embedding_request.model,
        **litellm_kwargs,
    )

    data = []
    for datum in embedding_response.data:
        embedding = Embedding(**datum)
        data.append(embedding)

    if type(embedding_response.usage) is dict:
        usage = CreateEmbeddingResponseUsage(**embedding_response.usage)
    else:
        usage = embedding_response.usage.dict()
    embedding_response = CreateEmbeddingResponse(
        data=data,
        model=create_embedding_request.model,
        object=embedding_response.object,
        usage=usage,
    )
    return embedding_response


@router.post(
    "/audio/speech",
    responses={'200': {'description': 'OK', 'headers': {'Transfer-Encoding': {'schema': {'type': 'string'}, 'description': 'chunked'}}, 'content': {'application/octet-stream': {'schema': {'type': 'string', 'format': 'binary'}}}}},
    tags=['Audio'],
    summary="""Generates audio from the input text.""",
    response_model_by_alias=True
)
async def create_speech(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.post(
    "/audio/transcriptions",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/CreateTranscriptionResponse'}}}}},
    tags=['Audio'],
    summary="""Transcribes audio into the input language.""",
    response_model_by_alias=True
)
async def create_transcription(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.post(
    "/audio/translations",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/CreateTranslationResponse'}}}}},
    tags=['Audio'],
    summary="""Translates audio into English.""",
    response_model_by_alias=True
)
async def create_translation(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.post(
    "/fine_tuning/jobs",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/FineTuningJob'}}}}},
    tags=['Fine-tuning'],
    summary="""Creates a job that fine-tunes a specified model from a given dataset.

Response includes details of the enqueued job including job status and the name of the fine-tuned models once complete.

[Learn more about fine-tuning](/docs/guides/fine-tuning)
""",
    response_model_by_alias=True
)
async def create_fine_tuning_job(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.get(
    "/fine_tuning/jobs",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ListPaginatedFineTuningJobsResponse'}}}}},
    tags=['Fine-tuning'],
    summary="""List your organization's fine-tuning jobs
""",
    response_model_by_alias=True
)
async def list_paginated_fine_tuning_jobs(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.get(
    "/fine_tuning/jobs/{fine_tuning_job_id}",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/FineTuningJob'}}}}},
    tags=['Fine-tuning'],
    summary="""Get info about a fine-tuning job.

[Learn more about fine-tuning](/docs/guides/fine-tuning)
""",
    response_model_by_alias=True
)
async def retrieve_fine_tuning_job(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.get(
    "/fine_tuning/jobs/{fine_tuning_job_id}/events",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ListFineTuningJobEventsResponse'}}}}},
    tags=['Fine-tuning'],
    summary="""Get status updates for a fine-tuning job.
""",
    response_model_by_alias=True
)
async def list_fine_tuning_events(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.post(
    "/fine_tuning/jobs/{fine_tuning_job_id}/cancel",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/FineTuningJob'}}}}},
    tags=['Fine-tuning'],
    summary="""Immediately cancel a fine-tune job.
""",
    response_model_by_alias=True
)
async def cancel_fine_tuning_job(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.get(
    "/fine_tuning/jobs/{fine_tuning_job_id}/checkpoints",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ListFineTuningJobCheckpointsResponse'}}}}},
    tags=['Fine-tuning'],
    summary="""List checkpoints for a fine-tuning job.
""",
    response_model_by_alias=True
)
async def list_fine_tuning_job_checkpoints(
        request: Request,
        using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)



@router.post(
    "/fine-tunes",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/FineTune'}}}}},
    tags=['Fine-tunes'],
    summary="""Creates a job that fine-tunes a specified model from a given dataset.

Response includes details of the enqueued job including job status and the name of the fine-tuned models once complete.

[Learn more about fine-tuning](/docs/guides/legacy-fine-tuning)
""",
    response_model_by_alias=True
)
async def create_fine_tune(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.get(
    "/fine-tunes",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ListFineTunesResponse'}}}}},
    tags=['Fine-tunes'],
    summary="""List your organization's fine-tuning jobs
""",
    response_model_by_alias=True
)
async def list_fine_tunes(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.get(
    "/fine-tunes/{fine_tune_id}",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/FineTune'}}}}},
    tags=['Fine-tunes'],
    summary="""Gets info about the fine-tune job.

[Learn more about fine-tuning](/docs/guides/legacy-fine-tuning)
""",
    response_model_by_alias=True
)
async def retrieve_fine_tune(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.post(
    "/fine-tunes/{fine_tune_id}/cancel",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/FineTune'}}}}},
    tags=['Fine-tunes'],
    summary="""Immediately cancel a fine-tune job.
""",
    response_model_by_alias=True
)
async def cancel_fine_tune(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.get(
    "/fine-tunes/{fine_tune_id}/events",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ListFineTuneEventsResponse'}}}}},
    tags=['Fine-tunes'],
    summary="""Get fine-grained status updates for a fine-tune job.
""",
    response_model_by_alias=True
)
async def list_fine_tune_events(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.get(
    "/models",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ListModelsResponse'}}}}},
    tags=['Models'],
    summary="""Lists the currently available models, and provides basic information about each one such as the owner and availability.""",
    response_model_by_alias=True
)
async def list_models(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.get(
    "/models/{model}",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/Model'}}}}},
    tags=['Models'],
    summary="""Retrieves a model instance, providing basic information about the model such as the owner and permissioning.""",
    response_model_by_alias=True
)
async def retrieve_model(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)


@router.delete(
    "/models/{model}",
    responses={'200': {'description': 'OK', 'content': {'application/json': {'schema': {'$ref': '#/components/schemas/DeleteModelResponse'}}}}},
    tags=['Models'],
    summary="""Delete a fine-tuned model. You must have the Owner role in your organization to delete a model.""",
    response_model_by_alias=True
)
async def delete_model(
    request: Request,
    using_openai: bool = Depends(check_if_using_openai),
) -> StreamingResponse:
    return await maybe_forward_request(request, using_openai)

