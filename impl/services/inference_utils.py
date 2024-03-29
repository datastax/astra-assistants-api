import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

import litellm
from litellm import (
    EmbeddingResponse,
    embedding as get_litellm_embedding, acompletion
)
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_random_exponential

litellm.add_function_to_prompt=True
litellm.telemetry = False
litellm.drop_params = True
litellm.verbose_logger.setLevel("WARN")

# TODO: Make these async
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_embeddings_response(
    texts: Union[str, List[str]],
    model: str,
    deployment_id: Optional[str] = None,
    **litellm_kwargs: Any,
) -> EmbeddingResponse:
    """
    Embed texts using OpenAI's ada model.

    Args:
        texts: The list of texts to embed.

    Returns:
        A list of embeddings, each of which is a list of floats.

    Raises:
        Exception: If the OpenAI API call fails.
    """
    if deployment_id is not None:
        raise NotImplementedError(f"Deployment id is currently not supported for embeddings")

    try:
        if "base_url" in litellm_kwargs:
            # LiteLLM `embedding` has slightly different signature than `completion`
            litellm_kwargs = litellm_kwargs.copy()
            litellm_kwargs["api_base"] = litellm_kwargs.pop("base_url")

        embeddings = get_litellm_embedding(
            model=model,
            input=texts,
            **litellm_kwargs,
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e


def get_embeddings(
    texts: Union[str, List[str]],
    model: str,
    deployment_id: Optional[str] = None,
    **litellm_kwargs: Any,
) -> List[List[float]]:
    response = get_embeddings_response(
        texts=texts,
        model=model,
        deployment_id=deployment_id,
        **litellm_kwargs,
    )

    # Return the embeddings as a list of lists of floats
    try:
        if litellm_kwargs.get("aws_access_key_id") is not None:
            # bedrock can't handle batches
            return [result['embedding'] for result in response.data]
        return [result.embedding for result in response.data]
    except AttributeError:
        return [result["embedding"] for result in response.data]


async def get_async_chat_completion_response(
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        deployment_id: Optional[str] = None,
        **litellm_kwargs: Any,
) -> AsyncGenerator:
    # call the LiteLLM chat completion router with the given messages
    if model is None and deployment_id is None:
        raise ValueError("Must provide either a model or a deployment id")

    try:
        if model is None:
            model = deployment_id

        completion = await acompletion(
            model=model,
            messages=messages,
            deployment_id=deployment_id,
            **litellm_kwargs
        )
        return completion
    except Exception as e:
        if "LLM Provider NOT provided" in e.args[0]:
            logger.error(f"Error: error {model} is not currently supported")
            raise ValueError(f"Model {model} is not currently supported")
        logger.error(f"Error: {e}")
        raise ValueError(f"Error: {e}")
    except asyncio.CancelledError:
        logger.error("litellm call cancelled")
        raise RuntimeError("litellm call cancelled")


async def get_chat_completion(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    deployment_id: Optional[str] = None,
    **litellm_kwargs: Any,
) -> Any:
    response = await get_async_chat_completion_response(
        messages=messages,
        model=model,
        deployment_id=deployment_id,
        **litellm_kwargs,
    )

    choices = response.choices
    message = choices[0].message
    return message
