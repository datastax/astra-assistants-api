import logging
import os

from .patch import patch

logger = logging.getLogger(__name__)

if "OPENAI_API_KEY" not in os.environ:
    logger.debug("OPENAI_API_KEY is unset. Setting it to 'dummy' so openai doesn't kill the "
                 "process when using other LLMs.\nIf you are using OpenAI models (including for embeddings)"
                 ", remember to set OPENAI_API_KEY to your actual key.")
    os.environ["OPENAI_API_KEY"] = "dummy"

