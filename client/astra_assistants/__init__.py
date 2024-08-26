import logging
import os

from openai import OpenAI

from .patch import patch

logger = logging.getLogger(__name__)


class OpenAIWithDefaultKey(OpenAI):
    def __init__(self, *args, **kwargs):
        key = os.environ.get("OPENAI_API_KEY", "dummy")
        if key == "dummy":
            logger.debug("OPENAI_API_KEY is unset. Setting it to 'dummy' so openai doesn't kill the "
                         "process when using other LLMs.\nIf you are using OpenAI models (including for embeddings)"
                         ", remember to set OPENAI_API_KEY to your actual key.")
        print(f"CustomLibraryClass initialized with args={args} and kwargs={kwargs}")
        super(OpenAIWithDefaultKey, self).__init__(*args, **kwargs)


OpenAI = OpenAIWithDefaultKey
