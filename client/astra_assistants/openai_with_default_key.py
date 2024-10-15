import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenAIWithDefaultKey(OpenAI):
    def __init__(self, *args, **kwargs):
        #if kwargs.get("api_key") is not None:
        #    return super(OpenAIWithDefaultKey, self).__init__(*args, **kwargs)
        key = os.environ.get("OPENAI_API_KEY", "dummy")
        if key == "dummy":
            logger.debug("OPENAI_API_KEY is unset. Setting it to 'dummy' so openai doesn't kill the "
                         "process when using other LLMs.\nIf you are using OpenAI models (including for embeddings)"
                         ", remember to set OPENAI_API_KEY to your actual key.")
            os.environ['OPENAI_API_KEY'] = key
        super(OpenAIWithDefaultKey, self).__init__(*args, **kwargs)
