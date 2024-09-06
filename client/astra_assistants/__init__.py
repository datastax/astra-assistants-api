import sys

import openai

from .async_openai_with_default_key import AsyncOpenAIWithDefaultKey
from .openai_with_default_key import OpenAIWithDefaultKey
from .patch import patch

OpenAI = OpenAIWithDefaultKey
openai.OpenAI = OpenAI
sys.modules['openai'].OpenAI = OpenAI

AsyncOpenAI = AsyncOpenAIWithDefaultKey
openai.AsyncOpenAI = AsyncOpenAI
sys.modules['openai'].AsyncOpenAI = AsyncOpenAI

__all__ = ["OpenAI", "AsyncOpenAI", "patch"]