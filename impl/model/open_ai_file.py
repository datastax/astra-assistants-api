from typing import Optional

from pydantic import StrictStr, Field, field_validator
from openapi_server_v2.models.open_ai_file import OpenAIFile as OpenAIFileGenerated


class OpenAIFile(OpenAIFileGenerated):
    purpose: StrictStr = Field(description="The intended purpose of the file. Supported values are `fine-tune`, `fine-tune-results`, `assistants`, `assistants_output` and `auth`.")
    embedding_model: Optional[str] = Field(default=None, description="The embedding model to use for the file. This is only required if the purpose is `assistants`.")

    @field_validator('purpose')
    def purpose_validate_enum(cls, value):
        """Validates the enum"""
        if value not in ('fine-tune', 'fine-tune-results', 'assistants', 'assistants_output', 'auth'):
            raise ValueError("must be one of enum values ('fine-tune', 'fine-tune-results', 'assistants', 'assistants_output' 'auth')")
        return value
