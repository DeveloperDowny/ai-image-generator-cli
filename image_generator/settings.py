"""Application configuration loading."""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Runtime configuration loaded from the environment.

    Attributes:
        hf_token: Hugging Face user access token.
        openai_api_key: OpenAI API key.
        openai_base_url: Optional OpenAI base URL override.
        google_api_key: Google API key for Gemini.
    """

    model_config = SettingsConfigDict(extra="ignore")

    hf_token: Optional[str] = Field(default=None, validation_alias="HF_TOKEN")
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, validation_alias="OPENAI_BASE_URL")
    google_api_key: Optional[str] = Field(default=None, validation_alias="GOOGLE_API_KEY")

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls()
