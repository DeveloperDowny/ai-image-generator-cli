"""Structured models used by the CLI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    """Validated inputs for an image generation request.

    Attributes:
        prompt: The text prompt to render.
        model: The model ID to use.
        provider: The image provider identifier (for example, "openai" or "hf").
        hf_provider: The Hugging Face inference provider name (or "auto").
        size: Optional image size (provider-specific).
        quality: Optional image quality setting (provider-specific).
        seed: Optional random seed for reproducibility.
        output_path: Where to save the generated image.
    """

    prompt: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    provider: str = Field(default="openai", min_length=1)
    hf_provider: str = Field(default="auto", min_length=1)
    size: Optional[str] = None
    quality: Optional[str] = None
    seed: Optional[int] = None
    output_path: Path

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        """Reject prompts that are only whitespace."""
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class GenerateResult(BaseModel):
    """Result metadata for a generated image.

    Attributes:
        model: The model ID used for generation.
        provider: The provider that handled the request.
        output_path: Where the image was written.
    """

    model: str
    provider: str
    output_path: Path
