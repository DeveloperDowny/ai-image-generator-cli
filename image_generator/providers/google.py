"""Google Gemini image provider implementation."""

from __future__ import annotations

import logging
from typing import Optional

from google import genai
from google.genai import types
from PIL import Image

from image_generator.models import GenerateRequest
from image_generator.providers.base import ProviderError

logger = logging.getLogger(__name__)


def _aspect_ratio_from_size(size: Optional[str]) -> Optional[str]:
    """Map common size strings to Gemini aspect ratios."""
    if not size:
        return None

    mapping = {
        "1024x1024": "1:1",
        "1024x1792": "9:16",
        "1792x1024": "16:9",
        "1024x768": "4:3",
        "768x1024": "3:4",
        "1200x800": "3:2",
        "800x1200": "2:3",
        "1280x1024": "5:4",
        "1024x1280": "4:5",
    }
    return mapping.get(size)


class GoogleProvider:
    """Provider for Google GenAI (Gemini) image generation."""

    def __init__(self, api_key: str) -> None:
        """Create a provider client.

        Args:
            api_key: Google API key.
        """
        self._client = genai.Client(api_key=api_key)

    def generate(self, request: GenerateRequest) -> Image.Image:
        """Generate an image using Google GenAI.

        Args:
            request: Validated generation request.

        Returns:
            The generated image.

        Raises:
            ProviderError: If the API request fails.
        """
        try:
            config_kwargs: dict[str, object] = {"response_modalities": ["IMAGE"]}
            aspect_ratio = _aspect_ratio_from_size(request.size)
            if aspect_ratio:
                config_kwargs["image_config"] = types.ImageConfig(aspect_ratio=aspect_ratio)

            response = self._client.models.generate_content(
                model=request.model,
                contents=[request.prompt],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            if not response.parts:
                raise ProviderError("Google response contained no content parts.")
            for part in response.parts:
                if part.inline_data is not None:
                    return part.as_image()
        except Exception as exc:  # pragma: no cover - provider-specific errors
            logger.exception("Google GenAI request failed")
            raise ProviderError(str(exc)) from exc

        raise ProviderError("Google response contained no image data.")
