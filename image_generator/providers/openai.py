"""OpenAI image provider implementation."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict

from openai import OpenAI
from PIL import Image

from image_generator.models import GenerateRequest
from image_generator.providers.base import ProviderError

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """Provider for OpenAI image generation."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        """Create a provider client.

        Args:
            api_key: OpenAI API key.
            base_url: Optional base URL override.
        """
        if base_url:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = OpenAI(api_key=api_key)

    def generate(self, request: GenerateRequest) -> Image.Image:
        """Generate an image using the OpenAI Images API.

        Args:
            request: Validated generation request.

        Returns:
            The generated image.

        Raises:
            ProviderError: If the API request fails.
        """
        try:
            params: Dict[str, Any] = {
                "model": request.model,
                "prompt": request.prompt,
            }
            if request.size:
                params["size"] = request.size
            if request.quality:
                params["quality"] = request.quality

            response = self._client.images.generate(**params)
            if not response.data:
                raise ProviderError("OpenAI response contained no image data.")

            image_b64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_b64)
            return Image.open(io.BytesIO(image_bytes))
        except Exception as exc:  # pragma: no cover - provider-specific errors
            logger.exception("OpenAI request failed")
            raise ProviderError(str(exc)) from exc
