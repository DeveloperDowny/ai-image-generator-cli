"""Hugging Face Inference Providers implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict

from huggingface_hub import InferenceClient
from PIL import Image

from image_generator.models import GenerateRequest
from image_generator.providers.base import ProviderError

logger = logging.getLogger(__name__)


class HuggingFaceProvider:
    """Provider for Hugging Face Inference Providers."""

    def __init__(self, token: str, provider: str = "auto") -> None:
        """Create a provider client.

        Args:
            token: Hugging Face user access token.
            provider: Provider name or "auto" for Hugging Face routing.
        """
        if provider == "auto":
            self._client = InferenceClient(api_key=token)
        else:
            self._client = InferenceClient(provider=provider, api_key=token)

    def generate(self, request: GenerateRequest) -> Image.Image:
        """Generate an image using Hugging Face Inference Providers.

        Args:
            request: Validated generation request.

        Returns:
            The generated image.

        Raises:
            ProviderError: If the API request fails.
        """
        try:
            kwargs: Dict[str, Any] = {
                "prompt": request.prompt,
                "model": request.model,
            }
            if request.seed is not None:
                kwargs["seed"] = request.seed
            return self._client.text_to_image(**kwargs)
        except Exception as exc:  # pragma: no cover - provider-specific errors
            logger.exception("Hugging Face request failed")
            raise ProviderError(str(exc)) from exc
