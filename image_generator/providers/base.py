"""Provider interfaces and exceptions."""

from __future__ import annotations

import logging
from typing import Protocol

from PIL import Image

from image_generator.models import GenerateRequest

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when a provider request fails."""


class ImageProvider(Protocol):
    """Protocol for image generation providers."""

    def generate(self, request: GenerateRequest) -> Image.Image:
        """Generate an image from a request."""
        ...
