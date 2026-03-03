"""Shared defaults for CLI and batch scripts."""

from __future__ import annotations

DEFAULT_PROVIDER = "google"
DEFAULT_MODEL_OPENAI = "gpt-image-1-mini"
DEFAULT_MODEL_HF = "black-forest-labs/FLUX.1-dev"

# ---- nano banana pro model name is latest one 
# DEFAULT_MODEL_GOOGLE = "gemini-2.5-flash-image""
DEFAULT_MODEL_GOOGLE = "gemini-3-pro-image-preview"
# DEFAULT_HF_PROVIDER = "auto"
# DEFAULT_HF_PROVIDER = "zai-org"
# DEFAULT_HF_PROVIDER = "hf-inference"
DEFAULT_HF_PROVIDER = "together"
DEFAULT_SIZE = "512x512"
DEFAULT_QUALITY = "medium"
# DEFAULT_SIZE = "1024x1024"
# DEFAULT_QUALITY = "high"