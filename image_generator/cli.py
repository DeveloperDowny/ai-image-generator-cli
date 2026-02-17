"""CLI entrypoint for image generation."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError

from image_generator.models import GenerateRequest, GenerateResult
from image_generator.providers.google import GoogleProvider
from image_generator.providers.hf import HuggingFaceProvider
from image_generator.providers.openai import OpenAIProvider
from image_generator.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL_OPENAI = "gpt-image-1-mini"
DEFAULT_MODEL_HF = "black-forest-labs/FLUX.1-dev"
DEFAULT_MODEL_GOOGLE = "gemini-2.5-flash-image"
DEFAULT_PROVIDER = "google"
DEFAULT_HF_PROVIDER = "auto"
DEFAULT_SIZE = "1024x1024"
DEFAULT_QUALITY = "high"

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Text prompt to render."),
    model: Optional[str] = typer.Option(
        None,
        help="Model ID to use. If omitted, uses the provider default.",
    ),
    provider: str = typer.Option(
        DEFAULT_PROVIDER,
        help="Image provider to use (openai, hf, or google).",
    ),
    hf_provider: str = typer.Option(
        DEFAULT_HF_PROVIDER,
        help="Hugging Face inference provider name (or 'auto').",
    ),
    size: Optional[str] = typer.Option(
        DEFAULT_SIZE,
        help="Image size for providers that support it (e.g., 1024x1024).",
    ),
    quality: Optional[str] = typer.Option(
        DEFAULT_QUALITY,
        help="Image quality setting for providers that support it.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Output image path. Defaults to a timestamped PNG.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        help="Optional random seed for reproducibility.",
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        envvar="HF_TOKEN",
        help="Hugging Face access token (or set HF_TOKEN).",
    ),
    openai_api_key: Optional[str] = typer.Option(
        None,
        envvar="OPENAI_API_KEY",
        help="OpenAI API key (or set OPENAI_API_KEY).",
    ),
    openai_base_url: Optional[str] = typer.Option(
        None,
        envvar="OPENAI_BASE_URL",
        help="Optional OpenAI base URL override.",
    ),
    google_api_key: Optional[str] = typer.Option(
        None,
        envvar="GOOGLE_API_KEY",
        help="Google API key for Gemini (or set GOOGLE_API_KEY).",
    ),
    verbose: bool = typer.Option(False, help="Enable verbose logging."),
) -> None:
    """Generate an image from a prompt using configured providers."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    settings = Settings.from_env()
    provider = provider.lower().strip()
    if provider not in {"openai", "hf", "google"}:
        raise typer.BadParameter("provider must be 'openai', 'hf', or 'google'.")

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path(f"generated_{timestamp}.png")

    try:
        if model is None:
            if provider == "openai":
                model = DEFAULT_MODEL_OPENAI
            elif provider == "hf":
                model = DEFAULT_MODEL_HF
            else:
                model = DEFAULT_MODEL_GOOGLE

        request = GenerateRequest(
            prompt=prompt,
            model=model,
            provider=provider,
            hf_provider=hf_provider,
            size=size,
            quality=quality,
            seed=seed,
            output_path=output,
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc

    output.parent.mkdir(parents=True, exist_ok=True)

    if provider == "openai":
        api_key = openai_api_key or settings.openai_api_key
        base_url = openai_base_url or settings.openai_base_url
        if not api_key:
            raise typer.BadParameter(
                "OpenAI API key is required. Set OPENAI_API_KEY or use --openai-api-key."
            )
        provider_client = OpenAIProvider(api_key=api_key, base_url=base_url)
        image = provider_client.generate(request)
    elif provider == "google":
        api_key = google_api_key or settings.google_api_key
        if not api_key:
            raise typer.BadParameter(
                "Google API key is required. Set GOOGLE_API_KEY or use --google-api-key."
            )
        provider_client = GoogleProvider(api_key=api_key)
        image = provider_client.generate(request)
    else:
        token = hf_token or settings.hf_token
        if not token:
            raise typer.BadParameter("HF token is required. Set HF_TOKEN or use --hf-token.")
        provider_client = HuggingFaceProvider(token=token, provider=request.hf_provider)
        image = provider_client.generate(request)
    image.save(output)

    result = GenerateResult(
        model=request.model,
        provider=request.provider,
        output_path=output,
    )
    typer.echo(f"Saved image to {result.output_path}")


if __name__ == "__main__":
    app()
