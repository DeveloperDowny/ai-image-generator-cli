"""Batch image generation script with prompt editing and retries."""

from __future__ import annotations

import logging
import os
import random
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable, List, Optional

import typer
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from image_generator.defaults import (
    DEFAULT_MODEL_GOOGLE,
    DEFAULT_MODEL_HF,
    DEFAULT_MODEL_OPENAI,
    DEFAULT_PROVIDER,
)

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, no_args_is_help=True)


class ScriptSettings(BaseSettings):
    """Environment-driven defaults for the batch script.

    Attributes:
        editor: Editor command to open prompt batches.
        cli_command: Base CLI command used for image generation.
        output_dir: Default output directory for generated images.
        metadata_path: Default metadata jsonl output path.
        env: Environment name for log metadata (for example, "dev" or "prod").
    """

    model_config = SettingsConfigDict(env_prefix="IMAGE_GEN_")

    editor: Optional[str] = None
    cli_command: str = "image-generator"
    output_dir: Path = Path("outputs")
    metadata_path: Optional[Path] = None
    env: str = "unknown"


class ImageMetadata(BaseModel):
    """Metadata captured for each generated image.

    Attributes:
        ts: ISO 8601 timestamp with timezone.
        level: Log severity (for example, "info").
        event: Machine-readable event name.
        message: Human-readable summary.
        service: Service name.
        version: CLI version.
        env: Environment name.
        run_id: Unique ID for the batch run.
        request_id: Unique ID for the generated image.
        provider: Provider used to generate the image (if known).
        model: Model used to generate the image (if known).
        prompt: The tweaked prompt used for generation.
        output_path: Full path to the generated image.
        file_name: File name of the generated image.
        batch_index: 1-based batch index for this prompt.
        prompt_index: 1-based index within the batch.
    """

    ts: datetime
    level: str
    event: str
    message: str
    service: str
    version: str
    env: str
    run_id: str
    request_id: str
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt: str
    output_path: Path
    file_name: str
    batch_index: int
    prompt_index: int


class GenerateConfig(BaseModel):
    """Configuration for invoking the image-generator CLI.

    Attributes:
        cli_command: Base CLI command (already split into args).
        provider: Optional provider override.
        model: Optional model override.
        size: Optional size override.
        quality: Optional quality override.
        seed: Optional seed override.
        hf_provider: Optional Hugging Face provider override.
    """

    cli_command: List[str] = Field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    seed: Optional[int] = None
    hf_provider: Optional[str] = None


class RetryConfig(BaseModel):
    """Retry/backoff configuration.

    Attributes:
        retries: Maximum number of retries before giving up.
        backoff_base: Exponential backoff base (seconds).
        max_sleep: Maximum sleep between retries (seconds).
        per_request_sleep: Sleep after a successful request (seconds).
    """

    retries: int = 5
    backoff_base: int = 10
    max_sleep: int = 60
    per_request_sleep: int = 0


def read_prompts(path: Path) -> List[str]:
    """Read non-empty prompts from a line-delimited file."""
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if cleaned:
            prompts.append(cleaned)
    return prompts


def chunk_prompts(prompts: List[str], batch_size: int) -> Iterable[List[str]]:
    """Yield prompts in fixed-size batches."""
    for i in range(0, len(prompts), batch_size):
        yield prompts[i : i + batch_size]


def parse_command(command: str) -> List[str]:
    """Split a command string into argv parts."""
    if os.name == "nt":
        return shlex.split(command, posix=False)
    return shlex.split(command, posix=True)


def open_editor(prompts: List[str], editor_command: List[str]) -> List[str]:
    """Open prompts in an editor and return the edited lines."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        delete=False,
    ) as handle:
        handle.write("\n".join(prompts))
        temp_path = Path(handle.name)

    try:
        logger.info("Opening editor for batch: %s", temp_path)
        subprocess.run([*editor_command, str(temp_path)], check=True)
        edited = read_prompts(temp_path)
        if not edited:
            logger.warning("Edited batch is empty. Skipping.")
        return edited
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Unable to remove temp file: %s", temp_path)


def open_prompt_editor(prompt: str, editor_command: List[str]) -> Optional[str]:
    """Open a single prompt in an editor and return the edited prompt."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        delete=False,
    ) as handle:
        handle.write(prompt)
        temp_path = Path(handle.name)

    try:
        logger.info("Opening editor for prompt: %s", temp_path)
        subprocess.run([*editor_command, str(temp_path)], check=True)
        edited = read_prompts(temp_path)
        if not edited:
            return None
        return edited[0]
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Unable to remove temp file: %s", temp_path)


def open_prompt_editors(
    prompts: List[str], editor_command: List[str]
) -> List[Optional[str]]:
    """Open one editor per prompt in parallel and return edited prompts."""
    temp_paths: List[Path] = []
    processes: List[subprocess.Popen[bytes] | subprocess.Popen[str]] = []

    try:
        for prompt in prompts:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".txt",
                delete=False,
            ) as handle:
                handle.write(prompt)
                temp_path = Path(handle.name)
            temp_paths.append(temp_path)

            logger.info("Opening editor for prompt: %s", temp_path)
            process = subprocess.Popen([*editor_command, str(temp_path)])
            processes.append(process)

        for process in processes:
            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, process.args)

        edited_prompts: List[Optional[str]] = []
        for temp_path in temp_paths:
            edited = read_prompts(temp_path)
            if not edited:
                edited_prompts.append(None)
                continue
            edited_prompts.append(edited[0])
        return edited_prompts
    finally:
        for temp_path in temp_paths:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Unable to remove temp file: %s", temp_path)


def open_image(path: Path) -> None:
    """Open the image file in the default viewer."""
    try:
        if os.name == "nt":
            subprocess.run(["cmd", "/c", "start", "", str(path)], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except OSError:
        logger.warning("Unable to open image viewer for %s", path)


def sanitize_name(value: str) -> str:
    """Convert a string into a filesystem-friendly name."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "img"


def unique_output_path(
    output_dir: Path,
    prompt: str,
    used_names: set[str],
) -> Path:
    """Create a unique output path using the first two prompt characters."""
    trimmed = prompt.strip()
    base = trimmed[:2] if trimmed else "img"
    safe_base = sanitize_name(base)
    candidate = output_dir / f"{safe_base}.png"
    if candidate.name not in used_names and not candidate.exists():
        used_names.add(candidate.name)
        return candidate

    index = 1
    while True:
        candidate = output_dir / f"{safe_base}_{index:02d}.png"
        if candidate.name not in used_names and not candidate.exists():
            used_names.add(candidate.name)
            return candidate
        index += 1


def build_cli_args(config: GenerateConfig, output_path: Path, prompt: str) -> List[str]:
    """Build CLI arguments for image generation."""
    args = list(config.cli_command)
    if config.provider:
        args.extend(["--provider", config.provider])
    if config.model:
        args.extend(["--model", config.model])
    if config.hf_provider:
        args.extend(["--hf-provider", config.hf_provider])
    if config.size:
        args.extend(["--size", config.size])
    if config.quality:
        args.extend(["--quality", config.quality])
    if config.seed is not None:
        args.extend(["--seed", str(config.seed)])
    args.extend(["--output", str(output_path), prompt])
    return args


def resolve_provider_model(config: GenerateConfig) -> tuple[str, str]:
    """Resolve provider and model defaults to match CLI behavior."""
    provider = config.provider or DEFAULT_PROVIDER
    if config.model:
        return provider, config.model
    if provider == "openai":
        return provider, DEFAULT_MODEL_OPENAI
    if provider == "hf":
        return provider, DEFAULT_MODEL_HF
    return provider, DEFAULT_MODEL_GOOGLE


def run_with_retries(
    command: List[str],
    retry_config: RetryConfig,
) -> None:
    """Run a command with retry and exponential backoff."""
    attempt = 0
    while True:
        logger.info("Running: %s", " ".join(command))
        result = subprocess.run(command)
        if result.returncode == 0:
            if retry_config.per_request_sleep > 0:
                time.sleep(retry_config.per_request_sleep)
            return

        attempt += 1
        if attempt > retry_config.retries:
            raise RuntimeError(f"Command failed after {retry_config.retries} retries.")

        sleep_s = min(retry_config.max_sleep, retry_config.backoff_base**attempt)
        sleep_s += random.uniform(0, 1)
        logger.warning(
            "Command failed (attempt %s/%s). Retrying in %.1fs.",
            attempt,
            retry_config.retries,
            sleep_s,
        )
        time.sleep(sleep_s)


def generate_prompt_image(
    command: List[str],
    retry_config: RetryConfig,
    prompt_index: int,
    prompt: str,
    output_path: Path,
) -> tuple[int, str, Path]:
    """Generate one image with retries and return generation details."""
    run_with_retries(command, retry_config)
    return prompt_index, prompt, output_path


def write_metadata(metadata_path: Path, metadata: ImageMetadata) -> None:
    """Append metadata to a jsonl file."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write(metadata.model_dump_json())
        handle.write("\n")


@app.command()
def batch_generate(
    prompts_file: Path = typer.Argument(..., help="File with line-delimited prompts."),
    batch_size: int = typer.Option(5, help="Number of prompts per batch."),
    output_dir: Optional[Path] = typer.Option(
        None, help="Directory to store generated images."
    ),
    metadata_path: Optional[Path] = typer.Option(
        None, help="JSONL file path for metadata."
    ),
    editor: Optional[str] = typer.Option(None, help="Editor command for batch edits."),
    cli_command: Optional[str] = typer.Option(
        None, help="Base command for image generation."
    ),
    provider: Optional[str] = typer.Option(None, help="Provider override."),
    model: Optional[str] = typer.Option(None, help="Model override."),
    hf_provider: Optional[str] = typer.Option(None, help="HF provider override."),
    size: Optional[str] = typer.Option(None, help="Image size override."),
    quality: Optional[str] = typer.Option(None, help="Image quality override."),
    seed: Optional[int] = typer.Option(None, help="Random seed override."),
    retries: int = typer.Option(5, help="Retry attempts per prompt."),
    backoff_base: int = typer.Option(2, help="Exponential backoff base."),
    max_sleep: int = typer.Option(60, help="Maximum retry sleep in seconds."),
    per_request_sleep: int = typer.Option(0, help="Sleep after each success."),
    show_image: bool = typer.Option(True, help="Open generated images for review."),
    verbose: bool = typer.Option(False, help="Enable verbose logging."),
) -> None:
    """Batch-generate images with prompt editing and retries."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    settings = ScriptSettings()
    run_id = str(uuid.uuid4())
    try:
        version = importlib_metadata.version("image-generator")
    except importlib_metadata.PackageNotFoundError:
        version = "unknown"

    if not prompts_file.exists():
        raise typer.BadParameter(f"Prompts file not found: {prompts_file}")
    if batch_size <= 0:
        raise typer.BadParameter("batch-size must be greater than zero")

    editor_value = editor or settings.editor or os.environ.get("EDITOR") or "notepad"
    editor_cmd = parse_command(editor_value)
    if not editor_cmd:
        raise typer.BadParameter("editor command is empty")

    cli_value = cli_command or settings.cli_command
    cli_cmd = parse_command(cli_value)
    if not cli_cmd:
        raise typer.BadParameter("cli-command is empty")

    resolved_output_dir = output_dir or settings.output_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_metadata = metadata_path or settings.metadata_path
    if resolved_metadata is None:
        resolved_metadata = resolved_output_dir / "metadata.jsonl"

    generate_config = GenerateConfig(
        cli_command=cli_cmd,
        provider=provider,
        model=model,
        hf_provider=hf_provider,
        size=size,
        quality=quality,
        seed=seed,
    )
    retry_config = RetryConfig(
        retries=retries,
        backoff_base=backoff_base,
        max_sleep=max_sleep,
        per_request_sleep=per_request_sleep,
    )
    effective_provider, effective_model = resolve_provider_model(generate_config)

    prompts = read_prompts(prompts_file)
    if not prompts:
        logger.warning("No prompts found in %s", prompts_file)
        return

    used_names: set[str] = set()
    for batch_index, batch in enumerate(chunk_prompts(prompts, batch_size), start=1):
        edited_prompts = open_editor(batch, editor_cmd)
        if not edited_prompts:
            continue

        pending_prompts: List[tuple[int, str]] = [
            (prompt_index, prompt)
            for prompt_index, prompt in enumerate(edited_prompts, start=1)
        ]

        while pending_prompts:
            jobs: List[tuple[int, str, Path, List[str]]] = []
            for prompt_index, current_prompt in pending_prompts:
                output_path = unique_output_path(
                    resolved_output_dir, current_prompt, used_names
                )
                cmd = build_cli_args(generate_config, output_path, current_prompt)
                jobs.append((prompt_index, current_prompt, output_path, cmd))

            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                future_to_job = {
                    executor.submit(
                        generate_prompt_image,
                        cmd,
                        retry_config,
                        prompt_index,
                        current_prompt,
                        output_path,
                    ): (prompt_index, current_prompt, output_path)
                    for prompt_index, current_prompt, output_path, cmd in jobs
                }

                for future in as_completed(future_to_job):
                    prompt_index, current_prompt, output_path = future.result()
                    meta_entry = ImageMetadata(
                        ts=datetime.now(timezone.utc),
                        level="info",
                        event="generate.success",
                        message="image generated",
                        service="image-generator",
                        version=version,
                        env=settings.env,
                        run_id=run_id,
                        request_id=str(uuid.uuid4()),
                        provider=effective_provider,
                        model=effective_model,
                        prompt=current_prompt,
                        output_path=output_path,
                        file_name=output_path.name,
                        batch_index=batch_index,
                        prompt_index=prompt_index,
                    )
                    write_metadata(resolved_metadata, meta_entry)
                    logger.info("Wrote metadata for %s", output_path.name)

                    if show_image:
                        open_image(output_path)

            edited_round = open_prompt_editors(
                [prompt for _, prompt in pending_prompts],
                editor_cmd,
            )

            next_round: List[tuple[int, str]] = []
            for (prompt_index, current_prompt), edited_prompt in zip(
                pending_prompts,
                edited_round,
            ):
                if not edited_prompt:
                    logger.info("Prompt %s cleared; keeping last result.", prompt_index)
                    continue
                if edited_prompt.strip() == current_prompt.strip():
                    continue
                next_round.append((prompt_index, edited_prompt))

            pending_prompts = next_round


if __name__ == "__main__":
    app()
