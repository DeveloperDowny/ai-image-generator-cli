## About this project

CLI tool to generate images from a prompt using hosted APIs. The default provider
is OpenAI Images.

## Quick start (uv)

1. Create an OpenAI account and create an API key.
2. Export it as `OPENAI_API_KEY`.
3. Install dependencies with uv.
4. Run the CLI.

```bash
uv sync
```

```bash
$env:OPENAI_API_KEY = "sk_..."
uv run image-generator generate "A cozy cabin in a snowy forest"
```

To use Hugging Face Inference Providers instead:

```bash
$env:HF_TOKEN = "hf_..."
uv run image-generator generate --provider hf --model black-forest-labs/FLUX.1-dev "A cozy cabin in a snowy forest"
```

To use Google Gemini instead:

```bash
$env:GOOGLE_API_KEY = "AIza..."
uv run image-generator generate --provider google --model gemini-2.5-flash-image "A cozy cabin in a snowy forest"
```

## Configuration

- `OPENAI_API_KEY`: OpenAI API key used for image generation.
- `OPENAI_BASE_URL`: Optional base URL override for OpenAI.
- `GOOGLE_API_KEY`: Google API key used for Gemini image generation.
- `HF_TOKEN`: Hugging Face access token used for Inference Providers (when using `--provider hf`).

## Defaults

- Provider: `openai`
- Model: `gpt-image-1-mini`
