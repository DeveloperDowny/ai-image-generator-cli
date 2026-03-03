# AGENTS.md

## Project Preferences
- Prefer clean architecture and separation of concerns.
- Use Pydantic models for structured data validation and serialization.
- Use pydantic-settings for env
- Use type hints everywhere.
- Use Google-style docstrings.
- Keep code readable and maintainable with small, focused functions.
- Prefer Typer for CLI applications.
- Add structured logging via `logger = logging.getLogger(__name__)`.
- JSON logs (JSONL) best practices:
- One JSON object per line, newline-terminated.
- Stable schema with consistent field names and types.
- Include `ts` (ISO 8601 with timezone), `level`, `event`, `message`.
- Include context: `service`, `version`, `env`, `run_id`, `request_id`.
- Include `provider`, `model`, `output_path` when relevant.
- Log errors as structured objects (`error.type`, `error.message`, `error.stack`).
- Avoid secrets/PII; full prompts are allowed in logs.
- Keep nesting shallow and payloads small.

## Project Intent
- Build a CLI tool to generate AI images from a prompt using hosted APIs/models.
- Prefer providers with monthly free credits; fully free options count as free.
