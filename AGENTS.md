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

## Project Intent
- Build a CLI tool to generate AI images from a prompt using hosted APIs/models.
- Prefer providers with monthly free credits; fully free options count as free.
