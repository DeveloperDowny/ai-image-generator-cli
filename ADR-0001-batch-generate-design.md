# ADR-0001: Batch Generate Design Choice

- Date: 2026-03-03
- Status: Proposed
- Owners: image-generator maintainers

## Context

The batch flow in `scripts/batch_generate.py` currently orchestrates:

- reading/chunking prompts,
- interactive prompt editing,
- retries with exponential backoff,
- output path generation,
- metadata JSONL writing,
- optional image preview.

For image generation itself, the script invokes the public CLI command via `subprocess` instead of directly calling provider classes or a shared in-process service.

The project goals emphasize clean architecture, structured metadata, type safety, and maintainable small functions.

## Decision

Keep the current **CLI-as-boundary orchestration** for `batch_generate` in the short term, and evolve toward a shared in-process generation service in phases.

### Why keep it now

- Reuses the same public entrypoint users run (`image-generator generate`).
- Keeps provider-specific logic centralized in the main CLI path.
- Reduces immediate refactor risk while preserving current behavior.

### Why not stop here forever

- Process boundaries lose typed exceptions and rich provider error details.
- Retry policy cannot easily branch on retryable vs non-retryable failure classes.
- Subprocess startup adds overhead for large batches.

## Alternatives Considered

### A) Subprocess CLI orchestration (current)

**Pros**

- Strong decoupling from provider implementation details.
- Stable if CLI contract remains stable.
- Minimal code sharing complexity.

**Cons**

- Coarse failure handling (`returncode` only).
- Harder structured telemetry for failures.
- Less efficient at scale.

### B) Direct provider calls from batch script

**Pros**

- Full typed control, richer error handling, lower overhead.

**Cons**

- Duplicates option/default resolution now owned by CLI path.
- Couples script tightly to provider internals.

### C) Shared application service used by both CLI and batch (target)

**Pros**

- Best balance: single generation logic, typed errors, no duplication.
- CLI and batch become thin adapters.

**Cons**

- Requires refactor and migration testing.

## Consequences

Short term:

- Behavior remains stable and user-facing workflows are unchanged.
- Architecture debt remains around failure classification and process overhead.

Medium term:

- Introducing a shared generation service reduces duplication and improves observability.

## Phased Migration Plan

### Phase 0 (now): Stabilize current boundary

1. Keep subprocess architecture.
2. Add structured failure metadata records (for example `generate.error`) with:
   - `error.type`,
   - `error.message`,
   - command context (`provider`, `model`, `output_path`, `prompt`).
3. Align retry defaults in model and CLI options to avoid drift.

### Phase 1: Improve operability without boundary change

1. Add explicit `--non-interactive` mode for unattended runs.
2. Introduce deterministic output naming strategy (prompt index and/or short hash).
3. Add counters/log summary per run (total, success, failed, retries).

### Phase 2: Extract shared in-process generation service

1. Create an application-level `generate_image(...)` service with typed input/output models.
2. Move provider/model resolution and common error mapping to service layer.
3. Keep CLI command behavior unchanged by routing it through the service.

### Phase 3: Migrate batch script to service path

1. Replace subprocess invocation with service calls.
2. Keep a temporary `--use-subprocess` fallback for rollback safety.
3. Remove fallback after validation window.

## Validation / Exit Criteria

- Batch output parity with current behavior for the same prompt set.
- Metadata schema remains stable and newline-delimited JSONL.
- Retry behavior is deterministic and documented.
- No regression in default provider/model behavior.

## Non-goals

- No immediate UX redesign of the batch workflow.
- No provider-specific feature expansion in this ADR.
- No breaking changes to the public CLI contract.
