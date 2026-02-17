#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <prompts_file> [jobs]" >&2
  exit 1
fi

prompts_file="$1"
jobs="${2:-1}"
# Rate limit/retry settings (override via env)
retries="${RETRIES:-3}"
backoff_base="${BACKOFF_BASE:-2}"
max_sleep="${MAX_SLEEP:-30}"
per_request_sleep="${PER_REQUEST_SLEEP:-0}"
if [[ ! -f "$prompts_file" ]]; then
  echo "Error: file not found: $prompts_file" >&2
  exit 1
fi

generate_one() {
  local line="$1"
  local base="${line%%,*}"
  base="${base#"${base%%[![:space:]]*}"}"
  base="${base%"${base##*[![:space:]]}"}"
  if [[ -z "${base//[[:space:]]/}" ]]; then
    base="prompt"
  fi
  local out_file="${base}.png"
  local attempt=0
  while true; do
    if uv run image-generator --output "$out_file" "$line"; then
      if [[ "$per_request_sleep" -gt 0 ]]; then
        sleep "$per_request_sleep"
      fi
      return 0
    fi
    attempt=$((attempt + 1))
    if [[ "$attempt" -gt "$retries" ]]; then
      echo "Error: failed after ${retries} retries for prompt: $line" >&2
      return 1
    fi
    # Exponential backoff with jitter (in seconds)
    local sleep_s=$((backoff_base ** attempt))
    if [[ "$sleep_s" -gt "$max_sleep" ]]; then
      sleep_s="$max_sleep"
    fi
    local jitter=$((RANDOM % 3))
    sleep_s=$((sleep_s + jitter))
    echo "Retrying in ${sleep_s}s (attempt ${attempt}/${retries})..." >&2
    sleep "$sleep_s"
  done
}

export -f generate_one

if [[ "$jobs" -gt 1 ]]; then
  if command -v parallel >/dev/null 2>&1; then
    parallel -j "$jobs" --linebuffer generate_one :::: "$prompts_file"
  else
    xargs -P "$jobs" -n 1 -I {} bash -lc 'generate_one "$@"' _ {} < "$prompts_file"
  fi
else
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ -z "${line//[[:space:]]/}" ]]; then
      continue
    fi
    generate_one "$line"
  done < "$prompts_file"
fi
