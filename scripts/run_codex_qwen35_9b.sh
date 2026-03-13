#!/usr/bin/env bash
set -euo pipefail

# Lightweight wrapper for running Codex against local llama.cpp Qwen3.5-9B.
exec codex -c model_provider=qwen35_local -c model=qwen35-9b "$@"
