#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Download llama.cpp-compatible GGUF models for local comparison.

Usage:
  scripts/download_llama_models.sh qwen35-4b
  scripts/download_llama_models.sh qwen35-9b
  scripts/download_llama_models.sh qwen3-4b
  scripts/download_llama_models.sh qwen3-8b
  scripts/download_llama_models.sh all

Files are stored under ./artifacts/models.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PRESET="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${ROOT_DIR}/artifacts/models"
mkdir -p "${MODELS_DIR}"

download() {
  local filename="$1"
  local url="$2"
  local output="${MODELS_DIR}/${filename}"
  echo "Downloading ${filename} -> ${output}"
  curl -L -C - --fail -o "${output}" "${url}"
}

case "${PRESET}" in
  qwen35-4b)
    download "Qwen3.5-4B-Q4_K_M.gguf" \
      "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"
    ;;
  qwen35-9b)
    download "Qwen3.5-9B-Q4_K_M.gguf" \
      "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf"
    ;;
  qwen3-4b)
    download "Qwen3-4B-Q4_K_M.gguf" \
      "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
    ;;
  qwen3-8b)
    download "Qwen3-8B-Q4_K_M.gguf" \
      "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
    ;;
  all)
    download "Qwen3.5-4B-Q4_K_M.gguf" \
      "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"
    download "Qwen3.5-9B-Q4_K_M.gguf" \
      "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf"
    download "Qwen3-4B-Q4_K_M.gguf" \
      "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
    download "Qwen3-8B-Q4_K_M.gguf" \
      "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
    ;;
  *)
    echo "Unknown preset: ${PRESET}" >&2
    usage
    exit 1
    ;;
esac
