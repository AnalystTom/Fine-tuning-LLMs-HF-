#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start llama-server with a Qwen preset or a local GGUF.

Usage:
  scripts/start_llama_server.sh qwen35-4b
  scripts/start_llama_server.sh qwen35-9b
  scripts/start_llama_server.sh qwen3-4b
  scripts/start_llama_server.sh qwen3-8b
  scripts/start_llama_server.sh glm47-flash
  LLAMA_MODEL_PATH=/absolute/path/model.gguf scripts/start_llama_server.sh local

Optional environment variables:
  LLAMA_HOST=127.0.0.1
  LLAMA_PORT=8080
  LLAMA_CTX_SIZE=8192
  LLAMA_GPU_LAYERS=999
  LLAMA_THREADS=8
  LLAMA_THREADS_BATCH=8
  LLAMA_ALIAS=<model alias>
  LLAMA_HF_FILE=<exact gguf filename override>

Examples:
  scripts/start_llama_server.sh qwen35-4b
  LLAMA_PORT=8081 scripts/start_llama_server.sh qwen35-9b
  scripts/start_llama_server.sh qwen3-4b
  LLAMA_PORT=8081 scripts/start_llama_server.sh qwen3-8b
  scripts/start_llama_server.sh glm47-flash
  LLAMA_MODEL_PATH=/Users/tom/.lmstudio/models/unsloth/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_S.gguf scripts/start_llama_server.sh local
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PRESET="${1:-qwen3-4b}"
shift || true

HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8080}"
CTX_SIZE="${LLAMA_CTX_SIZE:-8192}"
GPU_LAYERS="${LLAMA_GPU_LAYERS:-999}"
THREADS="${LLAMA_THREADS:-8}"
THREADS_BATCH="${LLAMA_THREADS_BATCH:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${ROOT_DIR}/artifacts/models"
EXTRA_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --ctx-size|-c)
      CTX_SIZE="$2"
      shift 2
      ;;
    --gpu-layers|-ngl|--n-gpu-layers)
      GPU_LAYERS="$2"
      shift 2
      ;;
    --threads|-t)
      THREADS="$2"
      shift 2
      ;;
    --threads-batch|-tb)
      THREADS_BATCH="$2"
      shift 2
      ;;
    --alias|-a)
      LLAMA_ALIAS="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

case "${PRESET}" in
  qwen35-4b)
    HF_REPO="unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
    DEFAULT_ALIAS="qwen35-4b"
    LOCAL_MODEL_PATH="${MODELS_DIR}/Qwen3.5-4B-Q4_K_M.gguf"
    ;;
  qwen35-9b)
    HF_REPO="unsloth/Qwen3.5-9B-GGUF:Q4_K_M"
    DEFAULT_ALIAS="qwen35-9b"
    LOCAL_MODEL_PATH="${MODELS_DIR}/Qwen3.5-9B-Q4_K_M.gguf"
    ;;
  qwen3-4b)
    HF_REPO="Qwen/Qwen3-4B-GGUF:Q4_K_M"
    DEFAULT_ALIAS="qwen3-4b"
    LOCAL_MODEL_PATH="${MODELS_DIR}/Qwen3-4B-Q4_K_M.gguf"
    ;;
  qwen3-8b)
    HF_REPO="Qwen/Qwen3-8B-GGUF:Q4_K_M"
    DEFAULT_ALIAS="qwen3-8b"
    LOCAL_MODEL_PATH="${MODELS_DIR}/Qwen3-8B-Q4_K_M.gguf"
    ;;
  glm47-flash)
    DEFAULT_ALIAS="glm47-flash"
    LOCAL_MODEL_PATH="/Users/tom/.lmstudio/models/unsloth/GLM-4.7-Flash-GGUF/GLM-4.7-Flash-Q4_K_S.gguf"
    ;;
  local)
    MODEL_PATH="${LLAMA_MODEL_PATH:-}"
    if [[ -z "${MODEL_PATH}" ]]; then
      echo "LLAMA_MODEL_PATH must be set when using the 'local' preset." >&2
      exit 1
    fi
    if [[ ! -f "${MODEL_PATH}" ]]; then
      echo "Local model not found: ${MODEL_PATH}" >&2
      exit 1
    fi
    DEFAULT_ALIAS="$(basename "${MODEL_PATH}" .gguf)"
    ;;
  *)
    echo "Unknown preset: ${PRESET}" >&2
    usage
    exit 1
    ;;
esac

ALIAS="${LLAMA_ALIAS:-${DEFAULT_ALIAS}}"

cmd=(
  llama-server
  --host "${HOST}"
  --port "${PORT}"
  --alias "${ALIAS}"
  --ctx-size "${CTX_SIZE}"
  --gpu-layers "${GPU_LAYERS}"
  --threads "${THREADS}"
  --threads-batch "${THREADS_BATCH}"
  --flash-attn on
  --jinja
  --reasoning-format none
  --reasoning-budget 0
)

if [[ "${PRESET}" == "local" ]]; then
  cmd+=(-m "${MODEL_PATH}")
elif [[ -n "${LOCAL_MODEL_PATH:-}" && -f "${LOCAL_MODEL_PATH}" ]]; then
  cmd+=(-m "${LOCAL_MODEL_PATH}")
else
  cmd+=(--hf-repo "${HF_REPO}")
  if [[ -n "${LLAMA_HF_FILE:-}" ]]; then
    cmd+=(--hf-file "${LLAMA_HF_FILE}")
  fi
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "Starting ${ALIAS} on http://${HOST}:${PORT}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
