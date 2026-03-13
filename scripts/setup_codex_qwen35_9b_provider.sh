#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install a llama.cpp Qwen3.5-9B model provider entry into ~/.codex/config.toml.

Usage:
  scripts/setup_codex_qwen35_9b_provider.sh
  CONFIG=~/.codex/config.toml PROVIDER_ID=qwen35_local MODEL_ALIAS=qwen35-9b PORT=8001 scripts/setup_codex_qwen35_9b_provider.sh

Optional env vars:
  CONFIG=/path/to/config.toml
  PROVIDER_ID=qwen35_local
  MODEL_ALIAS=qwen35-9b
  HOST=127.0.0.1
  PORT=8001
  DRY_RUN=1 (print block only)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CONFIG="${CONFIG:-$HOME/.codex/config.toml}"
PROVIDER_ID="${PROVIDER_ID:-qwen35_local}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen35-9b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"

echo "[model_providers.${PROVIDER_ID}]"
echo "name = \"Local Qwen3.5-9B via llama.cpp\""
echo "base_url = \"http://${HOST}:${PORT}/v1\""
echo "wire_api = \"responses\""

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: not writing to ${CONFIG}"
  exit 0
fi

if [[ ! -f "${CONFIG}" ]]; then
  mkdir -p "$(dirname "${CONFIG}")"
  touch "${CONFIG}"
fi

if rg -q "^\[model_providers\.${PROVIDER_ID}\]" "${CONFIG}"; then
  echo "provider already present in ${CONFIG}: model_providers.${PROVIDER_ID}"
  echo "If needed, edit it manually to use host ${HOST}, port ${PORT}, alias ${MODEL_ALIAS}."
  exit 0
fi

BACKUP_PATH="${CONFIG}.codex_backup_$(date +%Y%m%d_%H%M%S)"
cp "${CONFIG}" "${BACKUP_PATH}"
{
  printf '\n'
  printf '[model_providers.%s]\n' "${PROVIDER_ID}"
  printf 'name = "%s"\n' "Local Qwen3.5-9B via llama.cpp"
  printf 'base_url = "http://%s:%s/v1"\n' "${HOST}" "${PORT}"
  printf 'wire_api = "responses"\n'
  printf '\n'
} >> "${CONFIG}"

cat <<EOF
Appended provider block to: ${CONFIG}
Backup at: ${BACKUP_PATH}

Use with:
  codex -c model_provider=${PROVIDER_ID} -c model=${MODEL_ALIAS}
  --search
EOF
