#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start qwen35-9b with llama-server on the default Codex port.

Usage:
  scripts/start_qwen35_9b_codex_server.sh
  LLAMA_HOST=127.0.0.1 LLAMA_PORT=8001 scripts/start_qwen35_9b_codex_server.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
export LLAMA_PORT="${LLAMA_PORT:-8001}"
export LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-8192}"
export LLAMA_ALIAS="${LLAMA_ALIAS:-qwen35-9b}"

"${SCRIPT_DIR}/start_llama_server.sh" qwen35-9b
