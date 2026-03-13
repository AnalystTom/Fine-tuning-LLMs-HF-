#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install (or update) heretic under ./tools/heretic for local safety red-team research.

Usage:
  scripts/install_heretic.sh
  HERETIC_DIR=/custom/path/scripts/install_heretic.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HERETIC_DIR="${HERETIC_DIR:-${ROOT_DIR}/tools/heretic}"

mkdir -p "$(dirname "${HERETIC_DIR}")"

if [[ -d "${HERETIC_DIR}/.git" ]]; then
  echo "Updating heretic in ${HERETIC_DIR}"
  git -C "${HERETIC_DIR}" pull --ff-only
else
  echo "Cloning heretic into ${HERETIC_DIR}"
  git clone https://github.com/p-e-w/heretic.git "${HERETIC_DIR}"
fi

echo "Done: ${HERETIC_DIR}"
