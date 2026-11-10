#!/usr/bin/env bash
# Codex CLI + DeepSeek, isolated from the user's ~/.codex configuration.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PATH="$HOME/.docker/bin:$PATH"
# The pinned Codex binary is linux/amd64 and is validated inside the player
# container on this macOS route-A host.
export ARC3_CODEX_VERSION_OVERRIDE="${ARC3_CODEX_VERSION_OVERRIDE:-codex-cli 0.145.0}"

set -a
. "$ROOT/.env"
set +a

exec "$ROOT/.venv/bin/vista" --profile arc3 --backend codex --provider deepseek "$@"
