#!/usr/bin/env bash
# Route A: run VISTA's Python harness on macOS while the Claude player runs in
# a linux/amd64 Docker container.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PATH="$HOME/.docker/bin:$PATH"

set -a
. "$ROOT/.env"
set +a

exec "$ROOT/.venv/bin/vista" --profile arc3 --backend claude --provider deepseek "$@"
