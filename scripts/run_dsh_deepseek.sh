#!/usr/bin/env bash
# DeepSeek Harness (dsh) + DeepSeek, isolated per run.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PATH="$HOME/.local/share/deepseek-harness/0.2.2-rc.1/node_modules/.bin:$PATH"

set -a
. "$ROOT/.env"
set +a

exec "$ROOT/.venv/bin/vista" --profile arc3 --backend dsh "$@"
