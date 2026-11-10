#!/usr/bin/env bash
# Wait until 12:00 Asia/Shanghai, then start the 3 x 25 game experiment.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_NOW=false
PRINT_TARGET=false
ARGS=()
for argument in "$@"; do
  case $argument in
    --run-now) RUN_NOW=true ;;
    --print-target) PRINT_TARGET=true ;;
    *) ARGS+=("$argument") ;;
  esac
done

TARGET_INFO=$(python3 - <<'PY'
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
zone = ZoneInfo("Asia/Shanghai")
now = datetime.now(zone)
target = datetime.combine(now.date(), time(12, 0), tzinfo=zone)
if target <= now:
    target += timedelta(days=1)
print(target.isoformat())
print(int(target.timestamp()))
PY
)
TARGET_ISO=$(printf '%s\n' "$TARGET_INFO" | sed -n '1p')
TARGET_EPOCH=$(printf '%s\n' "$TARGET_INFO" | sed -n '2p')

if [[ $PRINT_TARGET == true ]]; then
  printf 'target=%s epoch=%s\n' "$TARGET_ISO" "$TARGET_EPOCH"
  exit 0
fi

mkdir -p "$ROOT/runs"
LOG="$ROOT/runs/full_experiment_scheduler.log"
exec > >(tee -a "$LOG") 2>&1

# Keep the Mac awake until the experiment process exits.  This follows the
# shell PID through exec, so it also covers the Python orchestrator.
if [[ ${VISTA_EXPERIMENT_CAFFEINATE:-1} == 1 ]] && command -v caffeinate >/dev/null; then
  caffeinate -dimsu -w $$ &
fi

if [[ $RUN_NOW == false ]]; then
  printf '[%s] scheduled for %s (Asia/Shanghai)\n' \
    "$(date '+%F %T %Z')" "$TARGET_ISO"
  while true; do
    now=$(date +%s)
    remaining=$((TARGET_EPOCH - now))
    if (( remaining <= 0 )); then
      break
    fi
    printf '[%s] waiting for 12:00 Asia/Shanghai (%ss remaining)\n' \
      "$(date '+%F %T %Z')" "$remaining"
    if (( remaining > 300 )); then
      sleep 300
    else
      sleep "$remaining"
    fi
  done
fi

DEFAULT_ARGS=(
  --backends codex,claude,dsh
  --max-steps 3000
  --game-timeout 7200
  --max-invalid-retries 15
  --max-consecutive-failures 3
  --max-wall-hours 96
  --game-workers "${VISTA_EXPERIMENT_GAME_WORKERS:-2}"
)
if [[ ${VISTA_EXPERIMENT_PARALLEL_BACKENDS:-1} == 1 ]]; then
  DEFAULT_ARGS+=(--parallel-backends)
fi

printf '[%s] starting full experiment\n' "$(date '+%F %T %Z')"
if (( ${#ARGS[@]} )); then
  exec "$ROOT/.venv/bin/python" "$ROOT/scripts/run_full_experiment.py" \
    "${DEFAULT_ARGS[@]}" "${ARGS[@]}"
else
  exec "$ROOT/.venv/bin/python" "$ROOT/scripts/run_full_experiment.py" \
    "${DEFAULT_ARGS[@]}"
fi
