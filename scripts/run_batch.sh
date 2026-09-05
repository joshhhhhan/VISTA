#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT="$ROOT/scripts/run_batch.sh"
PYTHON=${ARC3_PYTHON:-"$ROOT/.venv/bin/python"}
ENV_FILE=${ARC3_ENV_FILE:-"$ROOT/.env"}
CODEX_BIN=${ARC3_CODEX_BIN:-"$HOME/.local/share/arc3-codex/0.145.0/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex"}
CLAUDE_BIN=${ARC3_CLAUDE_BIN:-"$HOME/.local/share/claude/versions/2.1.220"}
RUNS_DIR=${ARC3_COMPETITION_RUNS_DIR:-"$ROOT/runs/competition"}
SOCKET=${ARC3_TMUX_SOCKET:-arc3-competition}
RUNTIME=${ARC3_COMPETITION_RUNTIME:-codex}
OPERATION_MODE=${ARC3_COMPETITION_OPERATION_MODE:-competition}
MODEL=${ARC3_COMPETITION_MODEL:-}
EFFORT=${ARC3_COMPETITION_EFFORT:-}
MAX_STEPS=${ARC3_COMPETITION_MAX_STEPS:-2000}
CONCURRENCY=${ARC3_COMPETITION_CONCURRENCY:-}
CLAUDE_WORKERS_PER_ACCOUNT=${ARC3_CLAUDE_WORKERS_PER_ACCOUNT:-2}
TIMEOUT=${ARC3_COMPETITION_TIMEOUT:-300}
MAX_INVALID_RETRIES=${ARC3_COMPETITION_MAX_INVALID_RETRIES:-15}
BATCH_ID=${ARC3_COMPETITION_BATCH_ID:-vista}
MIN_FREE_GIB=${ARC3_COMPETITION_MIN_FREE_GIB:-100}
ALLOW_CONCURRENT_ONLINE=${ARC3_COMPETITION_ALLOW_CONCURRENT_ONLINE:-false}
ALLOW_CONCURRENT_COORDINATOR=${ARC3_COMPETITION_ALLOW_CONCURRENT_COORDINATOR:-false}
INTERNAL_SUPERVISOR=false
if [[ ${1:-} == __supervise ]]; then
  INTERNAL_SUPERVISOR=true
  shift
fi

fail() {
  printf 'competition runner error: %s\n' "$*" >&2
  exit 1
}

if [[ $INTERNAL_SUPERVISOR != true ]]; then
  mode_set=false
  for argument in "$@"; do
    case $argument in
      --mode|--operation-mode|-h|--help)
        mode_set=true
        ;;
    esac
  done
  [[ $mode_set == true ]] \
    || fail "choose --mode offline, online, or competition"
fi

configure_runtime() {
  case $RUNTIME in
    codex)
      MODEL=${MODEL:-gpt-5.6-sol}
      EFFORT=${EFFORT:-max}
      CONCURRENCY=${CONCURRENCY:-2}
      PLAYER_IMAGE=arc3-codex-player:0.1
      PLAYER_DOCKERFILE="$ROOT/Dockerfile.codex-player"
      ;;
    claude)
      MODEL=${MODEL:-opus}
      EFFORT=${EFFORT:-xhigh}
      CONCURRENCY=${CONCURRENCY:-2}
      PLAYER_IMAGE=arc3-claude-player:0.1
      PLAYER_DOCKERFILE="$ROOT/Dockerfile.claude-player"
      ;;
    *)
      fail "--runtime must be codex or claude"
      ;;
  esac
}

usage() {
  cat <<EOF
Usage: ./scripts/run_batch.sh [options]

Run every available environment on one shared ARC scorecard.

Options:
  --runtime RUNTIME
  --mode offline|online|competition
  --operation-mode MODE       Alias for --mode.
  -j, --concurrency N
  --claude-workers-per-account N
  --model MODEL
  --effort LEVEL
  --max-steps N
  --timeout SECONDS
  --max-invalid-retries N
  --batch-id ID
  --runs-dir PATH
  -h, --help

By default, the command refuses to start while another ARC run is active. Set
ARC3_COMPETITION_ALLOW_CONCURRENT_ONLINE=true to permit separate online runs.
Set ARC3_COMPETITION_ALLOW_CONCURRENT_COORDINATOR=true only when account capacity
is coordinated explicitly. A non-scoring preflight runs immediately before the
scorecard is opened.
EOF
}

while (($# > 0)); do
  case $1 in
    --runtime)
      RUNTIME=${2:?}
      shift 2
      ;;
    --mode|--operation-mode)
      OPERATION_MODE=${2:?}
      shift 2
      ;;
    -j|--concurrency)
      CONCURRENCY=${2:?}
      shift 2
      ;;
    --claude-workers-per-account)
      CLAUDE_WORKERS_PER_ACCOUNT=${2:?}
      shift 2
      ;;
    --model)
      MODEL=${2:?}
      shift 2
      ;;
    --effort)
      EFFORT=${2:?}
      shift 2
      ;;
    --max-steps)
      MAX_STEPS=${2:?}
      shift 2
      ;;
    --timeout)
      TIMEOUT=${2:?}
      shift 2
      ;;
    --max-invalid-retries)
      MAX_INVALID_RETRIES=${2:?}
      shift 2
      ;;
    --batch-id)
      BATCH_ID=${2:?}
      shift 2
      ;;
    --runs-dir)
      RUNS_DIR=${2:?}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

configure_runtime
[[ $OPERATION_MODE == offline \
  || $OPERATION_MODE == online \
  || $OPERATION_MODE == competition ]] \
  || fail "--mode must be offline, online, or competition"
[[ $CONCURRENCY =~ ^[1-9][0-9]*$ ]] || fail "invalid concurrency"
[[ $CLAUDE_WORKERS_PER_ACCOUNT =~ ^[1-9][0-9]*$ ]] \
  || fail "invalid Claude workers per account"
[[ $MAX_STEPS =~ ^[1-9][0-9]*$ ]] || fail "invalid max steps"
[[ $TIMEOUT =~ ^[1-9][0-9]*$ ]] || fail "invalid timeout"
[[ $MAX_INVALID_RETRIES =~ ^[0-9]+$ ]] || fail "invalid retry count"
[[ $BATCH_ID =~ ^[A-Za-z0-9_-]+$ ]] || fail "invalid batch id"
[[ $ALLOW_CONCURRENT_ONLINE == true || $ALLOW_CONCURRENT_ONLINE == false ]] \
  || fail "ARC3_COMPETITION_ALLOW_CONCURRENT_ONLINE must be true or false"
[[ $ALLOW_CONCURRENT_COORDINATOR == true \
  || $ALLOW_CONCURRENT_COORDINATOR == false ]] \
  || fail "ARC3_COMPETITION_ALLOW_CONCURRENT_COORDINATOR must be true or false"
if [[ $RUNTIME == claude ]]; then
  [[ $EFFORT =~ ^(low|medium|high|xhigh|max)$ ]] \
    || fail "invalid Claude effort"
else
  [[ $EFFORT =~ ^(minimal|low|medium|high|xhigh|max|ultra)$ ]] \
    || fail "invalid Codex effort"
fi

load_environment() {
  local selected_mode=$OPERATION_MODE
  [[ -r "$ENV_FILE" ]] || fail "ARC environment file missing: $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  export OPERATION_MODE="$selected_mode"
}

env_file_has_value() {
  "$PYTHON" - "$ENV_FILE" "$1" <<'PY'
import sys
from dotenv import dotenv_values

value = dotenv_values(sys.argv[1]).get(sys.argv[2])
raise SystemExit(0 if value else 1)
PY
}

runtime_version() {
  if [[ $RUNTIME == codex ]]; then
    "$CODEX_BIN" --version
  else
    local -a clean_env=(env -u ARC_API_KEY -u ANTHROPIC_API_KEY)
    local name
    while IFS= read -r name; do
      clean_env+=(-u "$name")
    done < <(compgen -A variable CLAUDE_CODE_OAUTH_TOKEN)
    "${clean_env[@]}" "$CLAUDE_BIN" --version
  fi
}

check_prerequisites() {
  if [[ $INTERNAL_SUPERVISOR == true ]]; then
    load_environment
  else
    [[ -r "$ENV_FILE" ]] || fail "ARC environment file missing: $ENV_FILE"
  fi
  [[ -x "$PYTHON" ]] || fail "Python runtime not found: $PYTHON"
  command -v docker >/dev/null || fail "docker is not installed"
  command -v git >/dev/null || fail "git is not installed"
  command -v tmux >/dev/null || fail "tmux is not installed"
  docker info >/dev/null 2>&1 || fail "Docker is unavailable"
  docker image inspect "$PLAYER_IMAGE" >/dev/null 2>&1 \
    || fail "Docker image $PLAYER_IMAGE is missing"
  if [[ $RUNTIME == codex ]]; then
    [[ -r "$HOME/.codex/auth.json" ]] || fail "Codex auth missing under HOME=$HOME"
    [[ -x "$CODEX_BIN" ]] || fail "Codex runtime is missing: $CODEX_BIN"
  else
    [[ -x "$CLAUDE_BIN" ]] || fail "Claude Code runtime is missing: $CLAUDE_BIN"
    [[ $(runtime_version) == "2.1.220 (Claude Code)" ]] \
      || fail "Claude Code 2.1.220 is required"
    if [[ $INTERNAL_SUPERVISOR == true ]]; then
      [[ -n ${CLAUDE_CODE_OAUTH_TOKEN:-} ]] \
        || fail "CLAUDE_CODE_OAUTH_TOKEN is missing; run claude setup-token"
    else
      env_file_has_value CLAUDE_CODE_OAUTH_TOKEN \
        || fail "CLAUDE_CODE_OAUTH_TOKEN is missing from $ENV_FILE"
    fi
  fi
  if [[ $OPERATION_MODE != offline ]]; then
    if [[ $INTERNAL_SUPERVISOR == true ]]; then
      [[ -n ${ARC_API_KEY:-} ]] || fail "ARC_API_KEY is missing"
    else
      env_file_has_value ARC_API_KEY \
        || fail "ARC_API_KEY is missing from $ENV_FILE"
    fi
  fi
  mkdir -p "$RUNS_DIR"
  local available_kib
  available_kib=$(df --output=avail -k "$RUNS_DIR" | awk 'NR == 2 {print $1}')
  ((available_kib >= MIN_FREE_GIB * 1024 * 1024)) \
    || fail "less than $MIN_FREE_GIB GiB is free under $RUNS_DIR"
}

single_game_runs_exist() {
  ps -eo comm=,args= | awk '
    $1 ~ /^python([0-9.]*)?$/ && /scripts\/run_arc3_(codex|claude).py/ { found = 1 }
    END { exit !found }
  '
}

check_online_run_policy() {
  if [[ $OPERATION_MODE == offline ]]; then
    return
  fi
  if ! single_game_runs_exist; then
    return
  fi
  if [[ $ALLOW_CONCURRENT_ONLINE == true ]]; then
    printf '[%s] Concurrent online ARC run detected; proceeding by explicit opt-in.\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    return
  fi
  fail "single-game ARC runs are still active"
}

competition_run_exists() {
  ps -eo comm=,args= | awk '
    $1 ~ /^python([0-9.]*)?$/ && /scripts\/run_arc3_batch.py/ { found = 1 }
    END { exit !found }
  '
}

check_competition_run_policy() {
  if ! competition_run_exists; then
    return
  fi
  if [[ $ALLOW_CONCURRENT_COORDINATOR == true ]]; then
    printf '[%s] Another competition coordinator is active; proceeding by explicit opt-in.\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    return
  fi
  fail "another competition coordinator is running"
}

source_is_unchanged() {
  [[ $(git -C "$ROOT" rev-parse HEAD) == "$PINNED_COMMIT" ]] \
    && git -C "$ROOT" diff --quiet -- \
      "$PLAYER_DOCKERFILE" pyproject.toml src scripts \
    && git -C "$ROOT" diff --cached --quiet -- \
      "$PLAYER_DOCKERFILE" pyproject.toml src scripts \
    && [[ -z $(git -C "$ROOT" ls-files --others --exclude-standard -- src scripts) ]] \
    && [[ $(docker image inspect "$PLAYER_IMAGE" --format '{{.Id}}') \
      == "$PINNED_IMAGE_ID" ]] \
    && [[ $(runtime_version) == "$PINNED_RUNTIME_VERSION" ]]
}

run_python() {
  local -a command=(
    "$PYTHON" "$ROOT/scripts/run_arc3_batch.py"
    --runtime "$RUNTIME"
    --operation-mode "$OPERATION_MODE"
    --model "$MODEL"
    --max-steps "$MAX_STEPS"
    --concurrency "$CONCURRENCY"
    --claude-workers-per-account "$CLAUDE_WORKERS_PER_ACCOUNT"
    --timeout "$TIMEOUT"
    --max-invalid-retries "$MAX_INVALID_RETRIES"
    --batch-id "$BATCH_ID"
    --runs-dir "$RUNS_DIR"
  )
  command+=(--effort "$EFFORT")
  command+=("$@")
  "${command[@]}"
}

supervise() {
  check_prerequisites
  check_competition_run_policy
  check_online_run_policy
  source_is_unchanged || fail "pinned source or runtime changed while waiting"
  printf '[%s] Running non-scoring preflight.\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_python --preflight
  source_is_unchanged || fail "pinned source or runtime changed after preflight"
  check_online_run_policy
  check_competition_run_policy
  printf '[%s] Opening the %s scorecard.\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OPERATION_MODE"
  run_python
}

if [[ $INTERNAL_SUPERVISOR == true ]]; then
  PINNED_COMMIT=${ARC3_PINNED_COMMIT:?}
  PINNED_IMAGE_ID=${ARC3_PINNED_IMAGE_ID:?}
  PINNED_RUNTIME_VERSION=${ARC3_PINNED_RUNTIME_VERSION:?}
  supervise
  exit
fi

check_prerequisites
check_competition_run_policy
check_online_run_policy
PINNED_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
PINNED_IMAGE_ID=$(docker image inspect "$PLAYER_IMAGE" --format '{{.Id}}')
PINNED_RUNTIME_VERSION=$(runtime_version)
source_is_unchanged || fail "competition source has uncommitted changes"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LAUNCH_DIR="$RUNS_DIR/launch_${BATCH_ID}_$STAMP"
SESSION="arc3_competition_${BATCH_ID}_$STAMP"
mkdir -p "$LAUNCH_DIR"

printf -v command '%q ' \
  env \
  "HOME=$HOME" \
  "PATH=$PATH" \
  "ARC3_PYTHON=$PYTHON" \
  "ARC3_ENV_FILE=$ENV_FILE" \
  "ARC3_CODEX_BIN=$CODEX_BIN" \
  "ARC3_CLAUDE_BIN=$CLAUDE_BIN" \
  "ARC3_COMPETITION_RUNS_DIR=$RUNS_DIR" \
  "ARC3_TMUX_SOCKET=$SOCKET" \
  "ARC3_COMPETITION_RUNTIME=$RUNTIME" \
  "ARC3_COMPETITION_OPERATION_MODE=$OPERATION_MODE" \
  "ARC3_COMPETITION_MODEL=$MODEL" \
  "ARC3_COMPETITION_EFFORT=$EFFORT" \
  "ARC3_COMPETITION_MAX_STEPS=$MAX_STEPS" \
  "ARC3_COMPETITION_CONCURRENCY=$CONCURRENCY" \
  "ARC3_CLAUDE_WORKERS_PER_ACCOUNT=$CLAUDE_WORKERS_PER_ACCOUNT" \
  "ARC3_COMPETITION_TIMEOUT=$TIMEOUT" \
  "ARC3_COMPETITION_MAX_INVALID_RETRIES=$MAX_INVALID_RETRIES" \
  "ARC3_COMPETITION_BATCH_ID=$BATCH_ID" \
  "ARC3_COMPETITION_MIN_FREE_GIB=$MIN_FREE_GIB" \
  "ARC3_COMPETITION_ALLOW_CONCURRENT_ONLINE=$ALLOW_CONCURRENT_ONLINE" \
  "ARC3_COMPETITION_ALLOW_CONCURRENT_COORDINATOR=$ALLOW_CONCURRENT_COORDINATOR" \
  "ARC3_PINNED_COMMIT=$PINNED_COMMIT" \
  "ARC3_PINNED_IMAGE_ID=$PINNED_IMAGE_ID" \
  "ARC3_PINNED_RUNTIME_VERSION=$PINNED_RUNTIME_VERSION" \
  "$SCRIPT" __supervise
printf -v log_path '%q' "$LAUNCH_DIR/supervisor.log"
command+=" >>$log_path 2>&1"

tmux -L "$SOCKET" new-session -d -s "$SESSION" -c "$ROOT" "$command"
sleep 1
tmux -L "$SOCKET" has-session -t "$SESSION" 2>/dev/null \
  || fail "competition supervisor failed; inspect $LAUNCH_DIR/supervisor.log"

printf 'Supervisor: %s:%s\n' "$SOCKET" "$SESSION"
printf 'Launch log: %s\n' "$LAUNCH_DIR/supervisor.log"
printf 'Status: tail -f %q\n' "$LAUNCH_DIR/supervisor.log"
