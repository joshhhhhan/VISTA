#!/usr/bin/env python3
"""Run the full Codex/Claude/DSH x 25-game ARC-AGI-3 experiment.

Official completion/failure semantics
-------------------------------------
`arc_agi.scorecard` marks a game completed only when its state is `WIN`:

    completed = state == GameState.WIN

`GAME_OVER` and `NOT_FINISHED` are not wins.  VISTA can continue after a
`GAME_OVER` by sending `RESET`, so a game is not allowed to run forever.  Each
attempt is bounded by:

* `--max-steps` (VISTA's action budget; 0 disables the budget),
* `--max-invalid-retries` (controller retry budget),
* `--game-timeout` (outer wall-clock kill),
* `--max-consecutive-failures` per backend,
* optional `--max-total-tokens-per-backend`,
* optional global `--max-wall-hours`,
* a `STOP` file in the batch directory.

A bounded game that does not reach `WIN` is recorded as incomplete/game_over/
timeout.  RHAE still receives the official partial credit for levels completed
before the stop; unfinished levels score zero.

Reports
-------
The batch directory contains `batch.json`, `summary.json`, `summary.csv`, and
`summary.md`.  Token usage comes from the backend's own records:
Claude `claude_segments[].usage`, Codex `codex_segments[].usage`, and DSH
`status/step_end` events in `dsh_io/*.stdout.jsonl`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

STANDARD_GAME_ORDER = (
    "bp35",
    "sk48",
    "lf52",
    "wa30",
    "sp80",
    "re86",
    "su15",
    "vc33",
    "ka59",
    "dc22",
    "ls20",
    "tn36",
    "g50t",
    "sc25",
    "tu93",
    "s5i5",
    "cn04",
    "tr87",
    "ar25",
    "m0r0",
    "r11l",
    "lp85",
    "ft09",
    "cd82",
    "sb26",
)

BACKEND_WRAPPERS = {
    "codex": ROOT / "scripts" / "run_codex_deepseek.sh",
    "claude": ROOT / "scripts" / "run_route_a.sh",
    "dsh": ROOT / "scripts" / "run_dsh_deepseek.sh",
}
BACKEND_LABELS = {
    "codex": "Codex CLI + DeepSeek",
    "claude": "Claude Code + DeepSeek",
    "dsh": "DeepSeek Harness + DeepSeek",
}
INFRA_FAILURES = {"setup_failed", "error", "timeout"}
FINAL_STATUSES = {
    "win",
    "incomplete",
    "game_over",
    "timeout",
    "setup_failed",
    "error",
    "skipped",
}

DEFAULT_MAX_STEPS = 3_000
DEFAULT_GAME_TIMEOUT = 7_200
DEFAULT_MAX_INVALID_RETRIES = 15
DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_MAX_WALL_HOURS = 96.0


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def duration_seconds(manifest: dict[str, Any], fallback: float) -> float:
    started = parse_time(manifest.get("created_at"))
    finished = parse_time(manifest.get("finished_at"))
    if started is not None and finished is not None:
        return max(0.0, (finished - started).total_seconds())
    return max(0.0, fallback)


def token_usage(backend: str, run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "reasoning": 0,
        "total": 0,
        "cost_usd": 0.0,
    }

    if backend == "claude":
        segments = manifest.get("claude_segments")
        if not isinstance(segments, list) or not segments:
            single = manifest.get("claude")
            segments = [single] if isinstance(single, dict) else []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            values = segment.get("usage")
            if not isinstance(values, dict):
                continue
            usage["input"] += as_int(values.get("input_tokens"))
            usage["output"] += as_int(values.get("output_tokens"))
            usage["cache_read"] += as_int(values.get("cache_read_input_tokens"))
            usage["cache_write"] += as_int(values.get("cache_creation_input_tokens"))
            usage["cost_usd"] += float(values.get("cost_usd") or 0.0)
        usage["total"] = (
            usage["input"]
            + usage["output"]
            + usage["cache_read"]
            + usage["cache_write"]
        )
        return usage

    if backend == "codex":
        segments = manifest.get("codex_segments")
        if not isinstance(segments, list) or not segments:
            single = manifest.get("codex")
            segments = [single] if isinstance(single, dict) else []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            values = segment.get("usage")
            if not isinstance(values, dict):
                continue
            usage["input"] += as_int(values.get("input_tokens"))
            usage["output"] += as_int(values.get("output_tokens"))
            usage["cache_read"] += as_int(values.get("cached_input_tokens"))
            usage["reasoning"] += as_int(values.get("reasoning_output_tokens"))
        # Codex reports cached input as a subset of input tokens.
        usage["total"] = usage["input"] + usage["output"]
        return usage

    # DSH emits incremental usage on every status/step_end event.
    for path in sorted((run_dir / "private" / "dsh_io").glob("segment_*.stdout.jsonl")):
        try:
            stream = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") != "status" or event.get("phase") != "step_end":
                    continue
                values = event.get("usage")
                if not isinstance(values, dict):
                    continue
                usage["input"] += as_int(values.get("inputTokens"))
                usage["output"] += as_int(values.get("outputTokens"))
                usage["cache_read"] += as_int(values.get("cacheReadTokens"))
                usage["cache_write"] += as_int(values.get("cacheWriteTokens"))
                usage["total"] += as_int(values.get("totalTokens"))
    if usage["total"] == 0:
        usage["total"] = (
            usage["input"]
            + usage["output"]
            + usage["cache_read"]
            + usage["cache_write"]
        )
    return usage


def scorecard_run(manifest: dict[str, Any]) -> dict[str, Any]:
    summary = manifest.get("scorecard_summary")
    if not isinstance(summary, dict):
        summary = manifest.get("scorecard_preclose_summary")
    if not isinstance(summary, dict):
        return {}
    environments = summary.get("environments")
    if not isinstance(environments, list) or not environments:
        return {}
    environment = environments[0]
    if not isinstance(environment, dict):
        return {}
    runs = environment.get("runs")
    if not isinstance(runs, list) or not runs:
        return {}
    run = runs[0]
    return run if isinstance(run, dict) else {}


def newest_run_dir(attempt_dir: Path) -> Path | None:
    candidates = [path for path in attempt_dir.glob("run_*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def collect_result(
    *,
    backend: str,
    game_id: str,
    attempt_dir: Path,
    run_dir: Path | None,
    wall_seconds: float,
    timed_out: bool,
    returncode: int | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    if run_dir is not None:
        loaded = load_json(run_dir / "manifest.json", {})
        if isinstance(loaded, dict):
            manifest = loaded

    official = scorecard_run(manifest)
    state = official.get("state")
    if not isinstance(state, str) or not state:
        progress = manifest.get("progress")
        completed_levels = (
            progress.get("completed") if isinstance(progress, dict) else None
        )
        total_levels = progress.get("total") if isinstance(progress, dict) else None
        if (
            manifest.get("terminal")
            and isinstance(completed_levels, int)
            and isinstance(total_levels, int)
            and completed_levels == total_levels
        ):
            state = "WIN"
        else:
            state = "UNKNOWN"

    win = state == "WIN"
    completed = official.get("completed")
    if completed is None:
        completed = win
    progress = manifest.get("progress")
    if not isinstance(progress, dict):
        progress = {}
    levels_completed = official.get("levels_completed")
    if levels_completed is None:
        levels_completed = progress.get("completed")
    level_count = official.get("level_count")
    if not isinstance(level_count, int) or level_count <= 0:
        # The scorecard omits level_count on some runs; the per-level arrays
        # still carry the full level count even when only some levels are done.
        for key in ("level_scores", "level_baseline_actions", "level_actions"):
            values = official.get(key)
            if isinstance(values, list) and values:
                level_count = len(values)
                break
    if not isinstance(level_count, int) or level_count <= 0:
        level_count = progress.get("total")
    actions = official.get("actions")
    if actions is None:
        actions = manifest.get("steps_attempted")
    score = official.get("score")
    score = float(score) if isinstance(score, (int, float)) else 0.0

    if timed_out:
        status = "timeout"
    elif win:
        status = "win"
    elif state == "GAME_OVER":
        status = "game_over"
    elif manifest.get("error"):
        status = "error"
    elif not manifest:
        status = "setup_failed"
    elif manifest.get("finished_at"):
        status = "incomplete"
    else:
        status = "error"

    failure_parts = []
    if not win:
        if manifest.get("error"):
            failure_parts.append(str(manifest["error"]))
        if manifest.get("termination_reason"):
            failure_parts.append(str(manifest["termination_reason"]))
        if timed_out:
            failure_parts.append("outer_wall_timeout")
        if state and state not in {"WIN", "UNKNOWN"}:
            failure_parts.append(f"state={state}")
        if returncode not in (None, 0):
            failure_parts.append(f"returncode={returncode}")

    return {
        "backend": backend,
        "game_id": game_id,
        "status": status,
        "win": bool(win),
        "completed": bool(completed),
        "state": state,
        "levels_completed": as_int(levels_completed),
        "level_count": as_int(level_count),
        "actions": as_int(actions),
        "resets": as_int(official.get("resets")),
        "score": score,
        "level_actions": official.get("level_actions") or [],
        "level_baseline_actions": official.get("level_baseline_actions") or [],
        "level_scores": official.get("level_scores") or [],
        "tokens": token_usage(backend, run_dir, manifest) if run_dir else {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "reasoning": 0,
            "total": 0,
            "cost_usd": 0.0,
        },
        "duration_seconds": duration_seconds(manifest, wall_seconds),
        "wall_seconds": round(wall_seconds, 3),
        "timed_out": bool(timed_out),
        "returncode": returncode,
        "error": manifest.get("error"),
        "termination_reason": manifest.get("termination_reason"),
        "failure_reason": "; ".join(failure_parts),
        "requested_model": manifest.get("requested_model"),
        "reasoning_effort": manifest.get("reasoning_effort"),
        "scorecard_id": manifest.get("scorecard_id"),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "attempt_dir": str(attempt_dir),
        "finished_at": manifest.get("finished_at"),
    }


def fallback_result(
    *,
    backend: str,
    game_id: str,
    attempt_dir: Path,
    error: str,
    wall_seconds: float = 0.0,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "game_id": game_id,
        "status": "setup_failed",
        "win": False,
        "completed": False,
        "state": "UNKNOWN",
        "levels_completed": 0,
        "level_count": 0,
        "actions": 0,
        "resets": 0,
        "score": 0.0,
        "level_actions": [],
        "level_baseline_actions": [],
        "level_scores": [],
        "tokens": {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "reasoning": 0,
            "total": 0,
            "cost_usd": 0.0,
        },
        "duration_seconds": wall_seconds,
        "wall_seconds": wall_seconds,
        "timed_out": False,
        "returncode": None,
        "error": error,
        "termination_reason": None,
        "failure_reason": error,
        "requested_model": None,
        "reasoning_effort": None,
        "scorecard_id": None,
        "run_dir": None,
        "attempt_dir": str(attempt_dir),
        "finished_at": None,
    }


def skipped_result(
    *,
    backend: str,
    game_id: str,
    attempt_dir: Path,
    reason: str,
) -> dict[str, Any]:
    result = fallback_result(
        backend=backend,
        game_id=game_id,
        attempt_dir=attempt_dir,
        error=reason,
    )
    result["status"] = "skipped"
    result["failure_reason"] = reason
    return result


def stop_process_group(process: subprocess.Popen[str]) -> None:
    for signal_number, wait_seconds in (
        (signal.SIGINT, 30),
        (signal.SIGTERM, 10),
        (signal.SIGKILL, 5),
    ):
        if process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal_number)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_seconds)
            return
        except subprocess.TimeoutExpired:
            continue


def next_attempt_dir(game_dir: Path) -> Path:
    game_dir.mkdir(parents=True, exist_ok=True)
    for index in range(10_000):
        candidate = game_dir / f"attempt_{index:02d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate an attempt directory under {game_dir}")


def build_command(
    *,
    backend: str,
    game_id: str,
    attempt_dir: Path,
    max_steps: int,
    game_timeout: int,
    max_invalid_retries: int,
    effort: str,
) -> list[str]:
    wrapper = BACKEND_WRAPPERS[backend]
    if not wrapper.is_file():
        raise FileNotFoundError(f"Missing backend launcher: {wrapper}")
    if not os.access(wrapper, os.X_OK):
        raise PermissionError(f"Backend launcher is not executable: {wrapper}")
    return [
        str(wrapper),
        "--game-id",
        game_id,
        "--operation-mode",
        "online",
        "--max-steps",
        str(max_steps),
        "--timeout",
        str(game_timeout),
        "--max-invalid-retries",
        str(max_invalid_retries),
        "--effort",
        effort,
        "--runs-dir",
        str(attempt_dir),
    ]


def run_one_game(
    *,
    backend: str,
    game_id: str,
    batch_dir: Path,
    max_steps: int,
    game_timeout: int,
    max_invalid_retries: int,
    effort: str,
) -> dict[str, Any]:
    game_dir = batch_dir / backend / game_id
    attempt_dir = next_attempt_dir(game_dir)
    command = build_command(
        backend=backend,
        game_id=game_id,
        attempt_dir=attempt_dir,
        max_steps=max_steps,
        game_timeout=game_timeout,
        max_invalid_retries=max_invalid_retries,
        effort=effort,
    )
    (attempt_dir / "command.json").write_text(
        json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    stdout_path = attempt_dir / "launcher.stdout.log"
    stderr_path = attempt_dir / "launcher.stderr.log"
    started = time.monotonic()
    timed_out = False
    returncode: int | None = None
    process: subprocess.Popen[str] | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=game_timeout + 300)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_process_group(process)
                returncode = process.poll()
    finally:
        if process is not None and process.poll() is None:
            stop_process_group(process)
            returncode = process.poll()
    wall_seconds = time.monotonic() - started
    run_dir = newest_run_dir(attempt_dir)
    result = collect_result(
        backend=backend,
        game_id=game_id,
        attempt_dir=attempt_dir,
        run_dir=run_dir,
        wall_seconds=wall_seconds,
        timed_out=timed_out,
        returncode=returncode,
    )
    result["command"] = command
    result["launcher_stdout"] = str(stdout_path)
    result["launcher_stderr"] = str(stderr_path)
    return result


def summarize_backend(
    backend: str, results: list[dict[str, Any]], game_count: int
) -> dict[str, Any]:
    rows = [result for result in results if result.get("backend") == backend]
    attempted = [
        result for result in rows if result.get("status") not in {"skipped", "running"}
    ]
    wins = sum(1 for result in attempted if result.get("win"))
    scores = [float(result.get("score") or 0.0) for result in attempted]
    total_levels = sum(as_int(result.get("level_count")) for result in attempted)
    completed_levels = sum(
        as_int(result.get("levels_completed")) for result in attempted
    )
    tokens = {
        "input": sum(as_int((r.get("tokens") or {}).get("input")) for r in attempted),
        "output": sum(as_int((r.get("tokens") or {}).get("output")) for r in attempted),
        "cache_read": sum(
            as_int((r.get("tokens") or {}).get("cache_read")) for r in attempted
        ),
        "cache_write": sum(
            as_int((r.get("tokens") or {}).get("cache_write")) for r in attempted
        ),
        "reasoning": sum(
            as_int((r.get("tokens") or {}).get("reasoning")) for r in attempted
        ),
        "total": sum(as_int((r.get("tokens") or {}).get("total")) for r in attempted),
        "cost_usd": sum(
            float((r.get("tokens") or {}).get("cost_usd") or 0.0)
            for r in attempted
        ),
    }
    duration = sum(float(result.get("duration_seconds") or 0.0) for result in attempted)
    status_counts: dict[str, int] = {}
    for result in attempted:
        status = str(result.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    failure_counts: dict[str, int] = {}
    for result in attempted:
        reason = str(result.get("failure_reason") or "").strip()
        if not reason:
            continue
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    return {
        "backend": backend,
        "label": BACKEND_LABELS[backend],
        "games": game_count,
        "attempted": len(attempted),
        "wins": wins,
        "completion_rate": wins / game_count if game_count else 0.0,
        "rhae": sum(scores) / game_count if game_count else 0.0,
        "level_completion_rate": (
            completed_levels / total_levels if total_levels else 0.0
        ),
        "levels_completed": completed_levels,
        "levels_total": total_levels,
        "actions": sum(as_int(result.get("actions")) for result in attempted),
        "resets": sum(as_int(result.get("resets")) for result in attempted),
        "tokens": tokens,
        "duration_seconds": duration,
        "mean_duration_seconds": duration / len(attempted) if attempted else 0.0,
        "status_counts": status_counts,
        "failure_counts": failure_counts,
    }


def summarize_overall(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    items = list(summaries.values())
    games = sum(int(item["games"]) for item in items)
    attempted = sum(int(item["attempted"]) for item in items)
    wins = sum(int(item["wins"]) for item in items)
    levels_completed = sum(int(item["levels_completed"]) for item in items)
    levels_total = sum(int(item["levels_total"]) for item in items)
    duration = sum(float(item["duration_seconds"]) for item in items)
    tokens = {
        "input": sum(int(item["tokens"]["input"]) for item in items),
        "output": sum(int(item["tokens"]["output"]) for item in items),
        "cache_read": sum(int(item["tokens"]["cache_read"]) for item in items),
        "cache_write": sum(int(item["tokens"]["cache_write"]) for item in items),
        "reasoning": sum(int(item["tokens"]["reasoning"]) for item in items),
        "total": sum(int(item["tokens"]["total"]) for item in items),
        "cost_usd": sum(float(item["tokens"]["cost_usd"]) for item in items),
    }
    status_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    for item in items:
        for status, count in item["status_counts"].items():
            status_counts[status] = status_counts.get(status, 0) + int(count)
        for reason, count in item["failure_counts"].items():
            failure_counts[reason] = failure_counts.get(reason, 0) + int(count)
    return {
        "label": "ALL BACKENDS",
        "games": games,
        "attempted": attempted,
        "wins": wins,
        "completion_rate": wins / games if games else 0.0,
        "rhae": (
            sum(float(item["rhae"]) * int(item["games"]) for item in items) / games
            if games
            else 0.0
        ),
        "level_completion_rate": (
            levels_completed / levels_total if levels_total else 0.0
        ),
        "levels_completed": levels_completed,
        "levels_total": levels_total,
        "actions": sum(int(item["actions"]) for item in items),
        "resets": sum(int(item["resets"]) for item in items),
        "tokens": tokens,
        "duration_seconds": duration,
        "mean_duration_seconds": duration / attempted if attempted else 0.0,
        "status_counts": status_counts,
        "failure_counts": failure_counts,
    }


def format_duration(seconds: float | int) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def format_int(value: int | float) -> str:
    return f"{int(value):,}"


def write_reports(
    batch_dir: Path,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    backend_status: dict[str, Any],
) -> dict[str, Any]:
    game_count = len(config["games"])
    summaries = {
        backend: summarize_backend(backend, results, game_count)
        for backend in config["backends"]
    }
    overall = summarize_overall(summaries)
    summary = {
        "batch_dir": str(batch_dir),
        "generated_at": now_utc(),
        "config": config,
        "backend_status": backend_status,
        "overall": overall,
        "summaries": summaries,
        "results": results,
    }
    write_json(batch_dir / "summary.json", summary)

    csv_fields = [
        "backend",
        "game_id",
        "status",
        "win",
        "state",
        "levels_completed",
        "level_count",
        "actions",
        "resets",
        "score",
        "tokens_input",
        "tokens_output",
        "tokens_cache_read",
        "tokens_cache_write",
        "tokens_reasoning",
        "tokens_total",
        "duration_seconds",
        "failure_reason",
        "error",
        "termination_reason",
        "run_dir",
    ]
    with (batch_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for result in results:
            tokens = result.get("tokens") or {}
            writer.writerow(
                {
                    "backend": result.get("backend"),
                    "game_id": result.get("game_id"),
                    "status": result.get("status"),
                    "win": result.get("win"),
                    "state": result.get("state"),
                    "levels_completed": result.get("levels_completed"),
                    "level_count": result.get("level_count"),
                    "actions": result.get("actions"),
                    "resets": result.get("resets"),
                    "score": result.get("score"),
                    "tokens_input": tokens.get("input", 0),
                    "tokens_output": tokens.get("output", 0),
                    "tokens_cache_read": tokens.get("cache_read", 0),
                    "tokens_cache_write": tokens.get("cache_write", 0),
                    "tokens_reasoning": tokens.get("reasoning", 0),
                    "tokens_total": tokens.get("total", 0),
                    "duration_seconds": result.get("duration_seconds", 0),
                    "failure_reason": result.get("failure_reason"),
                    "error": result.get("error"),
                    "termination_reason": result.get("termination_reason"),
                    "run_dir": result.get("run_dir"),
                }
            )

    lines = [
        "# VISTA full experiment",
        "",
        f"- Batch: `{batch_dir}`",
        f"- Generated: `{summary['generated_at']}`",
        f"- Games per backend: `{game_count}`",
        f"- Game workers: `{config.get('game_workers', 1)} per backend x "
        f"{len(config['backends']) if config.get('parallel_backends') else 1} "
        f"backend(s) = "
        f"{int(config.get('game_workers', 1)) * (len(config['backends']) if config.get('parallel_backends') else 1)} "
        f"concurrent; remaining games are queued`",
        "",
        "Official completion is `state == WIN` from the ARC scorecard. "
        "`GAME_OVER` and `NOT_FINISHED` are not wins; unfinished levels receive "
        "zero RHAE credit. Each game is bounded by the configured action, "
        "time, and failure guards.",
        "",
        f"## Overall ({overall['games']} games)",
        "",
        "| Games | Wins | Completion | RHAE | Levels | Actions | Tokens | Duration |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {games} | {wins}/{games} | {completion:.1%} | {rhae:.2f} | "
        "{levels}/{levels_total} | {actions:,} | {tokens:,} | {duration} |".format(
            games=overall["games"],
            wins=overall["wins"],
            completion=overall["completion_rate"],
            rhae=overall["rhae"],
            levels=overall["levels_completed"],
            levels_total=overall["levels_total"],
            actions=overall["actions"],
            tokens=overall["tokens"]["total"],
            duration=format_duration(overall["duration_seconds"]),
        ),
        "",
        "## Backend summary",
        "",
        "| Backend | Wins | Completion | RHAE | Levels | Actions | Tokens | Duration |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend in config["backends"]:
        item = summaries[backend]
        lines.append(
            "| {label} | {wins}/{games} | {completion:.1%} | {rhae:.2f} | "
            "{levels}/{levels_total} | {actions:,} | {tokens:,} | {duration} |".format(
                label=item["label"],
                wins=item["wins"],
                games=item["games"],
                completion=item["completion_rate"],
                rhae=item["rhae"],
                levels=item["levels_completed"],
                levels_total=item["levels_total"],
                actions=item["actions"],
                tokens=item["tokens"]["total"],
                duration=format_duration(item["duration_seconds"]),
            )
        )
    lines.extend(
        [
            "",
            "## Token breakdown",
            "",
            "| Backend | Input | Output | Cache read | Cache write | Reasoning | Total | Cost USD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for backend in config["backends"]:
        tokens = summaries[backend]["tokens"]
        lines.append(
            "| {label} | {input:,} | {output:,} | {cache_read:,} | "
            "{cache_write:,} | {reasoning:,} | {total:,} | {cost:.4f} |".format(
                label=BACKEND_LABELS[backend],
                input=tokens["input"],
                output=tokens["output"],
                cache_read=tokens["cache_read"],
                cache_write=tokens["cache_write"],
                reasoning=tokens["reasoning"],
                total=tokens["total"],
                cost=tokens["cost_usd"],
            )
        )
    tokens = overall["tokens"]
    lines.append(
        "| **ALL BACKENDS** | {input:,} | {output:,} | {cache_read:,} | "
        "{cache_write:,} | {reasoning:,} | {total:,} | {cost:.4f} |".format(
            input=tokens["input"],
            output=tokens["output"],
            cache_read=tokens["cache_read"],
            cache_write=tokens["cache_write"],
            reasoning=tokens["reasoning"],
            total=tokens["total"],
            cost=tokens["cost_usd"],
        )
    )
    lines.extend(
        [
            "",
            "## Per-game results",
            "",
            "| Backend | Game | Status | State | Levels | Actions | Score | Tokens | Duration | Failure |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for result in results:
        tokens = result.get("tokens") or {}
        lines.append(
            "| {backend} | {game} | {status} | {state} | {levels}/{total} | "
            "{actions} | {score:.2f} | {tokens:,} | {duration} | {failure} |".format(
                backend=result.get("backend"),
                game=result.get("game_id"),
                status=result.get("status"),
                state=result.get("state"),
                levels=result.get("levels_completed", 0),
                total=result.get("level_count", 0),
                actions=result.get("actions", 0),
                score=float(result.get("score") or 0.0),
                tokens=tokens.get("total", 0),
                duration=format_duration(result.get("duration_seconds") or 0),
                failure=(result.get("failure_reason") or "").replace("|", "\\|"),
            )
        )
    lines.append("")
    (batch_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def should_skip(existing: Any, *, retry_failed: bool) -> bool:
    if not isinstance(existing, dict):
        return False
    status = existing.get("status")
    if status in (None, "running"):
        return False
    if existing.get("win"):
        return True
    if retry_failed:
        return False
    return status in FINAL_STATUSES


def check_stop(
    *,
    stop_file: Path,
    wall_deadline: float | None,
) -> str | None:
    if stop_file.exists():
        return "stop_file"
    if wall_deadline is not None and time.monotonic() >= wall_deadline:
        return "wall_budget_exhausted"
    return None


def run_backend(
    *,
    backend: str,
    games: list[str],
    config: dict[str, Any],
    state: dict[str, Any],
    state_lock: threading.Lock,
    batch_dir: Path,
    wall_deadline: float | None,
) -> dict[str, Any]:
    backend_status: dict[str, Any] = {
        "state": "running",
        "started_at": now_utc(),
        "finished_at": None,
        "reason": None,
        "consecutive_failures": 0,
        "tokens_total": 0,
    }
    positions = {game_id: index for index, game_id in enumerate(games, start=1)}
    pending: list[str] = []
    for game_id in games:
        key = f"{backend}/{game_id}"
        with state_lock:
            existing = state["results"].get(key)
        if should_skip(existing, retry_failed=bool(config["retry_failed"])):
            print(
                f"[{backend}] {positions[game_id]}/{len(games)} {game_id}: skip "
                f"({existing.get('status')})",
                flush=True,
            )
            continue
        pending.append(game_id)
    if not pending:
        backend_status["state"] = "finished"
        backend_status["finished_at"] = now_utc()
        return backend_status

    stop_event = threading.Event()
    counters = {
        "tokens": 0,
        "consecutive_failures": 0,
        "reason": None,
    }
    counters_lock = threading.Lock()
    workers = max(1, min(int(config["game_workers"]), len(pending)))

    def record_skip(game_id: str, reason: str) -> dict[str, Any]:
        key = f"{backend}/{game_id}"
        result = skipped_result(
            backend=backend,
            game_id=game_id,
            attempt_dir=batch_dir / backend / game_id,
            reason=reason,
        )
        with state_lock:
            state["results"][key] = result
            state["updated_at"] = now_utc()
            write_json(batch_dir / "batch.json", state)
        print(
            f"[{backend}] {positions[game_id]}/{len(games)} {game_id}: skipped "
            f"({reason})",
            flush=True,
        )
        return result

    def task(game_id: str) -> dict[str, Any]:
        if stop_event.is_set():
            with counters_lock:
                reason = counters["reason"] or "backend_stopped"
            return record_skip(game_id, reason)
        stop_reason = check_stop(
            stop_file=Path(config["stop_file"]), wall_deadline=wall_deadline
        )
        if stop_reason is not None:
            with counters_lock:
                counters["reason"] = stop_reason
            stop_event.set()
            return record_skip(game_id, stop_reason)
        with counters_lock:
            token_total = counters["tokens"]
        if (
            config["max_total_tokens_per_backend"] > 0
            and token_total >= config["max_total_tokens_per_backend"]
        ):
            with counters_lock:
                counters["reason"] = "token_budget_exhausted"
            stop_event.set()
            return record_skip(game_id, "token_budget_exhausted")

        key = f"{backend}/{game_id}"
        print(
            f"[{backend}] {positions[game_id]}/{len(games)} {game_id}: start "
            f"(tokens so far {token_total:,}, workers={workers})",
            flush=True,
        )
        with state_lock:
            state["results"][key] = {
                "backend": backend,
                "game_id": game_id,
                "status": "running",
                "started_at": now_utc(),
            }
            write_json(batch_dir / "batch.json", state)
        try:
            result = run_one_game(
                backend=backend,
                game_id=game_id,
                batch_dir=batch_dir,
                max_steps=int(config["max_steps"]),
                game_timeout=int(config["game_timeout"]),
                max_invalid_retries=int(config["max_invalid_retries"]),
                effort=str(config["effort"]),
            )
        except BaseException as exc:
            result = fallback_result(
                backend=backend,
                game_id=game_id,
                attempt_dir=batch_dir / backend / game_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        with state_lock:
            state["results"][key] = result
            state["updated_at"] = now_utc()
            write_json(batch_dir / "batch.json", state)
        with counters_lock:
            counters["tokens"] += as_int((result.get("tokens") or {}).get("total"))
            if result["status"] in INFRA_FAILURES:
                counters["consecutive_failures"] += 1
            else:
                counters["consecutive_failures"] = 0
            token_total = counters["tokens"]
            consecutive_failures = counters["consecutive_failures"]
            if (
                config["max_total_tokens_per_backend"] > 0
                and token_total >= config["max_total_tokens_per_backend"]
            ):
                counters["reason"] = "token_budget_exhausted"
                stop_event.set()
            if consecutive_failures >= int(config["max_consecutive_failures"]):
                counters["reason"] = "consecutive_infrastructure_failures"
                stop_event.set()
        print(
            f"[{backend}] {positions[game_id]}/{len(games)} {game_id}: "
            f"{result['status']} levels={result['levels_completed']}/"
            f"{result['level_count']} score={result['score']:.2f} "
            f"actions={result['actions']} tokens={token_total:,} "
            f"duration={format_duration(result['duration_seconds'])}",
            flush=True,
        )
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(task, game_id): game_id for game_id in pending}
        for future in as_completed(futures):
            game_id = futures[future]
            try:
                future.result()
            except BaseException as exc:
                result = fallback_result(
                    backend=backend,
                    game_id=game_id,
                    attempt_dir=batch_dir / backend / game_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                with state_lock:
                    state["results"][f"{backend}/{game_id}"] = result
                    state["updated_at"] = now_utc()
                    write_json(batch_dir / "batch.json", state)
                with counters_lock:
                    counters["consecutive_failures"] += 1
                    if counters["consecutive_failures"] >= int(
                        config["max_consecutive_failures"]
                    ):
                        counters["reason"] = "consecutive_infrastructure_failures"
                        stop_event.set()
                print(
                    f"[{backend}] {positions[game_id]}/{len(games)} {game_id}: "
                    f"setup_failed ({type(exc).__name__}: {exc})",
                    flush=True,
                )

    with counters_lock:
        token_total = counters["tokens"]
        consecutive_failures = counters["consecutive_failures"]
        reason = counters["reason"]
    backend_status["consecutive_failures"] = consecutive_failures
    backend_status["tokens_total"] = token_total
    backend_status["state"] = "stopped" if stop_event.is_set() else "finished"
    backend_status["reason"] = reason
    backend_status["finished_at"] = now_utc()
    return backend_status


def serialize_config(args: argparse.Namespace, batch_dir: Path) -> dict[str, Any]:
    return {
        "batch_dir": str(batch_dir),
        "backends": list(args.backends),
        "games": list(args.games),
        "max_steps": args.max_steps,
        "game_timeout": args.game_timeout,
        "max_invalid_retries": args.max_invalid_retries,
        "effort": args.effort,
        "game_workers": args.game_workers,
        "max_consecutive_failures": args.max_consecutive_failures,
        "max_total_tokens_per_backend": args.max_total_tokens_per_backend,
        "max_wall_hours": args.max_wall_hours,
        "parallel_backends": args.parallel_backends,
        "retry_failed": args.retry_failed,
        "stop_file": str(args.stop_file or batch_dir / "STOP"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing batch directory; skip recorded final results",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="With --resume, rerun non-WIN results",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only rebuild summary files from batch.json",
    )
    parser.add_argument(
        "--backends",
        default="codex,claude,dsh",
        help="Comma-separated: codex,claude,dsh",
    )
    parser.add_argument(
        "--games",
        default=",".join(STANDARD_GAME_ORDER),
        help="Comma-separated game IDs (default: all 25 public games)",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--game-timeout", type=int, default=DEFAULT_GAME_TIMEOUT)
    parser.add_argument(
        "--max-invalid-retries", type=int, default=DEFAULT_MAX_INVALID_RETRIES
    )
    parser.add_argument(
        "--game-workers",
        type=int,
        default=2,
        help="Concurrent games per backend (default 2; 25 gives 75 with 3 backends)",
    )
    parser.add_argument("--effort", default="max")
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
    )
    parser.add_argument(
        "--max-total-tokens-per-backend",
        type=int,
        default=0,
        help="0 disables the token budget; otherwise stop a backend after this many tokens",
    )
    parser.add_argument(
        "--max-wall-hours",
        type=float,
        default=DEFAULT_MAX_WALL_HOURS,
        help="Global wall-clock budget; 0 disables it",
    )
    parser.add_argument(
        "--parallel-backends",
        action="store_true",
        help="Run codex/claude/dsh loops concurrently",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="Create this file to stop after the current game",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    backends = tuple(part.strip() for part in args.backends.split(",") if part.strip())
    games = tuple(part.strip() for part in args.games.split(",") if part.strip())
    unknown_backends = [backend for backend in backends if backend not in BACKEND_WRAPPERS]
    if unknown_backends:
        parser.error(f"unknown backends: {', '.join(unknown_backends)}")
    unknown_games = [game for game in games if game not in STANDARD_GAME_ORDER]
    if unknown_games:
        parser.error(f"unknown games: {', '.join(unknown_games)}")
    args.backends = backends
    args.games = games
    if args.max_steps < 0:
        parser.error("--max-steps must be >= 0")
    if args.game_timeout < 1:
        parser.error("--game-timeout must be positive")
    if args.game_workers < 1:
        parser.error("--game-workers must be positive")
    if args.max_consecutive_failures < 1:
        parser.error("--max-consecutive-failures must be positive")
    if args.max_total_tokens_per_backend < 0:
        parser.error("--max-total-tokens-per-backend must be >= 0")
    if args.max_wall_hours < 0:
        parser.error("--max-wall-hours must be >= 0")

    if args.batch_dir is None:
        batch_dir = ROOT / "runs" / f"full_experiment_{utc_stamp()}"
    else:
        batch_dir = args.batch_dir.expanduser().resolve()
    if args.resume or args.report_only:
        if not (batch_dir / "batch.json").is_file():
            parser.error(f"no batch.json under {batch_dir}")
    else:
        if batch_dir.exists() and any(batch_dir.iterdir()):
            parser.error(
                f"batch directory is not empty: {batch_dir}; use --resume or a new path"
            )
        batch_dir.mkdir(parents=True, exist_ok=True)

    config = serialize_config(args, batch_dir)
    if args.stop_file is None:
        args.stop_file = batch_dir / "STOP"
    config["stop_file"] = str(args.stop_file)

    if args.report_only:
        state = load_json(batch_dir / "batch.json", {})
        results = list(state.get("results", {}).values())
        results.sort(key=lambda item: (item.get("backend", ""), item.get("game_id", "")))
        summary = write_reports(
            batch_dir,
            state.get("config", config),
            results,
            state.get("backend_status", {}),
        )
        print(f"Rebuilt reports under {batch_dir}")
        print(
            json.dumps(
                {
                    "overall": summary["overall"],
                    "backends": summary["summaries"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.dry_run:
        for backend in backends:
            for game_id in games:
                attempt_dir = batch_dir / backend / game_id / "attempt_00"
                command = build_command(
                    backend=backend,
                    game_id=game_id,
                    attempt_dir=attempt_dir,
                    max_steps=args.max_steps,
                    game_timeout=args.game_timeout,
                    max_invalid_retries=args.max_invalid_retries,
                    effort=args.effort,
                )
                print(" ".join(command))
        return 0

    if args.resume:
        state = load_json(batch_dir / "batch.json", {})
        if not isinstance(state, dict):
            parser.error(f"batch.json is not a mapping: {batch_dir / 'batch.json'}")
        state["config"] = config
        state["updated_at"] = now_utc()
    else:
        state = {
            "created_at": now_utc(),
            "updated_at": now_utc(),
            "config": config,
            "results": {},
            "backend_status": {},
        }
        write_json(batch_dir / "batch.json", state)

    wall_deadline = (
        time.monotonic() + args.max_wall_hours * 3600
        if args.max_wall_hours > 0
        else None
    )
    total_workers = args.game_workers * (
        len(backends) if args.parallel_backends else 1
    )
    print(
        f"Batch: {batch_dir}\n"
        f"Backends: {', '.join(backends)}\n"
        f"Games: {len(games)} per backend; total {len(games) * len(backends)}\n"
        f"Concurrency: {args.game_workers} per backend, "
        f"{total_workers} total game workers\n"
        f"Guards: max_steps={args.max_steps}, game_timeout={args.game_timeout}s, "
        f"invalid_retries={args.max_invalid_retries}, "
        f"consecutive_failures={args.max_consecutive_failures}, "
        f"token_budget={args.max_total_tokens_per_backend}, "
        f"wall_hours={args.max_wall_hours}",
        flush=True,
    )
    if total_workers > 12:
        docker_note = ""
        try:
            probe = subprocess.run(
                ["docker", "info", "--format", "{{.MemTotal}} {{.NCPU}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if probe.returncode == 0:
                memory_bytes, cpus = probe.stdout.split()
                docker_note = (
                    f" Docker VM: {int(memory_bytes) / 1024**3:.1f} GiB / "
                    f"{cpus} CPUs."
                )
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        print(
            "WARNING: high local concurrency. Each Claude/Codex game starts a "
            "linux/amd64 container requesting 4 CPUs and up to 16 GiB. "
            f"{total_workers} concurrent games can exhaust the local Docker VM "
            f"and cause OOM/timeouts.{docker_note}",
            flush=True,
        )
    state_lock = threading.Lock()

    if args.parallel_backends:
        with ThreadPoolExecutor(max_workers=len(backends)) as executor:
            futures = {
                executor.submit(
                    run_backend,
                    backend=backend,
                    games=list(games),
                    config=config,
                    state=state,
                    state_lock=state_lock,
                    batch_dir=batch_dir,
                    wall_deadline=wall_deadline,
                ): backend
                for backend in backends
            }
            for future in as_completed(futures):
                backend = futures[future]
                try:
                    status = future.result()
                except BaseException as exc:
                    status = {
                        "state": "crashed",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "finished_at": now_utc(),
                    }
                with state_lock:
                    state["backend_status"][backend] = status
                    state["updated_at"] = now_utc()
                    write_json(batch_dir / "batch.json", state)
    else:
        for backend in backends:
            status = run_backend(
                backend=backend,
                games=list(games),
                config=config,
                state=state,
                state_lock=state_lock,
                batch_dir=batch_dir,
                wall_deadline=wall_deadline,
            )
            with state_lock:
                state["backend_status"][backend] = status
                state["updated_at"] = now_utc()
                write_json(batch_dir / "batch.json", state)

    results = list(state.get("results", {}).values())
    results.sort(key=lambda item: (item.get("backend", ""), item.get("game_id", "")))
    summary = write_reports(
        batch_dir,
        config,
        results,
        state.get("backend_status", {}),
    )
    print(f"\nReports written under {batch_dir}")
    for backend in backends:
        item = summary["summaries"][backend]
        print(
            f"{item['label']}: wins={item['wins']}/{item['games']} "
            f"RHAE={item['rhae']:.2f} tokens={item['tokens']['total']:,} "
            f"duration={format_duration(item['duration_seconds'])}"
        )
    overall = summary["overall"]
    print(
        f"ALL BACKENDS: wins={overall['wins']}/{overall['games']} "
        f"completion={overall['completion_rate']:.1%} "
        f"RHAE={overall['rhae']:.2f} "
        f"levels={overall['levels_completed']}/{overall['levels_total']} "
        f"tokens={overall['tokens']['total']:,} "
        f"duration={format_duration(overall['duration_seconds'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
