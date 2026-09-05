"""Runtime-neutral run storage and ARC scorecard helpers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAX_TASK_TIMEOUT_SECONDS = 2_000_000


@dataclass(frozen=True)
class GameRunOutcome:
    run_dir: Path
    scorecard_id: str | None
    session_id: str | None
    failure: BaseException | None


def bounded_task_timeout(tool_timeout: int, maximum_calls: int) -> int:
    return min(tool_timeout * maximum_calls, MAX_TASK_TIMEOUT_SECONDS)


def open_evaluation_scorecard(arc: Any) -> str:
    return str(arc.open_scorecard(tags=[]))


def share_online_cookie_jar(arc: Any, env: Any | None) -> bool:
    """Keep scorecard and environment requests on the same remote session."""
    arc_session = getattr(arc, "_session", None)
    shared = getattr(arc_session, "cookies", None)
    if shared is None:
        return False

    env_session = getattr(env, "_session", None)
    lock = getattr(arc, "_cookie_lock", None)
    context = lock if lock is not None else nullcontext()
    with context:
        old_master = getattr(arc, "_master_cookie_jar", None)
        if old_master is not None and old_master is not shared:
            shared.update(old_master)
        if env_session is not None:
            shared.update(env_session.cookies)

        arc._master_cookie_jar = shared
        if env is not None and hasattr(env, "_master_cookie_jar"):
            env._master_cookie_jar = shared
    return True


def finalize_scorecard(
    *,
    arc: Any,
    env: Any | None,
    scorecard_id: str,
    operation_mode: str,
) -> tuple[dict[str, Any], BaseException | None]:
    """Close a remote scorecard without losing a run to a stale cookie."""
    outcome: dict[str, Any] = {"scorecard_closed": False}
    snapshot: Any | None = None

    share_online_cookie_jar(arc, env)
    if operation_mode == "online":
        try:
            snapshot = arc.get_scorecard(scorecard_id)
        except BaseException as exc:
            outcome["scorecard_preclose_error"] = f"{type(exc).__name__}: {exc}"
        else:
            if snapshot is not None:
                outcome["scorecard_preclose_summary"] = safe_model_dump(snapshot)

    share_online_cookie_jar(arc, env)
    try:
        final_scorecard = arc.close_scorecard(scorecard_id)
    except BaseException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        outcome["scorecard_close_error"] = f"{type(exc).__name__}: {exc}"
        outcome["scorecard_close_status_code"] = status_code
        if status_code == 404 and snapshot is not None:
            outcome["scorecard_summary"] = safe_model_dump(snapshot)
            outcome["scorecard_summary_source"] = "preclose_snapshot"
            outcome["scorecard_finalization"] = "close_not_found_after_snapshot"
            return outcome, None
        return outcome, exc

    outcome["scorecard_closed"] = True
    outcome["scorecard_finalization"] = "closed"
    if final_scorecard is not None:
        outcome["scorecard_summary"] = safe_model_dump(final_scorecard)
        outcome["scorecard_summary_source"] = "close_response"
    elif snapshot is not None:
        outcome["scorecard_summary"] = safe_model_dump(snapshot)
        outcome["scorecard_summary_source"] = "preclose_snapshot"
    return outcome, None


def create_run_dir(runs_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    counter = 0
    while True:
        suffix = "" if counter == 0 else f"_{counter}"
        run_dir = runs_dir / f"run_{stamp}{suffix}"
        try:
            run_dir.mkdir(parents=True)
        except FileExistsError:
            counter += 1
            continue
        return run_dir


def safe_model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [safe_model_dump(item) for item in value]
    if isinstance(value, dict):
        return {str(key): safe_model_dump(item) for key, item in value.items()}
    return str(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
