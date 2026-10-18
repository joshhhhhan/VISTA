"""Private game controller for the sanitized player tools."""

from __future__ import annotations

import base64
import io
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from arcengine import GameAction, GameState
from PIL import Image

from vista.benchmarks.arc3.claude.tools import (
    ACTION_NAMES,
    MAX_GUIDE_CHARS,
    MAX_INSPECTION_LABEL_CHARS,
    MAX_INSPECTION_QUESTION_CHARS,
    MAX_INSPECTION_VIEWS,
    MAX_PIXEL_READOUT_SAMPLES,
    MAX_PIXEL_READOUT_VIEWS,
    MAX_WORKING_CHARS,
)
from vista.benchmarks.arc3.render import render_frame_image, save_frame_png
from vista.core.session import (
    CompactRecovery,
    RetryRecovery,
    RuntimeRecovery,
    SessionController,
)

DISPLAY_SIZE = 1024
GRID_SIZE = 64
ACTION_BY_NAME = {
    "RESET": GameAction.RESET,
    "ACTION1": GameAction.ACTION1,
    "ACTION2": GameAction.ACTION2,
    "ACTION3": GameAction.ACTION3,
    "ACTION4": GameAction.ACTION4,
    "ACTION5": GameAction.ACTION5,
    "ACTION6": GameAction.ACTION6,
    "ACTION7": GameAction.ACTION7,
}
NAME_BY_ACTION = {action: name for name, action in ACTION_BY_NAME.items()}
PIXEL_READOUT_SYMBOLS = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
)


@dataclass(frozen=True)
class ToolActionValidation:
    ok: bool
    action: GameAction | None = None
    data: dict[str, int] | None = None
    parsed: dict[str, Any] | None = None
    retry_state: str | None = None
    error: str | None = None


def available_action_names(actions: Iterable[GameAction]) -> list[str]:
    available = {
        NAME_BY_ACTION[action] for action in actions if action in NAME_BY_ACTION
    }
    return [name for name in ACTION_NAMES if name in available]


def validate_tool_action(
    arguments: object,
    legal_actions: Iterable[GameAction],
    display_size: int = DISPLAY_SIZE,
    *,
    reset_requires_retry_state: bool = False,
) -> ToolActionValidation:
    if type(display_size) is not int or display_size < 1:
        raise ValueError("display_size must be a positive integer")
    if not isinstance(arguments, dict):
        return ToolActionValidation(ok=False, error="Tool arguments must be an object.")
    allowed_arguments = {"action", "x", "y"}
    if reset_requires_retry_state:
        allowed_arguments.add("retry_state")
    if set(arguments) - allowed_arguments:
        return ToolActionValidation(ok=False, error="Unexpected tool argument.")
    action_name = arguments.get("action")
    if not isinstance(action_name, str) or action_name not in ACTION_BY_NAME:
        return ToolActionValidation(ok=False, error="Unknown action.")
    action = ACTION_BY_NAME[action_name]
    legal = set(legal_actions)
    if action not in legal:
        return ToolActionValidation(
            ok=False,
            error=f"{action_name} is not currently available.",
        )

    x = arguments.get("x")
    y = arguments.get("y")
    retry_state = arguments.get("retry_state")
    if action == GameAction.RESET:
        if x is not None or y is not None:
            return ToolActionValidation(
                ok=False,
                error="x and y are only used with ACTION6.",
            )
        if reset_requires_retry_state:
            if (
                not isinstance(retry_state, str)
                or not retry_state.strip()
                or len(retry_state) > MAX_WORKING_CHARS
            ):
                return ToolActionValidation(
                    ok=False,
                    error=(
                        "RESET requires retry_state containing the continuation "
                        "state for the next attempt."
                    ),
                )
            retry_state = retry_state.strip()
        return ToolActionValidation(
            ok=True,
            action=action,
            data=None,
            parsed={
                "action": action_name,
                "x": None,
                "y": None,
            },
            retry_state=retry_state,
        )

    if "retry_state" in arguments:
        return ToolActionValidation(
            ok=False,
            error="retry_state is only used with RESET.",
        )

    if action == GameAction.ACTION6:
        if type(x) is not int or type(y) is not int:
            return ToolActionValidation(
                ok=False, error="ACTION6 requires integer x and y."
            )
        if not (0 <= x < display_size and 0 <= y < display_size):
            return ToolActionValidation(
                ok=False,
                error=f"ACTION6 coordinates must be in 0..{display_size - 1}.",
            )
        environment_x = min(GRID_SIZE - 1, x * GRID_SIZE // display_size)
        environment_y = min(GRID_SIZE - 1, y * GRID_SIZE // display_size)
        return ToolActionValidation(
            ok=True,
            action=action,
            data={"x": environment_x, "y": environment_y},
            parsed={
                "action": action_name,
                "x": x,
                "y": y,
                "environment_x": environment_x,
                "environment_y": environment_y,
            },
        )

    if x is not None or y is not None:
        return ToolActionValidation(
            ok=False,
            error="x and y are only used with ACTION6.",
        )
    return ToolActionValidation(
        ok=True,
        action=action,
        data=None,
        parsed={
            "action": action_name,
            "x": None,
            "y": None,
        },
    )


class GameController(SessionController):
    def _atomic_write_text(self, path: Path, content: str, *, mode: int) -> None:
        atomic_write_text(path, content, mode=mode)

    def __init__(
        self,
        *,
        env: Any,
        initial_observation: Any,
        private_dir: Path,
        frames_dir: Path,
        max_steps: int,
        max_invalid_retries: int,
        observation_mode: str,
        render_scale: int,
        render_grid: bool = False,
        compact_restore_marker: Path | None = None,
        compact_guide_snapshot: Path | None = None,
        compact_checkpoint_marker: Path | None = None,
        compact_checkpoint_ready: Path | None = None,
        retry_boundary_marker: Path | None = None,
        reset_starts_fresh_session: bool = False,
        working_path: Path | None = None,
        guide_path: Path | None = None,
        agent_name: str = "claude-code",
        interface_name: str = "native-mcp-tool-loop",
    ) -> None:
        self.env = env
        self.obs = initial_observation
        self.private_dir = private_dir
        self.frames_dir = frames_dir
        self.max_steps = max_steps
        self.max_invalid_retries = max_invalid_retries
        self.observation_mode = observation_mode
        self.render_scale = render_scale
        if type(render_scale) is not int or render_scale < 1:
            raise ValueError("render_scale must be a positive integer")
        self.display_size = GRID_SIZE * render_scale
        self.render_grid = render_grid
        self.compact_restore_marker = compact_restore_marker
        self.compact_guide_snapshot = compact_guide_snapshot
        self.compact_checkpoint_marker = compact_checkpoint_marker
        self.compact_checkpoint_ready = compact_checkpoint_ready
        self.retry_boundary_marker = retry_boundary_marker
        self.reset_starts_fresh_session = reset_starts_fresh_session
        self.working_path = working_path
        self.guide_path = guide_path
        self.agent_name = agent_name
        self.interface_name = interface_name
        self.working_provenance_path = private_dir / "working_memory_provenance.json"
        self.working_archive_dir = private_dir / "working_memory_archive"
        self.retry_state_staging_path = private_dir / "retry_state.pending"
        self.action_log_path = private_dir / "action_log.jsonl"
        self.error_log_path = private_dir / "controller_errors.jsonl"
        self.recovery_log_path = private_dir / "compact_recovery.jsonl"
        self.step_index = 0
        self.attempt_index = 0
        self.total_attempts = 0
        self._events: list[dict[str, Any]] = []
        self._pixel_readout_symbol_by_color: dict[tuple[int, int, int], str] = {}
        self._level_start_turn = 0
        self._retry_boundary_step: int | None = None
        self.forced_termination_reason: str | None = None
        self._current_images: list[Path] = []
        self._lock = threading.Lock()
        self._store_observation()

    @property
    def terminal(self) -> bool:
        return (
            self.forced_termination_reason is not None
            or self.obs is None
            or self.obs.state == GameState.WIN
        )

    @property
    def termination_reason(self) -> str | None:
        if self.forced_termination_reason:
            return self.forced_termination_reason
        if self.obs is None:
            return "environment_ended"
        if self.obs.state == GameState.WIN:
            return state_name(self.obs.state)
        return None

    def _available_actions(self) -> list[GameAction]:
        if self.obs is not None and self.obs.state == GameState.GAME_OVER:
            return [GameAction.RESET]
        return list(self.env.action_space)

    @property
    def retry_boundary_pending(self) -> bool:
        return self._retry_boundary_step is not None

    @property
    def retry_boundary_step(self) -> int | None:
        return self._retry_boundary_step

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            method = request.get("method")
            if self.retry_boundary_pending:
                return self._response(
                    ok=False,
                    action_applied=False,
                    error="Retry recovery has not been delivered yet.",
                    include_image=False,
                )
            if method == "inspect":
                return self._inspect(request.get("arguments"))
            if method == "read_pixels":
                return self._read_pixels(request.get("arguments"))
            if method == "history":
                return self._history(request.get("arguments"))
            if method == "save_compact_checkpoint":
                return self._save_compact_checkpoint(request.get("arguments"))
            if method != "play":
                return self._response(
                    ok=False,
                    action_applied=False,
                    error="Unknown controller method.",
                    include_image=False,
                )
            if self.terminal:
                return self._response(
                    ok=True,
                    action_applied=False,
                    include_image=False,
                )
            if (
                self.compact_restore_marker is not None
                and self.compact_restore_marker.exists()
            ):
                return self._response(
                    ok=False,
                    action_applied=False,
                    error="Compact recovery has not been delivered yet.",
                    include_image=False,
                )

            if (
                self.compact_checkpoint_marker is not None
                and self.compact_checkpoint_marker.exists()
            ):
                return self._response(
                    ok=False,
                    action_applied=False,
                    error="Save the compact checkpoint before playing.",
                    include_image=False,
                )

            arguments = request.get("arguments")
            retry = self.attempt_index
            self.total_attempts += 1
            validation = validate_tool_action(
                arguments,
                self._available_actions(),
                display_size=self.display_size,
                reset_requires_retry_state=self.reset_starts_fresh_session,
            )
            log_entry: dict[str, Any] = {
                "protocol": self.interface_name,
                "step": self.step_index,
                "retry": retry,
                "tool_call_id": request.get("request_id"),
                "requested": arguments,
                "validation": validation_to_log(validation),
            }
            if not validation.ok:
                self.attempt_index += 1
                if retry >= self.max_invalid_retries:
                    self.attempt_index = 0
                    log_entry["recovery_boundary"] = "invalid_action_limit"
                    append_jsonl(self.action_log_path, log_entry)
                    return self._response(
                        ok=False,
                        action_applied=False,
                        error=validation.error,
                        include_image=False,
                        invalid_action_limit=True,
                    )
                append_jsonl(self.action_log_path, log_entry)
                return self._response(
                    ok=False,
                    action_applied=False,
                    error=validation.error,
                    include_image=False,
                )

            assert validation.action is not None
            assert validation.parsed is not None
            current_step = self.step_index
            previous_levels_completed = getattr(self.obs, "levels_completed", None)
            try:
                if validation.action == GameAction.RESET:
                    retry_provenance = (
                        self._stage_retry_state(validation.retry_state)
                        if validation.retry_state is not None
                        else None
                    )
                    next_observation = self.env.reset()
                else:
                    retry_provenance = None
                    next_observation = self.env.step(
                        validation.action,
                        data=validation.data,
                    )
                if next_observation is None:
                    self.forced_termination_reason = "environment_response_missing"
                    raise RuntimeError(
                        "The environment returned no observation; remote state is unknown."
                    )
                self.obs = next_observation
                self.step_index += 1
                if retry_provenance is not None:
                    self._commit_retry_state(retry_provenance)
                self.attempt_index = 0
                if self.obs is not None:
                    self._store_observation()
                else:
                    self._current_images = []
                if (
                    self.max_steps > 0
                    and self.step_index >= self.max_steps
                    and not self.terminal
                ):
                    self.forced_termination_reason = "action_budget_exhausted"
                level_boundary = self._crossed_level_boundary(previous_levels_completed)
                if level_boundary:
                    self._level_start_turn = self.step_index
                    self._archive_level_working(previous_levels_completed)
                reset_applied = validation.action == GameAction.RESET
                retry_boundary = reset_applied and self.reset_starts_fresh_session
                if retry_boundary:
                    self._mark_retry_boundary()
                    if not self.terminal:
                        self._retry_boundary_step = self.step_index
                result_metadata = self._metadata(
                    action_applied=True,
                    level_boundary=level_boundary,
                    retry_boundary=retry_boundary,
                )
                log_entry["result"] = result_metadata
                self._events.append(
                    self._public_event(
                        step=current_step,
                        parsed_action=validation.parsed,
                        result=result_metadata,
                    )
                )
                append_jsonl(self.action_log_path, log_entry)
            except Exception as exc:
                self.retry_state_staging_path.unlink(missing_ok=True)
                error = {
                    "step": current_step,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                append_jsonl(self.error_log_path, error)
                log_entry["controller_error"] = error
                append_jsonl(self.action_log_path, log_entry)
                if self.forced_termination_reason is None:
                    self.forced_termination_reason = "controller_error"
                raise
            return self._response(
                ok=True,
                action_applied=True,
                include_image=True,
                level_boundary=level_boundary,
                retry_boundary=retry_boundary,
            )

    def _crossed_level_boundary(self, previous_completed: object) -> bool:
        if self.terminal or self.obs is None:
            return False
        current_completed = getattr(self.obs, "levels_completed", None)
        return (
            type(previous_completed) is int
            and type(current_completed) is int
            and current_completed > previous_completed
        )

    def _archive_level_working(self, previous_completed: object) -> None:
        if self.working_path is None or not self.working_path.is_file():
            if self.working_provenance_path.exists():
                self.working_provenance_path.unlink()
            return
        working = self.working_path.read_text(encoding="utf-8", errors="replace")
        if not working.strip():
            self._clear_working()
            return
        provenance = self._read_working_provenance()
        current_completed = getattr(self.obs, "levels_completed", None)
        total = getattr(self.obs, "win_levels", None)
        archive = {
            "boundary_turn": self.step_index,
            "saved_at": provenance,
            "previous_progress": {
                "completed": (
                    previous_completed if type(previous_completed) is int else None
                ),
                "total": total if type(total) is int else None,
            },
            "current_progress": {
                "completed": (
                    current_completed if type(current_completed) is int else None
                ),
                "total": total if type(total) is int else None,
            },
        }
        self.working_archive_dir.mkdir(parents=True, exist_ok=True)
        archive_stem = f"turn_{self.step_index:06d}"
        atomic_write_text(
            self.working_archive_dir / f"{archive_stem}.md",
            working,
            mode=0o600,
        )
        atomic_write_text(
            self.working_archive_dir / f"{archive_stem}.json",
            json.dumps(archive, separators=(",", ":"), sort_keys=True) + "\n",
            mode=0o600,
        )
        append_jsonl(
            self.recovery_log_path,
            {
                "step": self.step_index,
                "action_applied": False,
                "event": "level_working_archived",
                "boundary_turn": self.step_index,
                "working_chars": len(working.strip()),
            },
        )
        self._clear_working()

    def _stage_retry_state(self, retry_state: str) -> dict[str, Any]:
        if self.working_path is None:
            raise RuntimeError("RESET continuation storage is unavailable.")
        metadata = self._metadata(action_applied=False)
        history = self._attempt_history()
        attempts = history["attempts"]
        provenance = {
            "source": "reset_handoff",
            "turn": self.step_index,
            "attempt": attempts[-1]["attempt"] if attempts else 1,
            "state": metadata["state"],
            "progress": metadata["progress"],
            "level_start_turn": self._level_start_turn,
        }
        atomic_write_text(
            self.retry_state_staging_path,
            retry_state.strip() + "\n",
            mode=0o600,
        )
        return provenance

    def _commit_retry_state(self, provenance: dict[str, Any]) -> None:
        if self.working_path is None or not self.retry_state_staging_path.is_file():
            raise RuntimeError("RESET continuation state is unavailable.")
        self.working_path.parent.mkdir(parents=True, exist_ok=True)
        self.retry_state_staging_path.replace(self.working_path)
        provenance = {**provenance, "reset_turn": self.step_index}
        atomic_write_text(
            self.working_provenance_path,
            json.dumps(provenance, separators=(",", ":"), sort_keys=True) + "\n",
            mode=0o600,
        )

    def _record_working_provenance(self) -> dict[str, Any]:
        metadata = self._metadata(action_applied=False)
        provenance = {
            "turn": self.step_index,
            "state": metadata["state"],
            "progress": metadata["progress"],
            "level_start_turn": self._level_start_turn,
        }
        atomic_write_text(
            self.working_provenance_path,
            json.dumps(provenance, separators=(",", ":"), sort_keys=True) + "\n",
            mode=0o600,
        )
        return provenance

    @staticmethod
    def _public_event(
        *,
        step: int,
        parsed_action: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        action = {"action": parsed_action["action"]}
        if parsed_action["action"] == "ACTION6":
            action["x"] = parsed_action["x"]
            action["y"] = parsed_action["y"]
        public_result = {key: value for key, value in result.items() if key != "grid"}
        return {
            "step": step,
            "turn": result["turn"],
            "action": action,
            "result": public_result,
        }

    def _history(self, arguments: object) -> dict[str, Any]:
        if not isinstance(arguments, dict) or set(arguments) - {
            "view",
            "start_turn",
            "end_turn",
            "limit",
        }:
            return {"ok": False, "metadata": {"error": "Invalid arguments."}}
        view = arguments.get("view")
        if view not in {"attempts", "events"}:
            return {"ok": False, "metadata": {"error": "Invalid history view."}}
        limit = arguments.get("limit", 64)
        if type(limit) is not int or not 1 <= limit <= 128:
            return {"ok": False, "metadata": {"error": "Invalid history limit."}}

        if view == "attempts":
            if set(arguments) - {"view", "limit"}:
                return {
                    "ok": False,
                    "metadata": {
                        "error": ("start_turn and end_turn are only valid for events.")
                    },
                }
            history = self._attempt_history()
            attempts = history["attempts"]
            history["attempts"] = attempts[-limit:]
            history["total_attempts"] = len(attempts)
            history["truncated"] = len(attempts) > limit
            return {"ok": True, "metadata": history}

        start_turn = arguments.get("start_turn")
        end_turn = arguments.get("end_turn")
        if start_turn is not None and (
            type(start_turn) is not int or not 0 <= start_turn <= 999999
        ):
            return {"ok": False, "metadata": {"error": "Invalid start_turn."}}
        if end_turn is not None and (
            type(end_turn) is not int or not 0 <= end_turn <= 999999
        ):
            return {"ok": False, "metadata": {"error": "Invalid end_turn."}}
        if (
            isinstance(start_turn, int)
            and isinstance(end_turn, int)
            and start_turn > end_turn
        ):
            return {
                "ok": False,
                "metadata": {"error": "start_turn must not exceed end_turn."},
            }

        matching = [
            event
            for event in self._events
            if (start_turn is None or event["turn"] >= start_turn)
            and (end_turn is None or event["turn"] <= end_turn)
        ]
        if start_turn is None and end_turn is None:
            selected = matching[-limit:]
        else:
            selected = matching[:limit]
        turns = [event["turn"] for event in self._events]
        return {
            "ok": True,
            "metadata": {
                "view": "events",
                "source": "environment action results",
                "events": selected,
                "available_turn_range": ([turns[0], turns[-1]] if turns else None),
                "total_matching": len(matching),
                "truncated": len(matching) > len(selected),
            },
        }

    def _attempt_history(self) -> dict[str, Any]:
        events = [
            event for event in self._events if event["turn"] > self._level_start_turn
        ]
        attempts: list[dict[str, Any]] = []
        attempt_number = 1
        start_kind = "level_entry"
        start_turn = self._level_start_turn
        game_actions = 0

        for event in events:
            action = event["action"]["action"]
            result = event["result"]
            if action == "RESET":
                attempt_number += 1
                start_kind = "reset"
                start_turn = event["turn"]
                game_actions = 0
                continue
            game_actions += 1
            if result["state"] in {"GAME_OVER", "WIN", "TERMINAL"}:
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "start": start_kind,
                        "start_turn": start_turn,
                        "outcome": result["state"],
                        "game_actions": game_actions,
                        "end_turn": event["turn"],
                    }
                )

        current_state = "TERMINAL" if self.obs is None else state_name(self.obs.state)
        if current_state not in {"GAME_OVER", "WIN", "TERMINAL"}:
            attempts.append(
                {
                    "attempt": attempt_number,
                    "start": start_kind,
                    "start_turn": start_turn,
                    "outcome": current_state,
                    "game_actions": game_actions,
                    "end_turn": self.step_index,
                }
            )
        return {
            "view": "attempts",
            "source": "environment action results",
            "level_start_turn": self._level_start_turn,
            "progress": self._metadata(action_applied=False)["progress"],
            "attempts": attempts,
        }

    def _store_observation(self) -> None:
        save_private_observation(self.private_dir, self.step_index, self.obs)
        self._current_images = render_observation_frames(
            self.frames_dir,
            self.step_index,
            self.obs,
            self.render_scale,
            self.render_grid,
        )

    def _selected_images(self) -> list[Path]:
        return self._current_images[-1:] if self._current_images else []

    def _metadata(
        self,
        *,
        action_applied: bool,
        error: str | None = None,
        level_boundary: bool = False,
        retry_boundary: bool = False,
        invalid_action_limit: bool = False,
    ) -> dict[str, Any]:
        state = "TERMINAL" if self.obs is None else state_name(self.obs.state)
        completed = None if self.obs is None else self.obs.levels_completed
        total = None if self.obs is None else self.obs.win_levels
        metadata: dict[str, Any] = {
            "turn": self.step_index,
            "state": state,
            "progress": {"completed": completed, "total": total},
            "available_actions": (
                []
                if self.terminal
                else available_action_names(self._available_actions())
            ),
            "action_applied": action_applied,
            "terminal": self.terminal,
        }
        if self.termination_reason:
            metadata["termination_reason"] = self.termination_reason
        if error:
            metadata["error"] = error
        if level_boundary:
            metadata["level_boundary"] = True
        if retry_boundary:
            metadata["retry_boundary"] = True
        if invalid_action_limit:
            metadata["invalid_action_limit"] = True
        selected = self._selected_images()
        if selected:
            metadata["visual"] = {
                "turn": self.step_index,
                "frame_count": len(self._current_images),
                "final_frame": self._frame_number(selected[-1]),
                "width": self.display_size,
                "height": self.display_size,
            }
        if self.observation_mode in {"grid", "both"} and self.obs is not None:
            metadata["grid"] = format_grid_observation(self.obs.frame)
        return metadata

    def _response(
        self,
        *,
        ok: bool,
        action_applied: bool,
        error: str | None = None,
        include_image: bool = True,
        level_boundary: bool = False,
        retry_boundary: bool = False,
        invalid_action_limit: bool = False,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "ok": ok,
            "metadata": self._metadata(
                action_applied=action_applied,
                error=error,
                level_boundary=level_boundary,
                retry_boundary=retry_boundary,
                invalid_action_limit=invalid_action_limit,
            ),
        }
        selected = self._selected_images()
        if include_image and self.observation_mode in {"vision", "both"} and selected:
            response["image"] = {
                "mime_type": "image/png",
                "detail": "original",
                "data": base64.b64encode(selected[-1].read_bytes()).decode("ascii"),
            }
        return response


def render_observation_frames(
    frames_dir: Path,
    step_index: int,
    obs: Any,
    scale: int,
    grid_lines: bool = False,
) -> list[Path]:
    paths = []
    for frame_index, frame in enumerate(obs.frame):
        path = frames_dir / f"turn_{step_index:03d}_frame_{frame_index:02d}.png"
        save_frame_png(frame, path, scale=scale, grid_lines=grid_lines)
        paths.append(path)
    return paths


def save_private_observation(private_dir: Path, step_index: int, obs: Any) -> None:
    response_dir = private_dir / "responses"
    response_dir.mkdir(exist_ok=True)
    write_json(response_dir / f"step_{step_index:04d}.json", serialize_observation(obs))


def serialize_observation(obs: Any) -> dict[str, Any]:
    return {
        "game_id": obs.game_id,
        "guid": obs.guid,
        "state": state_name(obs.state),
        "levels_completed": obs.levels_completed,
        "win_levels": obs.win_levels,
        "available_actions": list(obs.available_actions),
        "action_input": safe_model_dump(obs.action_input),
        "full_reset": obs.full_reset,
        "frame": [frame.tolist() for frame in obs.frame],
    }


def format_grid_observation(frames: Iterable[Any]) -> dict[str, Any]:
    return {
        "frames": [
            [[int(cell) for cell in row] for row in frame.tolist()]
            for frame in list(frames)[-1:]
        ]
    }


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


def validation_to_log(validation: ToolActionValidation) -> dict[str, Any]:
    return {
        "ok": validation.ok,
        "action_id": (
            int(validation.action.value) if validation.action is not None else None
        ),
        "data": validation.data,
        "parsed": validation.parsed,
        "error": validation.error,
    }


def state_name(state: Any) -> str:
    return state.name if hasattr(state, "name") else str(state)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, data: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, separators=(",", ":"), sort_keys=True))
        handle.write("\n")


def atomic_write_text(path: Path, content: str, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(mode)
    temporary.replace(path)
