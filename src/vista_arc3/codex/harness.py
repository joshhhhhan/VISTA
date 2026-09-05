"""Run a visual-game evaluation with checkpointed Codex context."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc3_mplconfig")

import arc_agi
from arc_agi import Arcade, OperationMode
from dotenv import load_dotenv

from .runner import (
    CodexResult,
    CompactThresholds,
    DockerCodexRunner,
    default_codex_auth_file,
    default_codex_bin,
    default_native_codex_bin,
    prepare_codex_home,
    resolve_compact_thresholds,
    validate_codex_auth_file,
)
from .controller import GameController, write_json
from .recovery import TASK_OBJECTIVE

ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "runs"
GRID_SIZE = 64
RENDER_SCALE = 8
DEFAULT_RENDER_GRID = True
DISPLAY_SIZE = GRID_SIZE * RENDER_SCALE
DEFAULT_MAX_STEPS = 2_000
DEFAULT_MAX_INVALID_RETRIES = 15
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "max"
CODEX_EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")
EMPTY_GUIDE_MODEL = "No reliable model yet."
COMPACT_RESTORE_MARKER = Path("compact_restore.pending")
COMPACT_GUIDE_SNAPSHOT = Path("compact_guide.snapshot")
COMPACT_CHECKPOINT_MARKER = Path("compact_checkpoint.requested")
COMPACT_CHECKPOINT_READY = Path("compact_checkpoint.ready")
RETRY_BOUNDARY_MARKER = Path("retry_boundary.pending")
WORKING_MEMORY_FILE = Path("WORKING.md")
CHECKPOINT_PROMPT = (
    '<codex_internal_context source="compact_checkpoint">\n'
    "Call save_compact_checkpoint now as your first and only action, then end "
    "this turn.\n"
    "</codex_internal_context>"
)
MAX_CHECKPOINT_ATTEMPTS = 2
MAX_COMPACT_CYCLES_WITHOUT_ACTION = 3
MAX_RUNTIME_RECOVERIES_PER_STEP = 8
MAX_TASK_TIMEOUT_SECONDS = 2_000_000
NONTERMINAL_CONTINUATION_PROMPT = (
    "The environment is still active. "
    "Context limits are handled automatically. Continue."
)
INFRASTRUCTURE_TERMINATION_REASONS = frozenset(
    {
        "compact_recovery_failed",
        "controller_error",
        "environment_response_missing",
        "retry_boundary_interrupt_failed",
        "visual_transport_failed",
    }
)
ACTION_COMMENTARY_INSTRUCTION = (
    "Before each `play` call, briefly state what you expect to happen visibly."
)
CHANGE_COMMENTARY_INSTRUCTION = (
    "Before each `play` call, briefly state the relevant visual change after the "
    "previous action when applicable, and what you expect to happen visibly from "
    "the action you are taking now."
)


@dataclass(frozen=True)
class GameRunOutcome:
    run_dir: Path
    scorecard_id: str | None
    session_id: str | None
    failure: BaseException | None


EnvironmentFactory = Callable[[str, str, Path], Any | None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--operation-mode",
        choices=["online", "offline"],
        default="online",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Codex model",
    )
    parser.add_argument("--effort", choices=CODEX_EFFORT_LEVELS)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--max-invalid-retries",
        type=int,
        default=DEFAULT_MAX_INVALID_RETRIES,
    )
    parser.set_defaults(
        observation_mode="vision",
        describe_changes=False,
        render_grid=DEFAULT_RENDER_GRID,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.operation_mode != "offline" and not os.getenv("ARC_API_KEY"):
        parser.error("ARC_API_KEY is not set. Put it in .env or export it.")
    if args.max_steps < 1:
        parser.error("--max-steps must be >= 1")
    if args.timeout < 1:
        parser.error("--timeout must be >= 1")
    if args.max_invalid_retries < 0:
        parser.error("--max-invalid-retries must be >= 0")

    outcome = run_game(args)
    print(f"Run directory: {outcome.run_dir}")
    print(f"Scorecard id: {outcome.scorecard_id}")
    print(f"Codex session id: {outcome.session_id}")
    if outcome.failure is not None:
        raise outcome.failure
    return 0


def run_game(
    args: argparse.Namespace,
    *,
    shared_arc: Arcade | None = None,
    shared_scorecard_id: str | None = None,
    auth_file: Path | None = None,
    run_dir: Path | None = None,
    environment_factory: EnvironmentFactory | None = None,
) -> GameRunOutcome:
    """Run one game, optionally under a coordinator-owned scorecard."""
    if (shared_arc is None) != (shared_scorecard_id is None):
        raise ValueError("shared_arc and shared_scorecard_id must be provided together")

    args = argparse.Namespace(**vars(args))
    args.model, reasoning_effort = resolve_model_spec(
        args.model,
        getattr(args, "effort", None),
    )
    compact_thresholds = resolve_compact_thresholds(args.model)
    if run_dir is None:
        run_dir = create_run_dir(RUNS_DIR)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    private_dir = run_dir / "private"
    visible_dir = run_dir / "codex_visible"
    frames_dir = visible_dir / "screenshots"
    io_dir = private_dir / "codex_io"
    recordings_dir = private_dir / "recordings"
    for directory in (
        private_dir,
        visible_dir,
        frames_dir,
        io_dir,
        recordings_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    private_dir.chmod(0o700)

    write_player_agents(
        Path(__file__).with_name("prompt.md"),
        visible_dir / "AGENTS.md",
        describe_changes=args.describe_changes,
    )
    write_guide(visible_dir / "GUIDE.md", current_model=EMPTY_GUIDE_MODEL)
    codex_home = prepare_codex_home(private_dir / "codex_home")
    write_compact_restore_hook(visible_dir, codex_home)

    manifest = base_manifest(
        args,
        run_dir,
        shared_scorecard_id,
        reasoning_effort,
        compact_thresholds,
    )
    write_json(run_dir / "manifest.json", manifest)

    arc: Arcade | None = shared_arc
    scorecard_id: str | None = shared_scorecard_id
    owns_scorecard = shared_arc is None
    env: Any | None = None
    controller: GameController | None = None
    result: CodexResult | None = None
    results: list[CodexResult] = []
    failure: BaseException | None = None

    try:
        auth_file = validate_codex_auth_file(
            auth_file if auth_file is not None else default_codex_auth_file()
        )
        if arc is None:
            os.environ["OPERATION_MODE"] = args.operation_mode
            arc = Arcade(
                operation_mode=OperationMode(args.operation_mode),
                recordings_dir=str(recordings_dir),
            )
        if not arc.get_environments():
            raise RuntimeError(
                "No environments were fetched. Check network, credentials, and endpoint settings."
            )

        if scorecard_id is None:
            scorecard_id = open_evaluation_scorecard(arc)
            manifest["scorecard_id"] = scorecard_id
            write_json(run_dir / "manifest.json", manifest)

        if environment_factory is None:
            env = arc.make(
                args.game_id,
                scorecard_id=scorecard_id,
                save_recording=True,
                include_frame_data=True,
            )
        else:
            env = environment_factory(args.game_id, scorecard_id, recordings_dir)
        if env is None:
            raise RuntimeError(f"Failed to make environment {args.game_id!r}")
        share_online_cookie_jar(arc, env)
        write_json(run_dir / "manifest.json", manifest)
        observation = env.observation_space
        if observation is None:
            raise RuntimeError("Environment did not provide an initial observation")

        controller = GameController(
            env=env,
            initial_observation=observation,
            private_dir=private_dir,
            frames_dir=frames_dir,
            max_steps=args.max_steps,
            max_invalid_retries=args.max_invalid_retries,
            observation_mode=args.observation_mode,
            render_scale=RENDER_SCALE,
            render_grid=args.render_grid,
            compact_restore_marker=codex_home / COMPACT_RESTORE_MARKER,
            compact_guide_snapshot=codex_home / COMPACT_GUIDE_SNAPSHOT,
            compact_checkpoint_marker=codex_home / COMPACT_CHECKPOINT_MARKER,
            compact_checkpoint_ready=codex_home / COMPACT_CHECKPOINT_READY,
            retry_boundary_marker=codex_home / RETRY_BOUNDARY_MARKER,
            working_path=visible_dir / WORKING_MEMORY_FILE,
            guide_path=visible_dir / "GUIDE.md",
        )
        runner = build_runner(
            args,
            visible_dir,
            codex_home,
            auth_file,
            io_dir,
            controller,
            reasoning_effort,
            compact_thresholds,
        )

        run_task_with_compact_checkpoints(
            runner=runner,
            controller=controller,
            prompt=build_initial_prompt(controller.initial_metadata()),
            checkpoint_marker=codex_home / COMPACT_CHECKPOINT_MARKER,
            checkpoint_ready=codex_home / COMPACT_CHECKPOINT_READY,
            working_path=visible_dir / WORKING_MEMORY_FILE,
            results=results,
        )
        result = results[-1]
        manifest["codex"] = result_to_manifest(result)
        manifest["codex_segments"] = [
            result_to_manifest(segment) for segment in results
        ]
        if result.returncode != 0 and not controller.terminal:
            raise RuntimeError(
                f"Codex task failed with exit code {result.returncode}; "
                f"stderr: {result.stderr_path}"
            )
        if controller.termination_reason in INFRASTRUCTURE_TERMINATION_REASONS:
            raise RuntimeError(
                "Game stopped after an infrastructure failure: "
                f"{controller.termination_reason}"
            )
        if not controller.terminal:
            controller.forced_termination_reason = "agent_stopped_early"
            raise RuntimeError("Codex stopped before the game or action budget ended")
    except BaseException as exc:
        if result is None and results:
            result = results[-1]
            manifest["codex"] = result_to_manifest(result)
            manifest["codex_segments"] = [
                result_to_manifest(segment) for segment in results
            ]
        failure = exc
    finally:
        if owns_scorecard and arc is not None and scorecard_id is not None:
            scorecard_result, scorecard_failure = finalize_scorecard(
                arc=arc,
                env=env,
                scorecard_id=scorecard_id,
                operation_mode=args.operation_mode,
            )
            manifest.update(scorecard_result)
            if scorecard_failure is not None and failure is None:
                failure = scorecard_failure
        elif arc is not None and scorecard_id is not None:
            manifest["scorecard_closed"] = False
            manifest["scorecard_finalization"] = "managed_by_competition_coordinator"
        else:
            manifest["scorecard_closed"] = False

        manifest["finished_at"] = utc_now()
        manifest["steps_attempted"] = controller.step_index if controller else 0
        manifest["action_attempts"] = controller.total_attempts if controller else 0
        manifest["terminal"] = controller.terminal if controller else False
        manifest["termination_reason"] = (
            controller.termination_reason if controller else "setup_failed"
        )
        if result is not None:
            manifest["session_segments"] = len(results)
        if failure is not None:
            manifest["error"] = type(failure).__name__
            (private_dir / "error.txt").write_text(
                f"{type(failure).__name__}: {failure}\n",
                encoding="utf-8",
            )
        write_json(run_dir / "manifest.json", manifest)

    return GameRunOutcome(
        run_dir=run_dir,
        scorecard_id=scorecard_id,
        session_id=result.session_id if result else None,
        failure=failure,
    )


def build_runner(
    args: argparse.Namespace,
    visible_dir: Path,
    codex_home: Path,
    auth_file: Path,
    io_dir: Path,
    controller: GameController,
    reasoning_effort: str | None,
    compact_thresholds: CompactThresholds,
) -> DockerCodexRunner:
    maximum_calls = args.max_steps * (args.max_invalid_retries + 1) + 1
    return DockerCodexRunner(
        visible_dir=visible_dir,
        codex_home=codex_home,
        auth_file=auth_file,
        io_dir=io_dir,
        controller=controller,
        codex_bin=default_native_codex_bin(),
        model=args.model,
        reasoning_effort=reasoning_effort,
        compact_thresholds=compact_thresholds,
        task_timeout=bounded_task_timeout(args.timeout, maximum_calls),
        display_size=DISPLAY_SIZE,
    )


def bounded_task_timeout(tool_timeout: int, maximum_calls: int) -> int:
    return min(tool_timeout * maximum_calls, MAX_TASK_TIMEOUT_SECONDS)


def build_initial_prompt(metadata: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Current observation:",
            json.dumps(metadata, separators=(",", ":")),
            "",
            TASK_OBJECTIVE,
        ]
    )


def run_task_with_compact_checkpoints(
    *,
    runner: DockerCodexRunner,
    controller: GameController,
    prompt: str,
    checkpoint_marker: Path,
    checkpoint_ready: Path,
    working_path: Path,
    results: list[CodexResult] | None = None,
) -> list[CodexResult]:
    if results is None:
        results = []
    results.append(runner.run_task(prompt))
    session_id = results[-1].session_id
    checkpoints = 0
    retry_boundaries = 0
    last_checkpoint_step: int | None = None
    compact_cycles_without_action = 0
    runtime_recovery_step: int | None = None
    runtime_recoveries_at_step = 0
    nonterminal_continuation_scope: tuple[str, int | None] | None = None

    def record_runtime_failure(result: CodexResult) -> None:
        nonlocal runtime_recovery_step, runtime_recoveries_at_step
        current_step = controller.step_index
        if runtime_recovery_step != current_step:
            runtime_recovery_step = current_step
            runtime_recoveries_at_step = 0
        runtime_recoveries_at_step += 1
        if runtime_recoveries_at_step > MAX_RUNTIME_RECOVERIES_PER_STEP:
            raise RuntimeError(
                "Codex player runtime failed repeatedly at the same game state; "
                f"last stderr: {result.stderr_path}"
            )

    def resume_player_runtime() -> None:
        nonlocal session_id
        failed = results[-1]
        record_runtime_failure(failed)
        if session_id is None:
            raise RuntimeError(
                "Codex player runtime failed without a resumable session id; "
                f"stderr: {failed.stderr_path}"
            )
        expected_session_id = session_id
        recovery = controller.begin_runtime_recovery()
        resumed = runner.resume_runtime_recovery(
            expected_session_id,
            recovery,
        )
        results.append(resumed)
        session_id = resumed.session_id
        if session_id != expected_session_id:
            raise RuntimeError(
                "Runtime recovery did not resume the exact Codex session"
            )

    while True:
        if controller.terminal:
            controller.discard_compact_recovery()
            return results

        if getattr(controller, "retry_boundary_pending", False):
            retry_boundaries += 1
            if retry_boundaries > controller.max_steps:
                raise RuntimeError("Retry boundaries made no bounded progress")
            previous_session_id = session_id
            recovery = controller.begin_retry_recovery()
            recovery_session_id: str | None = None

            while getattr(controller, "retry_boundary_pending", False):
                if recovery_session_id is None:
                    recovered = runner.run_retry_task(recovery)
                else:
                    recovered = runner.resume_retry_task(
                        recovery_session_id,
                        recovery,
                    )
                results.append(recovered)

                if recovered.session_id is not None:
                    if recovery_session_id is None:
                        if recovered.session_id == previous_session_id:
                            raise RuntimeError(
                                "Retry recovery did not create its required fresh session"
                            )
                        recovery_session_id = recovered.session_id
                    elif recovered.session_id != recovery_session_id:
                        raise RuntimeError(
                            "Retry recovery did not resume the exact Codex session"
                        )

                boundary_pending = bool(
                    getattr(controller, "retry_boundary_pending", False)
                )
                pending_boundary_step = (
                    getattr(controller, "retry_boundary_step", None)
                    if boundary_pending
                    else None
                )
                if boundary_pending and (
                    type(pending_boundary_step) is not int
                    or pending_boundary_step < recovery.boundary_step
                ):
                    raise RuntimeError("Retry boundary state is inconsistent")
                same_boundary_pending = (
                    pending_boundary_step == recovery.boundary_step
                )
                new_boundary_pending = (
                    type(pending_boundary_step) is int
                    and pending_boundary_step > recovery.boundary_step
                )

                if recovered.returncode != 0:
                    record_runtime_failure(recovered)
                    if controller.terminal:
                        break
                    if new_boundary_pending:
                        break
                    continue
                if same_boundary_pending:
                    raise RuntimeError("Codex did not enter retry recovery")
                break

            if controller.terminal:
                continue
            if recovery_session_id is None:
                raise RuntimeError("Retry recovery created no Codex session")
            session_id = recovery_session_id
            continue

        if checkpoint_marker.exists():
            if session_id is None:
                raise RuntimeError(
                    "Codex stopped for compact without a resumable session id"
                )

            current_step = controller.step_index
            if current_step == last_checkpoint_step:
                compact_cycles_without_action += 1
            else:
                last_checkpoint_step = current_step
                compact_cycles_without_action = 1
            if compact_cycles_without_action >= MAX_COMPACT_CYCLES_WITHOUT_ACTION:
                raise RuntimeError(
                    "Compact checkpoint loop repeated without an environment action"
                )

            if not checkpoint_ready.exists() or not working_path.is_file():
                checkpoint_step = controller.step_index
                completed_attempts = 0
                while (
                    not checkpoint_ready.exists()
                    or not working_path.is_file()
                ):
                    if completed_attempts >= MAX_CHECKPOINT_ATTEMPTS:
                        raise RuntimeError("Codex did not write the compact checkpoint")
                    checkpoint = runner.resume_checkpoint(session_id, CHECKPOINT_PROMPT)
                    results.append(checkpoint)
                    if checkpoint.session_id != session_id:
                        raise RuntimeError(
                            "Compact checkpoint did not resume the exact Codex session"
                        )
                    if controller.terminal:
                        break
                    if controller.step_index != checkpoint_step:
                        raise RuntimeError(
                            "Compact checkpoint turn applied an environment action"
                        )
                    if checkpoint.returncode != 0:
                        record_runtime_failure(checkpoint)
                        continue
                    completed_attempts += 1
                if controller.terminal:
                    continue

            checkpoints += 1
            if checkpoints > controller.max_steps + 1:
                raise RuntimeError("Compact checkpoint loop made no bounded progress")

            previous_session_id = session_id
            recovery = controller.begin_compact_recovery()
            restore_marker = controller.compact_restore_marker
            recovery_session_id: str | None = None

            while restore_marker is not None and restore_marker.exists():
                if recovery_session_id is None:
                    recovered = runner.run_recovery_task(recovery)
                else:
                    recovered = runner.resume_recovery_task(
                        recovery_session_id,
                        recovery,
                    )
                results.append(recovered)

                if recovered.session_id is not None:
                    if recovery_session_id is None:
                        if recovered.session_id == previous_session_id:
                            raise RuntimeError(
                                "Compact recovery did not create its required fresh session"
                            )
                        recovery_session_id = recovered.session_id
                    elif recovered.session_id != recovery_session_id:
                        raise RuntimeError(
                            "Compact recovery did not resume the exact Codex session"
                        )

                if recovered.returncode != 0:
                    record_runtime_failure(recovered)
                    if controller.terminal:
                        break
                    continue
                if restore_marker.exists():
                    raise RuntimeError("Codex did not enter compact recovery")

            if controller.terminal:
                continue
            if recovery_session_id is None:
                raise RuntimeError("Compact recovery created no Codex session")
            session_id = recovery_session_id
            continue

        current_result = results[-1]
        if getattr(current_result, "returncode", 0) != 0:
            if session_id is None:
                if controller.step_index != 0:
                    raise RuntimeError(
                        "Codex runtime failed after game actions without a session id"
                    )
                record_runtime_failure(current_result)
                results.append(runner.run_task(prompt))
                session_id = results[-1].session_id
                continue
            resume_player_runtime()
            continue

        current = results[-1]
        if current.returncode != 0 or session_id is None:
            return results

        progress = controller.initial_metadata().get("progress")
        completed = progress.get("completed") if isinstance(progress, dict) else None
        level_progress = completed if type(completed) is int else None
        continuation_scope = (session_id, level_progress)
        if continuation_scope != nonterminal_continuation_scope:
            nonterminal_continuation_scope = continuation_scope
            continued = runner.resume_nonterminal_task(
                session_id,
                NONTERMINAL_CONTINUATION_PROMPT,
            )
            results.append(continued)
            if (
                continued.session_id is not None
                and continued.session_id != session_id
            ):
                raise RuntimeError(
                    "Nonterminal continuation changed Codex session"
                )
            session_id = continued.session_id or session_id
            continue

        controller.request_fresh_recovery()


def parse_model_spec(model_spec: str | None) -> tuple[str | None, str | None]:
    if not model_spec:
        return None, None
    efforts = set(CODEX_EFFORT_LEVELS)
    model, separator, suffix = model_spec.rpartition("-")
    if separator and suffix in efforts:
        return model, suffix
    return model_spec, None


def resolve_model_spec(
    model_spec: str | None,
    effort: str | None,
) -> tuple[str | None, str]:
    model, suffix_effort = parse_model_spec(model_spec)
    if effort and suffix_effort and effort != suffix_effort:
        raise ValueError("Conflicting reasoning efforts in --model and --effort")
    return model, effort or suffix_effort or DEFAULT_REASONING_EFFORT


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

        # arc-agi 0.9.x otherwise keeps an open-time cookie snapshot and copies
        # it back over the refreshed environment cookies during scorecard close.
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
    """Close a remote scorecard without losing a completed run to a stale cookie."""
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


def base_manifest(
    args: argparse.Namespace,
    run_dir: Path,
    scorecard_id: str | None,
    reasoning_effort: str | None,
    compact_thresholds: CompactThresholds,
) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "arc_agi_version": getattr(arc_agi, "__version__", None),
        "game_id": args.game_id,
        "operation_mode": args.operation_mode,
        "scorecard_id": scorecard_id,
        "runtime": "codex",
        "codex_model": args.model,
        "codex_reasoning_effort": reasoning_effort,
        "describe_visual_changes": args.describe_changes,
        "observation_mode": args.observation_mode,
        "render_grid": args.render_grid,
        "max_steps": args.max_steps,
        "max_invalid_retries": args.max_invalid_retries,
        "codex_version": codex_version(),
    }


def codex_version() -> str | None:
    try:
        proc = subprocess.run(
            [str(default_codex_bin()), "--version"],
            text=True,
            capture_output=True,
            env=codex_host_environment(),
            timeout=20,
            check=False,
        )
    except Exception:
        return None
    return (proc.stdout or proc.stderr).strip() or None


def codex_host_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ARC_")
    }


def write_guide(path: Path, *, current_model: str) -> None:
    path.write_text(
        (current_model.strip() or EMPTY_GUIDE_MODEL) + "\n",
        encoding="utf-8",
    )


def write_player_agents(
    source: Path,
    destination: Path,
    *,
    describe_changes: bool,
) -> None:
    text = source.read_text(encoding="utf-8")
    if describe_changes:
        if ACTION_COMMENTARY_INSTRUCTION not in text:
            raise RuntimeError("AGENTS.md is missing the action commentary instruction")
        text = text.replace(
            ACTION_COMMENTARY_INSTRUCTION,
            CHANGE_COMMENTARY_INSTRUCTION,
            1,
        )
    destination.write_text(text, encoding="utf-8")


def write_compact_restore_hook(visible_dir: Path, codex_home: Path) -> None:
    codex_dir = visible_dir / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        codex_home / "hooks.json",
        {
            "hooks": {
                "PreCompact": [
                    {
                        "matcher": "auto|manual",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "sh .codex/pre_compact_checkpoint.sh",
                                "statusMessage": "Saving game checkpoint",
                            }
                        ],
                    }
                ]
            }
        },
    )
    pre_script_path = codex_dir / "pre_compact_checkpoint.sh"
    pre_script_path.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "set -eu",
                "umask 077",
                "cat >/dev/null",
                'retry="$CODEX_HOME/retry_boundary.pending"',
                'if [ -f "$retry" ]; then',
                "  printf '%s\\n' '{\"continue\":false,\"stopReason\":\"Starting a fresh retry context.\"}'",
                "  exit 0",
                "fi",
                'request="$CODEX_HOME/compact_checkpoint.requested"',
                ': > "$request"',
                "printf '%s\\n' '{\"continue\":false,\"stopReason\":\"Save the compact checkpoint before continuing.\"}'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    pre_script_path.chmod(0o555)


def result_to_manifest(result: CodexResult) -> dict[str, Any]:
    return {
        "segment": result.segment,
        "returncode": result.returncode,
        "usage": result.usage.__dict__ if result.usage else None,
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
