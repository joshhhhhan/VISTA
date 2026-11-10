"""Run one ARC-AGI-3 game through DeepSeek Harness headless mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc3_mplconfig")

import arc_agi
from arc_agi import Arcade, OperationMode

from vista.backends.dsh.runner import (
    DEFAULT_DSH_EFFORT,
    DEFAULT_DSH_MODEL,
    DshRunner,
    dsh_bin,
)
from vista.benchmarks.arc3.claude.controller import (
    ACTION_BY_NAME,
    GameController,
    render_observation_frames,
    write_json,
)
from vista.benchmarks.arc3.claude.recovery import TASK_OBJECTIVE
from vista.benchmarks.arc3.claude.runner import CLAUDE_EFFORT_LEVELS
from vista.benchmarks.arc3.http import configure_arc_http
from vista.benchmarks.arc3.run import (
    GameRunOutcome,
    create_run_dir,
    finalize_scorecard,
    load_run_environment,
    open_evaluation_scorecard,
    report_run_failure,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[5]
GRID_SIZE = 64
RENDER_SCALE = 8
DEFAULT_RENDER_GRID = True
DISPLAY_SIZE = GRID_SIZE * RENDER_SCALE
DEFAULT_MAX_STEPS = 3_000
DEFAULT_MAX_INVALID_RETRIES = 15
EMPTY_GUIDE_MODEL = "No reliable model yet."
FRAME_NAME = re.compile(r"^turn_(\d+)_frame_00\.png$")


def build_initial_prompt(metadata: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are playing an ARC-AGI-3 game.",
            "You may only use the mcp__game__* tools. Do not use shell, filesystem, web, or subagent tools.",
            "First call mcp__game__inspect with question 'Initial game state' and views "
            "[{\"label\": \"initial\", \"turn\": 0}] to see the current visual state.",
            "Then make exactly one game action per mcp__game__play call. "
            "Use only actions listed in available_actions.",
            "Use mcp__game__read_pixels or mcp__game__inspect when exact visual details matter.",
            "GUIDE.md and WORKING.md are available through mcp__game__read_guide, "
            "mcp__game__write_guide, mcp__game__read_working, and mcp__game__write_working.",
            "Do not stop until the game is terminal (WIN or GAME_OVER).",
            "",
            "Current observation:",
            json.dumps(metadata, separators=(",", ":")),
            "",
            TASK_OBJECTIVE,
        ]
    )


def build_resume_prompt(
    metadata: dict[str, Any],
    *,
    completed: int | None,
    total: int | None,
    replayed_turns: int,
) -> str:
    return "\n".join(
        [
            "You are resuming an in-progress ARC-AGI-3 game.",
            f"The game is already at {completed} of {total} levels completed "
            f"after replaying {replayed_turns} recorded actions.",
            "Do not restart or reset the game.",
            "Continue from the current observation and finish the remaining levels.",
            "The previous player's GUIDE.md and WORKING.md are restored.",
            "First call mcp__game__inspect with question 'Current game state' and views "
            "[{\"label\": \"current\", \"turn\": 0}] to see the current visual state.",
            "Then make exactly one game action per mcp__game__play call. "
            "Use only actions listed in available_actions.",
            "Use mcp__game__read_pixels or mcp__game__inspect when exact visual details matter.",
            "Do not stop until the game is terminal (WIN or GAME_OVER).",
            "",
            "Current observation:",
            json.dumps(metadata, separators=(",", ":")),
            "",
            TASK_OBJECTIVE,
        ]
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_actions(path: Path) -> list[tuple[str, dict[str, int] | None]]:
    actions: dict[int, tuple[str, dict[str, int] | None]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        result = entry.get("result") or {}
        validation = entry.get("validation") or {}
        if not result.get("action_applied") or not validation.get("ok"):
            continue
        turn = result.get("turn")
        if not isinstance(turn, int):
            continue
        parsed = validation.get("parsed") or {}
        name = parsed.get("action")
        if not isinstance(name, str):
            continue
        actions[turn] = (name, validation.get("data"))
    return [actions[turn] for turn in sorted(actions)]


def final_frame_turn(frames_dir: Path) -> int:
    turns = []
    for path in frames_dir.glob("turn_*_frame_00.png"):
        match = FRAME_NAME.match(path.name)
        if match is not None:
            turns.append(int(match.group(1)))
    if not turns:
        raise RuntimeError(f"No final frames found in {frames_dir}")
    return max(turns)


def copy_player_state(previous_run: Path, visible: Path) -> None:
    for name in ("GUIDE.md", "WORKING.md"):
        source = previous_run / "player" / name
        if source.is_file():
            shutil.copy2(source, visible / name)


def replay_previous_run(
    previous_run: Path,
    *,
    game_id: str,
    check_dir: Path,
) -> tuple[Arcade, Any, Any, int, str]:
    action_log = previous_run / "private" / "action_log.jsonl"
    if not action_log.is_file():
        raise FileNotFoundError(f"Missing previous action log: {action_log}")
    actions = recorded_actions(action_log)
    if not actions:
        raise RuntimeError("The previous run has no successful actions to replay.")

    # NORMAL downloads the game source if needed, then runs it locally.  Replay
    # does not touch the original scorecard or consume remote game actions.
    arc = Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=str(ROOT / "environment_files"),
    )
    configure_arc_http(arc)
    env = arc.make(game_id, save_recording=True, include_frame_data=True)
    if env is None:
        raise RuntimeError(f"Failed to create local replay environment {game_id!r}")
    obs = env.observation_space
    if obs is None:
        raise RuntimeError("The local environment returned no initial observation.")
    for turn, (name, data) in enumerate(actions, start=1):
        action = ACTION_BY_NAME.get(name)
        if action is None:
            raise RuntimeError(
                f"Previous run action {turn} is not a known game action: {name!r}"
            )
        obs = env.step(action, data=data)
        if obs is None:
            raise RuntimeError(f"Replay failed at action {turn} ({name}).")

    replayed_turns = final_frame_turn(previous_run / "player" / "screenshots")
    render_observation_frames(
        check_dir,
        replayed_turns,
        obs,
        RENDER_SCALE,
        DEFAULT_RENDER_GRID,
    )
    previous_final = (
        previous_run
        / "player"
        / "screenshots"
        / f"turn_{replayed_turns:03d}_frame_00.png"
    )
    local_final = check_dir / f"turn_{replayed_turns:03d}_frame_00.png"
    if not previous_final.is_file() or not local_final.is_file():
        raise RuntimeError("Could not locate the previous and replayed final frames.")
    previous_hash = digest(previous_final)
    if previous_hash != digest(local_final):
        raise RuntimeError(
            "Local replay did not reproduce the previous final frame; refusing "
            "to resume from an unverified state."
        )
    return arc, env, obs, replayed_turns, previous_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--operation-mode", choices=["online", "offline"], default="online"
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--max-invalid-retries", type=int, default=DEFAULT_MAX_INVALID_RETRIES)
    parser.add_argument("--model", default=DEFAULT_DSH_MODEL)
    parser.add_argument("--effort", choices=CLAUDE_EFFORT_LEVELS, default=DEFAULT_DSH_EFFORT)
    parser.add_argument("--observation-mode", choices=["vision", "grid", "both"], default="vision")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    return parser


def base_manifest(
    args: argparse.Namespace,
    run_dir: Path,
    scorecard_id: str | None,
) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "arc_agi_version": getattr(arc_agi, "__version__", None),
        "game_id": args.game_id,
        "operation_mode": args.operation_mode,
        "scorecard_id": scorecard_id,
        "runtime": "dsh",
        "requested_model": args.model,
        "reasoning_effort": args.effort,
        "observation_mode": args.observation_mode,
        "render_grid": DEFAULT_RENDER_GRID,
        "max_steps": args.max_steps,
        "max_invalid_retries": args.max_invalid_retries,
        "dsh_version": None,
        "resume_from_run": None,
        "resume_from_turn": None,
    }


def run_game(
    args: argparse.Namespace,
    *,
    shared_arc: Arcade | None = None,
    shared_scorecard_id: str | None = None,
    run_dir: Path | None = None,
) -> GameRunOutcome:
    load_run_environment()
    if args.max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if args.timeout < 1:
        raise ValueError("timeout must be positive")

    previous_run: Path | None = None
    resume_from = getattr(args, "resume_from", None)
    if resume_from is not None:
        previous_run = Path(resume_from).expanduser().resolve()
        previous_manifest_path = previous_run / "manifest.json"
        if not previous_manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing previous run manifest: {previous_manifest_path}"
            )
        previous_manifest = json.loads(
            previous_manifest_path.read_text(encoding="utf-8")
        )
        if not args.game_id:
            args.game_id = previous_manifest.get("game_id")
        if not args.game_id:
            raise ValueError(
                "--game-id is required when the previous manifest has no game_id"
            )
    elif not args.game_id:
        raise ValueError("--game-id is required unless --resume-from is given")

    if run_dir is None:
        run_dir = create_run_dir(args.runs_dir)
    else:
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    private_dir = run_dir / "private"
    visible_dir = run_dir / "player"
    frames_dir = visible_dir / "screenshots"
    private_dir.mkdir(parents=True)
    visible_dir.mkdir()
    frames_dir.mkdir()
    private_dir.chmod(0o700)
    (visible_dir / "GUIDE.md").write_text(EMPTY_GUIDE_MODEL + "\n", encoding="utf-8")
    (visible_dir / "WORKING.md").write_text("", encoding="utf-8")
    if previous_run is not None:
        copy_player_state(previous_run, visible_dir)

    arc = shared_arc
    scorecard_id = shared_scorecard_id
    owns_scorecard = arc is None
    env: Any = None
    controller: GameController | None = None
    failure: BaseException | None = None
    result = None
    replayed_turns = 0
    resume_completed: int | None = None
    resume_total: int | None = None
    manifest = base_manifest(args, run_dir, scorecard_id)
    if previous_run is not None:
        manifest["operation_mode"] = "offline-resume"
        manifest["resume_from_run"] = str(previous_run)
    write_json(run_dir / "manifest.json", manifest)
    print(f"Run directory: {run_dir}", flush=True)
    try:
        if previous_run is not None:
            if shared_arc is not None:
                raise ValueError(
                    "--resume-from cannot be combined with a shared Arcade"
                )
            arc, env, observation, replayed_turns, previous_hash = replay_previous_run(
                previous_run,
                game_id=args.game_id,
                check_dir=private_dir / "resume_check",
            )
            scorecard_id = getattr(arc, "_default_scorecard_id", None)
            owns_scorecard = scorecard_id is not None
            resume_completed = getattr(observation, "levels_completed", None)
            resume_total = getattr(observation, "win_levels", None)
            manifest.update(
                {
                    "scorecard_id": scorecard_id,
                    "resume_from_run": str(previous_run),
                    "resume_from_turn": replayed_turns,
                    "resume_progress": {
                        "completed": resume_completed,
                        "total": resume_total,
                    },
                    "resume_frame_sha256": previous_hash,
                }
            )
            write_json(run_dir / "manifest.json", manifest)
            print(
                f"Resumed at level {resume_completed}/{resume_total} after "
                f"replaying {replayed_turns} actions.",
                flush=True,
            )
        else:
            if arc is None:
                os.environ["OPERATION_MODE"] = args.operation_mode
                arc = Arcade(operation_mode=OperationMode(args.operation_mode))
                configure_arc_http(arc)
                if not arc.get_environments():
                    raise RuntimeError(
                        "No environments were fetched. Check network and ARC credentials."
                    )
            if scorecard_id is None:
                scorecard_id = open_evaluation_scorecard(arc)
                manifest["scorecard_id"] = scorecard_id
                write_json(run_dir / "manifest.json", manifest)
            env = arc.make(
                args.game_id,
                scorecard_id=scorecard_id,
                save_recording=True,
                include_frame_data=True,
            )
            if env is None:
                raise RuntimeError(f"Failed to make environment {args.game_id!r}")
            configure_arc_http(env)
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
            render_grid=DEFAULT_RENDER_GRID,
            reset_starts_fresh_session=False,
            working_path=visible_dir / "WORKING.md",
            guide_path=visible_dir / "GUIDE.md",
            agent_name="dsh",
            interface_name="dsh-mcp-tool-loop",
        )
        runner = DshRunner(
            visible_dir=visible_dir,
            private_dir=private_dir,
            controller=controller,
            model=args.model,
            effort=args.effort,
            dsh_bin_path=dsh_bin(),
            task_timeout=args.timeout,
            display_size=DISPLAY_SIZE,
        )
        prompt = (
            build_resume_prompt(
                controller.initial_metadata(),
                completed=resume_completed,
                total=resume_total,
                replayed_turns=replayed_turns,
            )
            if previous_run is not None
            else build_initial_prompt(controller.initial_metadata())
        )
        result = runner.run(prompt)
        manifest["dsh"] = {
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "stdout": str(result.stdout_path),
            "stderr": str(result.stderr_path),
            "profile_dir": str(result.profile_dir),
            "final_message": result.final_message,
        }
        if result.timed_out:
            failure = TimeoutError("DeepSeek Harness timed out")
        elif result.returncode != 0 and not controller.terminal:
            failure = RuntimeError(
                f"DeepSeek Harness exited with code {result.returncode} before terminal"
            )
    except BaseException as exc:
        failure = exc
    finally:
        finalization_mode = (
            "offline-resume" if previous_run is not None else args.operation_mode
        )
        if owns_scorecard and arc is not None and scorecard_id is not None:
            scorecard_result, scorecard_failure = finalize_scorecard(
                arc=arc,
                env=env,
                scorecard_id=scorecard_id,
                operation_mode=finalization_mode,
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
        manifest["progress"] = {
            "completed": (
                None
                if controller is None or controller.obs is None
                else controller.obs.levels_completed
            ),
            "total": (
                None
                if controller is None or controller.obs is None
                else controller.obs.win_levels
            ),
        }
        if failure is not None:
            manifest["error"] = type(failure).__name__
            (private_dir / "error.txt").write_text(
                f"{type(failure).__name__}: {failure}\n", encoding="utf-8"
            )
        write_json(run_dir / "manifest.json", manifest)
    return GameRunOutcome(
        run_dir=run_dir,
        scorecard_id=scorecard_id,
        session_id=None,
        failure=failure,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume_from is not None:
        args.resume_from = args.resume_from.expanduser().resolve()
        action_log = args.resume_from / "private" / "action_log.jsonl"
        if not action_log.is_file():
            parser.error(f"not a resumable run (missing {action_log})")
        if not args.game_id:
            manifest_path = args.resume_from / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                parser.error(f"cannot read {manifest_path}: {exc}")
            args.game_id = manifest.get("game_id")
        if not args.game_id:
            parser.error("--game-id is required when the previous manifest has no game_id")
    elif not args.game_id:
        parser.error("--game-id is required unless --resume-from is given")
    outcome = run_game(args)
    print(f"Finished run directory: {outcome.run_dir}")
    print(f"Scorecard id: {outcome.scorecard_id}")
    if outcome.failure is not None:
        report_run_failure(outcome.run_dir, "dsh")
        raise outcome.failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
