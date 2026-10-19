"""Resume an ARC-AGI-3 run from a previous run's recorded action log.

The previous actions are replayed locally to reconstruct the exact game
state, then the Claude Code player continues from that state in a new run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from arc_agi import Arcade, OperationMode
from arcengine import GameAction

from vista.backends.claude.lifecycle import run_task_with_native_context
from vista.backends.claude.provider import DEEPSEEK_PROVIDER
from vista.backends.claude.runner import claude_version, prepare_claude_config
from vista.benchmarks.arc3.claude import harness
from vista.benchmarks.arc3.claude.controller import (
    GameController,
    render_observation_frames,
    write_json,
)
from vista.benchmarks.arc3.claude.runner import ClaudeCodeRunner, default_claude_bin
from vista.benchmarks.arc3.run import create_run_dir, load_run_environment, utc_now
from vista.core.timeouts import bounded_task_timeout

ROOT = Path(__file__).resolve().parents[1]


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


def copy_player_state(previous_run: Path, visible: Path) -> None:
    for name in ("AGENTS.md", "GUIDE.md", "WORKING.md"):
        source = previous_run / "player" / name
        destination = visible / name
        if source.is_file():
            shutil.copy2(source, destination)
        elif name == "AGENTS.md":
            harness.write_player_agents(
                Path(harness.__file__).with_name("prompt.md"),
                destination,
                describe_changes=False,
            )
        elif name == "GUIDE.md":
            destination.write_text(harness.EMPTY_GUIDE_MODEL + "\n", encoding="utf-8")
        else:
            destination.write_text("", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--game-id", default="s5i5")
    parser.add_argument("--model", default=DEEPSEEK_PROVIDER.default_model)
    parser.add_argument("--effort", default=DEEPSEEK_PROVIDER.default_effort or "max")
    parser.add_argument("--max-invalid-retries", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    load_run_environment()
    args = build_parser().parse_args()
    previous_run = args.previous_run.expanduser().resolve()
    action_log = previous_run / "private" / "action_log.jsonl"
    if not action_log.is_file():
        raise FileNotFoundError(f"Missing previous action log: {action_log}")
    credential = os.environ.get(DEEPSEEK_PROVIDER.credential_env)
    if not credential:
        raise RuntimeError(
            f"{DEEPSEEK_PROVIDER.credential_env} is not set; cannot start DeepSeek."
        )
    base_url = (
        os.environ.get("ANTHROPIC_BASE_URL") or DEEPSEEK_PROVIDER.default_base_url
    )
    actions = recorded_actions(action_log)
    if not actions:
        raise RuntimeError("The previous run has no successful actions to replay.")

    run_dir = create_run_dir(args.runs_dir)
    private_dir = run_dir / "private"
    visible_dir = run_dir / "player"
    frames_dir = visible_dir / "screenshots"
    io_dir = private_dir / "claude_io"
    config_dir = private_dir / "claude_config"
    recordings_dir = private_dir / "recordings"
    for directory in (
        private_dir,
        visible_dir,
        frames_dir,
        io_dir,
        config_dir,
        recordings_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    private_dir.chmod(0o700)
    prepare_claude_config(config_dir)
    copy_player_state(previous_run, visible_dir)

    os.environ["OPERATION_MODE"] = "offline"
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ROOT / "environment_files"),
        recordings_dir=str(recordings_dir),
    )
    env = arc.make(args.game_id, save_recording=True, include_frame_data=True)
    if env is None:
        raise RuntimeError(f"Failed to create local environment {args.game_id!r}")
    obs = env.observation_space
    if obs is None:
        raise RuntimeError("The local environment returned no initial observation.")
    for turn, (name, data) in enumerate(actions, start=1):
        action = getattr(GameAction, name)
        obs = env.step(action, data=data)
        if obs is None:
            raise RuntimeError(f"Replay failed at action {turn} ({name}).")

    scorecard_id = getattr(arc, "_default_scorecard_id", None)
    check_dir = private_dir / "resume_check"
    render_observation_frames(
        check_dir, len(actions), obs, harness.RENDER_SCALE, harness.DEFAULT_RENDER_GRID
    )
    previous_final = (
        previous_run
        / "player"
        / "screenshots"
        / f"turn_{len(actions):03d}_frame_00.png"
    )
    local_final = check_dir / f"turn_{len(actions):03d}_frame_00.png"
    if not previous_final.is_file():
        raise RuntimeError(f"Missing previous final frame: {previous_final}")
    if digest(previous_final) != digest(local_final):
        raise RuntimeError(
            "Local replay did not reproduce the previous final frame; refusing "
            "to resume from an unverified state."
        )

    controller = GameController(
        env=env,
        initial_observation=obs,
        private_dir=private_dir,
        frames_dir=frames_dir,
        max_steps=0,
        max_invalid_retries=args.max_invalid_retries,
        observation_mode="vision",
        render_scale=harness.RENDER_SCALE,
        render_grid=harness.DEFAULT_RENDER_GRID,
        compact_restore_marker=io_dir / harness.COMPACT_RESTORE_MARKER,
        compact_guide_snapshot=io_dir / harness.COMPACT_GUIDE_SNAPSHOT,
        compact_checkpoint_marker=io_dir / harness.COMPACT_CHECKPOINT_MARKER,
        compact_checkpoint_ready=io_dir / harness.COMPACT_CHECKPOINT_READY,
        retry_boundary_marker=private_dir / harness.RETRY_BOUNDARY_MARKER,
        reset_starts_fresh_session=False,
        working_path=visible_dir / harness.WORKING_MEMORY_FILE,
        guide_path=visible_dir / "GUIDE.md",
        agent_name="claude-code",
        interface_name="native-mcp-tool-loop",
    )

    claude_bin = default_claude_bin()
    runner = ClaudeCodeRunner(
        visible_dir=visible_dir,
        claude_config_dir=config_dir,
        io_dir=io_dir,
        controller=controller,
        claude_bin=claude_bin,
        model=args.model,
        effort=args.effort,
        expected_model_id=args.model,
        oauth_token=None,
        provider=DEEPSEEK_PROVIDER,
        provider_credential=credential,
        provider_base_url=base_url,
        auto_compact_window=DEEPSEEK_PROVIDER.auto_compact_window,
        task_timeout=bounded_task_timeout(args.timeout, 1_000_000),
        display_size=harness.DISPLAY_SIZE,
        host_runtime=True,
    )

    manifest = {
        "created_at": utc_now(),
        "game_id": args.game_id,
        "operation_mode": "offline-resume",
        "scorecard_id": scorecard_id,
        "runtime": "claude",
        "requested_model": args.model,
        "reasoning_effort": args.effort,
        "resolved_model_ids": [],
        "observation_mode": "vision",
        "render_grid": harness.DEFAULT_RENDER_GRID,
        "max_steps": 0,
        "max_invalid_retries": args.max_invalid_retries,
        "claude_code_version": claude_version(claude_bin),
        "resume_from_run": str(previous_run),
        "resume_from_turn": len(actions),
        "resume_progress": {
            "completed": obs.levels_completed,
            "total": obs.win_levels,
        },
        "resume_frame_sha256": digest(previous_final),
    }
    write_json(run_dir / "manifest.json", manifest)
    print(f"Run directory: {run_dir}", flush=True)
    print(
        f"Resumed at level {obs.levels_completed}/{obs.win_levels} "
        f"after replaying {len(actions)} actions.",
        flush=True,
    )
    print("Initial metadata:", json.dumps(controller.initial_metadata()), flush=True)

    if args.dry_run:
        print("Dry run: model not started.", flush=True)
        arc.close_scorecard(scorecard_id)
        return 0

    prompt = "\n".join(
        [
            "You are resuming an in-progress ARC-AGI-3 game.",
            "The game is already at 7 of 8 levels completed. Do not restart or reset it.",
            "Continue from the current observation and finish the remaining level.",
            "The previous player's GUIDE.md and WORKING.md are restored in the workspace.",
            "",
            harness.build_initial_prompt(controller.initial_metadata()),
        ]
    )
    results = []
    failure: BaseException | None = None
    try:
        run_task_with_native_context(
            runner=runner,
            controller=controller,
            prompt=prompt,
            results=results,
        )
    except BaseException as exc:
        failure = exc
    finally:
        if results:
            manifest["claude"] = harness.result_to_manifest(results[-1])
            manifest["claude_segments"] = [
                harness.result_to_manifest(result) for result in results
            ]
            manifest["resolved_model_ids"] = sorted(
                {args.model}
                | {model for result in results for model in result.resolved_models}
            )
        manifest["finished_at"] = utc_now()
        manifest["steps_attempted"] = controller.step_index
        manifest["action_attempts"] = controller.total_attempts
        manifest["terminal"] = controller.terminal
        manifest["termination_reason"] = controller.termination_reason
        manifest["progress"] = {
            "completed": None if controller.obs is None else controller.obs.levels_completed,
            "total": None if controller.obs is None else controller.obs.win_levels,
        }
        if failure is not None:
            manifest["error"] = type(failure).__name__
            (private_dir / "error.txt").write_text(
                f"{type(failure).__name__}: {failure}\n", encoding="utf-8"
            )
        try:
            arc.close_scorecard(scorecard_id)
            manifest["scorecard_closed"] = True
        except Exception as exc:
            manifest["scorecard_closed"] = False
            manifest["scorecard_close_error"] = f"{type(exc).__name__}: {exc}"
        write_json(run_dir / "manifest.json", manifest)
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
