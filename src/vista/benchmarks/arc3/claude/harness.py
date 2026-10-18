"""Run an ARC visual-game evaluation with Claude Code."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc3_mplconfig")

import arc_agi
from arc_agi import Arcade, OperationMode

from vista.backends.claude.lifecycle import (
    MAX_COMPACT_CYCLES_WITHOUT_ACTION as MAX_COMPACT_CYCLES_WITHOUT_ACTION,
    MAX_DECISION_LEASE_RECOVERIES_PER_STEP as MAX_DECISION_LEASE_RECOVERIES_PER_STEP,
    MAX_RUNTIME_RECOVERIES_PER_STEP as MAX_RUNTIME_RECOVERIES_PER_STEP,
    NONTERMINAL_CONTINUATION_PROMPT as NONTERMINAL_CONTINUATION_PROMPT,
    PROVIDER_5XX_BACKOFF_SECONDS as PROVIDER_5XX_BACKOFF_SECONDS,
    run_task_with_native_context as run_task_with_native_context,
)
from vista.backends.claude.provider import (
    ANTHROPIC_PROVIDER,
    DEEPSEEK_PROVIDER,
    ClaudeCodeProvider,
)
from vista.backends.claude.quota import ClaudeCredentialLease, ClaudeRateLimitEvent
from vista.benchmarks.arc3.claude.controller import GameController, write_json
from vista.benchmarks.arc3.claude.recovery import TASK_OBJECTIVE
from vista.benchmarks.arc3.claude.runner import (
    CLAUDE_AUTO_COMPACT_WINDOW,
    CLAUDE_EFFORT_LEVELS,
    DEFAULT_CLAUDE_EFFORT,
    PINNED_CLAUDE_VERSION,
    RUNTIME_PREFLIGHT_PROMPT,
    ClaudeCodeRunner,
    ClaudePreflightController,
    ClaudeResult,
    claude_version,
    default_claude_bin,
    prepare_claude_config,
    resolved_model_id,
    validate_claude_runtime,
    validate_instruction_envelope,
)
from vista.benchmarks.arc3.http import configure_arc_http
from vista.benchmarks.arc3.run import (
    GameRunOutcome,
    bounded_task_timeout,
    create_run_dir,
    finalize_scorecard,
    load_run_environment,
    open_evaluation_scorecard,
    report_run_failure,
    share_online_cookie_jar,
    utc_now,
)

GRID_SIZE = 64
RENDER_SCALE = 8
DEFAULT_RENDER_GRID = True
DISPLAY_SIZE = GRID_SIZE * RENDER_SCALE
DEFAULT_MAX_STEPS = 2_000
DEFAULT_MAX_INVALID_RETRIES = 15
EMPTY_GUIDE_MODEL = "No reliable model yet."
PROVIDERS = {
    ANTHROPIC_PROVIDER.name: ANTHROPIC_PROVIDER,
    DEEPSEEK_PROVIDER.name: DEEPSEEK_PROVIDER,
}
COMPACT_RESTORE_MARKER = Path("compact_restore.pending")
COMPACT_GUIDE_SNAPSHOT = Path("compact_guide.snapshot")
COMPACT_CHECKPOINT_MARKER = Path("compact_checkpoint.requested")
COMPACT_CHECKPOINT_READY = Path("compact_checkpoint.ready")
RETRY_BOUNDARY_MARKER = Path("retry_boundary.pending")
WORKING_MEMORY_FILE = Path("WORKING.md")
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


def build_initial_prompt(metadata: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Current observation:",
            json.dumps(metadata, separators=(",", ":")),
            "",
            TASK_OBJECTIVE,
        ]
    )


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Run output directory (default: ./runs in the current working directory)",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--operation-mode", choices=["online", "offline"], default="online"
    )
    parser.add_argument(
        "--provider",
        choices=("auto", *PROVIDERS),
        default=None,
        help="Model provider (default: auto-detect from the environment)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model alias or full model ID (default: provider default)",
    )
    parser.add_argument(
        "--effort",
        choices=CLAUDE_EFFORT_LEVELS,
        default=None,
        help="Reasoning effort (default: provider default)",
    )
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


def resolve_provider(
    args: argparse.Namespace,
) -> tuple[ClaudeCodeProvider, str, str | None, str, str]:
    name = (
        getattr(args, "provider", None)
        or os.getenv("VISTA_CLAUDE_PROVIDER")
        or "auto"
    ).strip().lower()
    if name == "auto":
        if os.getenv("ANTHROPIC_BASE_URL") and os.getenv("ANTHROPIC_AUTH_TOKEN"):
            name = DEEPSEEK_PROVIDER.name
        else:
            name = ANTHROPIC_PROVIDER.name
    provider = PROVIDERS.get(name)
    if provider is None:
        raise ValueError(f"Unknown Claude provider {name!r}")
    model = getattr(args, "model", None) or provider.default_model
    if not provider.supports_model(model):
        raise ValueError(f"Unsupported {provider.name} model: {model}")
    effort = (
        getattr(args, "effort", None)
        or provider.default_effort
        or DEFAULT_CLAUDE_EFFORT
    )
    credential = os.getenv(provider.credential_env)
    if not credential:
        raise RuntimeError(
            f"{provider.credential_env} is not set for the {provider.name} provider."
        )
    base_url = None
    if not provider.native:
        base_url = os.getenv("ANTHROPIC_BASE_URL") or provider.default_base_url
        if not base_url:
            raise RuntimeError(
                f"The {provider.name} provider requires an Anthropic-compatible "
                "base URL."
            )
    return provider, credential, base_url, model, effort


def main(argv: Sequence[str] | None = None) -> int:
    load_run_environment()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.operation_mode != "offline" and not os.getenv("ARC_API_KEY"):
        print(
            "ARC_API_KEY is not set; using the ARC anonymous API key.",
            file=sys.stderr,
            flush=True,
        )
    try:
        _, _, _, model, effort = resolve_provider(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    args.model = model
    args.effort = effort
    if args.max_steps < 1:
        parser.error("--max-steps must be >= 1")
    if args.timeout < 1:
        parser.error("--timeout must be >= 1")
    if args.max_invalid_retries < 0:
        parser.error("--max-invalid-retries must be >= 0")

    outcome = run_game(args)
    print(f"Run directory: {outcome.run_dir}")
    print(f"Scorecard id: {outcome.scorecard_id}")
    print(f"Claude session id: {outcome.session_id}")
    if outcome.failure is not None:
        report_run_failure(outcome.run_dir, "claude")
        raise outcome.failure
    return 0


def run_game(
    args: argparse.Namespace,
    *,
    shared_arc: Arcade | None = None,
    shared_scorecard_id: str | None = None,
    preflight_model_id: str | None = None,
    oauth_token: str | None = None,
    credential_lease: ClaudeCredentialLease | None = None,
    run_dir: Path | None = None,
    environment_factory: Any | None = None,
) -> GameRunOutcome:
    if (shared_arc is None) != (shared_scorecard_id is None):
        raise ValueError("shared_arc and shared_scorecard_id must be provided together")
    if preflight_model_id is not None and shared_arc is None:
        raise ValueError(
            "preflight_model_id requires a coordinator-owned Arcade and scorecard"
        )

    args = argparse.Namespace(**vars(args))
    provider, resolved_credential, provider_base_url, model, effort = resolve_provider(
        args
    )
    args.model = model
    args.effort = effort
    if provider.native:
        provider_credential = None
    else:
        provider_credential = resolved_credential
        if credential_lease is not None:
            raise ValueError(
                "Claude credential leases are only defined for the Anthropic provider"
            )
        if oauth_token is not None:
            raise ValueError(
                "oauth_token is only defined for the Anthropic provider"
            )
    if credential_lease is not None:
        if oauth_token is not None and oauth_token != credential_lease.oauth_token:
            raise ValueError("Claude credential lease and OAuth token do not match")
        oauth_token = credential_lease.oauth_token
    elif provider.native and oauth_token is None:
        oauth_token = resolved_credential
    if run_dir is None:
        run_dir = create_run_dir(getattr(args, "runs_dir", Path("runs")))
    else:
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
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
        recordings_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    private_dir.chmod(0o700)
    prepare_claude_config(config_dir)

    write_player_agents(
        Path(__file__).with_name("prompt.md"),
        visible_dir / "AGENTS.md",
        describe_changes=args.describe_changes,
    )
    write_guide(visible_dir / "GUIDE.md", current_model=EMPTY_GUIDE_MODEL)
    manifest = base_manifest(args, run_dir, shared_scorecard_id)
    write_json(run_dir / "manifest.json", manifest)

    arc: Arcade | None = shared_arc
    scorecard_id: str | None = shared_scorecard_id
    owns_scorecard = shared_arc is None
    env: Any | None = None
    controller: GameController | None = None
    result: ClaudeResult | None = None
    results: list[ClaudeResult] = []
    concrete_model_id: str | None = None
    failure: BaseException | None = None

    try:
        claude_bin = validate_claude_runtime(
            default_claude_bin(),
            oauth_token=oauth_token,
            provider=provider,
            provider_credential=provider_credential,
        )
        if preflight_model_id is None:
            _, concrete_model_id = run_runtime_preflight(
                visible_dir=visible_dir,
                config_dir=config_dir / "preflight",
                io_dir=io_dir / "preflight",
                claude_bin=claude_bin,
                model=args.model,
                effort=args.effort,
                timeout=args.timeout,
                oauth_token=oauth_token,
                provider=provider,
                provider_credential=provider_credential,
                provider_base_url=provider_base_url,
            )
        else:
            concrete_model_id = preflight_model_id
        manifest["resolved_model_id"] = concrete_model_id
        manifest["resolved_model_ids"] = [concrete_model_id]
        write_json(run_dir / "manifest.json", manifest)
        if arc is None:
            os.environ["OPERATION_MODE"] = args.operation_mode
            arc = Arcade(
                operation_mode=OperationMode(args.operation_mode),
                recordings_dir=str(recordings_dir),
            )
            configure_arc_http(arc)
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
            configure_arc_http(env)
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
            compact_restore_marker=io_dir / COMPACT_RESTORE_MARKER,
            compact_guide_snapshot=io_dir / COMPACT_GUIDE_SNAPSHOT,
            compact_checkpoint_marker=io_dir / COMPACT_CHECKPOINT_MARKER,
            compact_checkpoint_ready=io_dir / COMPACT_CHECKPOINT_READY,
            retry_boundary_marker=private_dir / RETRY_BOUNDARY_MARKER,
            reset_starts_fresh_session=False,
            working_path=visible_dir / WORKING_MEMORY_FILE,
            guide_path=visible_dir / "GUIDE.md",
            agent_name="claude-code",
            interface_name="native-mcp-tool-loop",
        )
        maximum_calls = args.max_steps * (args.max_invalid_retries + 1) + 1
        runner = ClaudeCodeRunner(
            visible_dir=visible_dir,
            claude_config_dir=config_dir,
            io_dir=io_dir,
            controller=controller,
            claude_bin=claude_bin,
            model=args.model,
            effort=args.effort,
            expected_model_id=concrete_model_id,
            oauth_token=oauth_token,
            provider=provider,
            provider_credential=provider_credential,
            provider_base_url=provider_base_url,
            auto_compact_window=(
                provider.auto_compact_window
                if provider.auto_compact_window is not None
                else CLAUDE_AUTO_COMPACT_WINDOW
            ),
            task_timeout=bounded_task_timeout(args.timeout, maximum_calls),
            display_size=DISPLAY_SIZE,
            rate_limit_observer=(
                credential_lease.observe_rate_limit
                if credential_lease is not None
                else None
            ),
            credential_yield_requested=(
                credential_lease.should_yield if credential_lease is not None else None
            ),
        )
        run_task_with_native_context(
            runner=runner,
            controller=controller,
            prompt=build_initial_prompt(controller.initial_metadata()),
            results=results,
            credential_relay=(
                credential_lease.relay if credential_lease is not None else None
            ),
            credential_relay_needed=(
                credential_lease.should_yield if credential_lease is not None else None
            ),
        )
        result = results[-1]
        manifest["claude"] = result_to_manifest(result)
        manifest["claude_segments"] = [
            result_to_manifest(segment) for segment in results
        ]
        manifest["resolved_model_ids"] = sorted(
            {concrete_model_id}
            | {model for segment in results for model in segment.resolved_models}
        )
        if result.returncode != 0 and not controller.terminal:
            raise RuntimeError(
                f"Claude task failed with exit code {result.returncode}; "
                f"stderr: {result.stderr_path}"
            )
        if controller.termination_reason in INFRASTRUCTURE_TERMINATION_REASONS:
            raise RuntimeError(
                "Game stopped after an infrastructure failure: "
                f"{controller.termination_reason}"
            )
        if not controller.terminal:
            controller.forced_termination_reason = "agent_stopped_early"
            raise RuntimeError("Claude stopped before the game or action budget ended")
    except BaseException as exc:
        if result is None and results:
            result = results[-1]
            manifest["claude"] = result_to_manifest(result)
            manifest["claude_segments"] = [
                result_to_manifest(segment) for segment in results
            ]
            manifest["resolved_model_ids"] = sorted(
                ({concrete_model_id} if concrete_model_id is not None else set())
                | {model for segment in results for model in segment.resolved_models}
            )
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


def run_runtime_preflight(
    *,
    visible_dir: Path,
    config_dir: Path,
    io_dir: Path,
    claude_bin: Path,
    model: str,
    effort: str,
    timeout: int,
    oauth_token: str | None = None,
    rate_limit_observer: Callable[[ClaudeRateLimitEvent], None] | None = None,
    provider: ClaudeCodeProvider = ANTHROPIC_PROVIDER,
    provider_credential: str | None = None,
    provider_base_url: str | None = None,
) -> tuple[ClaudeResult, str]:
    runner = ClaudeCodeRunner(
        visible_dir=visible_dir,
        claude_config_dir=config_dir,
        io_dir=io_dir,
        controller=ClaudePreflightController(),
        claude_bin=claude_bin,
        model=model,
        effort=effort,
        oauth_token=oauth_token,
        provider=provider,
        provider_credential=provider_credential,
        provider_base_url=provider_base_url,
        auto_compact_window=(
            provider.auto_compact_window
            if provider.auto_compact_window is not None
            else CLAUDE_AUTO_COMPACT_WINDOW
        ),
        task_timeout=max(60, timeout),
        display_size=DISPLAY_SIZE,
        rate_limit_observer=rate_limit_observer,
    )
    result = runner.run_task(RUNTIME_PREFLIGHT_PROMPT)
    if result.returncode != 0:
        raise RuntimeError(
            f"Claude runtime preflight failed; stderr: {result.stderr_path}"
        )
    validate_instruction_envelope(result.init_envelope, display_size=DISPLAY_SIZE)
    concrete_model_id = resolved_model_id(result)
    if result.final_message.strip() != "READY":
        raise RuntimeError("Claude runtime preflight returned an unexpected result.")
    return result, concrete_model_id


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
        "runtime": "claude",
        "requested_model": args.model,
        "reasoning_effort": args.effort,
        "resolved_model_ids": [],
        "describe_visual_changes": args.describe_changes,
        "observation_mode": args.observation_mode,
        "render_grid": args.render_grid,
        "max_steps": args.max_steps,
        "max_invalid_retries": args.max_invalid_retries,
        "claude_code_version": _safe_claude_version(),
    }


def result_to_manifest(result: ClaudeResult) -> dict[str, Any]:
    usage = None
    if result.usage is not None:
        usage = {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "cache_creation_input_tokens": result.usage.cache_creation_input_tokens,
            "cache_read_input_tokens": result.usage.cache_read_input_tokens,
            "cost_usd": result.usage.cost_usd,
            "num_turns": result.usage.num_turns,
        }
    return {
        "segment": result.segment,
        "returncode": result.returncode,
        "api_error_status": result.api_error_status,
        "resolved_models": list(result.resolved_models),
        "usage": usage,
    }


def _safe_claude_version() -> str | None:
    try:
        return claude_version(default_claude_bin())
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
