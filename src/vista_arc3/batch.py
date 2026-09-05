"""Run every available ARC environment on one shared scorecard."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from arc_agi import Arcade, OperationMode
from dotenv import load_dotenv

from .claude.harness import (
    run_game as run_claude_game,
    run_runtime_preflight as probe_claude_runtime,
)
from .claude.quota import (
    ClaudeCredentialLease,
    ClaudeCredentialPool,
    ClaudeRateLimitEvent,
)
from .claude.runner import (
    CLAUDE_EFFORT_LEVELS,
    DEFAULT_CLAUDE_MODEL,
    default_claude_bin,
    validate_claude_runtime,
)
from .codex.harness import (
    CODEX_EFFORT_LEVELS,
    DEFAULT_MAX_INVALID_RETRIES,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL,
    resolve_model_spec,
    run_game as run_codex_game,
    write_compact_restore_hook,
)
from .codex.runner import (
    DockerCodexRunner,
    default_codex_auth_file,
    default_native_codex_bin,
    prepare_codex_home,
    resolve_compact_thresholds,
    validate_codex_auth_file,
)
from .shared.http import configure_arc_http as configure_competition_http
from .shared.io import write_json
from .shared.run import (
    GameRunOutcome,
    finalize_scorecard,
    open_evaluation_scorecard,
    share_online_cookie_jar,
    utc_now,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = Path(
    os.getenv(
        "ARC3_COMPETITION_RUNS_DIR",
        str(ROOT / "runs" / "competition"),
    )
)
DEFAULT_CONCURRENCY = 2
DEFAULT_CLAUDE_CONCURRENCY = 2
DEFAULT_CLAUDE_WORKERS_PER_ACCOUNT = 2
DEFAULT_CLAUDE_COMPETITION_EFFORT = "xhigh"
SCHEDULER_POLL_SECONDS = 5.0
DEFAULT_RUNTIME = "codex"
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


@dataclass(frozen=True)
class CompetitionConfig:
    model: str
    max_steps: int
    concurrency: int
    timeout: int
    max_invalid_retries: int
    batch_id: str
    runs_dir: Path
    operation_mode: str = "competition"
    runtime: str = DEFAULT_RUNTIME
    effort: str | None = None
    claude_workers_per_account: int = DEFAULT_CLAUDE_WORKERS_PER_ACCOUNT


@dataclass(frozen=True)
class CompetitionGameResult:
    game_id: str
    error: str | None
    finished_at: str


@dataclass(frozen=True)
class ClaudeAccountSchedule:
    account_games: tuple[tuple[str, ...], ...]


class RuntimeProbeController:
    compact_restore_marker = None
    compact_checkpoint_marker = None
    retry_boundary_marker = None
    forced_termination_reason = None

    @staticmethod
    def initial_image_paths() -> tuple[Path, ...]:
        return ()

    @staticmethod
    def handle(request: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": False,
            "metadata": {"error": "Tools are unavailable during runtime preflight."},
        }


def run_codex_runtime_preflight(
    config: CompetitionConfig,
    auth_file: Path,
) -> None:
    """Verify the exact container, account, model, and app-server path."""
    model, reasoning_effort = resolve_model_spec(config.model, config.effort)
    with tempfile.TemporaryDirectory(prefix="arc3_codex_preflight_") as raw_root:
        root = Path(raw_root)
        visible_dir = root / "visible"
        io_dir = root / "io"
        codex_home = prepare_codex_home(root / "codex_home")
        (visible_dir / "screenshots").mkdir(parents=True)
        (visible_dir / "AGENTS.md").write_text(
            "Reply exactly READY.\n",
            encoding="utf-8",
        )
        (visible_dir / "GUIDE.md").write_text("\n", encoding="utf-8")
        write_compact_restore_hook(visible_dir, codex_home)
        runner = DockerCodexRunner(
            visible_dir=visible_dir,
            codex_home=codex_home,
            auth_file=auth_file,
            io_dir=io_dir,
            controller=RuntimeProbeController(),
            codex_bin=default_native_codex_bin(),
            model=model,
            reasoning_effort=reasoning_effort,
            compact_thresholds=resolve_compact_thresholds(model),
            task_timeout=config.timeout,
        )
        result = runner.run_task("Reply exactly READY.")
        if result.returncode != 0 or result.final_message.strip() != "READY":
            detail = result.stderr_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()
            raise RuntimeError(
                "Codex runtime preflight failed"
                + (f": {detail}" if detail else "")
            )
        if result.session_id is None:
            raise RuntimeError("Codex runtime preflight returned no session id")
        resumed = runner.resume_checkpoint(
            result.session_id,
            "Reply exactly RESUMED.",
        )
        if (
            resumed.returncode != 0
            or resumed.session_id != result.session_id
            or resumed.final_message.strip().rstrip(".") != "RESUMED"
        ):
            detail = resumed.stderr_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()
            observed = json.dumps(
                {
                    "returncode": resumed.returncode,
                    "session_preserved": resumed.session_id == result.session_id,
                    "final_message": resumed.final_message,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
            raise RuntimeError(
                f"Codex session-resume preflight failed: {observed}"
                + (f": {detail}" if detail else "")
            )


def claude_effort(config: CompetitionConfig) -> str:
    return config.effort or DEFAULT_CLAUDE_COMPETITION_EFFORT


def claude_oauth_token_pool(
    environ: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Load a contiguous, unique Claude credential pool without exposing values."""

    source = os.environ if environ is None else environ
    primary = source.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not primary:
        raise RuntimeError("CLAUDE_CODE_OAUTH_TOKEN is not set")

    numbered: dict[int, str] = {}
    pattern = re.compile(r"^CLAUDE_CODE_OAUTH_TOKEN_(\d+)$")
    for name, value in source.items():
        match = pattern.fullmatch(name)
        if match is None:
            continue
        index = int(match.group(1))
        if index < 2:
            raise RuntimeError(
                "Numbered Claude credentials must start at "
                "CLAUDE_CODE_OAUTH_TOKEN_2"
            )
        if not value:
            raise RuntimeError(f"{name} is empty")
        numbered[index] = value

    if numbered:
        expected = set(range(2, max(numbered) + 1))
        missing = sorted(expected - set(numbered))
        if missing:
            names = ", ".join(
                f"CLAUDE_CODE_OAUTH_TOKEN_{index}" for index in missing
            )
            raise RuntimeError(f"Claude credential pool has missing slots: {names}")

    tokens = (primary, *(numbered[index] for index in sorted(numbered)))
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("Claude credential pool contains duplicate tokens")
    return tokens


def claude_worker_slots(
    oauth_tokens: Sequence[str],
    *,
    workers_per_account: int,
    concurrency: int,
) -> tuple[tuple[int, str], ...]:
    if not oauth_tokens:
        raise ValueError("At least one Claude credential is required")
    if workers_per_account < 1:
        raise ValueError("Claude workers per account must be positive")
    capacity = len(oauth_tokens) * workers_per_account
    if concurrency > capacity:
        raise ValueError(
            f"Claude concurrency {concurrency} exceeds the credential pool "
            f"capacity {capacity}"
        )
    slots = tuple(
        (account_index, token)
        for _ in range(workers_per_account)
        for account_index, token in enumerate(oauth_tokens, start=1)
    )
    return slots[:concurrency]


def claude_account_schedule(
    game_ids: Sequence[str],
    *,
    account_count: int,
) -> ClaudeAccountSchedule:
    if account_count < 1:
        raise ValueError("At least one Claude account is required")
    if len(set(game_ids)) != len(game_ids):
        raise ValueError("Claude schedule contains duplicate game ids")

    account_games: list[list[str]] = [[] for _ in range(account_count)]
    for index, game_id in enumerate(game_ids):
        account_index = index % account_count
        account_games[account_index].append(game_id)

    return ClaudeAccountSchedule(
        account_games=tuple(tuple(games) for games in account_games),
    )


def pop_claude_scheduled_game(
    account_slot: int,
    queues: dict[int, deque[str]],
) -> str | None:
    own_queue = queues[account_slot]
    if own_queue:
        return own_queue.popleft()

    donors = [
        donor
        for donor, queue in queues.items()
        if donor != account_slot and queue
    ]
    if not donors:
        return None
    donor = max(
        donors,
        key=lambda candidate: (len(queues[candidate]), -candidate),
    )
    return queues[donor].popleft()


def run_claude_runtime_preflight(
    config: CompetitionConfig,
    *,
    oauth_token: str,
    rate_limit_observer: Callable[[ClaudeRateLimitEvent], None] | None = None,
) -> str:
    """Verify Claude before opening the coordinator-owned scorecard."""

    claude_bin = validate_claude_runtime(
        default_claude_bin(),
        oauth_token=oauth_token,
    )
    with tempfile.TemporaryDirectory(prefix="arc3_claude_competition_preflight_") as raw_root:
        root = Path(raw_root)
        visible_dir = root / "visible"
        (visible_dir / "screenshots").mkdir(parents=True)
        (visible_dir / "AGENTS.md").write_text(
            (Path(__file__).with_name("claude") / "prompt.md").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        (visible_dir / "GUIDE.md").write_text(
            "No reliable model yet.\n",
            encoding="utf-8",
        )
        result, concrete_model_id = probe_claude_runtime(
            visible_dir=visible_dir,
            config_dir=root / "config",
            io_dir=root / "io",
            claude_bin=claude_bin,
            model=config.model,
            effort=claude_effort(config),
            timeout=config.timeout,
            oauth_token=oauth_token,
            rate_limit_observer=rate_limit_observer,
        )
        if result.session_id is None:
            raise RuntimeError("Claude runtime preflight returned no session id")
        return concrete_model_id


def run_claude_runtime_preflights(
    config: CompetitionConfig,
    oauth_tokens: Sequence[str],
    *,
    rate_limit_observer: Callable[[int, ClaudeRateLimitEvent], None] | None = None,
) -> str:
    model_ids = []
    for account_slot, token in enumerate(oauth_tokens, start=1):
        observer = None
        if rate_limit_observer is not None:
            def observer(
                event: ClaudeRateLimitEvent,
                slot: int = account_slot,
            ) -> None:
                rate_limit_observer(slot, event)
        model_ids.append(
            run_claude_runtime_preflight(
                config,
                oauth_token=token,
                rate_limit_observer=observer,
            )
        )
    model_ids = tuple(model_ids)
    if not model_ids:
        raise RuntimeError("Claude credential pool is empty")
    if len(set(model_ids)) != 1:
        raise RuntimeError(
            "Claude account preflights resolved different concrete model IDs"
        )
    return model_ids[0]


class CompetitionEnvironmentFactory:
    """Create each environment once while preserving per-game recordings."""

    def __init__(self, arcade: Arcade) -> None:
        self.arcade = arcade
        self._lock = threading.RLock()
        self.arcade._cookie_lock = self._lock
        self._made: set[str] = set()

    @property
    def made(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._made)

    def __call__(self, game_id: str, scorecard_id: str, recordings_dir: Path) -> Any:
        with self._lock:
            if game_id in self._made:
                raise RuntimeError(f"Competition environment already made: {game_id}")
            self._made.add(game_id)
            previous_recordings_dir = self.arcade.recordings_dir
            self.arcade.recordings_dir = str(recordings_dir)
            try:
                environment = self.arcade.make(
                    game_id,
                    scorecard_id=scorecard_id,
                    save_recording=True,
                    include_frame_data=True,
                )
                if environment is None:
                    raise RuntimeError(
                        f"Competition environment could not be created: {game_id}"
                    )
                configure_competition_http(environment)
                if getattr(environment, "observation_space", None) is None:
                    raise RuntimeError(
                        f"Competition environment has no initial observation: {game_id}"
                    )
                share_online_cookie_jar(self.arcade, environment)
                return environment
            finally:
                self.arcade.recordings_dir = previous_recordings_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        choices=["codex", "claude"],
        default=DEFAULT_RUNTIME,
    )
    parser.add_argument(
        "--mode",
        "--operation-mode",
        dest="operation_mode",
        choices=["offline", "online", "competition"],
        default="competition",
    )
    parser.add_argument("--model")
    parser.add_argument("--effort", choices=CODEX_EFFORT_LEVELS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("-j", "--concurrency", type=int)
    parser.add_argument(
        "--claude-workers-per-account",
        type=int,
        default=DEFAULT_CLAUDE_WORKERS_PER_ACCOUNT,
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--max-invalid-retries",
        type=int,
        default=DEFAULT_MAX_INVALID_RETRIES,
    )
    parser.add_argument("--batch-id", default="vista")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate configuration and discover games without opening a scorecard.",
    )
    return parser


def validate_config(config: CompetitionConfig) -> None:
    if config.operation_mode not in {"offline", "online", "competition"}:
        raise ValueError(
            "--operation-mode must be offline, online, or competition"
        )
    if config.runtime not in {"codex", "claude"}:
        raise ValueError("--runtime must be codex or claude")
    if config.runtime == "codex":
        if config.effort is not None and config.effort not in CODEX_EFFORT_LEVELS:
            raise ValueError(
                "--effort must be " + ", ".join(CODEX_EFFORT_LEVELS)
            )
        resolve_model_spec(config.model, config.effort)
    if config.runtime == "claude" and claude_effort(config) not in CLAUDE_EFFORT_LEVELS:
        raise ValueError(
            "--effort must be " + ", ".join(CLAUDE_EFFORT_LEVELS)
        )
    if config.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if config.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if config.claude_workers_per_account < 1:
        raise ValueError("--claude-workers-per-account must be positive")
    if config.timeout < 1:
        raise ValueError("--timeout must be positive")
    if config.max_invalid_retries < 0:
        raise ValueError("--max-invalid-retries must be non-negative")
    if not config.batch_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in config.batch_id
    ):
        raise ValueError("--batch-id may contain only letters, numbers, hyphens, and underscores")


def config_from_args(args: argparse.Namespace) -> CompetitionConfig:
    model = args.model
    if model is None:
        model = DEFAULT_CLAUDE_MODEL if args.runtime == "claude" else DEFAULT_MODEL
    concurrency = args.concurrency
    if concurrency is None:
        concurrency = (
            DEFAULT_CLAUDE_CONCURRENCY
            if args.runtime == "claude"
            else DEFAULT_CONCURRENCY
        )
    return CompetitionConfig(
        model=model,
        max_steps=args.max_steps,
        concurrency=concurrency,
        timeout=args.timeout,
        max_invalid_retries=args.max_invalid_retries,
        batch_id=args.batch_id,
        runs_dir=args.runs_dir.expanduser().resolve(),
        operation_mode=args.operation_mode,
        runtime=args.runtime,
        effort=args.effort,
        claude_workers_per_account=args.claude_workers_per_account,
    )


def available_game_ids(arcade: Arcade) -> list[str]:
    by_base: dict[str, str] = {}
    for environment in arcade.get_environments():
        full_id = str(environment.game_id)
        base_id = full_id.split("-", 1)[0]
        if base_id in by_base and by_base[base_id] != full_id:
            raise RuntimeError(
                f"Multiple available versions for {base_id}: "
                f"{by_base[base_id]} and {full_id}"
            )
        by_base[base_id] = full_id
    if not by_base:
        raise RuntimeError("ARC API returned no available environments")

    ordered = [game for game in STANDARD_GAME_ORDER if game in by_base]
    ordered.extend(sorted(set(by_base) - set(ordered)))
    return ordered


def game_args(config: CompetitionConfig, game_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        game_id=game_id,
        max_steps=config.max_steps,
        operation_mode=config.operation_mode,
        model=config.model,
        effort=(claude_effort(config) if config.runtime == "claude" else config.effort),
        timeout=config.timeout,
        max_invalid_retries=config.max_invalid_retries,
        observation_mode="vision",
        describe_changes=False,
        render_grid=True,
    )


def create_batch_dir(config: CompetitionConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    for counter in range(10_000):
        suffix = "" if counter == 0 else f"_{counter}"
        candidate = config.runs_dir / f"{config.batch_id}_{stamp}{suffix}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        (candidate / "games").mkdir()
        return candidate
    raise RuntimeError("Could not allocate a competition batch directory")


def write_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, separators=(",", ":"))
        handle.write("\n")


def run_worker(
    *,
    config: CompetitionConfig,
    game_id: str,
    arcade: Arcade,
    scorecard_id: str,
    auth_file: Path | None,
    preflight_model_id: str | None,
    environment_factory: CompetitionEnvironmentFactory,
    batch_dir: Path,
    claude_oauth_token: str | None = None,
    claude_account_slot: int | None = None,
    claude_credential_lease: ClaudeCredentialLease | None = None,
) -> GameRunOutcome:
    common = {
        "shared_arc": arcade,
        "shared_scorecard_id": scorecard_id,
        "run_dir": batch_dir / "games" / game_id,
        "environment_factory": environment_factory,
    }
    if config.runtime == "claude":
        if preflight_model_id is None:
            raise RuntimeError("Claude worker has no coordinator preflight model ID")
        if claude_credential_lease is None and (
            claude_oauth_token is None or claude_account_slot is None
        ):
            raise RuntimeError("Claude worker has no assigned account credential")
        return run_claude_game(
            game_args(config, game_id),
            preflight_model_id=preflight_model_id,
            oauth_token=claude_oauth_token,
            credential_lease=claude_credential_lease,
            **common,
        )
    if auth_file is None:
        raise RuntimeError("Codex worker has no authentication file")
    return run_codex_game(
        game_args(config, game_id),
        auth_file=auth_file,
        **common,
    )


def run_competition(
    config: CompetitionConfig,
    *,
    arcade_factory: Callable[..., Arcade] = Arcade,
    game_runner: Callable[..., GameRunOutcome] = run_worker,
    preflight: bool = False,
) -> tuple[Path | None, int]:
    validate_config(config)
    auth_file: Path | None = None
    preflight_model_id: str | None = None
    claude_oauth_tokens: tuple[str, ...] = ()
    claude_slots: tuple[tuple[int, str], ...] = ()
    claude_schedule: ClaudeAccountSchedule | None = None
    claude_credential_pool: ClaudeCredentialPool | None = None
    if config.runtime == "codex":
        auth_file = validate_codex_auth_file(default_codex_auth_file())
    else:
        claude_oauth_tokens = claude_oauth_token_pool()
        claude_slots = claude_worker_slots(
            claude_oauth_tokens,
            workers_per_account=config.claude_workers_per_account,
            concurrency=config.concurrency,
        )
        active_accounts = {account for account, _ in claude_slots}
        expected_accounts = set(range(1, len(claude_oauth_tokens) + 1))
        if active_accounts != expected_accounts:
            raise ValueError(
                "Claude concurrency must provide at least one worker per account"
            )
        claude_credential_pool = ClaudeCredentialPool(
            claude_oauth_tokens,
            (config.claude_workers_per_account,) * len(claude_oauth_tokens),
        )

    operation_mode = OperationMode(config.operation_mode)
    if preflight and operation_mode == OperationMode.COMPETITION:
        operation_mode = OperationMode.ONLINE
    os.environ["OPERATION_MODE"] = operation_mode.value
    arcade = arcade_factory(operation_mode=operation_mode)
    configure_competition_http(arcade)
    game_ids = available_game_ids(arcade)
    missing_standard = sorted(set(STANDARD_GAME_ORDER) - set(game_ids))
    if missing_standard:
        raise RuntimeError(
            "Expected public environments are unavailable: " + ", ".join(missing_standard)
        )
    if config.runtime == "claude":
        claude_schedule = claude_account_schedule(
            game_ids,
            account_count=len(claude_oauth_tokens),
        )

    print(f"Available games ({len(game_ids)}): {' '.join(game_ids)}")
    print(
        f"Runtime: {config.runtime}, {config.model}"
        + (
            f"/{claude_effort(config)}"
            if config.runtime == "claude"
            else ""
        )
        + f", {config.max_steps} actions, "
        f"{config.concurrency} workers"
    )
    if config.runtime == "claude":
        assert claude_credential_pool is not None
        preflight_model_id = run_claude_runtime_preflights(
            config,
            claude_oauth_tokens,
            rate_limit_observer=(
                claude_credential_pool.observe_account_rate_limit
            ),
        )
        if not claude_credential_pool.healthy_account_slots():
            raise RuntimeError(
                "Every Claude account is at or above the protected quota threshold"
            )
        print(
            "Claude container runtime: ready "
            f"({preflight_model_id}; {len(claude_oauth_tokens)} accounts; "
            f"{config.concurrency} total worker slots)"
        )
        assert claude_schedule is not None
        for account, games in enumerate(claude_schedule.account_games, start=1):
            print(f"Claude account {account}: {len(games)} planned games")
    if preflight:
        if config.runtime == "codex":
            assert auth_file is not None
            run_codex_runtime_preflight(config, auth_file)
            print("Codex container runtime: ready")
        print("Preflight passed; no scorecard was opened.")
        return None, 0

    batch_dir = create_batch_dir(config)
    batch_manifest: dict[str, Any] = {
        "created_at": utc_now(),
        "finished_at": None,
        "operation_mode": config.operation_mode,
        "scorecard_id": None,
        "scorecard_closed": False,
        "runtime": config.runtime,
        "model": config.model,
        "reasoning_effort": (
            claude_effort(config) if config.runtime == "claude" else None
        ),
        "resolved_model_id": preflight_model_id,
        "max_steps_per_game": config.max_steps,
        "concurrency": config.concurrency,
        "game_ids": game_ids,
        "runtime_version": os.getenv("ARC3_PINNED_RUNTIME_VERSION"),
        "results": {},
    }
    if config.runtime == "claude":
        assert claude_credential_pool is not None
        assert claude_schedule is not None
        batch_manifest.update(
            {
                "claude_account_count": len(claude_oauth_tokens),
                "claude_workers_per_account": config.claude_workers_per_account,
            }
        )
    write_json(batch_dir / "manifest.json", batch_manifest)

    scorecard_id: str | None = None
    close_failure: BaseException | None = None
    game_results: dict[str, CompetitionGameResult] = {}
    try:
        scorecard_id = open_evaluation_scorecard(arcade)
        batch_manifest["scorecard_id"] = scorecard_id
        write_json(batch_dir / "manifest.json", batch_manifest)
        print(f"{config.operation_mode.title()} scorecard opened: {scorecard_id}")
        print(f"Batch directory: {batch_dir}")
        environment_factory = CompetitionEnvironmentFactory(arcade)

        def execute_game(
            game_id: str,
            account_slot: int | None = None,
            oauth_token: str | None = None,
            credential_lease: ClaudeCredentialLease | None = None,
        ) -> CompetitionGameResult:
            try:
                outcome = game_runner(
                    config=config,
                    game_id=game_id,
                    arcade=arcade,
                    scorecard_id=scorecard_id,
                    auth_file=auth_file,
                    preflight_model_id=preflight_model_id,
                    environment_factory=environment_factory,
                    batch_dir=batch_dir,
                    claude_oauth_token=oauth_token,
                    claude_account_slot=account_slot,
                    claude_credential_lease=credential_lease,
                )
            except BaseException as exc:
                return CompetitionGameResult(
                    game_id=game_id,
                    error=type(exc).__name__,
                    finished_at=utc_now(),
                )
            return CompetitionGameResult(
                game_id=game_id,
                error=(
                    type(outcome.failure).__name__
                    if outcome.failure is not None
                    else None
                ),
                finished_at=utc_now(),
            )

        def record_result(result: CompetitionGameResult) -> None:
            game_results[result.game_id] = result
            batch_manifest["results"] = {
                key: asdict(value) for key, value in sorted(game_results.items())
            }
            write_jsonl(batch_dir / "results.jsonl", asdict(result))
            write_json(batch_dir / "manifest.json", batch_manifest)
            status = "failed" if result.error else "finished"
            print(
                f"{result.game_id}: {status} "
                f"({len(game_results)}/{len(game_ids)})"
            )

        with ThreadPoolExecutor(
            max_workers=config.concurrency,
            thread_name_prefix="arc3-game",
        ) as executor:
            if config.runtime == "claude":
                assert claude_schedule is not None
                assert claude_credential_pool is not None
                account_queues = {
                    account: deque(games)
                    for account, games in enumerate(
                        claude_schedule.account_games,
                        start=1,
                    )
                }
                account_slots = tuple(account_queues)
                futures: dict[
                    Future[CompetitionGameResult], ClaudeCredentialLease
                ] = {}

                def submit_next(account_slot: int) -> bool:
                    credential_lease = claude_credential_pool.try_acquire(
                        account_slot
                    )
                    if credential_lease is None:
                        return False
                    game_id = pop_claude_scheduled_game(
                        account_slot,
                        account_queues,
                    )
                    if game_id is None:
                        credential_lease.release()
                        return False
                    future = executor.submit(
                        execute_game,
                        game_id,
                        account_slot,
                        credential_lease.oauth_token,
                        credential_lease,
                    )
                    futures[future] = credential_lease
                    return True

                while any(account_queues.values()) or futures:
                    submitted = True
                    while submitted and len(futures) < config.concurrency:
                        submitted = False
                        for account_slot in account_slots:
                            if len(futures) >= config.concurrency:
                                break
                            submitted = submit_next(account_slot) or submitted

                    if not futures:
                        time.sleep(SCHEDULER_POLL_SECONDS)
                        continue
                    completed, _ = wait(
                        tuple(futures),
                        timeout=SCHEDULER_POLL_SECONDS,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in completed:
                        credential_lease = futures.pop(future)
                        try:
                            result = future.result()
                        finally:
                            credential_lease.release()
                        record_result(result)
            else:
                codex_futures = {
                    executor.submit(execute_game, game_id): game_id
                    for game_id in game_ids
                }
                for future in as_completed(codex_futures):
                    record_result(future.result())

        batch_manifest["made_game_ids"] = sorted(environment_factory.made)
        if environment_factory.made != frozenset(game_ids):
            batch_manifest["make_invariant_error"] = {
                "missing": sorted(set(game_ids) - set(environment_factory.made)),
                "unexpected": sorted(set(environment_factory.made) - set(game_ids)),
            }
    finally:
        if scorecard_id is not None:
            scorecard_result, close_failure = finalize_scorecard(
                arc=arcade,
                env=None,
                scorecard_id=scorecard_id,
                operation_mode=config.operation_mode,
            )
            batch_manifest.update(scorecard_result)
        batch_manifest["finished_at"] = utc_now()
        if close_failure is not None:
            batch_manifest["scorecard_close_failure"] = type(close_failure).__name__
        write_json(batch_dir / "manifest.json", batch_manifest)

    failures = sum(result.error is not None for result in game_results.values())
    if len(game_results) != len(game_ids):
        failures += len(game_ids) - len(game_results)
    if close_failure is not None:
        failures += 1
    print(f"Batch complete: {len(game_results)}/{len(game_ids)} games")
    print(f"Scorecard closed: {batch_manifest.get('scorecard_closed', False)}")
    return batch_dir, 0 if failures == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env_file = Path(os.getenv("ARC3_ENV_FILE", ROOT / ".env"))
    load_dotenv(env_file, override=True)
    if args.operation_mode != "offline" and not os.getenv("ARC_API_KEY"):
        raise RuntimeError(f"ARC_API_KEY is absent from {env_file}")
    _, status = run_competition(config_from_args(args), preflight=args.preflight)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
