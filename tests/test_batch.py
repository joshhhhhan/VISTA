from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest
from arc_agi import OperationMode
from requests import Session
from requests.adapters import HTTPAdapter

from vista_arc3 import batch as competition
from vista_arc3.claude.quota import ClaudeRateLimitEvent
from vista_arc3.shared.http import (
    ARC_CONNECT_RETRIES,
    ARC_CONNECT_TIMEOUT_SECONDS,
    ARC_READ_TIMEOUT_SECONDS,
    ArcHTTPAdapter,
)
from vista_arc3.shared.run import GameRunOutcome


class FakeArcade:
    def __init__(self, game_ids: tuple[str, ...]) -> None:
        self.recordings_dir = "recordings"
        self.environments = [SimpleNamespace(game_id=game_id) for game_id in game_ids]
        self.open_calls = 0
        self.close_calls = 0
        self.make_calls: list[tuple[str, str, str]] = []

    def get_environments(self):
        return self.environments

    def open_scorecard(self, **kwargs):
        self.open_calls += 1
        self.open_kwargs = kwargs
        return "competition-card"

    def make(self, game_id, *, scorecard_id, save_recording, include_frame_data):
        assert save_recording is True
        assert include_frame_data is True
        self.make_calls.append((game_id, scorecard_id, self.recordings_dir))
        return SimpleNamespace(game_id=game_id, observation_space=object())

    def get_scorecard(self, scorecard_id):  # pragma: no cover - safety tripwire
        raise AssertionError("competition scorecards must not be queried in flight")

    def close_scorecard(self, scorecard_id):
        self.close_calls += 1
        return {"card_id": scorecard_id, "closed": True}


def config(tmp_path: Path) -> competition.CompetitionConfig:
    return competition.CompetitionConfig(
        model="gpt-5.6-sol-max",
        max_steps=2_000,
        concurrency=8,
        timeout=300,
        max_invalid_retries=2,
        batch_id="test",
        runs_dir=tmp_path / "runs",
    )


def claude_config(tmp_path: Path) -> competition.CompetitionConfig:
    return replace(
        config(tmp_path),
        runtime="claude",
        model="opus",
        effort="xhigh",
        concurrency=6,
        batch_id="claude-test",
    )


def install_fake_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    auth = tmp_path / "auth.json"
    auth.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(competition, "default_codex_auth_file", lambda: auth)
    return auth


def install_fake_claude_tokens(
    monkeypatch: pytest.MonkeyPatch,
    count: int = 3,
) -> tuple[str, ...]:
    for name in tuple(competition.os.environ):
        if name.startswith("CLAUDE_CODE_OAUTH_TOKEN"):
            monkeypatch.delenv(name, raising=False)
    tokens = tuple(f"claude-token-{index}" for index in range(1, count + 1))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", tokens[0])
    for index, token in enumerate(tokens[1:], start=2):
        monkeypatch.setenv(f"CLAUDE_CODE_OAUTH_TOKEN_{index}", token)
    return tokens


def test_competition_owns_one_scorecard_and_makes_each_game_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_auth(monkeypatch, tmp_path)
    fake = FakeArcade(competition.STANDARD_GAME_ORDER)

    def fake_runner(**kwargs):
        game_id = kwargs["game_id"]
        run_dir = kwargs["batch_dir"] / "games" / game_id
        run_dir.mkdir()
        kwargs["environment_factory"](
            game_id,
            kwargs["scorecard_id"],
            run_dir / "private" / "recordings",
        )
        return GameRunOutcome(
            run_dir=run_dir,
            scorecard_id=kwargs["scorecard_id"],
            session_id=f"session-{game_id}",
            failure=None,
        )

    batch_dir, status = competition.run_competition(
        config(tmp_path),
        arcade_factory=lambda **kwargs: fake,
        game_runner=fake_runner,
    )

    assert status == 0
    assert batch_dir is not None
    assert fake.open_calls == 1
    assert fake.close_calls == 1
    assert len(fake.make_calls) == len(competition.STANDARD_GAME_ORDER)
    assert {call[0] for call in fake.make_calls} == set(competition.STANDARD_GAME_ORDER)
    assert len({call[0] for call in fake.make_calls}) == len(fake.make_calls)
    assert {call[1] for call in fake.make_calls} == {"competition-card"}
    assert all(call[2].endswith("private/recordings") for call in fake.make_calls)
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "codex_auth_file" not in manifest


def test_online_batch_owns_one_shared_scorecard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_auth(monkeypatch, tmp_path)
    fake = FakeArcade(competition.STANDARD_GAME_ORDER)
    fake.get_scorecard = lambda scorecard_id: {"card_id": scorecard_id}
    observed_modes: list[OperationMode] = []
    worker_modes: list[str] = []

    def factory(*, operation_mode):
        observed_modes.append(operation_mode)
        return fake

    def fake_runner(**kwargs):
        game_id = kwargs["game_id"]
        worker_modes.append(
            competition.game_args(kwargs["config"], game_id).operation_mode
        )
        run_dir = kwargs["batch_dir"] / "games" / game_id
        run_dir.mkdir()
        kwargs["environment_factory"](
            game_id,
            kwargs["scorecard_id"],
            run_dir / "private" / "recordings",
        )
        return GameRunOutcome(
            run_dir=run_dir,
            scorecard_id=kwargs["scorecard_id"],
            session_id=f"session-{game_id}",
            failure=None,
        )

    batch_dir, status = competition.run_competition(
        replace(config(tmp_path), operation_mode="online"),
        arcade_factory=factory,
        game_runner=fake_runner,
    )

    assert status == 0
    assert batch_dir is not None
    assert observed_modes == [OperationMode.ONLINE]
    assert worker_modes == ["online"] * len(competition.STANDARD_GAME_ORDER)
    assert fake.open_calls == 1
    assert fake.close_calls == 1
    manifest = json.loads(
        (batch_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["operation_mode"] == "online"


def test_offline_batch_owns_one_shared_local_scorecard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_auth(monkeypatch, tmp_path)
    fake = FakeArcade(competition.STANDARD_GAME_ORDER)
    observed_modes: list[OperationMode] = []
    worker_modes: list[str] = []

    def factory(*, operation_mode):
        observed_modes.append(operation_mode)
        return fake

    def fake_runner(**kwargs):
        game_id = kwargs["game_id"]
        worker_modes.append(competition.game_args(kwargs["config"], game_id).operation_mode)
        run_dir = kwargs["batch_dir"] / "games" / game_id
        run_dir.mkdir()
        kwargs["environment_factory"](
            game_id,
            kwargs["scorecard_id"],
            run_dir / "private" / "recordings",
        )
        return GameRunOutcome(
            run_dir=run_dir,
            scorecard_id=kwargs["scorecard_id"],
            session_id=f"session-{game_id}",
            failure=None,
        )

    batch_dir, status = competition.run_competition(
        replace(config(tmp_path), operation_mode="offline"),
        arcade_factory=factory,
        game_runner=fake_runner,
    )

    assert status == 0
    assert batch_dir is not None
    assert observed_modes == [OperationMode.OFFLINE]
    assert worker_modes == ["offline"] * len(competition.STANDARD_GAME_ORDER)
    assert fake.open_calls == 1
    assert fake.close_calls == 1
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["operation_mode"] == "offline"


def test_preflight_never_opens_scorecard_or_makes_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_auth(monkeypatch, tmp_path)
    fake = FakeArcade(competition.STANDARD_GAME_ORDER)
    observed_modes: list[OperationMode] = []
    runtime_checks: list[tuple[competition.CompetitionConfig, Path]] = []

    def factory(*, operation_mode):
        observed_modes.append(operation_mode)
        return fake

    monkeypatch.setattr(
        competition,
        "run_codex_runtime_preflight",
        lambda run_config, auth_file: runtime_checks.append((run_config, auth_file)),
    )

    batch_dir, status = competition.run_competition(
        config(tmp_path),
        arcade_factory=factory,
        preflight=True,
    )

    assert status == 0
    assert batch_dir is None
    assert observed_modes == [OperationMode.ONLINE]
    assert fake.open_calls == 0
    assert fake.make_calls == []
    assert fake.close_calls == 0
    assert len(runtime_checks) == 1


def test_offline_preflight_stays_offline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_auth(monkeypatch, tmp_path)
    fake = FakeArcade(competition.STANDARD_GAME_ORDER)
    observed_modes: list[OperationMode] = []

    def factory(*, operation_mode):
        observed_modes.append(operation_mode)
        return fake

    monkeypatch.setattr(
        competition,
        "run_codex_runtime_preflight",
        lambda *_: None,
    )

    batch_dir, status = competition.run_competition(
        replace(config(tmp_path), operation_mode="offline"),
        arcade_factory=factory,
        preflight=True,
    )

    assert status == 0
    assert batch_dir is None
    assert observed_modes == [OperationMode.OFFLINE]
    assert fake.open_calls == 0
    assert fake.make_calls == []
    assert fake.close_calls == 0


def test_claude_preflight_never_uses_codex_auth_or_opens_scorecard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokens = install_fake_claude_tokens(monkeypatch)
    fake = FakeArcade(competition.STANDARD_GAME_ORDER)
    runtime_checks: list[tuple[competition.CompetitionConfig, str]] = []
    monkeypatch.setattr(
        competition,
        "default_codex_auth_file",
        lambda: (_ for _ in ()).throw(AssertionError("Codex auth was accessed")),
    )
    monkeypatch.setattr(
        competition,
        "run_claude_runtime_preflight",
        lambda run_config, *, oauth_token, rate_limit_observer=None: (
            runtime_checks.append((run_config, oauth_token))
        )
        or "claude-opus-5",
    )

    batch_dir, status = competition.run_competition(
        claude_config(tmp_path),
        arcade_factory=lambda **kwargs: fake,
        preflight=True,
    )

    assert status == 0
    assert batch_dir is None
    assert runtime_checks == [
        (claude_config(tmp_path), token) for token in tokens
    ]
    assert fake.open_calls == 0
    assert fake.make_calls == []


def test_claude_competition_reuses_coordinator_preflight_for_every_game(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokens = install_fake_claude_tokens(monkeypatch)
    fake = FakeArcade(competition.STANDARD_GAME_ORDER)
    preflight_calls: list[tuple[competition.CompetitionConfig, str]] = []
    worker_models: list[str | None] = []
    worker_accounts: list[int | None] = []
    worker_tokens: list[str | None] = []
    worker_games: list[tuple[str, int | None]] = []

    def fake_preflight(run_config, *, oauth_token, rate_limit_observer=None):
        assert fake.open_calls == 0
        preflight_calls.append((run_config, oauth_token))
        return "claude-opus-5"

    monkeypatch.setattr(competition, "run_claude_runtime_preflight", fake_preflight)

    def fake_runner(**kwargs):
        assert kwargs["auth_file"] is None
        worker_models.append(kwargs["preflight_model_id"])
        worker_accounts.append(kwargs["claude_account_slot"])
        worker_tokens.append(kwargs["claude_oauth_token"])
        game_id = kwargs["game_id"]
        worker_games.append((game_id, kwargs["claude_account_slot"]))
        run_dir = kwargs["batch_dir"] / "games" / game_id
        run_dir.mkdir()
        kwargs["environment_factory"](
            game_id,
            kwargs["scorecard_id"],
            run_dir / "private" / "recordings",
        )
        return GameRunOutcome(
            run_dir=run_dir,
            scorecard_id=kwargs["scorecard_id"],
            session_id=f"claude-{game_id}",
            failure=None,
        )

    batch_dir, status = competition.run_competition(
        claude_config(tmp_path),
        arcade_factory=lambda **kwargs: fake,
        game_runner=fake_runner,
    )

    assert status == 0
    assert batch_dir is not None
    assert preflight_calls == [
        (claude_config(tmp_path), token) for token in tokens
    ]
    assert worker_models == ["claude-opus-5"] * len(competition.STANDARD_GAME_ORDER)
    assert set(worker_accounts) == {1, 2, 3}
    assert set(worker_tokens) == set(tokens)
    assert all(
        token == tokens[account - 1]
        for account, token in zip(worker_accounts, worker_tokens, strict=True)
        if account is not None and token is not None
    )
    assert fake.open_calls == 1
    assert fake.close_calls == 1
    assert len(fake.make_calls) == len(competition.STANDARD_GAME_ORDER)
    assert fake.open_kwargs == {"tags": []}
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime"] == "claude"
    assert manifest["resolved_model_id"] == "claude-opus-5"
    assert manifest["claude_account_count"] == 3
    assert manifest["claude_workers_per_account"] == 2
    assert "claude-token" not in json.dumps(manifest)


def test_preflight_quota_protection_runs_before_scorecard_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_claude_tokens(monkeypatch)
    fake = FakeArcade(competition.STANDARD_GAME_ORDER)

    def fake_preflight(
        run_config,
        *,
        oauth_token,
        rate_limit_observer=None,
    ):
        assert rate_limit_observer is not None
        rate_limit_observer(
            ClaudeRateLimitEvent(
                status="allowed_warning",
                rate_limit_type="five_hour",
                resets_at=2_000_000_000,
                utilization=0.90,
            )
        )
        return "claude-opus-5"

    monkeypatch.setattr(
        competition,
        "run_claude_runtime_preflight",
        fake_preflight,
    )

    with pytest.raises(RuntimeError, match="protected quota threshold"):
        competition.run_competition(
            claude_config(tmp_path),
            arcade_factory=lambda **kwargs: fake,
        )

    assert fake.open_calls == 0


def test_preflight_draining_account_gets_no_competition_games(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokens = install_fake_claude_tokens(monkeypatch)
    fake = FakeArcade(competition.STANDARD_GAME_ORDER)

    def fake_preflight(
        run_config,
        *,
        oauth_token,
        rate_limit_observer=None,
    ):
        if oauth_token == tokens[0]:
            assert rate_limit_observer is not None
            rate_limit_observer(
                ClaudeRateLimitEvent(
                    status="allowed_warning",
                    rate_limit_type="five_hour",
                    resets_at=2_000_000_000,
                    utilization=0.90,
                )
            )
        return "claude-opus-5"

    monkeypatch.setattr(
        competition,
        "run_claude_runtime_preflight",
        fake_preflight,
    )
    started_accounts = []

    def fake_runner(**kwargs):
        account = kwargs["claude_account_slot"]
        started_accounts.append(account)
        game_id = kwargs["game_id"]
        run_dir = kwargs["batch_dir"] / "games" / game_id
        run_dir.mkdir()
        kwargs["environment_factory"](
            game_id,
            kwargs["scorecard_id"],
            run_dir / "private" / "recordings",
        )
        return GameRunOutcome(
            run_dir=run_dir,
            scorecard_id=kwargs["scorecard_id"],
            session_id=f"claude-{game_id}",
            failure=None,
        )

    batch_dir, status = competition.run_competition(
        claude_config(tmp_path),
        arcade_factory=lambda **kwargs: fake,
        game_runner=fake_runner,
    )

    assert status == 0
    assert batch_dir is not None
    assert set(started_accounts) == {2, 3}
    assert len(started_accounts) == len(competition.STANDARD_GAME_ORDER)


def test_running_game_relays_before_new_work_uses_freed_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_claude_tokens(monkeypatch, count=2)
    games = ("gma1", "gmb2", "gmc3")
    monkeypatch.setattr(competition, "STANDARD_GAME_ORDER", games)
    monkeypatch.setattr(
        competition,
        "run_claude_runtime_preflight",
        lambda run_config, *, oauth_token, rate_limit_observer=None: (
            "claude-opus-5"
        ),
    )
    fake = FakeArcade(games)
    run_config = replace(
        claude_config(tmp_path),
        concurrency=2,
        claude_workers_per_account=1,
        batch_id="quota-relay-test",
    )
    lock = Lock()
    warning_sent = False
    starts: list[tuple[str, int]] = []
    histories: dict[str, tuple[int, ...]] = {}

    def fake_runner(**kwargs):
        nonlocal warning_sent
        game_id = kwargs["game_id"]
        account = kwargs["claude_account_slot"]
        lease = kwargs["claude_credential_lease"]
        assert account is not None
        assert lease is not None
        with lock:
            starts.append((game_id, account))
            should_relay = account == 1 and not warning_sent
            if should_relay:
                warning_sent = True
        if should_relay:
            lease.observe_rate_limit(
                ClaudeRateLimitEvent(
                    status="allowed_warning",
                    rate_limit_type="five_hour",
                    resets_at=2_000_000_000,
                    utilization=0.90,
                )
            )
            assert lease.relay() == "claude-token-2"
        histories[game_id] = lease.account_history
        run_dir = kwargs["batch_dir"] / "games" / game_id
        run_dir.mkdir()
        kwargs["environment_factory"](
            game_id,
            kwargs["scorecard_id"],
            run_dir / "private" / "recordings",
        )
        return GameRunOutcome(
            run_dir=run_dir,
            scorecard_id=kwargs["scorecard_id"],
            session_id=f"claude-{game_id}",
            failure=None,
        )

    batch_dir, status = competition.run_competition(
        run_config,
        arcade_factory=lambda **kwargs: fake,
        game_runner=fake_runner,
    )

    assert status == 0
    assert batch_dir is not None
    assert warning_sent is True
    assert any(history == (1, 2) for history in histories.values())
    warning_index = next(
        index for index, (_, account) in enumerate(starts) if account == 1
    )
    assert all(account == 2 for _, account in starts[warning_index + 1 :])
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["results"]) == set(games)


def test_competition_parser_selects_provider_defaults(tmp_path: Path) -> None:
    parser = competition.build_parser()
    codex = competition.config_from_args(
        parser.parse_args(["--runs-dir", str(tmp_path / "codex")])
    )
    claude = competition.config_from_args(
        parser.parse_args(
            [
                "--runtime",
                "claude",
                "--effort",
                "xhigh",
                "--runs-dir",
                str(tmp_path / "claude"),
            ]
        )
    )

    assert codex.runtime == "codex"
    assert codex.operation_mode == "competition"
    assert codex.model == "gpt-5.6-sol"
    assert codex.concurrency == 2
    assert codex.max_invalid_retries == 15
    assert claude.runtime == "claude"
    assert claude.operation_mode == "competition"
    assert claude.model == "opus"
    assert claude.effort == "xhigh"
    assert claude.concurrency == 2
    assert claude.max_invalid_retries == 15
    assert claude.claude_workers_per_account == 2
    online = competition.config_from_args(
        parser.parse_args(
            [
                "--operation-mode",
                "online",
                "--runs-dir",
                str(tmp_path / "online"),
            ]
        )
    )
    assert online.operation_mode == "online"
    offline = competition.config_from_args(
        parser.parse_args(
            [
                "--operation-mode",
                "offline",
                "--runs-dir",
                str(tmp_path / "offline"),
            ]
        )
    )
    assert offline.operation_mode == "offline"


def test_default_claude_concurrency_fits_one_credential() -> None:
    run_config = competition.config_from_args(
        competition.build_parser().parse_args(["--runtime", "claude"])
    )

    slots = competition.claude_worker_slots(
        ("token-one",),
        workers_per_account=run_config.claude_workers_per_account,
        concurrency=run_config.concurrency,
    )

    assert len(slots) == run_config.concurrency == 2


def test_worker_dispatches_claude_with_the_coordinator_model_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    expected = GameRunOutcome(
        run_dir=tmp_path / "run",
        scorecard_id="card",
        session_id="session",
        failure=None,
    )

    def fake_claude_run(args, **kwargs):
        observed["args"] = args
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(competition, "run_claude_game", fake_claude_run)
    factory = object()
    result = competition.run_worker(
        config=claude_config(tmp_path),
        game_id="s5i5",
        arcade=object(),
        scorecard_id="card",
        auth_file=None,
        preflight_model_id="claude-opus-5",
        environment_factory=factory,
        batch_dir=tmp_path,
        claude_oauth_token="assigned-token",
        claude_account_slot=2,
    )

    assert result is expected
    assert observed["preflight_model_id"] == "claude-opus-5"
    assert observed["environment_factory"] is factory
    assert observed["oauth_token"] == "assigned-token"
    assert observed["args"].effort == "xhigh"
    assert observed["args"].operation_mode == "competition"


def test_claude_pool_assigns_eight_workers_as_three_three_two() -> None:
    tokens = ("one", "two", "three")

    slots = competition.claude_worker_slots(
        tokens,
        workers_per_account=3,
        concurrency=8,
    )

    assert [account for account, _ in slots].count(1) == 3
    assert [account for account, _ in slots].count(2) == 3
    assert [account for account, _ in slots].count(3) == 2
    assert all(token == tokens[account - 1] for account, token in slots)


def test_claude_pool_assigns_nine_workers_as_three_each() -> None:
    slots = competition.claude_worker_slots(
        ("one", "two", "three"),
        workers_per_account=3,
        concurrency=9,
    )

    assert [account for account, _ in slots].count(1) == 3
    assert [account for account, _ in slots].count(2) == 3
    assert [account for account, _ in slots].count(3) == 3


def test_claude_pool_assigns_twelve_workers_as_three_each() -> None:
    slots = competition.claude_worker_slots(
        ("one", "two", "three", "four"),
        workers_per_account=3,
        concurrency=12,
    )

    assert [account for account, _ in slots].count(1) == 3
    assert [account for account, _ in slots].count(2) == 3
    assert [account for account, _ in slots].count(3) == 3
    assert [account for account, _ in slots].count(4) == 3


def test_claude_account_schedule_balances_three_accounts() -> None:
    schedule = competition.claude_account_schedule(
        competition.STANDARD_GAME_ORDER,
        account_count=3,
    )

    assert tuple(map(len, schedule.account_games)) == (9, 8, 8)
    assert set().union(*map(set, schedule.account_games)) == set(
        competition.STANDARD_GAME_ORDER
    )


def test_claude_worker_takes_its_own_scheduled_game_first() -> None:
    queues = {
        1: deque(["own"]),
        2: deque(["other"]),
        3: deque(),
    }

    assert competition.pop_claude_scheduled_game(1, queues) == "own"
    assert list(queues[2]) == ["other"]


def test_idle_claude_worker_takes_from_largest_remaining_queue() -> None:
    queues = {
        1: deque(),
        2: deque(["short"]),
        3: deque(["long-a", "long-b"]),
    }

    assert competition.pop_claude_scheduled_game(1, queues) == "long-a"
    assert list(queues[2]) == ["short"]
    assert list(queues[3]) == ["long-b"]


def test_idle_claude_worker_returns_none_when_all_queues_are_empty() -> None:
    queues = {
        1: deque(),
        2: deque(),
        3: deque(),
    }

    assert competition.pop_claude_scheduled_game(2, queues) is None


def test_claude_pool_rejects_gaps_duplicates_and_excess_concurrency() -> None:
    with pytest.raises(RuntimeError, match="missing slots"):
        competition.claude_oauth_token_pool(
            {
                "CLAUDE_CODE_OAUTH_TOKEN": "one",
                "CLAUDE_CODE_OAUTH_TOKEN_3": "three",
            }
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        competition.claude_oauth_token_pool(
            {
                "CLAUDE_CODE_OAUTH_TOKEN": "same",
                "CLAUDE_CODE_OAUTH_TOKEN_2": "same",
            }
        )
    with pytest.raises(ValueError, match="capacity 6"):
        competition.claude_worker_slots(
            ("one", "two"),
            workers_per_account=3,
            concurrency=8,
        )


def test_claude_competition_requires_one_worker_per_account(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_claude_tokens(monkeypatch)
    run_config = replace(claude_config(tmp_path), concurrency=2)

    with pytest.raises(ValueError, match="one worker per account"):
        competition.run_competition(
            run_config,
            arcade_factory=lambda **kwargs: FakeArcade(
                competition.STANDARD_GAME_ORDER
            ),
        )


def test_runtime_preflight_requires_a_successful_exact_reply(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    auth = install_fake_auth(monkeypatch, tmp_path)
    observed: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def run_task(self, prompt):
            observed["prompt"] = prompt
            stderr = tmp_path / "probe.stderr"
            stderr.write_text("", encoding="utf-8")
            return SimpleNamespace(
                returncode=0,
                final_message="READY",
                session_id="probe-session",
                stderr_path=stderr,
            )

        def resume_checkpoint(self, session_id, prompt):
            observed["resume_session_id"] = session_id
            observed["resume_prompt"] = prompt
            stderr = tmp_path / "resume.stderr"
            stderr.write_text("", encoding="utf-8")
            return SimpleNamespace(
                returncode=0,
                final_message="RESUMED",
                session_id=session_id,
                stderr_path=stderr,
            )

    monkeypatch.setattr(competition, "DockerCodexRunner", FakeRunner)
    monkeypatch.setattr(
        competition,
        "default_native_codex_bin",
        lambda: tmp_path / "codex",
    )

    competition.run_codex_runtime_preflight(config(tmp_path), auth)

    assert observed["model"] == "gpt-5.6-sol"
    assert observed["reasoning_effort"] == "max"
    assert observed["prompt"] == "Reply exactly READY."
    assert observed["resume_session_id"] == "probe-session"
    assert observed["resume_prompt"] == "Reply exactly RESUMED."


def test_environment_factory_rejects_duplicate_make(tmp_path: Path) -> None:
    fake = FakeArcade(("ls20",))
    factory = competition.CompetitionEnvironmentFactory(fake)
    factory("ls20", "card", tmp_path / "first")

    with pytest.raises(RuntimeError, match="already made"):
        factory("ls20", "card", tmp_path / "second")


def test_environment_factory_shares_cookies_under_one_lock(tmp_path: Path) -> None:
    fake = FakeArcade(("ls20",))
    fake._session = Session()
    environment = SimpleNamespace(
        game_id="ls20",
        observation_space=object(),
        _session=Session(),
        _master_cookie_jar=None,
    )
    fake.make = lambda *args, **kwargs: environment
    factory = competition.CompetitionEnvironmentFactory(fake)

    returned = factory("ls20", "card", tmp_path / "recordings")

    assert returned is environment
    assert fake._cookie_lock is factory._lock
    assert fake._master_cookie_jar is fake._session.cookies
    assert environment._master_cookie_jar is fake._session.cookies


def test_available_games_include_unknown_environments_after_known_order() -> None:
    fake = FakeArcade(("private2-version", "ls20-version", "private1-version"))

    assert competition.available_game_ids(fake) == ["ls20", "private1", "private2"]


def test_competition_http_uses_fresh_connections_and_long_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session()
    owner = SimpleNamespace(_session=session)
    sent: dict[str, object] = {}

    def fake_send(self, request, **kwargs):
        sent.update(kwargs)
        return object()

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)

    assert competition.configure_competition_http(owner) is True
    adapter = session.get_adapter("https://")
    assert isinstance(adapter, ArcHTTPAdapter)
    assert session.headers["Connection"] == "close"
    assert adapter.max_retries.connect == ARC_CONNECT_RETRIES
    assert adapter.max_retries.read == 0

    adapter.send(object(), timeout=10)
    assert sent["timeout"] == (
        ARC_CONNECT_TIMEOUT_SECONDS,
        ARC_READ_TIMEOUT_SECONDS,
    )


def test_competition_launcher_supports_claude_without_credential_arguments() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "run_batch.sh"
    ).read_text(encoding="utf-8")

    assert "ARC3_COMPETITION_RUNTIME" in script
    assert "ARC3_COMPETITION_OPERATION_MODE" in script
    assert "ARC3_COMPETITION_ALLOW_CONCURRENT_COORDINATOR" in script
    assert "ARC3_CLAUDE_WORKERS_PER_ACCOUNT" in script
    assert "--claude-workers-per-account" in script
    assert "--operation-mode" in script
    assert "--effort" in script
    assert 'EFFORT=${EFFORT:-xhigh}' in script
    assert 'CONCURRENCY=${CONCURRENCY:-2}' in script
    assert 'CLAUDE_CODE_OAUTH_TOKEN=$CLAUDE_CODE_OAUTH_TOKEN' not in script
    assert '-e "CLAUDE_CODE_OAUTH_TOKEN=' not in script


def test_public_batch_launcher_requires_an_explicit_mode() -> None:
    root = Path(__file__).parents[1]
    batch_script = (root / "scripts" / "run_batch.sh").read_text(
        encoding="utf-8"
    )
    assert "choose --mode offline, online, or competition" in batch_script
    assert 'SCRIPT="$ROOT/scripts/run_batch.sh"' in batch_script


@pytest.mark.parametrize("mode", ["offline", "online", "competition"])
@pytest.mark.parametrize("dotenv_mode", [None, "offline", "online", "competition"])
def test_batch_launcher_preserves_selected_mode(
    tmp_path: Path, mode: str, dotenv_mode: str | None
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = """#!/bin/sh
case "${0##*/}" in
  docker) printf '%s\\n' test-image ;;
  git)
    case "$*" in *rev-parse*) printf '%s\\n' test-commit ;; esac
    ;;
  claude) printf '%s\\n' '2.1.220 (Claude Code)' ;;
esac
"""
    for name in ("docker", "git", "tmux", "ps", "claude"):
        path = bin_dir / name
        path.write_text(stub, encoding="utf-8")
        path.chmod(0o755)
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print('CALL ' + json.dumps({'argv': sys.argv[1:], "
        "'mode': os.environ.get('OPERATION_MODE')}))\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ARC_API_KEY=test-key\nCLAUDE_CODE_OAUTH_TOKEN=test-token\n"
        + (f"OPERATION_MODE={dotenv_mode}\n" if dotenv_mode else ""),
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts" / "run_batch.sh"
    proc = subprocess.run(
        ["bash", str(script), "__supervise", "--runtime", "claude", "--mode", mode],
        env={
            "HOME": str(tmp_path),
            "PATH": f"{bin_dir}{os.pathsep}{os.defpath}",
            "ARC3_ENV_FILE": str(env_file),
            "ARC3_PYTHON": str(python),
            "ARC3_CLAUDE_BIN": str(bin_dir / "claude"),
            "ARC3_COMPETITION_RUNS_DIR": str(tmp_path / "runs"),
            "ARC3_COMPETITION_MIN_FREE_GIB": "0",
            "ARC3_PINNED_COMMIT": "test-commit",
            "ARC3_PINNED_IMAGE_ID": "test-image",
            "ARC3_PINNED_RUNTIME_VERSION": "2.1.220 (Claude Code)",
        },
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = [
        json.loads(line.removeprefix("CALL "))
        for line in proc.stdout.splitlines()
        if line.startswith("CALL ")
    ]
    assert len(calls) == 2
    assert "--preflight" in calls[0]["argv"]
    assert "--preflight" not in calls[1]["argv"]
    for call in calls:
        assert call["argv"][call["argv"].index("--operation-mode") + 1] == mode
        assert call["mode"] == mode
