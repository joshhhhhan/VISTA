import argparse
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from requests.cookies import RequestsCookieJar

import vista_arc3.codex.harness as harness_module
from vista_arc3.codex.harness import (
    CHECKPOINT_PROMPT,
    bounded_task_timeout,
    build_initial_prompt,
    build_parser,
    codex_host_environment,
    create_run_dir,
    finalize_scorecard,
    open_evaluation_scorecard,
    parse_model_spec,
    resolve_model_spec,
    run_game,
    run_task_with_compact_checkpoints,
    share_online_cookie_jar,
    write_compact_restore_hook,
    write_guide,
    write_player_agents,
)


def test_initial_prompt_contains_only_current_public_state() -> None:
    prompt = build_initial_prompt(
        {
            "turn": 0,
            "state": "NOT_FINISHED",
            "progress": {"completed": 1, "total": 3},
            "available_actions": ["ACTION1", "ACTION6"],
            "terminal": False,
        }
    )

    assert "Complete the game with as few game actions as possible." in prompt
    assert "terminal state" not in prompt
    assert '"available_actions":["ACTION1","ACTION6"]' in prompt
    assert "GUIDE.md" not in prompt
    assert "ARC" not in prompt
    assert "game-id" not in prompt


def test_cli_exposes_only_evaluation_choices() -> None:
    parser = build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in {"-h", "--help"}
    }

    assert options == {
        "--game-id",
        "--max-steps",
        "--operation-mode",
        "--model",
        "--effort",
        "--timeout",
        "--max-invalid-retries",
    }
    with pytest.raises(SystemExit):
        parser.parse_args([])
    assert parser.parse_args(
        ["--game-id", "test-game", "--model", "test-model"]
    ).model == "test-model"
    defaults = parser.parse_args(["--game-id", "test-game"])
    assert defaults.model == "gpt-5.6-sol"
    assert defaults.effort is None
    assert defaults.max_steps == 2_000
    assert defaults.max_invalid_retries == 15
    assert defaults.observation_mode == "vision"
    assert defaults.describe_changes is False
    assert defaults.render_grid is True
    assert parser.parse_args(
        ["--game-id", "test-game", "--operation-mode", "offline"]
    ).operation_mode == "offline"
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--game-id", "test-game", "--operation-mode", "competition"]
        )


def test_scorecard_creation_sends_no_evaluation_metadata() -> None:
    calls: list[dict] = []
    arc = SimpleNamespace(
        open_scorecard=lambda **kwargs: calls.append(kwargs) or "card-1"
    )

    assert open_evaluation_scorecard(arc) == "card-1"
    assert calls == [{"tags": []}]


def test_task_timeout_is_bounded_for_long_runs() -> None:
    assert bounded_task_timeout(300, 301) == 90_300
    assert bounded_task_timeout(300, 9_001) == 2_000_000


def test_scene_change_commentary_is_run_local_and_optional(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text(
        "Before each `play` call, briefly state what you expect to happen visibly.\n",
        encoding="utf-8",
    )
    plain = tmp_path / "plain.md"
    described = tmp_path / "described.md"

    write_player_agents(source, plain, describe_changes=False)
    write_player_agents(source, described, describe_changes=True)

    assert plain.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    described_text = described.read_text(encoding="utf-8")
    assert "relevant visual change after the previous action" in described_text
    assert "what you expect to happen visibly" in described_text
    assert "briefly state what you expect to happen visibly" not in described_text


def test_model_spec_only_splits_reasoning_effort() -> None:
    assert parse_model_spec("gpt-5.6-sol-xhigh") == ("gpt-5.6-sol", "xhigh")
    assert parse_model_spec("gpt-5.6-sol") == ("gpt-5.6-sol", None)
    assert parse_model_spec(None) == (None, None)
    assert resolve_model_spec("gpt-5.6-sol", None) == ("gpt-5.6-sol", "max")
    assert resolve_model_spec("gpt-5.6-sol-xhigh", None) == (
        "gpt-5.6-sol",
        "xhigh",
    )
    assert resolve_model_spec("gpt-5.6-sol", "high") == (
        "gpt-5.6-sol",
        "high",
    )
    with pytest.raises(ValueError, match="Conflicting reasoning efforts"):
        resolve_model_spec("gpt-5.6-sol-xhigh", "max")


def test_run_directory_creation_is_atomic_for_parallel_launches(
    tmp_path: Path,
) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        run_dirs = list(pool.map(lambda _: create_run_dir(tmp_path), range(16)))

    assert len(set(run_dirs)) == 16
    assert all(run_dir.is_dir() for run_dir in run_dirs)


def test_online_scorecard_and_environment_share_latest_cookie_jar() -> None:
    stale = RequestsCookieJar()
    stale.set("AWSALBAPP-0", "stale", domain="three.arcprize.org", path="/")
    arc_cookies = RequestsCookieJar()
    arc_cookies.set("AWSALBAPP-0", "open", domain="three.arcprize.org", path="/")
    env_cookies = RequestsCookieJar()
    env_cookies.set("AWSALBAPP-0", "latest", domain="three.arcprize.org", path="/")
    arc = SimpleNamespace(
        _session=SimpleNamespace(cookies=arc_cookies),
        _master_cookie_jar=stale,
        _cookie_lock=threading.Lock(),
    )
    env = SimpleNamespace(
        _session=SimpleNamespace(cookies=env_cookies),
        _master_cookie_jar=stale,
    )

    assert share_online_cookie_jar(arc, env) is True
    assert arc._master_cookie_jar is arc._session.cookies
    assert env._master_cookie_jar is arc._session.cookies
    assert arc._session.cookies.get("AWSALBAPP-0") == "latest"


def test_scorecard_404_uses_successful_preclose_snapshot() -> None:
    class NotFoundError(RuntimeError):
        response = SimpleNamespace(status_code=404)

    cookies = RequestsCookieJar()
    arc = SimpleNamespace(
        _session=SimpleNamespace(cookies=cookies),
        _master_cookie_jar=RequestsCookieJar(),
        _cookie_lock=threading.Lock(),
    )
    env = SimpleNamespace(
        _session=SimpleNamespace(cookies=RequestsCookieJar()),
        _master_cookie_jar=RequestsCookieJar(),
    )
    arc.get_scorecard = lambda scorecard_id: {"card_id": scorecard_id, "score": 4.2}

    def close_scorecard(scorecard_id: str):
        raise NotFoundError(f"scorecard {scorecard_id} not found")

    arc.close_scorecard = close_scorecard

    outcome, failure = finalize_scorecard(
        arc=arc,
        env=env,
        scorecard_id="card-1",
        operation_mode="online",
    )

    assert failure is None
    assert outcome["scorecard_closed"] is False
    assert outcome["scorecard_summary"] == {"card_id": "card-1", "score": 4.2}
    assert outcome["scorecard_summary_source"] == "preclose_snapshot"
    assert outcome["scorecard_finalization"] == "close_not_found_after_snapshot"


def test_compact_hook_configures_checkpoint_rollover(tmp_path: Path) -> None:
    visible = tmp_path / "visible"
    codex_home = tmp_path / "codex-home"
    visible.mkdir()
    codex_home.mkdir()
    write_compact_restore_hook(visible, codex_home)

    hooks = json.loads((codex_home / "hooks.json").read_text())
    pre_hook = hooks["hooks"]["PreCompact"][0]
    pre_script = (visible / ".codex" / "pre_compact_checkpoint.sh").read_text()

    assert set(hooks["hooks"]) == {"PreCompact"}
    assert pre_hook["matcher"] == "auto|manual"
    assert pre_hook["hooks"][0]["command"] == "sh .codex/pre_compact_checkpoint.sh"
    assert "cat >/dev/null" in pre_script
    assert '"continue":false' in pre_script
    assert '"$CODEX_HOME/compact_checkpoint.requested"' in pre_script
    assert not (visible / ".codex" / "post_compact_restore.sh").exists()


def test_compact_hook_always_pauses_native_compaction(tmp_path: Path) -> None:
    visible = tmp_path / "visible"
    codex_home = tmp_path / "codex-home"
    visible.mkdir(parents=True)
    codex_home.mkdir()
    write_guide(visible / "GUIDE.md", current_model="Durable model")
    write_compact_restore_hook(visible, codex_home)

    first = subprocess.run(
        ["sh", ".codex/pre_compact_checkpoint.sh"],
        cwd=visible,
        env={"CODEX_HOME": str(codex_home), "PATH": "/usr/bin:/bin"},
        input='{"hook_event_name":"PreCompact","trigger":"auto"}',
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(first.stdout) == {
        "continue": False,
        "stopReason": "Save the compact checkpoint before continuing.",
    }
    assert (codex_home / "compact_checkpoint.requested").is_file()
    assert not (codex_home / "compact_guide.snapshot").exists()

    (codex_home / "compact_checkpoint.ready").touch()
    second = subprocess.run(
        ["sh", ".codex/pre_compact_checkpoint.sh"],
        cwd=visible,
        env={"CODEX_HOME": str(codex_home), "PATH": "/usr/bin:/bin"},
        input='{"hook_event_name":"PreCompact","trigger":"auto"}',
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(second.stdout) == {
        "continue": False,
        "stopReason": "Save the compact checkpoint before continuing.",
    }
    assert not (codex_home / "compact_restore.pending").exists()
    assert not (codex_home / "compact_guide.snapshot").exists()


def test_compact_hook_yields_to_retry_boundary_without_requesting_checkpoint(
    tmp_path: Path,
) -> None:
    visible = tmp_path / "visible"
    codex_home = tmp_path / "codex-home"
    visible.mkdir(parents=True)
    codex_home.mkdir()
    write_compact_restore_hook(visible, codex_home)
    (codex_home / "retry_boundary.pending").touch()

    result = subprocess.run(
        ["sh", ".codex/pre_compact_checkpoint.sh"],
        cwd=visible,
        env={"CODEX_HOME": str(codex_home), "PATH": "/usr/bin:/bin"},
        input='{"hook_event_name":"PreCompact","trigger":"auto"}',
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(result.stdout) == {
        "continue": False,
        "stopReason": "Starting a fresh retry context.",
    }
    assert not (codex_home / "compact_checkpoint.requested").exists()


def test_harness_starts_fresh_session_after_compact_checkpoint(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "compact_checkpoint.requested"
    ready = tmp_path / "compact_checkpoint.ready"
    working = tmp_path / "WORKING.md"
    restore = tmp_path / "compact_restore.pending"
    calls: list[tuple[str, str]] = []

    class FakeController:
        max_steps = 10
        terminal = False
        step_index = 0
        compact_restore_marker = restore

        def begin_compact_recovery(self):
            calls.append(("prepare", "fresh-thread"))
            restore.touch()
            return SimpleNamespace()

        def discard_compact_recovery(self):
            for path in (working, ready, marker):
                if path.exists():
                    path.unlink()

    controller = FakeController()

    class FakeRunner:
        def run_task(self, prompt: str):
            calls.append(("run", prompt))
            marker.touch()
            return SimpleNamespace(session_id="old-session")

        def resume_checkpoint(self, session_id: str, prompt: str):
            calls.append(("checkpoint", prompt))
            working.write_text("Local continuation state.\n", encoding="utf-8")
            ready.touch()
            return SimpleNamespace(
                session_id="old-session",
                returncode=0,
                stderr_path=tmp_path / "checkpoint.stderr",
            )

        def run_recovery_task(self, recovery):
            calls.append(("recovery", "fresh-session"))
            marker.unlink()
            ready.unlink()
            working.unlink()
            restore.unlink()
            controller.terminal = True
            return SimpleNamespace(
                session_id="fresh-session",
                returncode=0,
                stderr_path=tmp_path / "recovery.stderr",
            )

    results = run_task_with_compact_checkpoints(
        runner=FakeRunner(),
        controller=controller,
        prompt="Complete the game.",
        checkpoint_marker=marker,
        checkpoint_ready=ready,
        working_path=working,
    )

    assert len(results) == 3
    assert calls == [
        ("run", "Complete the game."),
        ("checkpoint", CHECKPOINT_PROMPT),
        ("prepare", "fresh-thread"),
        ("recovery", "fresh-session"),
    ]


def test_harness_starts_fresh_session_after_reset_boundary(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "compact_checkpoint.requested"
    ready = tmp_path / "compact_checkpoint.ready"
    working = tmp_path / "WORKING.md"
    calls: list[tuple[str, str]] = []

    class FakeController:
        max_steps = 10
        terminal = False
        step_index = 3
        retry_boundary_pending = True
        retry_boundary_step = 3

        def begin_retry_recovery(self):
            calls.append(("prepare", "reset"))
            return SimpleNamespace(boundary_step=3)

        def discard_compact_recovery(self):
            pass

    controller = FakeController()

    class FakeRunner:
        def run_task(self, prompt: str):
            calls.append(("run", prompt))
            return SimpleNamespace(session_id="failed-attempt-session")

        def run_retry_task(self, recovery):
            calls.append(("recovery", "fresh-retry-session"))
            controller.retry_boundary_pending = False
            controller.retry_boundary_step = None
            controller.terminal = True
            return SimpleNamespace(
                session_id="fresh-retry-session",
                returncode=0,
                stderr_path=tmp_path / "retry.stderr",
            )

    results = run_task_with_compact_checkpoints(
        runner=FakeRunner(),
        controller=controller,
        prompt="Complete the game.",
        checkpoint_marker=marker,
        checkpoint_ready=ready,
        working_path=working,
    )

    assert len(results) == 2
    assert calls == [
        ("run", "Complete the game."),
        ("prepare", "reset"),
        ("recovery", "fresh-retry-session"),
    ]


def test_nonterminal_stop_resumes_once_then_starts_fresh_player(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "compact_checkpoint.requested"
    ready = tmp_path / "compact_checkpoint.ready"
    working = tmp_path / "WORKING.md"
    calls: list[tuple[str, str]] = []

    class FakeController:
        max_steps = 10
        terminal = False
        step_index = 4
        retry_boundary_pending = False
        retry_boundary_step = None

        def initial_metadata(self):
            return {"progress": {"completed": 1, "total": 6}}

        def request_fresh_recovery(self):
            calls.append(("request", "fresh-player"))
            self.retry_boundary_pending = True
            self.retry_boundary_step = self.step_index

        def begin_retry_recovery(self):
            calls.append(("prepare", "unchanged-state"))
            return SimpleNamespace(boundary_step=self.step_index)

        def discard_compact_recovery(self):
            pass

    controller = FakeController()

    class FakeRunner:
        def run_task(self, prompt: str):
            calls.append(("run", prompt))
            return SimpleNamespace(session_id="session-a", returncode=0)

        def resume_nonterminal_task(self, session_id: str, prompt: str):
            calls.append(("continue", prompt))
            return SimpleNamespace(session_id=session_id, returncode=0)

        def run_retry_task(self, recovery):
            calls.append(("recovery", "fresh-player"))
            controller.retry_boundary_pending = False
            controller.retry_boundary_step = None
            controller.terminal = True
            return SimpleNamespace(
                session_id="session-b",
                returncode=0,
                stderr_path=tmp_path / "retry.stderr",
            )

    results = run_task_with_compact_checkpoints(
        runner=FakeRunner(),
        controller=controller,
        prompt="Complete the game.",
        checkpoint_marker=marker,
        checkpoint_ready=ready,
        working_path=working,
    )

    assert len(results) == 3
    assert calls == [
        ("run", "Complete the game."),
        (
            "continue",
            "The environment is still active. Context limits are handled "
            "automatically. Continue.",
        ),
        ("request", "fresh-player"),
        ("prepare", "unchanged-state"),
        ("recovery", "fresh-player"),
    ]


def test_initial_runtime_launch_retries_the_identical_task_input(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "compact_checkpoint.requested"
    ready = tmp_path / "compact_checkpoint.ready"
    working = tmp_path / "WORKING.md"
    prompts: list[str] = []

    class FakeController:
        max_steps = 10
        terminal = False
        step_index = 0
        retry_boundary_pending = False

        def discard_compact_recovery(self):
            pass

    controller = FakeController()

    class FakeRunner:
        def run_task(self, prompt: str):
            prompts.append(prompt)
            if len(prompts) == 1:
                return SimpleNamespace(
                    session_id=None,
                    returncode=125,
                    stderr_path=tmp_path / "launch.stderr",
                )
            controller.terminal = True
            return SimpleNamespace(
                session_id="created-session",
                returncode=0,
                stderr_path=tmp_path / "started.stderr",
            )

    results = run_task_with_compact_checkpoints(
        runner=FakeRunner(),
        controller=controller,
        prompt="Complete the game.",
        checkpoint_marker=marker,
        checkpoint_ready=ready,
        working_path=working,
    )

    assert len(results) == 2
    assert prompts == ["Complete the game.", "Complete the game."]
    assert controller.step_index == 0


def test_player_runtime_failure_resumes_context_without_replaying_an_action(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "compact_checkpoint.requested"
    ready = tmp_path / "compact_checkpoint.ready"
    working = tmp_path / "WORKING.md"
    calls: list[str] = []

    class FakeController:
        max_steps = 10
        terminal = False
        step_index = 4
        retry_boundary_pending = False

        def discard_compact_recovery(self):
            pass

        def begin_runtime_recovery(self):
            calls.append("snapshot-current-state")
            return SimpleNamespace(boundary_step=self.step_index)

    controller = FakeController()

    class FakeRunner:
        def run_task(self, prompt: str):
            calls.append("failed-segment")
            return SimpleNamespace(
                session_id="failed-session",
                returncode=133,
                stderr_path=tmp_path / "failed.stderr",
            )

        def resume_runtime_recovery(self, session_id, recovery):
            calls.append("resume-runtime-thread")
            assert session_id == "failed-session"
            assert recovery.boundary_step == 4
            controller.terminal = True
            return SimpleNamespace(
                session_id="failed-session",
                returncode=0,
                stderr_path=tmp_path / "recovered.stderr",
            )

    results = run_task_with_compact_checkpoints(
        runner=FakeRunner(),
        controller=controller,
        prompt="Complete the game.",
        checkpoint_marker=marker,
        checkpoint_ready=ready,
        working_path=working,
    )

    assert len(results) == 2
    assert calls == [
        "failed-segment",
        "snapshot-current-state",
        "resume-runtime-thread",
    ]
    assert controller.step_index == 4


def test_retry_process_failure_resumes_only_its_created_thread(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "compact_checkpoint.requested"
    ready = tmp_path / "compact_checkpoint.ready"
    working = tmp_path / "WORKING.md"
    calls: list[str] = []

    class FakeController:
        max_steps = 10
        terminal = False
        step_index = 3
        retry_boundary_pending = True
        retry_boundary_step = 3

        def begin_retry_recovery(self):
            return SimpleNamespace(boundary_step=3)

        def discard_compact_recovery(self):
            pass

    controller = FakeController()

    class FakeRunner:
        def run_task(self, prompt: str):
            return SimpleNamespace(
                session_id="failed-attempt-session",
                returncode=0,
                stderr_path=tmp_path / "initial.stderr",
            )

        def run_retry_task(self, recovery):
            calls.append("create-retry-thread")
            return SimpleNamespace(
                session_id="retry-session",
                returncode=133,
                stderr_path=tmp_path / "create.stderr",
            )

        def resume_retry_task(self, session_id, recovery):
            calls.append(f"resume:{session_id}")
            controller.retry_boundary_pending = False
            controller.retry_boundary_step = None
            controller.terminal = True
            return SimpleNamespace(
                session_id=session_id,
                returncode=0,
                stderr_path=tmp_path / "resume.stderr",
            )

    results = run_task_with_compact_checkpoints(
        runner=FakeRunner(),
        controller=controller,
        prompt="Complete the game.",
        checkpoint_marker=marker,
        checkpoint_ready=ready,
        working_path=working,
    )

    assert len(results) == 3
    assert calls == ["create-retry-thread", "resume:retry-session"]
    assert controller.step_index == 3


def test_compact_process_failure_resumes_only_its_created_thread(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "compact_checkpoint.requested"
    ready = tmp_path / "compact_checkpoint.ready"
    working = tmp_path / "WORKING.md"
    restore = tmp_path / "compact_restore.pending"
    calls: list[str] = []

    class FakeController:
        max_steps = 10
        terminal = False
        step_index = 7
        compact_restore_marker = restore

        def begin_compact_recovery(self):
            restore.touch()
            return SimpleNamespace(boundary_step=7)

        def discard_compact_recovery(self):
            for path in (marker, ready, working, restore):
                path.unlink(missing_ok=True)

    controller = FakeController()

    class FakeRunner:
        def run_task(self, prompt: str):
            marker.touch()
            return SimpleNamespace(
                session_id="old-session",
                returncode=0,
                stderr_path=tmp_path / "initial.stderr",
            )

        def resume_checkpoint(self, session_id: str, prompt: str):
            working.write_text("Continuation state.\n", encoding="utf-8")
            ready.touch()
            return SimpleNamespace(
                session_id=session_id,
                returncode=0,
                stderr_path=tmp_path / "checkpoint.stderr",
            )

        def run_recovery_task(self, recovery):
            calls.append("create-compact-thread")
            return SimpleNamespace(
                session_id="compact-session",
                returncode=133,
                stderr_path=tmp_path / "create.stderr",
            )

        def resume_recovery_task(self, session_id, recovery):
            calls.append(f"resume:{session_id}")
            for path in (marker, ready, working, restore):
                path.unlink(missing_ok=True)
            controller.terminal = True
            return SimpleNamespace(
                session_id=session_id,
                returncode=0,
                stderr_path=tmp_path / "resume.stderr",
            )

    results = run_task_with_compact_checkpoints(
        runner=FakeRunner(),
        controller=controller,
        prompt="Complete the game.",
        checkpoint_marker=marker,
        checkpoint_ready=ready,
        working_path=working,
    )

    assert len(results) == 4
    assert calls == ["create-compact-thread", "resume:compact-session"]
    assert controller.step_index == 7


def test_terminal_run_discards_unneeded_compact_request(tmp_path: Path) -> None:
    marker = tmp_path / "compact_checkpoint.requested"
    ready = tmp_path / "compact_checkpoint.ready"
    working = tmp_path / "WORKING.md"

    class FakeController:
        max_steps = 10
        terminal = True
        step_index = 0

        def discard_compact_recovery(self):
            for path in (working, ready, marker):
                if path.exists():
                    path.unlink()

    class FakeRunner:
        def run_task(self, prompt: str):
            marker.touch()
            return SimpleNamespace(session_id="same-session")

    results = run_task_with_compact_checkpoints(
        runner=FakeRunner(),
        controller=FakeController(),
        prompt="Complete the game.",
        checkpoint_marker=marker,
        checkpoint_ready=ready,
        working_path=working,
    )

    assert len(results) == 1
    assert not marker.exists()


def test_compact_checkpoint_loop_fails_after_repeating_without_action(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "compact_checkpoint.requested"
    ready = tmp_path / "compact_checkpoint.ready"
    working = tmp_path / "WORKING.md"
    restore = tmp_path / "compact_restore.pending"

    class FakeController:
        max_steps = 100
        terminal = False
        step_index = 7
        compact_restore_marker = restore

        def begin_compact_recovery(self):
            restore.touch()
            return SimpleNamespace()

        def discard_compact_recovery(self):
            pass

    class FakeRunner:
        session = 0

        def run_task(self, prompt: str):
            marker.touch()
            return SimpleNamespace(session_id="session-0")

        def resume_checkpoint(self, session_id: str, prompt: str):
            working.write_text("Continuation state.\n", encoding="utf-8")
            ready.touch()
            return SimpleNamespace(
                session_id=session_id,
                returncode=0,
                stderr_path=tmp_path / "checkpoint.stderr",
            )

        def run_recovery_task(self, recovery):
            self.session += 1
            marker.unlink()
            ready.unlink()
            working.unlink()
            restore.unlink()
            marker.touch()
            return SimpleNamespace(
                session_id=f"session-{self.session}",
                returncode=0,
                stderr_path=tmp_path / "recovery.stderr",
            )

    with pytest.raises(
        RuntimeError,
        match="without an environment action",
    ):
        run_task_with_compact_checkpoints(
            runner=FakeRunner(),
            controller=FakeController(),
            prompt="Complete the game.",
            checkpoint_marker=marker,
            checkpoint_ready=ready,
            working_path=working,
        )


def test_host_codex_environment_excludes_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARC_API_KEY", "secret")
    monkeypatch.setenv("ARC_BASE_URL", "private")
    monkeypatch.setenv("PATH", "/bin")

    environment = codex_host_environment()

    assert "ARC_API_KEY" not in environment
    assert "ARC_BASE_URL" not in environment
    assert environment["PATH"] == "/bin"


def test_shared_scorecard_run_never_opens_or_closes_scorecard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "game"
    made: list[tuple[str, str, Path]] = []

    class FakeArcade:
        def get_environments(self):
            return [SimpleNamespace(game_id="ls20")]

        def open_scorecard(self, **kwargs):  # pragma: no cover - safety tripwire
            raise AssertionError("worker opened a scorecard")

        def close_scorecard(self, scorecard_id):  # pragma: no cover - safety tripwire
            raise AssertionError("worker closed a scorecard")

    class FakeEnvironment:
        observation_space = object()

    class FakeController:
        terminal = True
        step_index = 0
        total_attempts = 0
        termination_reason = "win"

        def __init__(self, **kwargs):
            self.initial_observation = kwargs["initial_observation"]

        def initial_metadata(self):
            return {"turn": 0}

    result = SimpleNamespace(
        segment=0,
        returncode=0,
        final_message="",
        session_id="session",
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        output_path=tmp_path / "output",
        command_path=tmp_path / "command",
        requests_path=tmp_path / "requests",
        usage=None,
    )

    def fake_run_task(**kwargs):
        kwargs["results"].append(result)
        return kwargs["results"]

    monkeypatch.setattr(harness_module, "GameController", FakeController)
    monkeypatch.setattr(
        harness_module,
        "build_runner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        harness_module,
        "run_task_with_compact_checkpoints",
        fake_run_task,
    )
    monkeypatch.setattr(harness_module, "codex_version", lambda: "test")

    def environment_factory(game_id: str, scorecard_id: str, recordings: Path):
        made.append((game_id, scorecard_id, recordings))
        return FakeEnvironment()

    args = argparse.Namespace(
        game_id="ls20",
        max_steps=2_000,
        operation_mode="competition",
        model="gpt-5.6-sol-xhigh",
        timeout=300,
        max_invalid_retries=15,
        observation_mode="vision",
        describe_changes=False,
        render_grid=True,
    )
    outcome = run_game(
        args,
        shared_arc=FakeArcade(),
        shared_scorecard_id="competition-card",
        auth_file=auth,
        run_dir=run_dir,
        environment_factory=environment_factory,
    )

    assert outcome.failure is None
    assert outcome.scorecard_id == "competition-card"
    assert made == [
        ("ls20", "competition-card", run_dir / "private" / "recordings")
    ]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scorecard_closed"] is False
    assert manifest["scorecard_finalization"] == "managed_by_competition_coordinator"


def test_guide_is_plain_agent_owned_memory(tmp_path: Path) -> None:
    guide = tmp_path / "GUIDE.md"
    write_guide(guide, current_model="Current model")

    assert guide.read_text(encoding="utf-8") == "Current model\n"
