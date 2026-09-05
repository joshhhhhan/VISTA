import base64
import json
import time
import tomllib
from pathlib import Path

import pytest

from vista_arc3.codex.controller import CompactRecovery, RetryRecovery
from vista_arc3.codex.runner import (
    CHECKPOINT_AUTO_COMPACT_LIMIT,
    MIN_CHECKPOINT_TURN_HEADROOM,
    NORMAL_AUTO_COMPACT_LIMIT,
    PINNED_CODEX_CLI_VERSION,
    PLAYER_BASE_INSTRUCTIONS,
    PLAYER_TOOL_NAMESPACE,
    DockerCodexRunner,
    compact_thresholds_for_context,
    default_codex_bin,
    prepare_codex_home,
    resolve_compact_thresholds,
    write_codex_config,
)


class StubController:
    compact_restore_marker = None
    compact_checkpoint_marker = None
    forced_termination_reason = None

    def __init__(self) -> None:
        self.requests = []
        self.responses = []

    def initial_image_paths(self):
        return []

    def handle(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def write_fake_codex(path: Path, version: str) -> None:
    path.write_text(
        f"#!/bin/sh\nprintf '%s\\n' {version!r}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_default_codex_bin_uses_explicit_pinned_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex = tmp_path / "codex"
    write_fake_codex(codex, PINNED_CODEX_CLI_VERSION)
    monkeypatch.setenv("ARC3_CODEX_BIN", str(codex))

    assert default_codex_bin() == codex.resolve()


def test_default_codex_bin_rejects_another_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex = tmp_path / "codex"
    write_fake_codex(codex, "codex-cli 0.149.1")
    monkeypatch.setenv("ARC3_CODEX_BIN", str(codex))

    with pytest.raises(RuntimeError, match="Expected codex-cli 0.145.0"):
        default_codex_bin()


def make_runner(
    tmp_path: Path,
    *,
    controller: StubController | None = None,
    display_size: int = 1024,
) -> tuple[DockerCodexRunner, Path]:
    visible_dir = tmp_path / "visible"
    visible_dir.mkdir()
    (visible_dir / "GUIDE.md").write_text("Initial model\n", encoding="utf-8")
    (visible_dir / "AGENTS.md").write_text("Game instructions\n", encoding="utf-8")
    (visible_dir / "screenshots").mkdir()
    (visible_dir / ".codex").mkdir()
    auth_file = tmp_path / "shared-auth.json"
    auth_file.write_text("{}", encoding="utf-8")
    codex_vendor = tmp_path / "codex-vendor"
    codex_bin = codex_vendor / "bin" / "codex"
    codex_bin.parent.mkdir(parents=True)
    codex_bin.write_text("", encoding="utf-8")
    bundled_bwrap = codex_vendor / "codex-resources" / "bwrap"
    bundled_bwrap.parent.mkdir()
    bundled_bwrap.write_text("", encoding="utf-8")
    ca_bundle = tmp_path / "ca-certificates.crt"
    ca_bundle.write_text("test certificate bundle", encoding="utf-8")
    codex_home = prepare_codex_home(tmp_path / "codex-home")
    (codex_home / "hooks.json").write_text(
        '{"hooks": {}}\n',
        encoding="utf-8",
    )
    runner = DockerCodexRunner(
        visible_dir=visible_dir,
        codex_home=codex_home,
        auth_file=auth_file,
        io_dir=tmp_path / "io",
        controller=controller or StubController(),
        codex_bin=codex_bin,
        ca_bundle=ca_bundle,
        display_size=display_size,
    )
    return runner, visible_dir


def test_prepare_codex_home_does_not_copy_auth(tmp_path: Path) -> None:
    codex_home = prepare_codex_home(tmp_path / "codex-home")

    assert (codex_home / "sessions").is_dir()
    assert (codex_home / "tmp").is_dir()
    assert (codex_home / "auth.json").is_symlink()
    assert (codex_home / "auth.json").readlink() == Path("/run/codex-auth.json")


def test_compact_thresholds_follow_selected_model_context(tmp_path: Path) -> None:
    catalog = tmp_path / "models_cache.json"
    catalog.write_text(
        json.dumps(
            {
                "models": [
                    {"slug": "large", "context_window": 272_000},
                    {"slug": "small", "context_window": 128_000},
                ]
            }
        ),
        encoding="utf-8",
    )

    large = resolve_compact_thresholds("large", model_catalog=catalog)
    small = resolve_compact_thresholds("small", model_catalog=catalog)

    assert (large.normal_auto_compact_limit, large.checkpoint_auto_compact_limit) == (
        200_000,
        240_000,
    )
    assert (small.normal_auto_compact_limit, small.checkpoint_auto_compact_limit) == (
        56_000,
        96_000,
    )
    assert large.source == "codex-model-catalog"
    assert small.source == "codex-model-catalog"


def test_unknown_model_uses_conservative_compact_thresholds(tmp_path: Path) -> None:
    thresholds = resolve_compact_thresholds(
        "unknown",
        model_catalog=tmp_path / "missing.json",
    )

    assert thresholds.model_context_window == 128_000
    assert thresholds.normal_auto_compact_limit == 56_000
    assert thresholds.checkpoint_auto_compact_limit == 96_000
    assert thresholds.source == "conservative-fallback"


def test_known_frontier_family_does_not_depend_on_fresh_catalog(tmp_path: Path) -> None:
    thresholds = resolve_compact_thresholds(
        "gpt-5.6-sol",
        model_catalog=tmp_path / "missing.json",
    )

    assert thresholds.model_context_window == 272_000
    assert thresholds.normal_auto_compact_limit == 200_000
    assert thresholds.checkpoint_auto_compact_limit == 240_000
    assert thresholds.source == "known-model-family"


def test_compact_thresholds_reject_context_without_checkpoint_headroom() -> None:
    with pytest.raises(ValueError, match="too small"):
        compact_thresholds_for_context(80_000, source="test")


def test_config_disables_generic_and_mcp_tools_and_reserves_checkpoint_headroom(
    tmp_path: Path,
) -> None:
    config = write_codex_config(tmp_path).read_text(encoding="utf-8")
    parsed = tomllib.loads(config)

    assert (tmp_path / "player_instructions.md").read_text(encoding="utf-8") == (
        PLAYER_BASE_INSTRUCTIONS
    )
    assert 'model_instructions_file = "/codex-home/player_instructions.md"' in config
    assert 'personality = "none"' in config
    assert parsed["model_instructions_file"] == "/codex-home/player_instructions.md"
    assert parsed["personality"] == "none"
    assert parsed["web_search"] == "disabled"
    assert parsed["features"]["code_mode"]["enabled"] is False
    assert parsed["features"]["code_mode"]["direct_only_tool_namespaces"] == [
        PLAYER_TOOL_NAMESPACE
    ]
    assert parsed["features"]["code_mode_host"] is True
    assert parsed["features"]["shell_tool"] is False
    assert parsed["features"]["unified_exec"] is False
    assert parsed["agents"]["interrupt_message"] is False
    assert "mcp_servers" not in parsed
    assert "remote_compaction" not in config
    assert parsed["model_auto_compact_token_limit"] == NORMAL_AUTO_COMPACT_LIMIT
    assert parsed["model_auto_compact_token_limit_scope"] == "total"
    assert (
        CHECKPOINT_AUTO_COMPACT_LIMIT - NORMAL_AUTO_COMPACT_LIMIT
        >= MIN_CHECKPOINT_TURN_HEADROOM
    )
    assert "browser_use = false" in config
    assert "[features.code_mode]" in config


def test_dynamic_play_schema_uses_run_display_pixels(
    tmp_path: Path,
) -> None:
    runner, _ = make_runner(tmp_path, display_size=512)
    namespaces = runner._dynamic_tools()
    assert len(namespaces) == 1
    assert namespaces[0]["type"] == "namespace"
    assert namespaces[0]["name"] == PLAYER_TOOL_NAMESPACE
    tools = {tool["name"]: tool for tool in namespaces[0]["tools"]}
    action = tools["play"]["inputSchema"]

    assert set(tools) == {
        "play",
        "inspect",
        "read_pixels",
        "history",
        "read_guide",
        "write_guide",
        "read_working",
        "write_working",
        "save_compact_checkpoint",
    }
    assert action["properties"]["x"]["maximum"] == 511
    assert action["properties"]["y"]["maximum"] == 511


def test_read_pixels_is_a_text_only_controller_tool(tmp_path: Path) -> None:
    controller = StubController()
    controller.responses = [
        {
            "ok": True,
            "metadata": {
                "visual_kind": "pixel_readout",
                "current_state_unchanged": True,
                "sampling": "equal-bin-centers",
                "palette": {"0": "#000000", "1": "#FFFFFF"},
                "sample_count": 4,
                "view_count": 1,
                "views": [
                    {
                        "index": 1,
                        "turn": 0,
                        "frame": 0,
                        "region": {
                            "x": 0,
                            "y": 0,
                            "width": 40,
                            "height": 40,
                        },
                        "rows": 2,
                        "columns": 2,
                        "samples": ["01", "10"],
                    }
                ],
            },
        }
    ]
    runner, _ = make_runner(tmp_path, controller=controller, display_size=512)
    arguments = {
        "question": "What is the exact pattern?",
        "views": [
            {
                "label": "target",
                "turn": 0,
                "region": {"x": 0, "y": 0, "width": 40, "height": 40},
                "rows": 2,
                "columns": 2,
            }
        ],
    }

    execution = runner._execute_tool("read_pixels", arguments, "pixels-1")

    assert execution.success is True
    assert execution.images == ()
    assert json.loads(execution.text)["visual_kind"] == "pixel_readout"
    assert controller.requests == [
        {
            "method": "read_pixels",
            "request_id": "pixels-1",
            "arguments": arguments,
        }
    ]


def test_task_segments_roll_to_fresh_thread_after_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _ = make_runner(tmp_path)
    task_segments = []

    def fake_run(**kwargs):
        task_segments.append(kwargs)
        return object()

    monkeypatch.setattr(runner, "_run_segment", fake_run)
    recovery = CompactRecovery(
        guide="Durable model.\n",
        working_memory="Current plan.\n",
        current_observation={"turn": 12},
        last_action_result={
            "turn": 12,
            "action": {"action": "ACTION1"},
            "result": {"state": "NOT_FINISHED"},
        },
        objective_history={
            "view": "attempts",
            "attempts": [
                {
                    "attempt": 1,
                    "outcome": "NOT_FINISHED",
                    "game_actions": 12,
                }
            ],
        },
        image_paths=(tmp_path / "current.png",),
    )
    runner.run_task("Complete the game.")
    runner.resume_checkpoint("session-id", "Save checkpoint.")
    runner.run_recovery_task(recovery)

    assert task_segments[0]["session_id"] is None
    assert task_segments[1]["session_id"] == "session-id"
    assert task_segments[2]["session_id"] is None
    assert task_segments[2]["initial_images"] == recovery.image_paths
    assert task_segments[2]["recovery"] is recovery
    recovery_prompt = task_segments[2]["prompt"]
    assert recovery_prompt.endswith(
        "Continue the game with as few game actions as possible."
    )
    assert "Environment record:" in recovery_prompt
    assert '"last_action_result":' in recovery_prompt
    assert '"current_level_history":' in recovery_prompt
    assert "Agent-authored notes:" in recovery_prompt
    assert '"working_memory":"Current plan.\\n"' in recovery_prompt
    assert task_segments[0]["compact_limit"] == (
        runner.compact_thresholds.normal_auto_compact_limit
    )
    assert task_segments[1]["compact_limit"] == (
        runner.compact_thresholds.checkpoint_auto_compact_limit
    )
    assert task_segments[2]["compact_limit"] == (
        runner.compact_thresholds.normal_auto_compact_limit
    )


def test_retry_recovery_starts_a_fresh_thread_with_the_reset_visual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _ = make_runner(tmp_path)
    task_segments = []

    def fake_run(**kwargs):
        task_segments.append(kwargs)
        return object()

    monkeypatch.setattr(runner, "_run_segment", fake_run)
    recovery = RetryRecovery(
        boundary_step=17,
        guide="Durable model.\n",
        working_memory="Current retry plan.\n",
        working_provenance={
            "turn": 15,
            "state": "NOT_FINISHED",
            "progress": {"completed": 2, "total": 9},
            "level_start_turn": 10,
        },
        current_observation={
            "turn": 17,
            "state": "NOT_FINISHED",
            "progress": {"completed": 2, "total": 9},
        },
        last_action_result={
            "turn": 17,
            "action": {"action": "RESET"},
            "result": {"state": "NOT_FINISHED"},
        },
        objective_history={
            "view": "attempts",
            "attempts": [
                {
                    "attempt": 2,
                    "outcome": "NOT_FINISHED",
                    "game_actions": 0,
                }
            ],
        },
        image_paths=(tmp_path / "reset.png",),
    )

    runner.run_retry_task(recovery)

    assert len(task_segments) == 1
    segment = task_segments[0]
    assert segment["session_id"] is None
    assert segment["initial_images"] == recovery.image_paths
    assert segment["retry_recovery"] is recovery
    assert segment["compact_limit"] == (
        runner.compact_thresholds.normal_auto_compact_limit
    )
    assert "Environment record:" in segment["prompt"]
    assert '"last_action_result":' in segment["prompt"]
    assert '"current_level_history":' in segment["prompt"]
    assert "Agent-authored notes:" in segment["prompt"]
    assert '"content":"Current retry plan.\\n"' in segment["prompt"]
    assert '"saved_at":' in segment["prompt"]
    assert segment["prompt"].endswith(
        "Continue the game with as few game actions as possible."
    )


def test_nonterminal_continuation_resumes_same_thread_without_an_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _ = make_runner(tmp_path)
    segments = []

    def fake_run(**kwargs):
        segments.append(kwargs)
        return object()

    monkeypatch.setattr(runner, "_run_segment", fake_run)

    runner.resume_nonterminal_task("session-id", "Continue.")

    assert len(segments) == 1
    assert segments[0]["session_id"] == "session-id"
    assert segments[0]["initial_images"] == ()
    assert segments[0]["compact_limit"] == (
        runner.compact_thresholds.normal_auto_compact_limit
    )


def test_docker_mounts_only_player_files_and_no_visual_filesystem(tmp_path: Path) -> None:
    runner, _ = make_runner(tmp_path)
    command = runner._docker_command()

    assert f"{runner.auth_file}:/run/codex-auth.json:ro" in command
    assert f"{runner.codex_home}:/codex-home:rw" in command
    assert "/workspace:rw,nosuid,nodev,size=16m" in command
    assert not any(str(runner.guide_path) in value for value in command)
    assert not any(str(runner.agents_path) in value for value in command)
    assert f"{runner.codex_project_dir}:/workspace/.codex:ro" in command
    assert not any(str(runner.screenshots_dir) in value for value in command)
    assert not any("controller.sock" in value for value in command)
    assert not any("player_mcp.py" in value for value in command)
    assert not any("/codex-io" in value for value in command)
    assert "-i" in command
    assert command[-4:] == [
        "/opt/codex/bin/codex",
        "app-server",
        "--stdio",
        "--strict-config",
    ]
    assert "no-new-privileges" in command
    assert "seccomp=unconfined" in command
    assert "apparmor=unconfined" in command
    assert command[command.index("--memory") + 1] == "16g"
    assert command[command.index("--cpus") + 1] == "4"
    assert (runner.codex_home / "auth.json").is_symlink()


class FakeConnection:
    def __init__(self) -> None:
        self.sent = []

    def send(self, message):
        self.sent.append(message)

    def receive(self, deadline):
        steer = next(item for item in reversed(self.sent) if item.get("method") == "turn/steer")
        return {"id": steer["id"], "result": {"turnId": "turn-1"}}


def test_visual_is_steered_before_text_only_tool_response(tmp_path: Path) -> None:
    controller = StubController()
    image_bytes = b"exact-png-bytes"
    controller.responses = [
        {
            "ok": True,
            "metadata": {
                "turn": 1,
                "action_applied": True,
                "terminal": False,
                "visual": {
                    "turn": 1,
                    "frame_count": 3,
                    "final_frame": 2,
                    "width": 512,
                    "height": 512,
                },
            },
            "image": {
                "mime_type": "image/png",
                "detail": "original",
                "data": base64.b64encode(image_bytes).decode("ascii"),
            },
        }
    ]
    runner, _ = make_runner(tmp_path, controller=controller, display_size=512)
    connection = FakeConnection()
    runner._connection = connection
    runner._active_thread_id = "thread-1"
    runner._active_turn_id = "turn-1"

    runner._handle_tool_call(
        {
            "id": 60,
            "method": "item/tool/call",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "callId": "call-1",
                "namespace": PLAYER_TOOL_NAMESPACE,
                "tool": "play",
                "arguments": {"action": "ACTION1"},
            },
        },
        time.monotonic() + 10,
    )

    assert [item.get("method") for item in connection.sent] == ["turn/steer", None]
    steer, tool_response = connection.sent
    assert steer["params"]["threadId"] == "thread-1"
    assert steer["params"]["expectedTurnId"] == "turn-1"
    assert steer["params"]["input"][0] == {
        "type": "text",
            "text": (
                '<current_visual turn="1" frame="2" frame_count="3" '
                'size="512x512">'
                "Current game state after the action.</current_visual>"
            ),
    }
    assert steer["params"]["input"][1] == {
        "type": "image",
        "url": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
        "detail": "original",
    }
    assert tool_response["id"] == 60
    assert tool_response["result"]["success"] is True
    assert tool_response["result"]["contentItems"][0]["type"] == "inputText"
    assert all(
        item["type"] != "inputImage"
        for item in tool_response["result"]["contentItems"]
    )
    assert controller.requests == [
        {
            "method": "play",
            "request_id": "call-1",
            "arguments": {"action": "ACTION1"},
        }
    ]


def test_reset_visual_is_reserved_for_the_fresh_retry_thread(
    tmp_path: Path,
) -> None:
    controller = StubController()
    controller.responses = [
        {
            "ok": True,
            "metadata": {
                "turn": 2,
                "action_applied": True,
                "retry_boundary": True,
                "terminal": False,
                "visual": {
                    "turn": 2,
                    "frame_count": 1,
                    "final_frame": 0,
                    "width": 512,
                    "height": 512,
                },
            },
            "image": {
                "mime_type": "image/png",
                "detail": "original",
                "data": base64.b64encode(b"reset-visual").decode("ascii"),
            },
        }
    ]
    runner, _ = make_runner(tmp_path, controller=controller, display_size=512)

    class BoundaryConnection:
        def __init__(self) -> None:
            self.sent = []
            self.received = 0

        def send(self, message):
            self.sent.append(message)

        def receive(self, deadline):
            interrupt = next(
                item
                for item in self.sent
                if item.get("method") == "turn/interrupt"
            )
            self.received += 1
            if self.received == 1:
                return {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "old-thread",
                        "turn": {
                            "id": "old-turn",
                            "status": "interrupted",
                        },
                    },
                }
            return {"id": interrupt["id"], "result": {}}

    connection = BoundaryConnection()
    runner._connection = connection
    runner._active_thread_id = "old-thread"
    runner._active_turn_id = "old-turn"

    runner._handle_tool_call(
        {
            "id": 61,
            "method": "item/tool/call",
            "params": {
                "threadId": "old-thread",
                "turnId": "old-turn",
                "callId": "reset-1",
                "namespace": PLAYER_TOOL_NAMESPACE,
                "tool": "play",
                "arguments": {"action": "RESET"},
            },
        },
        time.monotonic() + 10,
    )

    assert len(connection.sent) == 2
    assert connection.sent[0].get("method") is None
    assert connection.sent[1]["method"] == "turn/interrupt"
    assert connection.sent[1]["params"] == {
        "threadId": "old-thread",
        "turnId": "old-turn",
    }
    metadata = json.loads(
        connection.sent[0]["result"]["contentItems"][0]["text"]
    )
    assert metadata["retry_boundary"] is True
    assert runner._completed_turn == {
        "id": "old-turn",
        "status": "interrupted",
    }
    assert controller.requests == [
        {
            "method": "play",
            "request_id": "reset-1",
            "arguments": {"action": "RESET"},
        }
    ]


def test_reset_boundary_accepts_precompact_winning_the_interrupt_race(
    tmp_path: Path,
) -> None:
    controller = StubController()
    controller.responses = [
        {
            "ok": True,
            "metadata": {
                "turn": 2,
                "action_applied": True,
                "retry_boundary": True,
                "terminal": False,
            },
        }
    ]
    runner, _ = make_runner(tmp_path, controller=controller, display_size=512)

    class BoundaryRaceConnection:
        def __init__(self) -> None:
            self.sent = []
            self.received = 0

        def send(self, message):
            self.sent.append(message)

        def receive(self, deadline):
            interrupt = next(
                item
                for item in self.sent
                if item.get("method") == "turn/interrupt"
            )
            self.received += 1
            if self.received == 1:
                return {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "old-thread",
                        "turn": {
                            "id": "old-turn",
                            "status": "failed",
                        },
                    },
                }
            return {
                "id": interrupt["id"],
                "error": {
                    "code": -32600,
                    "message": "no active turn to interrupt",
                },
            }

    connection = BoundaryRaceConnection()
    runner._connection = connection
    runner._active_thread_id = "old-thread"
    runner._active_turn_id = "old-turn"

    runner._handle_tool_call(
        {
            "id": 62,
            "method": "item/tool/call",
            "params": {
                "threadId": "old-thread",
                "turnId": "old-turn",
                "callId": "reset-race",
                "namespace": PLAYER_TOOL_NAMESPACE,
                "tool": "play",
                "arguments": {"action": "RESET"},
            },
        },
        time.monotonic() + 10,
    )

    assert [item.get("method") for item in connection.sent] == [
        None,
        "turn/interrupt",
    ]
    assert runner._completed_turn == {
        "id": "old-turn",
        "status": "failed",
    }
    assert controller.forced_termination_reason is None


def test_inspect_replies_before_starting_a_visual_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = StubController()
    first = base64.b64encode(b"inspection-one").decode("ascii")
    second = base64.b64encode(b"inspection-two").decode("ascii")
    controller.responses = [
        {
            "ok": True,
            "metadata": {
                "visual_kind": "inspection",
                "current_state_unchanged": True,
                "view_count": 2,
                "views": [
                    {
                        "index": 1,
                        "turn": 2,
                        "frame": 3,
                        "region": {"x": 4, "y": 5, "width": 7, "height": 6},
                        "image_size": {"width": 511, "height": 438},
                    },
                    {
                        "index": 2,
                        "turn": 4,
                        "frame": 1,
                        "region": {"x": 8, "y": 9, "width": 10, "height": 11},
                        "image_size": {"width": 465, "height": 511},
                    },
                ],
            },
            "images": [{"data": first}, {"data": second}],
        },
    ]
    runner, _ = make_runner(tmp_path, controller=controller)
    connection = FakeConnection()
    runner._connection = connection
    runner._active_thread_id = "thread-1"
    runner._active_turn_id = "turn-1"
    delivered = []

    def fake_start(images, deadline):
        assert len(connection.sent) == 1
        assert connection.sent[0]["id"] == 61
        delivered.append(images)

    monkeypatch.setattr(runner, "_start_inspection_turn", fake_start)
    monkeypatch.setattr(
        runner,
        "_wait_for_tool_completion",
        lambda call_id, deadline: None,
    )

    runner._handle_tool_call(
        {
            "id": 61,
            "method": "item/tool/call",
            "params": {
                "callId": "inspect-1",
                "namespace": PLAYER_TOOL_NAMESPACE,
                "tool": "inspect",
                "arguments": {
                    "question": "Which regions changed?",
                    "views": [
                        {
                            "label": "changed region",
                            "turn": 2,
                            "frame": 3,
                            "region": {"x": 4, "y": 5, "width": 7, "height": 6},
                        },
                        {"label": "control region", "turn": 4},
                    ]
                },
            },
        },
        time.monotonic() + 10,
    )

    assert len(connection.sent) == 1
    tool_response = connection.sent[0]
    metadata = json.loads(tool_response["result"]["contentItems"][0]["text"])
    assert metadata["current_state_unchanged"] is True
    assert all(
        item["type"] != "inputImage"
        for item in tool_response["result"]["contentItems"]
    )
    delivered_images = delivered[0]
    assert len(delivered_images) == 2
    assert delivered_images[0]["data"] == first
    assert delivered_images[0]["steer_text"] == (
        '<inspection_visual index="1" count="2" turn="2" frame="3" '
        'region="4,5,7,6" size="511x438"></inspection_visual>'
    )
    assert delivered_images[1]["data"] == second
    assert controller.requests == [
        {
            "method": "inspect",
            "request_id": "inspect-1",
            "arguments": {
                "question": "Which regions changed?",
                "views": [
                    {
                        "label": "changed region",
                        "turn": 2,
                        "frame": 3,
                        "region": {"x": 4, "y": 5, "width": 7, "height": 6},
                    },
                    {"label": "control region", "turn": 4},
                ]
            },
        }
    ]


def test_inspection_visuals_start_one_neutral_user_image_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _ = make_runner(tmp_path)
    runner._active_thread_id = "thread-1"
    runner._active_turn_id = "turn-1"
    calls = []

    class CompletionConnection:
        def receive(self, deadline):
            return {
                "method": "turn/completed",
                "params": {
                    "threadId": "thread-1",
                    "turn": {
                        "id": "turn-1",
                        "status": "interrupted",
                    },
                },
            }

    runner._connection = CompletionConnection()

    def fake_rpc(method, params, deadline):
        calls.append((method, params))
        if method == "turn/interrupt":
            return {}
        if method == "turn/start":
            return {"turn": {"id": "turn-2"}}
        raise AssertionError(method)

    monkeypatch.setattr(runner, "_rpc", fake_rpc)
    runner._start_inspection_turn(
        (
            {
                "data": base64.b64encode(b"one").decode("ascii"),
                "steer_text": '<inspection_visual index="1"></inspection_visual>',
                "detail": "original",
            },
            {
                "data": base64.b64encode(b"two").decode("ascii"),
                "steer_text": '<inspection_visual index="2"></inspection_visual>',
                "detail": "original",
            },
        ),
        time.monotonic() + 10,
    )

    assert [method for method, _ in calls] == ["turn/interrupt", "turn/start"]
    assert calls[0][1] == {"threadId": "thread-1", "turnId": "turn-1"}
    turn = calls[1][1]
    assert turn["threadId"] == "thread-1"
    assert [item["type"] for item in turn["input"]] == [
        "text",
        "text",
        "image",
        "text",
        "image",
    ]
    assert turn["input"][0]["text"] == (
        '<inspection_visuals count="2">'
        "Results of the completed inspect call follow in request order; "
        "the current game state is unchanged."
        "</inspection_visuals>"
    )
    assert turn["input"][1]["text"] == (
        '<inspection_visual index="1"></inspection_visual>'
    )
    assert turn["input"][2]["url"].endswith(
        base64.b64encode(b"one").decode("ascii")
    )
    assert turn["input"][3]["text"] == (
        '<inspection_visual index="2"></inspection_visual>'
    )
    assert turn["input"][4]["url"].endswith(
        base64.b64encode(b"two").decode("ascii")
    )
    assert runner._active_turn_id == "turn-2"
    assert runner._completed_turn is None


def test_working_memory_is_agent_owned_and_history_routes_to_controller(
    tmp_path: Path,
) -> None:
    controller = StubController()
    controller.responses = [
        {
            "ok": True,
            "metadata": {
                "view": "attempts",
                "attempts": [{"attempt": 1, "outcome": "NOT_FINISHED"}],
            },
        }
    ]
    runner, _ = make_runner(tmp_path, controller=controller)

    absent = runner._execute_tool("read_working", {}, "read-1")
    written = runner._execute_tool(
        "write_working",
        {"content": "  Current state and plan.  "},
        "write-1",
    )
    present = runner._execute_tool("read_working", {}, "read-2")
    history = runner._execute_tool(
        "history",
        {"view": "attempts"},
        "history-1",
    )

    assert absent.success is True
    assert absent.text == "WORKING.md does not exist."
    assert written.success is True
    assert runner.working_path.read_text(encoding="utf-8") == (
        "Current state and plan.\n"
    )
    assert present.success is True
    assert present.text == "Current state and plan.\n"
    assert history.success is True
    assert json.loads(history.text)["attempts"][0]["outcome"] == "NOT_FINISHED"
    assert controller.requests == [
        {
            "method": "history",
            "request_id": "history-1",
            "arguments": {"view": "attempts"},
        }
    ]


def test_dynamic_tool_call_rejects_the_wrong_namespace(tmp_path: Path) -> None:
    controller = StubController()
    runner, _ = make_runner(tmp_path, controller=controller)
    connection = FakeConnection()
    runner._connection = connection

    runner._handle_tool_call(
        {
            "id": 62,
            "method": "item/tool/call",
            "params": {
                "callId": "wrong-namespace",
                "namespace": "other",
                "tool": "play",
                "arguments": {"action": "ACTION1"},
            },
        },
        time.monotonic() + 10,
    )

    assert controller.requests == []
    assert connection.sent == [
        {
            "id": 62,
            "result": {
                "contentItems": [
                    {"type": "inputText", "text": "Invalid tool request."}
                ],
                "success": False,
            },
        }
    ]


def test_checkpoint_gate_rejects_play_without_executing_controller(tmp_path: Path) -> None:
    controller = StubController()
    controller.compact_checkpoint_marker = tmp_path / "checkpoint.requested"
    controller.compact_checkpoint_marker.touch()
    runner, _ = make_runner(tmp_path, controller=controller)

    result = runner._execute_tool(
        "play",
        {"action": "ACTION1"},
        "call-1",
    )

    assert result.success is False
    assert result.images == ()
    assert controller.requests == []


def test_recovery_gate_rejects_tools_until_new_turn_is_accepted(
    tmp_path: Path,
) -> None:
    controller = StubController()
    controller.compact_restore_marker = tmp_path / "compact_restore.pending"
    controller.compact_restore_marker.touch()
    controller.compact_checkpoint_marker = tmp_path / "checkpoint.requested"
    controller.compact_checkpoint_marker.touch()
    runner, _ = make_runner(tmp_path, controller=controller)

    result = runner._execute_tool(
        "play",
        {"action": "ACTION1"},
        "call-1",
    )

    assert result.text == "Compact recovery is being delivered."
    assert result.success is False
    assert result.images == ()
    assert controller.requests == []


def hook_list_response(trust_status: str) -> dict:
    hooks = [
        {
            "key": "/codex-home/hooks.json:pre_compact:0:0",
            "eventName": "preCompact",
            "sourcePath": "/codex-home/hooks.json",
            "currentHash": "sha256:pre_compact",
            "trustStatus": trust_status,
            "enabled": True,
        }
    ]
    return {"data": [{"cwd": "/workspace", "hooks": hooks}]}


def test_compact_hooks_are_trusted_before_thread_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _ = make_runner(tmp_path)
    calls = []
    responses = iter(
        [
            hook_list_response("untrusted"),
            {"status": "ok"},
            hook_list_response("trusted"),
        ]
    )

    def fake_rpc(method, params, deadline):
        calls.append((method, params))
        return next(responses)

    monkeypatch.setattr(runner, "_rpc", fake_rpc)
    runner._ensure_compact_hooks_trusted(time.monotonic() + 10)

    assert [method for method, _ in calls] == [
        "hooks/list",
        "config/batchWrite",
        "hooks/list",
    ]
    edit = calls[1][1]["edits"][0]
    assert edit["keyPath"] == "hooks.state"
    assert edit["mergeStrategy"] == "upsert"
    assert edit["value"] == {
        "/codex-home/hooks.json:pre_compact:0:0": {
            "trusted_hash": "sha256:pre_compact"
        }
    }
    assert calls[1][1]["reloadUserConfig"] is True


def test_missing_compact_hook_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="preCompact"):
        DockerCodexRunner._compact_hooks({"data": [{"hooks": []}]})
