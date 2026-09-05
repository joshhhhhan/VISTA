import base64
import json
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vista_arc3.claude.dispatcher import ToolExecution
from vista_arc3.claude.mcp_host import ClaudeGameMcpHost
from vista_arc3.claude.tools import build_tools


class StubController:
    compact_checkpoint_marker = None

    def __init__(self) -> None:
        self.checkpoint_reasons = []

    def request_compact_checkpoint(self, *, reason: str) -> None:
        self.checkpoint_reasons.append(reason)


class StubDispatcher:
    def __init__(self, *, interrupt: bool = False) -> None:
        self.interrupt = interrupt
        self.calls = []
        self.controller = StubController()

    def execute(self, name, arguments, call_id):
        self.calls.append((name, arguments, call_id))
        return ToolExecution(
            '{"turn":1}',
            True,
            (
                {
                    "mime_type": "image/png",
                    "data": base64.b64encode(b"png-data").decode("ascii"),
                    "steer_text": '<current_visual turn="1"></current_visual>',
                },
            ),
            self.interrupt,
        )


class FailingDispatcher(StubDispatcher):
    def execute(self, name, arguments, call_id):
        raise RuntimeError("private-controller-path:/secret/source.py")


def start_host(
    tmp_path: Path,
    dispatcher: StubDispatcher,
    *,
    enable_calls: bool = True,
    reset_requires_retry_state: bool = False,
):
    host = ClaudeGameMcpHost(
        socket_path=tmp_path / "controller.sock",
        token="secret",
        dispatcher=dispatcher,
        display_size=512,
        log_path=tmp_path / "bridge.jsonl",
        reset_requires_retry_state=reset_requires_retry_state,
    )
    host.start()
    if enable_calls:
        host.enable_calls()
    return host


def request(socket_path: Path, payload: dict) -> dict:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(str(socket_path))
    reader = connection.makefile("r", encoding="utf-8")
    writer = connection.makefile("w", encoding="utf-8")
    writer.write(json.dumps(payload) + "\n")
    writer.flush()
    response = json.loads(reader.readline())
    writer.close()
    reader.close()
    connection.close()
    return response


def test_host_tracks_only_unsettled_play_calls_as_state_uncertainty(
    tmp_path: Path,
) -> None:
    host = ClaudeGameMcpHost(
        socket_path=tmp_path / "successful.sock",
        token="secret",
        dispatcher=StubDispatcher(),
        display_size=512,
        log_path=tmp_path / "successful.jsonl",
    )
    host.enable_calls()

    host._handle(
        {
            "token": "secret",
            "method": "tools/call",
            "name": "play",
            "request_id": "play-1",
            "arguments": {"action": "ACTION1"},
        }
    )

    completed, in_flight, completed_at = host.tool_activity()
    assert (completed, in_flight) == (1, 0)
    assert completed_at is not None
    assert host.unconfirmed_play_calls == 0

    failed = ClaudeGameMcpHost(
        socket_path=tmp_path / "failed.sock",
        token="secret",
        dispatcher=FailingDispatcher(),
        display_size=512,
        log_path=tmp_path / "failed.jsonl",
    )
    failed.enable_calls()
    with pytest.raises(RuntimeError, match="private game controller failed"):
        failed._handle(
            {
                "token": "secret",
                "method": "tools/call",
                "name": "play",
                "request_id": "play-2",
                "arguments": {"action": "ACTION1"},
            }
        )
    assert failed.unconfirmed_play_calls == 1


def test_host_serves_the_exact_shared_tool_contracts(tmp_path: Path) -> None:
    host = start_host(tmp_path, StubDispatcher())
    try:
        response = request(
            host.socket_path,
            {"token": "secret", "method": "tools/list"},
        )
    finally:
        host.stop()

    expected = [
        {
            "name": tool["name"],
            "description": tool["description"],
            "inputSchema": tool["inputSchema"],
            "annotations": tool["annotations"],
        }
        for tool in build_tools(512, include_compact_checkpoint=False)
    ]
    assert response == {"result": {"tools": expected}}


def test_host_serves_same_session_reset_contract(tmp_path: Path) -> None:
    host = start_host(
        tmp_path,
        StubDispatcher(),
        reset_requires_retry_state=False,
    )
    try:
        response = request(
            host.socket_path,
            {"token": "secret", "method": "tools/list"},
        )
    finally:
        host.stop()

    play = {
        tool["name"]: tool for tool in response["result"]["tools"]
    }["play"]
    assert set(play["inputSchema"]["properties"]) == {"action", "x", "y"}
    assert "retry_state" not in play["description"]


def test_host_rejects_credentials_in_the_mcp_subprocess(tmp_path: Path) -> None:
    host = start_host(tmp_path, StubDispatcher())
    try:
        response = request(
            host.socket_path,
            {
                "token": "secret",
                "method": "tools/ready",
                "credential_environment_present": True,
            },
        )
    finally:
        host.stop()

    assert "inherited" in response["error"]
    assert not host.tools_ready.is_set()


def test_bridge_acknowledges_tool_delivery_after_writing_the_list(tmp_path: Path) -> None:
    dispatcher = StubDispatcher()
    host = start_host(tmp_path, dispatcher)
    bridge = (
        Path(__file__).parents[1]
        / "src"
        / "vista_arc3"
        / "claude"
        / "mcp_bridge.py"
    )
    process = subprocess.Popen(
        ["python3", str(bridge)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "ARC3_GAME_SOCKET": str(host.socket_path),
            "ARC3_GAME_TOKEN": "secret",
        },
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        process.stdin.write(
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"
        )
        process.stdin.flush()
        response = json.loads(process.stdout.readline())
        assert host.tools_ready.wait(timeout=2)
    finally:
        process.stdin.close()
        process.wait(timeout=5)
        host.stop()

    assert len(response["result"]["tools"]) == len(
        build_tools(512, include_compact_checkpoint=False)
    )


def test_host_authenticates_and_sanitizes_private_bridge_logs(tmp_path: Path) -> None:
    dispatcher = StubDispatcher()
    host = start_host(tmp_path, dispatcher)
    try:
        denied = request(
            host.socket_path,
            {"token": "wrong", "method": "tools/list"},
        )
        accepted = request(
            host.socket_path,
            {
                "token": "secret",
                "method": "tools/call",
                "request_id": 7,
                "name": "play",
                "arguments": {"action": "ACTION1"},
            },
        )
    finally:
        host.stop()

    assert "Unauthorized" in denied["error"]
    assert accepted["execution"]["success"] is True
    assert dispatcher.calls == [("play", {"action": "ACTION1"}, "7")]
    log = host.log_path.read_text(encoding="utf-8")
    assert "secret" not in log
    assert "png-data" not in log
    assert "<redacted>" in log
    assert "sha256" in log


def test_host_requests_compaction_before_an_action_would_reach_the_image_limit(
    tmp_path: Path,
) -> None:
    dispatcher = StubDispatcher()
    host = ClaudeGameMcpHost(
        socket_path=tmp_path / "controller.sock",
        token="secret",
        dispatcher=dispatcher,
        display_size=512,
        log_path=tmp_path / "bridge.jsonl",
        initial_image_blocks=599,
        image_block_limit=600,
        image_compact_at=599,
    )
    host.start()
    host.enable_calls()
    try:
        response = request(
            host.socket_path,
            {
                "token": "secret",
                "method": "tools/call",
                "request_id": 8,
                "name": "play",
                "arguments": {"action": "ACTION1"},
            },
        )
    finally:
        host.stop()

    execution = response["execution"]
    assert execution["success"] is False
    assert execution["interrupt_after"] is False
    assert execution["boundary_reason"] is None
    assert "Review and update WORKING.md" in execution["text"]
    assert "WORKING.md" in execution["text"]
    assert "No game action was executed" in execution["text"]
    assert dispatcher.calls == []
    assert dispatcher.controller.checkpoint_reasons == ["provider_image_limit"]
    assert host.failure is None


def test_host_preserves_the_early_run_path_at_593_images(tmp_path: Path) -> None:
    dispatcher = StubDispatcher()
    host = ClaudeGameMcpHost(
        socket_path=tmp_path / "controller.sock",
        token="secret",
        dispatcher=dispatcher,
        display_size=512,
        log_path=tmp_path / "bridge.jsonl",
        initial_image_blocks=593,
        image_block_limit=600,
        image_compact_at=599,
    )
    host.start()
    host.enable_calls()
    try:
        response = request(
            host.socket_path,
            {
                "token": "secret",
                "method": "tools/call",
                "request_id": 8,
                "name": "play",
                "arguments": {"action": "ACTION1"},
            },
        )
    finally:
        host.stop()

    assert response["execution"]["success"] is True
    assert response["execution"]["interrupt_after"] is False
    assert dispatcher.calls == [("play", {"action": "ACTION1"}, "8")]
    assert host.image_blocks_delivered == 594


def test_host_compacts_after_delivering_image_599(tmp_path: Path) -> None:
    dispatcher = StubDispatcher()
    host = ClaudeGameMcpHost(
        socket_path=tmp_path / "controller.sock",
        token="secret",
        dispatcher=dispatcher,
        display_size=512,
        log_path=tmp_path / "bridge.jsonl",
        initial_image_blocks=598,
        image_block_limit=600,
        image_compact_at=599,
    )
    host.start()
    host.enable_calls()
    try:
        response = request(
            host.socket_path,
            {
                "token": "secret",
                "method": "tools/call",
                "request_id": 8,
                "name": "play",
                "arguments": {"action": "ACTION1"},
            },
        )
    finally:
        host.stop()

    execution = response["execution"]
    assert execution["success"] is True
    assert execution["interrupt_after"] is False
    assert execution["boundary_reason"] is None
    assert "Review and update WORKING.md" in execution["text"]
    assert "WORKING.md" in execution["text"]
    assert len(execution["images"]) == 1
    assert dispatcher.calls == [("play", {"action": "ACTION1"}, "8")]
    assert dispatcher.controller.checkpoint_reasons == ["provider_image_limit"]


def test_native_compaction_resets_the_host_image_epoch(tmp_path: Path) -> None:
    dispatcher = StubDispatcher()
    host = start_host(tmp_path, dispatcher)
    try:
        first = request(
            host.socket_path,
            {
                "token": "secret",
                "method": "tools/call",
                "request_id": 9,
                "name": "play",
                "arguments": {"action": "ACTION1"},
            },
        )
        host.acknowledge_compaction()
    finally:
        host.stop()

    assert first["execution"]["success"] is True
    assert host.image_blocks_delivered == 0
    assert host.peak_image_blocks == 1


def test_host_namespaces_controller_call_ids_across_runtime_segments(
    tmp_path: Path,
) -> None:
    dispatcher = StubDispatcher()
    host = ClaudeGameMcpHost(
        socket_path=tmp_path / "controller.sock",
        token="secret",
        dispatcher=dispatcher,
        display_size=512,
        log_path=tmp_path / "bridge.jsonl",
        call_namespace="segment_003",
    )
    host.start()
    host.enable_calls()
    try:
        response = request(
            host.socket_path,
            {
                "token": "secret",
                "method": "tools/call",
                "request_id": 7,
                "name": "play",
                "arguments": {"action": "ACTION1"},
            },
        )
    finally:
        host.stop()

    assert response["execution"]["success"] is True
    assert dispatcher.calls == [
        ("play", {"action": "ACTION1"}, "segment_003:7")
    ]
    record = json.loads(host.log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["value"]["controller_call_id"] == "segment_003:7"


def test_host_fails_closed_without_exposing_controller_exceptions(
    tmp_path: Path,
) -> None:
    host = start_host(tmp_path, FailingDispatcher())
    try:
        response = request(
            host.socket_path,
            {
                "token": "secret",
                "method": "tools/call",
                "request_id": 9,
                "name": "play",
                "arguments": {"action": "ACTION1"},
            },
        )
    finally:
        host.stop()

    assert response == {"error": "RuntimeError: The private game controller failed."}
    assert isinstance(host.failure, RuntimeError)
    assert "private-controller-path" in str(host.failure)
    assert "private-controller-path" not in host.log_path.read_text(encoding="utf-8")


def test_host_blocks_game_calls_until_runtime_validation(tmp_path: Path) -> None:
    dispatcher = StubDispatcher()
    host = start_host(tmp_path, dispatcher, enable_calls=False)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                request,
                host.socket_path,
                {
                    "token": "secret",
                    "method": "tools/call",
                    "request_id": 1,
                    "name": "play",
                    "arguments": {"action": "ACTION1"},
                },
            )
            assert dispatcher.calls == []
            host.enable_calls()
            response = pending.result(timeout=2)
    finally:
        host.stop()

    assert response["execution"]["success"] is True
    assert dispatcher.calls == [("play", {"action": "ACTION1"}, "1")]


def test_host_allows_only_one_play_per_observation_cycle(tmp_path: Path) -> None:
    dispatcher = StubDispatcher()
    host = start_host(tmp_path, dispatcher)
    try:
        first = request(
            host.socket_path,
            {
                "token": "secret",
                "method": "tools/call",
                "request_id": 1,
                "name": "play",
                "arguments": {"action": "ACTION1"},
            },
        )
        second = request(
            host.socket_path,
            {
                "token": "secret",
                "method": "tools/call",
                "request_id": 2,
                "name": "play",
                "arguments": {"action": "ACTION1"},
            },
        )
        host.acknowledge_action_result()
        third = request(
            host.socket_path,
            {
                "token": "secret",
                "method": "tools/call",
                "request_id": 3,
                "name": "play",
                "arguments": {"action": "ACTION1"},
            },
        )
    finally:
        host.stop()

    assert first["execution"]["success"] is True
    assert second["execution"]["success"] is False
    assert "Observe the previous" in second["execution"]["text"]
    assert third["execution"]["success"] is True
    assert dispatcher.calls == [
        ("play", {"action": "ACTION1"}, "1"),
        ("play", {"action": "ACTION1"}, "3"),
    ]


def test_stdio_bridge_returns_native_mcp_image_content(tmp_path: Path) -> None:
    dispatcher = StubDispatcher()
    host = start_host(tmp_path, dispatcher)
    environment = {
        "ARC3_GAME_SOCKET": str(host.socket_path),
        "ARC3_GAME_TOKEN": "secret",
    }
    bridge = (
        Path(__file__).parents[1]
        / "src"
        / "vista_arc3"
        / "claude"
        / "mcp_bridge.py"
    )
    process = subprocess.Popen(
        ["python3", str(bridge)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        process.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18"},
                }
            )
            + "\n"
        )
        process.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "play",
                        "arguments": {"action": "ACTION1"},
                    },
                }
            )
            + "\n"
        )
        process.stdin.flush()
        initialized = json.loads(process.stdout.readline())
        result = json.loads(process.stdout.readline())
    finally:
        process.stdin.close()
        process.wait(timeout=5)
        host.stop()

    assert initialized["result"]["serverInfo"]["name"] == "game"
    assert result["result"] == {
        "content": [
            {"type": "text", "text": '{"turn":1}'},
            {
                "type": "text",
                "text": '<current_visual turn="1"></current_visual>',
            },
            {
                "type": "image",
                "data": base64.b64encode(b"png-data").decode("ascii"),
                "mimeType": "image/png",
            },
        ],
        "isError": False,
    }


def test_reset_result_withholds_visual_and_acknowledges_boundary(tmp_path: Path) -> None:
    dispatcher = StubDispatcher(interrupt=True)
    host = start_host(tmp_path, dispatcher)
    bridge = (
        Path(__file__).parents[1]
        / "src"
        / "vista_arc3"
        / "claude"
        / "mcp_bridge.py"
    )
    process = subprocess.Popen(
        ["python3", str(bridge)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "ARC3_GAME_SOCKET": str(host.socket_path),
            "ARC3_GAME_TOKEN": "secret",
        },
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        process.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "play",
                        "arguments": {"action": "RESET", "retry_state": "state"},
                    },
                }
            )
            + "\n"
        )
        process.stdin.flush()
        result = json.loads(process.stdout.readline())
        assert host.boundary_delivered.wait(timeout=2)
    finally:
        process.stdin.close()
        process.wait(timeout=5)
        host.stop()

    assert result["result"]["content"] == [
        {"type": "text", "text": '{"turn":1}'}
    ]


def test_image_limit_request_delivers_the_current_visual_without_interrupting(
    tmp_path: Path,
) -> None:
    dispatcher = StubDispatcher()
    host = ClaudeGameMcpHost(
        socket_path=tmp_path / "controller.sock",
        token="secret",
        dispatcher=dispatcher,
        display_size=512,
        log_path=tmp_path / "bridge.jsonl",
        initial_image_blocks=598,
        image_block_limit=600,
        image_compact_at=599,
    )
    host.start()
    host.enable_calls()
    bridge = (
        Path(__file__).parents[1]
        / "src"
        / "vista_arc3"
        / "claude"
        / "mcp_bridge.py"
    )
    process = subprocess.Popen(
        ["python3", str(bridge)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "ARC3_GAME_SOCKET": str(host.socket_path),
            "ARC3_GAME_TOKEN": "secret",
        },
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        process.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "play",
                        "arguments": {"action": "ACTION1"},
                    },
                }
            )
            + "\n"
        )
        process.stdin.flush()
        result = json.loads(process.stdout.readline())
        assert not host.boundary_delivered.wait(timeout=0.1)
    finally:
        process.stdin.close()
        process.wait(timeout=5)
        host.stop()

    assert result["result"]["content"][-1] == {
        "type": "image",
        "data": base64.b64encode(b"png-data").decode("ascii"),
        "mimeType": "image/png",
    }
