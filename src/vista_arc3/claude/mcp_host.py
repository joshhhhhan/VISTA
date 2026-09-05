"""Authenticated Unix-socket host for the Claude game MCP bridge."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .tools import build_tools
from .dispatcher import GameToolDispatcher, ToolExecution


COMPACT_CHECKPOINT_PROMPT = (
    "Review and update WORKING.md so it preserves relevant existing information "
    "and contains the complete continuation state needed after compaction."
)


class ClaudeGameMcpHost:
    def __init__(
        self,
        *,
        socket_path: Path,
        token: str,
        dispatcher: GameToolDispatcher,
        display_size: int,
        log_path: Path,
        call_namespace: str = "",
        initial_image_blocks: int = 0,
        image_block_limit: int | None = None,
        image_compact_at: int | None = None,
        include_compact_checkpoint: bool = False,
        reset_requires_retry_state: bool = False,
    ) -> None:
        self.socket_path = socket_path
        self.token = token
        self.dispatcher = dispatcher
        self.display_size = display_size
        self.log_path = log_path
        self.call_namespace = call_namespace
        self._image_lock = threading.Lock()
        self._activity_lock = threading.Lock()
        self.image_blocks_delivered = initial_image_blocks
        self.peak_image_blocks = initial_image_blocks
        self.image_block_limit = image_block_limit
        self.image_compact_at = image_compact_at
        self.include_compact_checkpoint = include_compact_checkpoint
        self.reset_requires_retry_state = reset_requires_retry_state
        self.boundary_reason: str | None = None
        self.boundary_delivered = threading.Event()
        self.listening = threading.Event()
        self.connected = threading.Event()
        self.tools_listed = threading.Event()
        self.tools_ready = threading.Event()
        self.calls_enabled = threading.Event()
        self.action_cycle_ready = threading.Event()
        self.action_cycle_ready.set()
        self.failure: BaseException | None = None
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._log_lock = threading.Lock()
        self._tool_calls_completed = 0
        self._tool_calls_in_flight = 0
        self._play_calls_in_flight = 0
        self._last_tool_completed_at: float | None = None
        self._play_calls_settled = threading.Event()
        self._play_calls_settled.set()

    def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.touch(mode=0o600, exist_ok=True)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        if not self.listening.wait(timeout=5):
            if self.failure is not None:
                raise RuntimeError(
                    f"Game MCP host failed: {type(self.failure).__name__}: {self.failure}"
                )
            raise TimeoutError("Game MCP host did not start listening.")

    def stop(self) -> None:
        self._stop.set()
        self.calls_enabled.set()
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.socket_path.unlink(missing_ok=True)

    def _serve(self) -> None:
        try:
            old_umask = os.umask(0o177)
            try:
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                listener.bind(str(self.socket_path))
            finally:
                os.umask(old_umask)
            self._listener = listener
            listener.listen(1)
            self.listening.set()
            listener.settimeout(0.2)
            while not self._stop.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                self.connected.set()
                with connection:
                    self._serve_connection(connection)
        except BaseException as exc:
            if not self._stop.is_set():
                self.failure = exc
        finally:
            self._listener = None

    def _serve_connection(self, connection: socket.socket) -> None:
        reader = connection.makefile("r", encoding="utf-8")
        writer = connection.makefile("w", encoding="utf-8")
        try:
            for line in reader:
                if self._stop.is_set():
                    break
                request: Any = None
                try:
                    request = json.loads(line)
                    response = self._handle(request)
                except Exception as exc:
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                self._record("request", request)
                self._record("response", response)
                writer.write(json.dumps(response, separators=(",", ":")) + "\n")
                writer.flush()
        finally:
            writer.close()
            reader.close()

    def _handle(self, request: object) -> dict[str, Any]:
        if not isinstance(request, dict) or request.get("token") != self.token:
            raise PermissionError("Unauthorized controller request.")
        method = request.get("method")
        if method == "tools/list":
            response = {
                "result": {
                    "tools": [
                        {
                            "name": tool["name"],
                            "description": tool["description"],
                            "inputSchema": tool["inputSchema"],
                            "annotations": tool.get("annotations", {}),
                        }
                        for tool in build_tools(
                            self.display_size,
                            include_compact_checkpoint=(
                                self.include_compact_checkpoint
                            ),
                            reset_requires_retry_state=(
                                self.reset_requires_retry_state
                            ),
                        )
                    ]
                }
            }
            self.tools_listed.set()
            return response
        if method == "tools/ready":
            if request.get("credential_environment_present") is not False:
                raise RuntimeError(
                    "The MCP subprocess inherited a provider or ARC credential."
                )
            self.tools_ready.set()
            return {"result": {"acknowledged": True}}
        if method == "tools/call":
            if not self.calls_enabled.wait(timeout=30) or self._stop.is_set():
                raise RuntimeError("Game tools were not enabled by runtime validation.")
            name = request.get("name")
            request_id = request.get("request_id")
            if not isinstance(name, str) or not isinstance(request_id, (str, int)):
                raise ValueError("Invalid tool request.")
            if (
                name == "save_compact_checkpoint"
                and not self.include_compact_checkpoint
            ):
                raise ValueError("Unknown tool.")
            controller_call_id = str(request_id)
            if self.call_namespace:
                controller_call_id = f"{self.call_namespace}:{controller_call_id}"
            request["controller_call_id"] = controller_call_id
            self._record_tool_started(name)
            try:
                response = self._execute_tool_call(
                    name=name,
                    arguments=request.get("arguments"),
                    controller_call_id=controller_call_id,
                )
            except BaseException:
                if name != "play":
                    self._record_tool_completed(name)
                raise
            self._record_tool_completed(name)
            return response
        if method == "boundary/delivered":
            self.boundary_delivered.set()
            return {"result": {"acknowledged": True}}
        raise ValueError("Unknown bridge method.")

    def _execute_tool_call(
        self,
        *,
        name: str,
        arguments: object,
        controller_call_id: str,
    ) -> dict[str, Any]:
        predicted_images = predicted_image_blocks(name, arguments)
        with self._image_lock:
            if (
                predicted_images
                and self.image_block_limit is not None
                and self.image_blocks_delivered + predicted_images
                >= self.image_block_limit
            ):
                self._request_compact_checkpoint(reason="provider_image_limit")
                return {
                    "execution": execution_payload(
                        ToolExecution(
                            f"{COMPACT_CHECKPOINT_PROMPT} "
                            "No game action was executed.",
                            False,
                        )
                    )
                }
        if name == "play" and not self.action_cycle_ready.is_set():
            return {
                "execution": execution_payload(
                    ToolExecution(
                        "Observe the previous play result before another action.",
                        False,
                    )
                )
            }
        if name == "play":
            self.action_cycle_ready.clear()
        try:
            execution = self.dispatcher.execute(
                name,
                arguments,
                controller_call_id,
            )
        except BaseException as exc:
            self.failure = exc
            self._stop.set()
            raise RuntimeError("The private game controller failed.") from None
        with self._image_lock:
            self.image_blocks_delivered += len(execution.images)
            self.peak_image_blocks = max(
                self.peak_image_blocks,
                self.image_blocks_delivered,
            )
            if execution.interrupt_after:
                self.boundary_reason = execution.boundary_reason or "reset"
            elif (
                execution.images
                and self.image_compact_at is not None
                and self.image_blocks_delivered >= self.image_compact_at
                and not getattr(
                    getattr(self.dispatcher, "controller", None),
                    "terminal",
                    False,
                )
            ):
                self._request_compact_checkpoint(reason="provider_image_limit")
                execution = replace(
                    execution,
                    text=f"{execution.text}\n\n{COMPACT_CHECKPOINT_PROMPT}",
                )
        return {"execution": execution_payload(execution)}

    def _record_tool_started(self, name: str) -> None:
        with self._activity_lock:
            self._tool_calls_in_flight += 1
            if name == "play":
                self._play_calls_in_flight += 1
                self._play_calls_settled.clear()

    def _record_tool_completed(self, name: str) -> None:
        with self._activity_lock:
            self._tool_calls_completed += 1
            self._tool_calls_in_flight -= 1
            self._last_tool_completed_at = time.monotonic()
            if name == "play":
                self._play_calls_in_flight -= 1
                if self._play_calls_in_flight == 0:
                    self._play_calls_settled.set()

    def tool_activity(self) -> tuple[int, int, float | None]:
        with self._activity_lock:
            return (
                self._tool_calls_completed,
                self._tool_calls_in_flight,
                self._last_tool_completed_at,
            )

    def wait_for_play_settlement(self, timeout: float) -> bool:
        return self._play_calls_settled.wait(timeout=timeout)

    @property
    def unconfirmed_play_calls(self) -> int:
        with self._activity_lock:
            return self._play_calls_in_flight

    def enable_calls(self) -> None:
        self.calls_enabled.set()

    def acknowledge_action_result(self) -> None:
        self.action_cycle_ready.set()

    def acknowledge_compaction(self) -> None:
        with self._image_lock:
            self.image_blocks_delivered = 0

    def disable_calls(self) -> None:
        self.calls_enabled.clear()

    def _request_compact_checkpoint(self, *, reason: str) -> None:
        controller = getattr(self.dispatcher, "controller", None)
        marker = getattr(controller, "compact_checkpoint_marker", None)
        if marker is not None and marker.is_file():
            return
        request = getattr(controller, "request_compact_checkpoint", None)
        if not callable(request):
            raise RuntimeError("Compact checkpoint control is unavailable.")
        request(reason=reason)

    def _record(self, direction: str, value: object) -> None:
        record = {"direction": direction, "value": sanitize(value)}
        with self._log_lock:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def execution_payload(execution: ToolExecution) -> dict[str, Any]:
    return {
        "text": execution.text,
        "success": execution.success,
        "images": list(execution.images),
        "interrupt_after": execution.interrupt_after,
        "boundary_reason": execution.boundary_reason,
    }


def predicted_image_blocks(name: str, arguments: object) -> int:
    if name == "play":
        return 1
    if name != "inspect" or not isinstance(arguments, dict):
        return 0
    views = arguments.get("views")
    return len(views) if isinstance(views, list) else 0


def sanitize(value: object) -> object:
    if isinstance(value, dict):
        sanitized: dict[str, object] = {}
        for key, item in value.items():
            if key == "token":
                sanitized[key] = "<redacted>"
            elif key == "data" and isinstance(item, str):
                try:
                    size = len(item.encode("ascii"))
                except UnicodeEncodeError:
                    size = len(item.encode("utf-8"))
                sanitized[key] = {
                    "encoded_chars": size,
                    "sha256": hashlib.sha256(item.encode("utf-8")).hexdigest(),
                }
            else:
                sanitized[key] = sanitize(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value
