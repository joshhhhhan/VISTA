#!/usr/bin/env python3
"""Minimal stdio MCP bridge to one authenticated host-side game controller."""

from __future__ import annotations

import json
import os
import socket
import sys
from typing import Any


SERVER_NAME = "game"
SERVER_VERSION = "1.0.0"
DEFAULT_PROTOCOL_VERSION = "2025-06-18"
FORBIDDEN_SUBPROCESS_CREDENTIALS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ARC_API_KEY",
)


class HostConnection:
    def __init__(self) -> None:
        socket_path = os.environ.get("ARC3_GAME_SOCKET")
        token = os.environ.get("ARC3_GAME_TOKEN")
        if not socket_path or not token:
            raise RuntimeError("The private game controller endpoint is unavailable.")
        self.token = token
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(socket_path)
        self.reader = self.socket.makefile("r", encoding="utf-8")
        self.writer = self.socket.makefile("w", encoding="utf-8")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = {"token": self.token, **payload}
        self.writer.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.writer.flush()
        line = self.reader.readline()
        if not line:
            raise RuntimeError("The private game controller disconnected.")
        response = json.loads(line)
        if not isinstance(response, dict):
            raise RuntimeError("The private game controller returned invalid data.")
        return response

    def close(self) -> None:
        self.writer.close()
        self.reader.close()
        self.socket.close()


def send(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def tool_result(execution: dict[str, Any]) -> dict[str, Any]:
    text = execution.get("text")
    if not isinstance(text, str):
        text = "The game controller returned invalid data."
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    interrupt_after = execution.get("interrupt_after") is True
    boundary_reason = execution.get("boundary_reason")
    images = execution.get("images")
    if (
        (not interrupt_after or boundary_reason == "compact_checkpoint")
        and isinstance(images, list)
    ):
        for image in images:
            if not isinstance(image, dict) or not isinstance(image.get("data"), str):
                continue
            label = image.get("steer_text")
            if isinstance(label, str) and label:
                content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image",
                    "data": image["data"],
                    "mimeType": image.get("mime_type", "image/png"),
                }
            )
    return {
        "content": content,
        "isError": execution.get("success") is not True,
    }


def handle(message: dict[str, Any], host: HostConnection) -> None:
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        params = message.get("params")
        requested = params.get("protocolVersion") if isinstance(params, dict) else None
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": (
                        requested if isinstance(requested, str) else DEFAULT_PROTOCOL_VERSION
                    ),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            }
        )
        return
    if method in {"notifications/initialized", "notifications/cancelled"}:
        return
    if method == "ping":
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
        return
    if method == "tools/list":
        response = host.request({"method": "tools/list"})
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("The host returned no tool list.")
        send({"jsonrpc": "2.0", "id": request_id, "result": result})
        ready = host.request(
            {
                "method": "tools/ready",
                "credential_environment_present": any(
                    os.environ.get(name)
                    for name in FORBIDDEN_SUBPROCESS_CREDENTIALS
                ),
            }
        )
        if not isinstance(ready.get("result"), dict):
            raise RuntimeError("The host rejected the MCP subprocess environment.")
        return
    if method == "tools/call":
        params = message.get("params")
        if not isinstance(params, dict):
            raise ValueError("Invalid tool call parameters.")
        response = host.request(
            {
                "method": "tools/call",
                "request_id": request_id,
                "name": params.get("name"),
                "arguments": params.get("arguments"),
            }
        )
        execution = response.get("execution")
        if not isinstance(execution, dict):
            raise RuntimeError("The host returned no tool result.")
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": tool_result(execution),
            }
        )
        if execution.get("interrupt_after") is True:
            host.request(
                {
                    "method": "boundary/delivered",
                    "request_id": request_id,
                }
            )
        return
    if request_id is not None:
        send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "Method not found"},
            }
        )


def main() -> int:
    host = HostConnection()
    try:
        for line in sys.stdin:
            message: object = None
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("Invalid JSON-RPC message.")
                handle(message, host)
            except Exception as exc:
                request_id = None
                try:
                    request_id = message.get("id")
                except Exception:
                    pass
                if request_id is not None:
                    send(
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32603, "message": str(exc)},
                        }
                    )
                else:
                    print(str(exc), file=sys.stderr, flush=True)
    finally:
        host.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
