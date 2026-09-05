#!/usr/bin/env python3
"""Pause Claude compaction until the player saves a continuation checkpoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def configured_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def main() -> int:
    event = json.load(sys.stdin)
    output = Path(
        os.environ.get(
            "ARC3_COMPACT_EVENT_LOG",
            "/claude-io/native_compact_events.jsonl",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        key: event.get(key)
        for key in (
            "session_id",
            "hook_event_name",
            "trigger",
            "compact_summary",
            "stop_hook_active",
        )
        if key in event
    }
    record["credential_environment_present"] = any(
        os.environ.get(name)
        for name in (
            "CLAUDE_CODE_OAUTH_TOKEN",
            "ANTHROPIC_API_KEY",
            "ARC_API_KEY",
        )
    )

    request = configured_path("ARC3_COMPACT_CHECKPOINT_REQUEST")
    ready = configured_path("ARC3_COMPACT_CHECKPOINT_READY")
    decision: dict[str, str] = {}
    event_name = event.get("hook_event_name")
    if event_name == "PreCompact" and request is not None:
        if ready is not None and ready.is_file():
            record["checkpoint_state"] = "ready"
            decision = {
                "decision": "block",
                "reason": "The checkpoint is saved; end this response for handoff.",
            }
        else:
            request.parent.mkdir(parents=True, exist_ok=True)
            request.touch(mode=0o600, exist_ok=True)
            record["checkpoint_state"] = "requested"
            decision = {
                "decision": "block",
                "reason": (
                    "Review and update WORKING.md so it preserves relevant existing "
                    "information and contains the complete continuation state needed "
                    "after compaction."
                ),
            }
    elif (
        event_name == "Stop"
        and request is not None
        and request.is_file()
        and (ready is None or not ready.is_file())
    ):
        record["checkpoint_state"] = "requested"
        decision = {
            "decision": "block",
            "reason": (
                "Review and update WORKING.md so it preserves relevant existing "
                "information and contains the complete continuation state needed "
                "after compaction."
            ),
        }
    elif event_name == "Stop" and ready is not None and ready.is_file():
        record["checkpoint_state"] = "ready"

    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(json.dumps(decision, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
