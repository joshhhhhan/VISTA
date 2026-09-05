import json
import os
import subprocess
import sys
from pathlib import Path

HOOK = (
    Path(__file__).parents[1]
    / "src"
    / "vista_arc3"
    / "claude"
    / "compact_hook.py"
)


def run_hook(tmp_path: Path, event: str) -> dict[str, object]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "ARC3_COMPACT_EVENT_LOG": str(tmp_path / "events.jsonl"),
        "ARC3_COMPACT_CHECKPOINT_REQUEST": str(tmp_path / "checkpoint.requested"),
        "ARC3_COMPACT_CHECKPOINT_READY": str(tmp_path / "checkpoint.ready"),
    }
    completed = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(
            {
                "session_id": "session",
                "hook_event_name": event,
                "trigger": "auto",
                **(
                    {"compact_summary": "summary"}
                    if event == "PostCompact"
                    else {}
                ),
            }
        ),
        text=True,
        capture_output=True,
        check=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def test_precompact_blocks_until_the_checkpoint_is_ready(
    tmp_path: Path,
) -> None:
    assert run_hook(tmp_path, "Stop") == {}
    assert run_hook(tmp_path, "PreCompact") == {
        "decision": "block",
        "reason": (
            "Review and update WORKING.md so it preserves relevant existing "
            "information and contains the complete continuation state needed after "
            "compaction."
        ),
    }
    assert (tmp_path / "checkpoint.requested").is_file()
    assert run_hook(tmp_path, "Stop") == {
        "decision": "block",
        "reason": (
            "Review and update WORKING.md so it preserves relevant existing "
            "information and contains the complete continuation state needed after "
            "compaction."
        ),
    }

    (tmp_path / "checkpoint.ready").touch()
    assert run_hook(tmp_path, "PreCompact") == {
        "decision": "block",
        "reason": "The checkpoint is saved; end this response for handoff.",
    }
    assert run_hook(tmp_path, "Stop") == {}
    assert run_hook(tmp_path, "PostCompact") == {}

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["hook_event_name"] for event in events] == [
        "Stop",
        "PreCompact",
        "Stop",
        "PreCompact",
        "Stop",
        "PostCompact",
    ]
    assert "compact_summary" not in events[1]
    assert events[-1]["compact_summary"] == "summary"
    assert [event.get("checkpoint_state") for event in events] == [
        None,
        "requested",
        "requested",
        "ready",
        "ready",
        None,
    ]
    assert all(event["credential_environment_present"] is False for event in events)
