#!/usr/bin/env python3
"""Extract model thinking, text, and tool calls from a VISTA Claude run.

The full stream-json is written by Claude Code under
<run>/private/claude_io/segment_*.stdout.jsonl. This script turns it into a
readable Markdown transcript.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_assistant_blocks(path: Path):
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get("type") != "assistant":
                continue
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "thinking":
                    text = block.get("thinking")
                    if isinstance(text, str) and text.strip():
                        yield event, "thinking", text.strip()
                elif kind == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        yield event, "text", text.strip()
                elif kind == "tool_use":
                    name = block.get("name") or "tool"
                    arguments = block.get("input")
                    payload = json.dumps(
                        arguments, indent=2, ensure_ascii=False, sort_keys=True
                    )
                    yield event, "tool", f"{name}\n```json\n{payload}\n```"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-tools",
        action="store_true",
        help="Include tool call names and arguments in the transcript",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    claude_io = run_dir / "private" / "claude_io"
    segments = sorted(claude_io.glob("segment_*.stdout.jsonl"))
    if not segments:
        raise FileNotFoundError(f"No Claude stream files under {claude_io}")
    output = args.output or (run_dir / "private" / "model_output.md")

    counts = {"thinking": 0, "text": 0, "tool": 0}
    with output.open("w", encoding="utf-8") as out:
        out.write(f"# Model output transcript\n\nRun: `{run_dir}`\n\n")
        for segment in segments:
            out.write(f"## Segment `{segment.name}`\n\n")
            current_event = None
            for event, kind, text in iter_assistant_blocks(segment):
                counts[kind] += 1
                if kind == "tool" and not args.include_tools:
                    continue
                marker = (event.get("uuid"), kind)
                if marker != current_event:
                    current_event = marker
                    timestamp = event.get("timestamp") or ""
                    heading = {
                        "thinking": "Thinking",
                        "text": "Output",
                        "tool": "Tool call",
                    }[kind]
                    out.write(f"### {heading}")
                    if timestamp:
                        out.write(f" — {timestamp}")
                    out.write("\n\n")
                out.write(text)
                out.write("\n\n")
    print(f"Wrote {output}")
    print(
        f"thinking={counts['thinking']} text={counts['text']} "
        f"tool_calls={counts['tool']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
