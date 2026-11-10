"""Run a VISTA game through the official DeepSeek Harness headless profile."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vista.backends.claude import mcp_bridge
from vista.backends.claude.mcp_host import ClaudeGameMcpHost
from vista.backends.claude.runner import create_short_socket_dir
from vista.benchmarks.arc3.claude.tools import build_tools
from vista.core.dispatcher import GameToolDispatcher

DEFAULT_DSH_HOME = (
    Path.home()
    / ".local"
    / "share"
    / "deepseek-harness"
    / "0.2.2-rc.1"
)
DEFAULT_DSH_BIN = DEFAULT_DSH_HOME / "node_modules" / ".bin" / "dsh"
DEFAULT_DSH_MODEL = "deepseek-flash"
DEFAULT_DSH_EFFORT = "max"

# These rows are part of the shipped dsh base/headless bundle.  Disabling the
# tool rows leaves only the MCP server registered below.
DISABLED_ROWS = (
    "agent-instructions",
    "command-compact",
    "command-feedback",
    "command-goal",
    "commands",
    "deepseek-account",
    "deepseek-llm-api-extensions",
    "fs-observation-policy",
    "llm-pi-ai",
    "mcp-resources",
    "otel",
    "plan-mode",
    "session-log-deepseek",
    "session-telemetry-otel",
    "skill",
    "skill-filesystem",
    "subagent",
    "subagent-fork-in-process",
    "subagent-spawn-in-process",
    "tool-bash",
    "tool-fs",
    "tool-fs-search",
    "tool-goal",
    "tool-jobs",
    "tool-plugin-manager",
    "tool-pwsh",
    "tool-skill",
    "tool-subagent",
    "tool-subagent-control",
    "tool-subagent-fork",
    "tool-todo",
    "tool-web",
    "tool-workflow",
    "tool-working-directory",
    "user-questions",
    "web",
    "web-deepseek-api",
    "web-fetch-http",
    "web-search-deepseek",
    "working-directory",
)


@dataclass(frozen=True)
class DshResult:
    returncode: int
    stdout_path: Path
    stderr_path: Path
    profile_dir: Path
    final_message: str
    timed_out: bool


def dsh_bin() -> Path:
    configured = os.getenv("ARC3_DSH_BIN")
    candidate = Path(configured).expanduser() if configured else DEFAULT_DSH_BIN
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(f"DeepSeek Harness CLI was not found at {candidate}")
    return candidate.resolve()


def _yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _write_profile(profile_dir: Path, *, socket_path: Path, token: str, model: str, effort: str) -> Path:
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "package.json").write_text(
        json.dumps(
            {
                "name": "dsh-profile-vista",
                "private": True,
                "dependencies": {},
                "dsh": {
                    "profile": {
                        "bundles": [
                            "@deepseek-ai/dsh-base",
                            "@deepseek-ai/dsh-headless",
                        ]
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (profile_dir / "cordis.yml").write_text("[]\n", encoding="utf-8")
    (profile_dir / "pnpm-workspace.yaml").write_text(
        "packages:\n  - .\n\nnodeLinker: hoisted\nautoInstallPeers: false\n",
        encoding="utf-8",
    )
    lines = [
        "- id: agent-default-model",
        "  config:",
        "    provider: deepseek-official",
        f"    model: {_yaml_quote(model)}",
        "- id: system-prompt",
        "  config:",
        '    personaPrefix: "You are a visual game player."',
    ]
    for row in DISABLED_ROWS:
        lines.extend((f"- id: {row}", "  disabled: true"))
    lines.extend(
        [
            "- insert:",
            "    - id: mcp-game",
            "      name: '@deepseek-ai/dsh-mcp-client'",
            "      config:",
            "        serverName: game",
            "        transport: stdio",
            f"        command: {_yaml_quote('/usr/bin/env')}",
            "        args:",
            f"          - {_yaml_quote('python3')}",
            f"          - {_yaml_quote(str(Path(mcp_bridge.__file__).resolve()))}",
            "        env:",
            f"          ARC3_GAME_SOCKET: {_yaml_quote(str(socket_path))}",
            f"          ARC3_GAME_TOKEN: {_yaml_quote(token)}",
            "        startupTimeoutMs: 30000",
            "        failOnStartupError: true",
        ]
    )
    patch_path = profile_dir / "cordis.patch.yml"
    patch_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    profile_dir.chmod(0o700)
    patch_path.chmod(0o600)
    return patch_path


class DshRunner:
    def __init__(
        self,
        *,
        visible_dir: Path,
        private_dir: Path,
        controller: Any,
        model: str = DEFAULT_DSH_MODEL,
        effort: str = DEFAULT_DSH_EFFORT,
        dsh_bin_path: Path | None = None,
        task_timeout: int = 3600,
        display_size: int = 512,
    ) -> None:
        self.visible_dir = visible_dir.resolve()
        self.private_dir = private_dir.resolve()
        self.controller = controller
        self.model = model
        self.effort = effort
        self.dsh_bin_path = (dsh_bin_path or dsh_bin()).resolve()
        self.task_timeout = task_timeout
        self.display_size = display_size
        self.io_dir = self.private_dir / "dsh_io"
        self.io_dir.mkdir(parents=True, exist_ok=True)
        self.dsh_home = self.private_dir / "dsh_home"
        self.work_dir = self.private_dir / "dsh_work"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path = self.io_dir / "segment_000.stdout.jsonl"
        self.stderr_path = self.io_dir / "segment_000.stderr.txt"
        self.bridge_log_path = self.io_dir / "segment_000.bridge.jsonl"
        self.socket_dir = create_short_socket_dir(0)
        self.socket_path = self.socket_dir / "controller.sock"
        self.token = secrets.token_urlsafe(32)
        dispatcher = GameToolDispatcher(
            controller=controller,
            guide_path=self.visible_dir / "GUIDE.md",
            working_path=self.visible_dir / "WORKING.md",
            interrupt_on_invalid_action_limit=False,
        )
        self.host = ClaudeGameMcpHost(
            socket_path=self.socket_path,
            token=self.token,
            dispatcher=dispatcher,
            display_size=self.display_size,
            log_path=self.bridge_log_path,
            call_namespace="dsh",
            include_compact_checkpoint=False,
            native_compaction=True,
            tool_definitions=build_tools(
                self.display_size,
                include_compact_checkpoint=False,
            ),
        )

    def _is_action_tool(self, name: str) -> bool:
        return name in self.host.action_tools or name.rsplit("__", 1)[-1] in self.host.action_tools

    def _watch_action_results(self, stop: threading.Event) -> None:
        """Re-open the action gate after a play result reaches the next step.

        Claude/Codex notify the host through their native event streams.  DSH
        has no equivalent callback, so follow its NDJSON stream and acknowledge
        at the end of a step that consumed a play tool result.
        """
        pending: set[str] = set()
        call_tools: dict[str, str] = {}
        buffer = ""
        try:
            with self.stdout_path.open("r", encoding="utf-8", errors="replace") as stream:
                while not stop.is_set():
                    chunk = stream.read()
                    if not chunk:
                        time.sleep(0.1)
                        continue
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        kind = event.get("type")
                        if kind == "tool_call":
                            call_id = event.get("callId")
                            name = event.get("tool")
                            if isinstance(call_id, str) and isinstance(name, str):
                                call_tools[call_id] = name
                        elif kind == "tool_result":
                            call_id = event.get("callId")
                            name = (
                                call_tools.pop(call_id, None)
                                if isinstance(call_id, str)
                                else None
                            )
                            if name is not None and self._is_action_tool(name):
                                pending.add(call_id)
                        elif kind == "status" and event.get("phase") == "step_end":
                            if pending:
                                pending.clear()
                                self.host.acknowledge_action_result()
        except OSError:
            return

    def _process_environment(self) -> dict[str, str]:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set for the dsh backend")
        environment = {
            name: value
            for name, value in os.environ.items()
            if name
            in {
                "PATH",
                "HOME",
                "TMPDIR",
                "TMP",
                "TEMP",
                "LANG",
                "LC_ALL",
                "TERM",
                "NO_COLOR",
            }
        }
        environment.update(
            {
                "DEEPSEEK_API_KEY": key,
                "DSH_HOME": str(self.dsh_home),
                "DSH_TELEMETRY_MODE": "DISABLED",
                "DSH_PERMISSION_MODE": "read-only",
            }
        )
        return environment

    def run(self, prompt: str) -> DshResult:
        profile_dir = self.dsh_home / "profiles" / "vista"
        _write_profile(
            profile_dir,
            socket_path=self.socket_path,
            token=self.token,
            model=self.model,
            effort=self.effort,
        )
        self.host.start()
        process: subprocess.Popen[str] | None = None
        watcher: threading.Thread | None = None
        stop_watch = threading.Event()
        timed_out = False
        try:
            command = [
                str(self.dsh_bin_path),
                "--profile",
                "vista",
                "--json",
                "--approval-policy",
                "never",
                "--permission-mode",
                "read-only",
                prompt,
            ]
            with self.stdout_path.open("w", encoding="utf-8") as stdout, self.stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=self.work_dir,
                    env=self._process_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                )
                watcher = threading.Thread(
                    target=self._watch_action_results,
                    args=(stop_watch,),
                    daemon=True,
                )
                watcher.start()
                if not self.host.connected.wait(timeout=30):
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"dsh exited with code {process.returncode} before MCP connected"
                        )
                    raise TimeoutError("dsh MCP bridge did not connect")
                if not self.host.tools_ready.wait(timeout=30):
                    raise TimeoutError("dsh MCP bridge did not deliver tools")
                self.host.enable_calls()
                deadline = time.monotonic() + self.task_timeout
                while process.poll() is None:
                    if getattr(self.controller, "terminal", False):
                        process.terminate()
                    if time.monotonic() >= deadline:
                        timed_out = True
                        process.kill()
                        break
                    time.sleep(0.25)
                returncode = process.wait()
        finally:
            stop_watch.set()
            if watcher is not None:
                watcher.join(timeout=1)
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            self.host.stop()
            shutil.rmtree(self.socket_dir, ignore_errors=True)
        final_message = ""
        try:
            lines = self.stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in reversed(lines):
                event = json.loads(line)
                if event.get("type") in {"agent_message", "message", "final"}:
                    text = event.get("text")
                    if isinstance(text, str) and text:
                        final_message = text
                        break
        except (OSError, ValueError, AttributeError):
            pass
        return DshResult(
            returncode=returncode,
            stdout_path=self.stdout_path,
            stderr_path=self.stderr_path,
            profile_dir=profile_dir,
            final_message=final_message,
            timed_out=timed_out,
        )
