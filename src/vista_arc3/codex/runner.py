"""Run one visual-game session through Codex app-server."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .controller import CompactRecovery, RetryRecovery, RuntimeRecovery
from .tools import build_tools
from .recovery import (
    compact_recovery_prompt,
    retry_recovery_prompt,
    runtime_recovery_prompt,
)
from .dispatcher import GameToolDispatcher, ToolExecution

CONTAINER_CODEX_BIN = "/opt/codex/bin/codex"
PLAYER_IMAGE = "arc3-codex-player:0.1"
PLAYER_BASE_INSTRUCTIONS = "You are a visual game player.\n"
PLAYER_BASE_INSTRUCTIONS_PATH = "/codex-home/player_instructions.md"
APP_SERVER_CLIENT_NAME = "vista_arc3"
PLAYER_TOOL_NAMESPACE = "game"
PINNED_CODEX_CLI_VERSION = "codex-cli 0.145.0"
DEFAULT_PINNED_CODEX_BIN = (
    Path.home()
    / ".local"
    / "share"
    / "arc3-codex"
    / "0.145.0"
    / "node_modules"
    / "@openai"
    / "codex-linux-x64"
    / "vendor"
    / "x86_64-unknown-linux-musl"
    / "bin"
    / "codex"
)
COMPACT_HOOK_EVENTS = frozenset({"preCompact"})
COMPACT_HOOK_SOURCE = "/codex-home/hooks.json"
DEFAULT_MODEL_CONTEXT_WINDOW = 272_000
FALLBACK_MODEL_CONTEXT_WINDOW = 128_000
DEFAULT_CONTEXT_MODEL_PREFIXES = ("gpt-5.4", "gpt-5.5", "gpt-5.6-")
CHECKPOINT_HARD_CONTEXT_RESERVE = 32_000
CHECKPOINT_TURN_HEADROOM = 40_000
MIN_NORMAL_AUTO_COMPACT_LIMIT = 16_000


@dataclass(frozen=True)
class CodexUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int


@dataclass(frozen=True)
class CompactThresholds:
    model_context_window: int
    normal_auto_compact_limit: int
    checkpoint_auto_compact_limit: int
    source: str


def compact_thresholds_for_context(
    context_window: int,
    *,
    source: str,
) -> CompactThresholds:
    checkpoint_limit = context_window - CHECKPOINT_HARD_CONTEXT_RESERVE
    normal_limit = checkpoint_limit - CHECKPOINT_TURN_HEADROOM
    if normal_limit < MIN_NORMAL_AUTO_COMPACT_LIMIT:
        raise ValueError(
            f"Model context window {context_window} is too small for compact checkpointing"
        )
    return CompactThresholds(
        model_context_window=context_window,
        normal_auto_compact_limit=normal_limit,
        checkpoint_auto_compact_limit=checkpoint_limit,
        source=source,
    )


def resolve_compact_thresholds(
    model: str | None,
    *,
    model_catalog: Path | None = None,
) -> CompactThresholds:
    catalog = model_catalog or (Path.home() / ".codex" / "models_cache.json")
    if model:
        try:
            payload = json.loads(catalog.read_text(encoding="utf-8"))
            models = payload.get("models", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            models = []
        for entry in models:
            if not isinstance(entry, dict) or entry.get("slug") != model:
                continue
            context_window = entry.get("context_window")
            if isinstance(context_window, int) and not isinstance(
                context_window, bool
            ):
                return compact_thresholds_for_context(
                    context_window,
                    source="codex-model-catalog",
                )

        if model.startswith(DEFAULT_CONTEXT_MODEL_PREFIXES):
            return compact_thresholds_for_context(
                DEFAULT_MODEL_CONTEXT_WINDOW,
                source="known-model-family",
            )

    return compact_thresholds_for_context(
        FALLBACK_MODEL_CONTEXT_WINDOW,
        source="conservative-fallback",
    )


DEFAULT_COMPACT_THRESHOLDS = compact_thresholds_for_context(
    DEFAULT_MODEL_CONTEXT_WINDOW,
    source="default",
)
NORMAL_AUTO_COMPACT_LIMIT = DEFAULT_COMPACT_THRESHOLDS.normal_auto_compact_limit
CHECKPOINT_AUTO_COMPACT_LIMIT = (
    DEFAULT_COMPACT_THRESHOLDS.checkpoint_auto_compact_limit
)
MIN_CHECKPOINT_TURN_HEADROOM = CHECKPOINT_TURN_HEADROOM


@dataclass(frozen=True)
class CodexResult:
    segment: int
    returncode: int
    final_message: str
    session_id: str | None
    stdout_path: Path
    stderr_path: Path
    output_path: Path
    command_path: Path
    requests_path: Path
    usage: CodexUsage | None


def default_codex_bin() -> Path:
    configured = os.getenv("ARC3_CODEX_BIN")
    candidate = (
        Path(configured).expanduser()
        if configured
        else DEFAULT_PINNED_CODEX_BIN
    )
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(
            f"Pinned Codex runtime not found: {candidate}. Install with: "
            "npm install --prefix ~/.local/share/arc3-codex/0.145.0 "
            "--omit=dev --no-audit --no-fund @openai/codex@0.145.0"
        )
    resolved = candidate.resolve()
    completed = subprocess.run(
        [str(resolved), "--version"],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    version = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 or version != PINNED_CODEX_CLI_VERSION:
        raise RuntimeError(
            f"Expected {PINNED_CODEX_CLI_VERSION}, found {version or 'unknown'} "
            f"at {resolved}"
        )
    return resolved


def default_native_codex_bin() -> Path:
    codex = default_codex_bin()
    if codex.name == "codex.js":
        root = codex.parent.parent
        native = (
            root
            / "node_modules"
            / "@openai"
            / "codex-linux-x64"
            / "vendor"
            / "x86_64-unknown-linux-musl"
            / "bin"
            / "codex"
        )
        if native.exists():
            return native.resolve()
    return codex


def default_codex_auth_file() -> Path:
    return Path.home() / ".codex" / "auth.json"


def default_ca_bundle() -> Path:
    configured = os.getenv("SSL_CERT_FILE")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path.resolve()
    for candidate in (
        Path("/etc/ssl/certs/ca-certificates.crt"),
        Path("/etc/pki/tls/certs/ca-bundle.crt"),
    ):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not find a host CA bundle; set SSL_CERT_FILE")


def prepare_codex_home(target: Path) -> Path:
    """Create writable per-run state without copying credentials."""
    target.mkdir(parents=True, exist_ok=True)
    target.chmod(0o700)
    for dirname in ("sessions", "tmp"):
        child = target / dirname
        child.mkdir(exist_ok=True)
        child.chmod(0o700)

    auth_link = target / "auth.json"
    if not auth_link.exists() and not auth_link.is_symlink():
        auth_link.symlink_to("/run/codex-auth.json")
    elif not auth_link.is_symlink() or os.readlink(auth_link) != "/run/codex-auth.json":
        raise RuntimeError(f"Unexpected auth entry in run-local Codex home: {auth_link}")
    return target


def write_codex_config(
    codex_home: Path,
    normal_auto_compact_limit: int = NORMAL_AUTO_COMPACT_LIMIT,
) -> Path:
    """Write the complete player-local Codex configuration."""
    instructions = codex_home / Path(PLAYER_BASE_INSTRUCTIONS_PATH).name
    instructions.write_text(PLAYER_BASE_INSTRUCTIONS, encoding="utf-8")
    instructions.chmod(0o600)

    config = codex_home / "config.toml"
    config.write_text(
        "\n".join(
            [
                f'model_instructions_file = "{PLAYER_BASE_INSTRUCTIONS_PATH}"',
                "model_auto_compact_token_limit = "
                f"{normal_auto_compact_limit}",
                'model_auto_compact_token_limit_scope = "total"',
                'personality = "none"',
                'web_search = "disabled"',
                "",
                '[projects."/workspace"]',
                'trust_level = "trusted"',
                "",
                "[agents]",
                "interrupt_message = false",
                "",
                "[skills]",
                "include_instructions = false",
                "",
                "[skills.bundled]",
                "enabled = false",
                "",
                "[features]",
                "apps = false",
                "browser_use = false",
                "computer_use = false",
                "code_mode_host = true",
                "goals = false",
                "hooks = true",
                "image_generation = false",
                "multi_agent = false",
                "plugins = false",
                "remote_plugin = false",
                "shell_snapshot = false",
                "shell_tool = false",
                "tool_suggest = false",
                "unified_exec = false",
                "",
                "[features.code_mode]",
                "enabled = false",
                f'direct_only_tool_namespaces = ["{PLAYER_TOOL_NAMESPACE}"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    return config


def validate_codex_auth_file(auth: Path | None = None) -> Path:
    auth = (auth or default_codex_auth_file()).expanduser().resolve()
    if not auth.exists() or not auth.is_file():
        raise FileNotFoundError(
            f"Codex auth file not found at {auth}; run `codex login` first"
        )
    return auth


_ToolExecution = ToolExecution


_STREAM_CLOSED = object()


class _AppServerConnection:
    def __init__(
        self,
        process: subprocess.Popen[str],
        *,
        stdout_path: Path,
        stderr_path: Path,
        requests_path: Path,
    ) -> None:
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise RuntimeError("app-server pipes are unavailable")
        self.process = process
        self.stdin = process.stdin
        self._messages: queue.Queue[object] = queue.Queue()
        self._request_log = requests_path.open("w", encoding="utf-8")
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(process.stdout, stdout_path),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._copy_stream,
            args=(process.stderr, stderr_path),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self, stream: Any, path: Path) -> None:
        with path.open("w", encoding="utf-8") as output:
            for line in stream:
                try:
                    logged = _sanitize_wire_message(json.loads(line))
                    output.write(json.dumps(logged, separators=(",", ":")) + "\n")
                except json.JSONDecodeError:
                    output.write(line)
                output.flush()
                self._messages.put(line)
        self._messages.put(_STREAM_CLOSED)

    @staticmethod
    def _copy_stream(stream: Any, path: Path) -> None:
        with path.open("w", encoding="utf-8") as output:
            for line in stream:
                output.write(line)
                output.flush()

    def send(self, message: dict[str, Any]) -> None:
        encoded = json.dumps(message, separators=(",", ":"))
        self.stdin.write(encoded + "\n")
        self.stdin.flush()
        self._request_log.write(
            json.dumps(_sanitize_wire_message(message), separators=(",", ":")) + "\n"
        )
        self._request_log.flush()

    def receive(self, deadline: float) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Codex app-server turn timed out")
        try:
            item = self._messages.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("Codex app-server turn timed out") from exc
        if item is _STREAM_CLOSED:
            raise RuntimeError(
                f"Codex app-server exited before the turn completed "
                f"(exit {self.process.poll()})"
            )
        try:
            message = json.loads(str(item))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Codex app-server emitted invalid JSON") from exc
        if not isinstance(message, dict):
            raise RuntimeError("Codex app-server emitted a non-object message")
        return message

    def close(self) -> int:
        try:
            self.stdin.close()
        except OSError:
            pass
        try:
            returncode = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                returncode = self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                returncode = self.process.wait(timeout=5)
        finally:
            self._stdout_thread.join(timeout=2)
            self._stderr_thread.join(timeout=2)
            self._request_log.close()
        return returncode


class DockerCodexRunner:
    def __init__(
        self,
        *,
        visible_dir: Path,
        codex_home: Path,
        auth_file: Path,
        io_dir: Path,
        controller: Any,
        codex_bin: Path | None = None,
        ca_bundle: Path | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        compact_thresholds: CompactThresholds | None = None,
        task_timeout: int = 3600,
        image: str = PLAYER_IMAGE,
        display_size: int = 1024,
    ) -> None:
        self.visible_dir = visible_dir.resolve()
        self.guide_path = self.visible_dir / "GUIDE.md"
        self.working_path = self.visible_dir / "WORKING.md"
        self.agents_path = self.visible_dir / "AGENTS.md"
        self.agents_instructions = self.agents_path.read_text(encoding="utf-8")
        self.screenshots_dir = self.visible_dir / "screenshots"
        self.codex_project_dir = self.visible_dir / ".codex"
        self.codex_home = codex_home.resolve()
        self.auth_file = validate_codex_auth_file(auth_file)
        self.io_dir = io_dir.resolve()
        self.exchange_dir = self.io_dir / "exchange"
        self.controller = controller
        self.tool_dispatcher = GameToolDispatcher(
            controller=controller,
            guide_path=self.guide_path,
            working_path=self.working_path,
        )
        self.codex_bin = codex_bin or default_native_codex_bin()
        self.codex_vendor_dir = self.codex_bin.parent.parent
        bundled_bwrap = self.codex_vendor_dir / "codex-resources" / "bwrap"
        if not bundled_bwrap.is_file():
            raise FileNotFoundError(
                f"Codex sandbox resource not found beside native binary: {bundled_bwrap}"
            )
        self.ca_bundle = (ca_bundle or default_ca_bundle()).resolve()
        self.model = model
        self.reasoning_effort = reasoning_effort
        if type(display_size) is not int or display_size < 1:
            raise ValueError("display_size must be a positive integer")
        self.display_size = display_size
        self.compact_thresholds = compact_thresholds or resolve_compact_thresholds(
            model
        )
        self.task_timeout = task_timeout
        self.image = image
        self._segment_index = 0
        self._rpc_id = 0
        self._connection: _AppServerConnection | None = None
        self._pending_responses: dict[Any, dict[str, Any]] = {}
        self._active_thread_id: str | None = None
        self._active_turn_id: str | None = None
        self._completed_turn: dict[str, Any] | None = None
        self._awaited_tool_call: str | None = None
        self._completed_tool_call: str | None = None
        self._final_message = ""
        self._segment_usage: CodexUsage | None = None
        self.io_dir.mkdir(parents=True, exist_ok=True)
        self.exchange_dir.mkdir(exist_ok=True)
        write_codex_config(
            self.codex_home,
            self.compact_thresholds.normal_auto_compact_limit,
        )

        required = (
            self.guide_path,
            self.agents_path,
            self.screenshots_dir,
            self.codex_project_dir,
            self.codex_home / "hooks.json",
            self.ca_bundle,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Incomplete player runtime: " + ", ".join(missing))

    def run_task(self, prompt: str) -> CodexResult:
        return self._run_segment(
            prompt=prompt,
            session_id=None,
            compact_limit=self.compact_thresholds.normal_auto_compact_limit,
            initial_images=self.controller.initial_image_paths(),
        )

    def run_recovery_task(self, recovery: CompactRecovery) -> CodexResult:
        return self._run_segment(
            prompt=compact_recovery_prompt(recovery),
            session_id=None,
            compact_limit=self.compact_thresholds.normal_auto_compact_limit,
            initial_images=recovery.image_paths,
            recovery=recovery,
        )

    def resume_recovery_task(
        self,
        session_id: str,
        recovery: CompactRecovery,
    ) -> CodexResult:
        return self._run_segment(
            prompt=compact_recovery_prompt(recovery),
            session_id=session_id,
            compact_limit=self.compact_thresholds.normal_auto_compact_limit,
            initial_images=recovery.image_paths,
            recovery=recovery,
        )

    def run_retry_task(self, recovery: RetryRecovery) -> CodexResult:
        return self._run_segment(
            prompt=retry_recovery_prompt(recovery),
            session_id=None,
            compact_limit=self.compact_thresholds.normal_auto_compact_limit,
            initial_images=recovery.image_paths,
            retry_recovery=recovery,
        )

    def resume_retry_task(
        self,
        session_id: str,
        recovery: RetryRecovery,
    ) -> CodexResult:
        return self._run_segment(
            prompt=retry_recovery_prompt(recovery),
            session_id=session_id,
            compact_limit=self.compact_thresholds.normal_auto_compact_limit,
            initial_images=recovery.image_paths,
            retry_recovery=recovery,
        )

    def resume_runtime_recovery(
        self,
        session_id: str,
        recovery: RuntimeRecovery,
    ) -> CodexResult:
        return self._run_segment(
            prompt=runtime_recovery_prompt(recovery),
            session_id=session_id,
            compact_limit=self.compact_thresholds.normal_auto_compact_limit,
            initial_images=recovery.image_paths,
            runtime_recovery=recovery,
        )

    def resume_checkpoint(self, session_id: str, prompt: str) -> CodexResult:
        return self._run_segment(
            prompt=prompt,
            session_id=session_id,
            compact_limit=self.compact_thresholds.checkpoint_auto_compact_limit,
            initial_images=(),
        )

    def resume_nonterminal_task(
        self,
        session_id: str,
        prompt: str,
    ) -> CodexResult:
        return self._run_segment(
            prompt=prompt,
            session_id=session_id,
            compact_limit=self.compact_thresholds.normal_auto_compact_limit,
            initial_images=(),
        )

    def _next_segment(self) -> int:
        segment = self._segment_index
        self._segment_index += 1
        return segment

    def _run_segment(
        self,
        *,
        prompt: str,
        session_id: str | None,
        compact_limit: int,
        initial_images: Any,
        recovery: CompactRecovery | None = None,
        retry_recovery: RetryRecovery | None = None,
        runtime_recovery: RuntimeRecovery | None = None,
    ) -> CodexResult:
        segment = self._next_segment()
        stem = f"segment_{segment:03d}"
        stdout_path = self.io_dir / f"{stem}.stdout.jsonl"
        stderr_path = self.io_dir / f"{stem}.stderr.txt"
        requests_path = self.io_dir / f"{stem}.requests.jsonl"
        command_path = self.io_dir / f"{stem}.command.json"
        output_path = self.exchange_dir / f"{stem}.last.txt"
        write_codex_config(self.codex_home, compact_limit)
        docker_cmd = self._docker_command()
        command_path.write_text(json.dumps(docker_cmd, indent=2), encoding="utf-8")

        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                docker_cmd,
                cwd=self.visible_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            connection = _AppServerConnection(
                process,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                requests_path=requests_path,
            )
        except Exception as exc:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            error_text = f"{type(exc).__name__}: {exc}"
            stderr_path.write_text(error_text + "\n", encoding="utf-8")
            stdout_path.touch()
            requests_path.touch()
            output_path.write_text("", encoding="utf-8")
            return CodexResult(
                segment=segment,
                returncode=125,
                final_message="",
                session_id=session_id,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_path=output_path,
                command_path=command_path,
                requests_path=requests_path,
                usage=None,
            )
        self._connection = connection
        self._pending_responses = {}
        self._active_thread_id = None
        self._active_turn_id = None
        self._completed_turn = None
        self._awaited_tool_call = None
        self._completed_tool_call = None
        self._final_message = ""
        self._segment_usage = None
        deadline = time.monotonic() + self.task_timeout
        returncode = 0
        error_text = ""
        resolved_session_id = session_id

        try:
            self._rpc(
                "initialize",
                {
                    "clientInfo": {
                        "name": APP_SERVER_CLIENT_NAME,
                        "title": "Visual Game Harness",
                        "version": "1.0.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
                deadline,
            )
            connection.send({"method": "initialized", "params": {}})
            self._ensure_compact_hooks_trusted(deadline)

            common = self._thread_parameters(compact_limit)

            if session_id is None:
                common["environments"] = []
                common["dynamicTools"] = self._dynamic_tools()
                thread_result = self._rpc("thread/start", common, deadline)
            else:
                thread_result = self._rpc(
                    "thread/resume",
                    {"threadId": session_id, "excludeTurns": True, **common},
                    deadline,
                )
            thread = thread_result.get("thread")
            if not isinstance(thread, dict) or not isinstance(thread.get("id"), str):
                raise RuntimeError("app-server returned no thread id")
            self._active_thread_id = thread["id"]
            resolved_session_id = self._active_thread_id

            turn_input: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for image_path in initial_images:
                turn_input.append(_image_input_from_path(Path(image_path)))
            turn_params: dict[str, Any] = {
                "threadId": self._active_thread_id,
                "input": turn_input,
            }
            if self.model:
                turn_params["model"] = self.model
            if self.reasoning_effort:
                turn_params["effort"] = self.reasoning_effort
            turn_result = self._rpc("turn/start", turn_params, deadline)
            turn = turn_result.get("turn")
            if not isinstance(turn, dict) or not isinstance(turn.get("id"), str):
                raise RuntimeError("app-server returned no turn id")
            self._active_turn_id = turn["id"]
            if recovery is not None:
                self.controller.complete_compact_recovery(recovery)
            if retry_recovery is not None:
                self.controller.complete_retry_recovery(retry_recovery)
            if runtime_recovery is not None:
                self.controller.complete_runtime_recovery(runtime_recovery)

            while self._completed_turn is None:
                self._dispatch(connection.receive(deadline), deadline)

            status = self._completed_turn.get("status")
            compact_paused = bool(
                self.controller.compact_checkpoint_marker is not None
                and self.controller.compact_checkpoint_marker.exists()
            )
            retry_marker = getattr(
                self.controller, "retry_boundary_marker", None
            )
            retry_paused = bool(
                retry_marker is not None and retry_marker.exists()
            )
            if (
                status not in {"completed", "interrupted"}
                and not compact_paused
                and not retry_paused
            ):
                returncode = 1
                error = self._completed_turn.get("error")
                error_text = f"Turn ended with status {status}: {error}"
        except TimeoutError as exc:
            returncode = 124
            error_text = str(exc)
        except Exception as exc:
            returncode = 1
            error_text = f"{type(exc).__name__}: {exc}"
        finally:
            process_returncode = connection.close()
            self._connection = None
            if returncode == 0 and process_returncode != 0:
                returncode = process_returncode
            if error_text:
                with stderr_path.open("a", encoding="utf-8") as stderr:
                    stderr.write(error_text + "\n")

        output_path.write_text(self._final_message, encoding="utf-8")
        return CodexResult(
            segment=segment,
            returncode=returncode,
            final_message=self._final_message,
            session_id=resolved_session_id,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            output_path=output_path,
            command_path=command_path,
            requests_path=requests_path,
            usage=self._segment_usage,
        )

    def _thread_parameters(self, compact_limit: int) -> dict[str, Any]:
        config: dict[str, Any] = {
            "model_auto_compact_token_limit": compact_limit,
        }
        if self.reasoning_effort:
            config["model_reasoning_effort"] = self.reasoning_effort
        common: dict[str, Any] = {
            "cwd": "/workspace",
            "approvalPolicy": "never",
            "sandbox": "workspace-write",
            "baseInstructions": PLAYER_BASE_INSTRUCTIONS,
            "developerInstructions": self.agents_instructions,
            "personality": "none",
            "config": config,
        }
        if self.model:
            common["model"] = self.model
        return common

    def _docker_command(self) -> list[str]:
        uid = str(os.getuid())
        gid = str(os.getgid())
        return [
            "docker",
            "run",
            "-i",
            "--rm",
            "--pull=never",
            "--network",
            "bridge",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--security-opt",
            "seccomp=unconfined",
            "--security-opt",
            "apparmor=unconfined",
            "--pids-limit",
            "256",
            "--memory",
            "16g",
            "--cpus",
            "4",
            "--user",
            f"{uid}:{gid}",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=256m",
            "--tmpfs",
            "/home/codex:rw,nosuid,nodev,size=64m",
            "--tmpfs",
            "/workspace:rw,nosuid,nodev,size=16m",
            "-e",
            "CODEX_HOME=/codex-home",
            "-e",
            "HOME=/home/codex",
            "-e",
            "TERM=dumb",
            "-e",
            "NO_COLOR=1",
            "-e",
            "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt",
            "-v",
            f"{self.codex_vendor_dir}:/opt/codex:ro",
            "-v",
            f"{self.ca_bundle}:/etc/ssl/certs/ca-certificates.crt:ro",
            "-v",
            f"{self.codex_home}:/codex-home:rw",
            "-v",
            f"{self.auth_file}:/run/codex-auth.json:ro",
            "-v",
            f"{self.codex_project_dir}:/workspace/.codex:ro",
            "-w",
            "/workspace",
            self.image,
            CONTAINER_CODEX_BIN,
            "app-server",
            "--stdio",
            "--strict-config",
        ]

    def _dynamic_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "namespace",
                "name": PLAYER_TOOL_NAMESPACE,
                "description": "Visual game tools.",
                "tools": [
                    {
                        "type": "function",
                        "name": tool["name"],
                        "description": tool["description"],
                        "inputSchema": tool["inputSchema"],
                    }
                    for tool in build_tools(self.display_size)
                ],
            }
        ]

    def _next_rpc_id(self) -> int:
        request_id = self._rpc_id
        self._rpc_id += 1
        return request_id

    def _ensure_compact_hooks_trusted(self, deadline: float) -> None:
        hooks = self._compact_hooks(
            self._rpc(
                "hooks/list",
                {"cwds": ["/workspace"]},
                deadline,
            )
        )
        untrusted = {
            hook["key"]: {"trusted_hash": hook["currentHash"]}
            for hook in hooks.values()
            if hook.get("trustStatus") not in {"trusted", "managed"}
        }
        if untrusted:
            self._rpc(
                "config/batchWrite",
                {
                    "edits": [
                        {
                            "keyPath": "hooks.state",
                            "value": untrusted,
                            "mergeStrategy": "upsert",
                        }
                    ],
                    "reloadUserConfig": True,
                },
                deadline,
            )
            hooks = self._compact_hooks(
                self._rpc(
                    "hooks/list",
                    {"cwds": ["/workspace"]},
                    deadline,
                )
            )

        invalid = {
            event: hook.get("trustStatus")
            for event, hook in hooks.items()
            if not hook.get("enabled")
            or hook.get("trustStatus") not in {"trusted", "managed"}
        }
        if invalid:
            raise RuntimeError(f"Compact hooks are not trusted and enabled: {invalid}")

    @staticmethod
    def _compact_hooks(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        data = result.get("data")
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                hooks = entry.get("hooks")
                if not isinstance(hooks, list):
                    continue
                for hook in hooks:
                    if not isinstance(hook, dict):
                        continue
                    event = hook.get("eventName")
                    if (
                        event in COMPACT_HOOK_EVENTS
                        and hook.get("sourcePath") == COMPACT_HOOK_SOURCE
                        and isinstance(hook.get("key"), str)
                        and isinstance(hook.get("currentHash"), str)
                    ):
                        found[event] = hook
        missing = COMPACT_HOOK_EVENTS.difference(found)
        if missing:
            raise RuntimeError(
                f"Required compact hooks were not discovered: {sorted(missing)}"
            )
        return found

    def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        deadline: float,
    ) -> dict[str, Any]:
        if self._connection is None:
            raise RuntimeError("app-server is not running")
        request_id = self._next_rpc_id()
        self._connection.send({"id": request_id, "method": method, "params": params})
        while request_id not in self._pending_responses:
            self._dispatch(self._connection.receive(deadline), deadline)
        response = self._pending_responses.pop(request_id)
        if "error" in response:
            raise RuntimeError(f"{method} failed: {response['error']}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"{method} returned no result object")
        return result

    def _dispatch(self, message: dict[str, Any], deadline: float) -> None:
        method = message.get("method")
        request_id = message.get("id")
        if isinstance(method, str) and request_id is not None:
            if method == "item/tool/call":
                self._handle_tool_call(message, deadline)
            elif self._connection is not None:
                self._connection.send(
                    {
                        "id": request_id,
                        "error": {"code": -32601, "message": "Unsupported request"},
                    }
                )
            return
        if method is None and request_id is not None:
            self._pending_responses[request_id] = message
            return
        if isinstance(method, str):
            self._observe_notification(method, message.get("params"))

    def _observe_notification(self, method: str, params: Any) -> None:
        if not isinstance(params, dict):
            return
        if method == "turn/started":
            turn = params.get("turn")
            if (
                self._active_turn_id is None
                and params.get("threadId") == self._active_thread_id
                and isinstance(turn, dict)
                and isinstance(turn.get("id"), str)
            ):
                self._active_turn_id = turn["id"]
        elif method == "turn/completed":
            turn = params.get("turn")
            if (
                isinstance(turn, dict)
                and turn.get("id") == self._active_turn_id
            ):
                self._completed_turn = turn
        elif method == "item/completed":
            item = params.get("item")
            if isinstance(item, dict):
                if item.get("type") == "dynamicToolCall":
                    call_id = item.get("id")
                    if call_id == self._awaited_tool_call:
                        self._completed_tool_call = call_id
                elif item.get("type") == "agentMessage":
                    text = item.get("text")
                    if isinstance(text, str):
                        self._final_message = text
        elif method == "thread/tokenUsage/updated":
            token_usage = params.get("tokenUsage")
            if (
                params.get("turnId") == self._active_turn_id
                and isinstance(token_usage, dict)
                and isinstance(token_usage.get("last"), dict)
            ):
                self._segment_usage = _add_usage(
                    self._segment_usage,
                    token_usage["last"],
                )

    def _handle_tool_call(
        self,
        message: dict[str, Any],
        deadline: float,
    ) -> None:
        if self._connection is None:
            raise RuntimeError("app-server is not running")
        request_id = message.get("id")
        params = message.get("params")
        if not isinstance(params, dict):
            self._connection.send(
                {
                    "id": request_id,
                    "result": {
                        "contentItems": [
                            {"type": "inputText", "text": "Invalid tool request."}
                        ],
                        "success": False,
                    },
                }
            )
            return
        tool = params.get("tool")
        namespace = params.get("namespace")
        call_id = params.get("callId")
        arguments = params.get("arguments")
        if (
            namespace != PLAYER_TOOL_NAMESPACE
            or not isinstance(tool, str)
            or not isinstance(call_id, str)
        ):
            execution = _ToolExecution("Invalid tool request.", False)
        else:
            execution = self._execute_tool(tool, arguments, call_id)

        response = {
            "id": request_id,
            "result": {
                "contentItems": [
                    {"type": "inputText", "text": execution.text}
                ],
                "success": execution.success,
            },
        }
        if tool == "inspect" and execution.images:
            self._awaited_tool_call = call_id
            self._connection.send(response)
            try:
                self._wait_for_tool_completion(call_id, deadline)
                self._start_inspection_turn(
                    execution.images,
                    deadline,
                )
            except Exception:
                raise
            return

        if execution.images and not execution.interrupt_after:
            try:
                for index, image in enumerate(execution.images):
                    self._steer_visual(
                        image,
                        f"{call_id}-{index + 1}",
                        deadline,
                    )
            except Exception:
                execution = _ToolExecution(
                    "The resulting visual could not be delivered.", False
                )

        self._connection.send(
            {
                **response,
                "result": {
                    "contentItems": [
                        {"type": "inputText", "text": execution.text}
                    ],
                    "success": execution.success,
                },
            }
        )
        if execution.interrupt_after:
            try:
                self._finish_retry_boundary(deadline)
            except Exception:
                raise

    def _finish_retry_boundary(self, deadline: float) -> None:
        if self._active_thread_id is None or self._active_turn_id is None:
            raise RuntimeError("No active turn to interrupt")
        turn_id = self._active_turn_id
        try:
            self._rpc(
                "turn/interrupt",
                {
                    "threadId": self._active_thread_id,
                    "turnId": turn_id,
                },
                deadline,
            )
        except RuntimeError as exc:
            if "no active turn to interrupt" not in str(exc):
                raise
        self._wait_for_turn_completion(turn_id, deadline)

    def _wait_for_turn_completion(self, turn_id: str, deadline: float) -> None:
        if self._connection is None:
            raise RuntimeError("app-server is not running")
        while (
            not isinstance(self._completed_turn, dict)
            or self._completed_turn.get("id") != turn_id
        ):
            self._dispatch(self._connection.receive(deadline), deadline)

    def _wait_for_tool_completion(self, call_id: str, deadline: float) -> None:
        if self._connection is None:
            raise RuntimeError("app-server is not running")
        while self._completed_tool_call != call_id:
            self._dispatch(self._connection.receive(deadline), deadline)
        self._awaited_tool_call = None
        self._completed_tool_call = None

    def _start_inspection_turn(
        self,
        images: tuple[dict[str, Any], ...],
        deadline: float,
    ) -> None:
        if self._active_thread_id is None or self._active_turn_id is None:
            raise RuntimeError("No active turn for inspection")

        previous_turn_id = self._active_turn_id
        self._rpc(
            "turn/interrupt",
            {
                "threadId": self._active_thread_id,
                "turnId": previous_turn_id,
            },
            deadline,
        )
        while (
            not isinstance(self._completed_turn, dict)
            or self._completed_turn.get("id") != previous_turn_id
        ):
            if self._connection is None:
                raise RuntimeError("app-server is not running")
            self._dispatch(self._connection.receive(deadline), deadline)
        if (
            not isinstance(self._completed_turn, dict)
            or self._completed_turn.get("status") not in {"interrupted", "completed"}
        ):
            raise RuntimeError("Inspection turn boundary was not completed")

        turn_input: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f'<inspection_visuals count="{len(images)}">'
                    "Results of the completed inspect call follow in request order; "
                    "the current game state is unchanged."
                    "</inspection_visuals>"
                ),
            }
        ]
        for image in images:
            data = image.get("data")
            steer_text = image.get("steer_text")
            if not isinstance(data, str) or not isinstance(steer_text, str):
                raise RuntimeError("Controller returned incomplete inspection visual")
            mime_type = image.get("mime_type", "image/png")
            turn_input.extend(
                [
                    {"type": "text", "text": steer_text},
                    {
                        "type": "image",
                        "url": f"data:{mime_type};base64,{data}",
                        "detail": image.get("detail", "original"),
                    },
                ]
            )

        self._completed_turn = None
        self._active_turn_id = None
        self._final_message = ""
        turn_params: dict[str, Any] = {
            "threadId": self._active_thread_id,
            "input": turn_input,
        }
        if self.model:
            turn_params["model"] = self.model
        if self.reasoning_effort:
            turn_params["effort"] = self.reasoning_effort
        result = self._rpc("turn/start", turn_params, deadline)
        turn = result.get("turn")
        if not isinstance(turn, dict) or not isinstance(turn.get("id"), str):
            raise RuntimeError("app-server returned no inspection turn id")
        self._active_turn_id = turn["id"]

    def _steer_visual(
        self,
        image: dict[str, Any],
        call_id: str,
        deadline: float,
    ) -> None:
        if self._active_thread_id is None or self._active_turn_id is None:
            raise RuntimeError("No active turn for the resulting visual")
        data = image.get("data")
        if not isinstance(data, str):
            raise RuntimeError("Controller returned no visual data")
        steer_text = image.get("steer_text")
        if not isinstance(steer_text, str) or not steer_text:
            raise RuntimeError("Controller returned no visual identity")
        mime_type = image.get("mime_type", "image/png")
        self._rpc(
            "turn/steer",
            {
                "threadId": self._active_thread_id,
                "expectedTurnId": self._active_turn_id,
                "clientUserMessageId": f"visual-{call_id}",
                "input": [
                    {"type": "text", "text": steer_text},
                    {
                        "type": "image",
                        "url": f"data:{mime_type};base64,{data}",
                        "detail": image.get("detail", "original"),
                    },
                ],
            },
            deadline,
        )

    def _execute_tool(
        self,
        tool: str,
        arguments: Any,
        call_id: str,
    ) -> ToolExecution:
        return self.tool_dispatcher.execute(tool, arguments, call_id)

def _image_input_from_path(path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image",
        "url": f"data:image/png;base64,{encoded}",
        "detail": "original",
    }


def _sanitize_wire_message(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_wire_message(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_wire_message(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        header, separator, payload = value.partition(",")
        if separator:
            try:
                raw = base64.b64decode(payload, validate=True)
            except ValueError:
                raw = payload.encode("ascii", errors="replace")
            digest = hashlib.sha256(raw).hexdigest()
            return f"<{header};bytes={len(raw)};sha256={digest}>"
    return value


def _add_usage(current: CodexUsage | None, raw: dict[str, Any]) -> CodexUsage:
    current = current or CodexUsage(0, 0, 0, 0)
    return CodexUsage(
        input_tokens=current.input_tokens + int(raw.get("inputTokens", 0)),
        cached_input_tokens=(
            current.cached_input_tokens + int(raw.get("cachedInputTokens", 0))
        ),
        output_tokens=current.output_tokens + int(raw.get("outputTokens", 0)),
        reasoning_output_tokens=(
            current.reasoning_output_tokens
            + int(raw.get("reasoningOutputTokens", 0))
        ),
    )
