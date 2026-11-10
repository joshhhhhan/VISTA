"""Optional non-OpenAI providers for the Codex CLI transport."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodexProvider:
    name: str
    base_url: str
    env_key: str
    wire_api: str
    model: str
    model_context_window: int | None = None
    model_catalog_json: Path | None = None


DEEPSEEK_CODEX_PROVIDER = CodexProvider(
    name="deepseek",
    base_url="https://api.deepseek.com/",
    env_key="DEEPSEEK_API_KEY",
    wire_api="responses",
    model="deepseek-flash",
    model_context_window=1_048_576,
    model_catalog_json=Path.home() / ".codex" / "deepseek-models.json",
)
