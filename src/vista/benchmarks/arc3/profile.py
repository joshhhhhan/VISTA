"""Resolve the ARC3 backend without translating its arguments."""

from __future__ import annotations

import importlib
from types import ModuleType

RELEASE_REVISION = "900aa3380e4f1120436d83b2ce1115a38ac29bf9"


def backend_module(backend: str) -> ModuleType:
    if backend not in {"codex", "claude", "dsh"}:
        raise ValueError(f"Unsupported ARC3 backend: {backend}")
    return importlib.import_module(f"vista.benchmarks.arc3.{backend}.harness")
