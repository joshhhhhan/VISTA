"""Fixed ARC-AGI-3 palette helpers."""

from __future__ import annotations

import os
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc3_mplconfig")

try:
    from arc_agi.rendering import COLOR_MAP as SDK_COLOR_MAP
except Exception:  # pragma: no cover - fallback only for broken installs.
    SDK_COLOR_MAP = {}


FALLBACK_COLOR_MAP: dict[int, str] = {
    0: "#FFFFFFFF",
    1: "#CCCCCCFF",
    2: "#999999FF",
    3: "#666666FF",
    4: "#333333FF",
    5: "#000000FF",
    6: "#E53AA3FF",
    7: "#FF7BCCFF",
    8: "#F93C31FF",
    9: "#1E93FFFF",
    10: "#88D8F1FF",
    11: "#FFDC00FF",
    12: "#FF851BFF",
    13: "#921231FF",
    14: "#4FCC30FF",
    15: "#A356D6FF",
}


COLOR_MAP: dict[int, str] = {
    idx: SDK_COLOR_MAP.get(idx, FALLBACK_COLOR_MAP[idx]) for idx in range(16)
}


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    raw = hex_color.lstrip("#")
    if len(raw) not in (6, 8):
        raise ValueError(f"Expected RGB/RGBA hex color, got {hex_color!r}")
    return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))


def palette_rgb() -> list[tuple[int, int, int]]:
    return [hex_to_rgb(COLOR_MAP[idx]) for idx in range(16)]


def palette_manifest() -> list[dict[str, object]]:
    return [
        {"index": idx, "hex": COLOR_MAP[idx], "rgb": list(hex_to_rgb(COLOR_MAP[idx]))}
        for idx in range(16)
    ]


def validate_palette_indices(values: Iterable[int]) -> None:
    invalid = sorted({int(value) for value in values if int(value) not in COLOR_MAP})
    if invalid:
        raise ValueError(f"Frame contains color indices outside 0..15: {invalid}")
