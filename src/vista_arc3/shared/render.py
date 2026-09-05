"""Lossless frame rendering for visual observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .palette import palette_rgb, validate_palette_indices


def normalize_frame(frame: object) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.int16)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got shape {arr.shape}")
    validate_palette_indices(arr.ravel())
    return arr.astype(np.uint8, copy=False)


def frame_to_rgb_array(frame: object) -> np.ndarray:
    arr = normalize_frame(frame)
    palette = np.asarray(palette_rgb(), dtype=np.uint8)
    return palette[arr]


def render_frame_image(
    frame: object,
    scale: int = 16,
    *,
    grid_lines: bool = False,
) -> Image.Image:
    if scale < 1:
        raise ValueError("scale must be >= 1")
    rgb = frame_to_rgb_array(frame)
    image = Image.fromarray(rgb, mode="RGB")
    if scale != 1:
        width, height = image.size
        image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    if grid_lines:
        pixels = np.asarray(image, dtype=np.uint16).copy()
        mask = np.zeros(pixels.shape[:2], dtype=bool)
        mask[::scale, :] = True
        mask[:, ::scale] = True
        pixels[mask] = (pixels[mask] * 3) // 4
        image = Image.fromarray(pixels.astype(np.uint8), mode="RGB")
    return image


def save_frame_png(
    frame: object,
    path: Path,
    scale: int = 16,
    *,
    grid_lines: bool = False,
) -> Path:
    image = render_frame_image(frame, scale=scale, grid_lines=grid_lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path
