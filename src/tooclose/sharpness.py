"""Shared fingertip sharpness (Python). Must match ``frontend/src/lib/sharpness.ts``.

Formula (see ``docs/feature_math.md`` §7):

1. Crop a square centered on the tip, side = 1.5 × DIP→TIP length in pixels.
2. Grayscale with BT.601 (``Y = 0.299 R + 0.587 G + 0.114 B``).
3. Bilinear resize to 64×64.
4. 3×3 Laplacian ``[[0,1,0],[1,-4,1],[0,1,0]]`` on interior pixels.
5. Population variance of the Laplacian.

OpenCV is used only as a convenient BGR ndarray container. Color conversion,
resize, and the Laplacian are explicit loops so the TypeScript port can copy
them line for line. Crops are never written to disk.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

CROP_SCALE = 1.5
RESIZE = 64
LAPLACIAN_KSIZE = 3  # documented 3×3 kernel; not cv2.Laplacian
BT601_R = 0.299
BT601_G = 0.587
BT601_B = 0.114


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def bgr_to_gray_bt601(b: float, g: float, r: float) -> float:
    """ITU-R BT.601 luma. Same coefficients as OpenCV ``COLOR_BGR2GRAY``."""
    return BT601_R * r + BT601_G * g + BT601_B * b


def sample_bilinear(gray: Sequence[Sequence[float]], x: float, y: float) -> float:
    """Edge-clamped bilinear sample from a 2D gray image."""
    height = len(gray)
    width = len(gray[0]) if height else 0
    if height == 0 or width == 0:
        return 0.0
    x_clamped = _clamp(x, 0.0, width - 1.0)
    y_clamped = _clamp(y, 0.0, height - 1.0)
    x0 = int(math.floor(x_clamped))
    y0 = int(math.floor(y_clamped))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    tx = x_clamped - x0
    ty = y_clamped - y0
    top = gray[y0][x0] * (1.0 - tx) + gray[y0][x1] * tx
    bot = gray[y1][x0] * (1.0 - tx) + gray[y1][x1] * tx
    return top * (1.0 - ty) + bot * ty


def resize_gray_bilinear(
    gray: Sequence[Sequence[float]],
    out_w: int = RESIZE,
    out_h: int = RESIZE,
) -> list[list[float]]:
    """Resize with pixel-center mapping: ``x = (i + 0.5) * src_w / out_w - 0.5``."""
    src_h = len(gray)
    src_w = len(gray[0]) if src_h else 0
    if src_h == 0 or src_w == 0:
        return [[0.0] * out_w for _ in range(out_h)]
    out: list[list[float]] = []
    for j in range(out_h):
        y = (j + 0.5) * src_h / out_h - 0.5
        row: list[float] = []
        for i in range(out_w):
            x = (i + 0.5) * src_w / out_w - 0.5
            row.append(sample_bilinear(gray, x, y))
        out.append(row)
    return out


def laplacian_variance(gray64: Sequence[Sequence[float]]) -> float:
    """Population variance of a 3×3 Laplacian on interior pixels of a 64×64 patch."""
    height = len(gray64)
    width = len(gray64[0]) if height else 0
    if height < 3 or width < 3:
        return 0.0
    total = 0.0
    total_sq = 0.0
    count = 0
    for y in range(1, height - 1):
        row = gray64[y]
        row_up = gray64[y - 1]
        row_dn = gray64[y + 1]
        for x in range(1, width - 1):
            value = row_up[x] + row_dn[x] + row[x - 1] + row[x + 1] - 4.0 * row[x]
            total += value
            total_sq += value * value
            count += 1
    if count == 0:
        return 0.0
    mean = total / count
    return total_sq / count - mean * mean


def crop_square_gray_bgr(
    frame_bgr: np.ndarray,
    cx: float,
    cy: float,
    side: float,
) -> list[list[float]]:
    """Clip an axis-aligned square out of a BGR image and return BT.601 gray."""
    height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    side_px = max(1, int(round(side)))
    x0 = int(math.floor(cx - side / 2.0))
    y0 = int(math.floor(cy - side / 2.0))
    x1 = x0 + side_px
    y1 = y0 + side_px
    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(width, x1)
    y1c = min(height, y1)
    if x1c <= x0c or y1c <= y0c:
        return []
    patch: list[list[float]] = []
    for y in range(y0c, y1c):
        row: list[float] = []
        for x in range(x0c, x1c):
            b, g, r = (float(v) for v in frame_bgr[y, x, :3])
            row.append(bgr_to_gray_bt601(b, g, r))
        patch.append(row)
    return patch


def fingertip_sharpness(
    frame_bgr: np.ndarray,
    tip_xy: tuple[float, float],
    dip_xy: tuple[float, float],
    *,
    crop_scale: float = CROP_SCALE,
    resize: int = RESIZE,
    laplacian_ksize: int = LAPLACIAN_KSIZE,
) -> float:
    """Variance of Laplacian on a 64×64 grayscale fingertip crop.

    ``laplacian_ksize`` is part of the shared contract (must be 3). The kernel
    is the explicit 3×3 4-neighbour Laplacian, not ``cv2.Laplacian``.
    """
    if laplacian_ksize != 3:
        raise ValueError("Shared sharpness definition requires a 3×3 Laplacian.")
    if frame_bgr.size == 0:
        return 0.0
    dip_tip = math.hypot(tip_xy[0] - dip_xy[0], tip_xy[1] - dip_xy[1])
    side = crop_scale * dip_tip
    if side < 1.0:
        return 0.0
    gray = crop_square_gray_bgr(frame_bgr, tip_xy[0], tip_xy[1], side)
    if not gray:
        return 0.0
    resized = resize_gray_bilinear(gray, resize, resize)
    return laplacian_variance(resized)


def all_fingertip_sharpness(
    frame_bgr: np.ndarray,
    landmarks_px: Mapping[str, tuple[tuple[float, float], tuple[float, float]]],
    *,
    crop_scale: float = CROP_SCALE,
) -> dict[str, float]:
    """Compute sharpness for each named finger. Values map name → ((tip), (dip))."""
    return {
        name: fingertip_sharpness(frame_bgr, tip, dip, crop_scale=crop_scale)
        for name, (tip, dip) in landmarks_px.items()
    }
