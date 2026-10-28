"""Shared sharpness: Python implementation and Python↔JS agreement."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tooclose.sharpness import (
    BT601_B,
    BT601_G,
    BT601_R,
    bgr_to_gray_bt601,
    fingertip_sharpness,
    laplacian_variance,
    resize_gray_bilinear,
)

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "frontend" / "src" / "lib" / "sharpness_cli.ts"


def checker_bgr(width: int = 160, height: int = 120) -> np.ndarray:
    """Deterministic high-frequency pattern (must match sharpness.test.ts)."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image[y, x, 0] = (x * 13 + y * 7) % 256  # B
            image[y, x, 1] = (x * 3 + y * 17) % 256  # G
            image[y, x, 2] = (x * y) % 256  # R
    return image


def test_bt601_matches_documented_coefficients() -> None:
    gray = bgr_to_gray_bt601(10.0, 20.0, 30.0)
    assert gray == pytest.approx(BT601_R * 30.0 + BT601_G * 20.0 + BT601_B * 10.0)


def test_uniform_patch_has_near_zero_sharpness() -> None:
    frame = np.full((80, 80, 3), 128, dtype=np.uint8)
    score = fingertip_sharpness(frame, (40.0, 40.0), (40.0, 20.0))
    assert score == pytest.approx(0.0, abs=1e-9)


def test_textured_patch_has_positive_sharpness() -> None:
    frame = checker_bgr()
    score = fingertip_sharpness(frame, (80.0, 60.0), (80.0, 20.0))
    assert score > 100.0


def test_identity_resize_when_crop_is_64() -> None:
    """DIP→TIP chosen so crop side = 64; bilinear is a pixel-center identity."""
    gray = [[float(x + 10 * y) for x in range(64)] for y in range(64)]
    out = resize_gray_bilinear(gray, 64, 64)
    for y in range(64):
        for x in range(64):
            assert out[y][x] == pytest.approx(gray[y][x], abs=1e-9)


def test_laplacian_on_impulse() -> None:
    gray = [[0.0] * 5 for _ in range(5)]
    gray[2][2] = 1.0
    # Interior 3x3 of a 5x5. Center contribution: -4, neighbors 1.
    var = laplacian_variance(gray)
    assert var > 0.0


def test_tiny_dip_tip_returns_zero() -> None:
    frame = checker_bgr()
    assert fingertip_sharpness(frame, (10.0, 10.0), (10.0, 10.0)) == 0.0


def test_python_js_agreement_on_fixed_pattern() -> None:
    frame = checker_bgr(160, 120)
    tip = (80.0, 60.0)
    dip = (80.0, 20.0)
    python_score = fingertip_sharpness(frame, tip, dip)

    rgb = np.ascontiguousarray(frame[:, :, ::-1]).reshape(-1).astype(int).tolist()
    payload = {
        "width": 160,
        "height": 120,
        "rgb": rgb,
        "tip": {"x": tip[0], "y": tip[1]},
        "dip": {"x": dip[0], "y": dip[1]},
    }
    case_path = REPO / "frontend" / ".cache-sharpness-case.json"
    case_path.parent.mkdir(exist_ok=True)
    case_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        proc = subprocess.run(
            [
                "node",
                "--experimental-strip-types",
                "--no-warnings",
                str(CLI),
                str(case_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(REPO / "frontend"),
        )
    finally:
        case_path.unlink(missing_ok=True)

    assert proc.returncode == 0, proc.stderr
    js_score = json.loads(proc.stdout)["sharpness"]
    # Same formula, same bilinear; allow tiny float drift.
    assert js_score == pytest.approx(python_score, rel=1e-9, abs=1e-6)
