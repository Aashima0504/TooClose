"""Deterministic 21-landmark hands for unit tests (no MediaPipe, no images)."""

from __future__ import annotations

from tooclose.hand_tracking import (
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_DIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_DIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    mirror_landmarks_x,
)

Landmark = tuple[float, float, float]

# Camera view, palm toward lens, fingers up:
#   Right: thumb on the left  (smaller x) → index x < pinky x
#   Left:  thumb on the right (larger x)  → index x > pinky x
_FINGER_SPREAD = (-0.06, 0.0, 0.06, 0.11)  # index, middle, ring, pinky from center


def _blank() -> list[Landmark]:
    return [(0.5, 0.5, 0.0)] * 21


def open_palm(
    handedness: str,
    *,
    palm_facing: bool = True,
    wrist: tuple[float, float] = (0.50, 0.78),
    finger_len: float = 0.28,
) -> list[Landmark]:
    """Planar open hand. ``palm_facing`` swaps index/pinky to show the dorsum."""
    points = _blank()
    wx, wy = wrist
    physical = handedness.strip().capitalize()
    side = 1.0 if physical == "Right" else -1.0
    if not palm_facing:
        side *= -1.0
    z = 0.0

    def chain(mcp_x: float, length: float) -> tuple[Landmark, Landmark, Landmark, Landmark]:
        mcp = (mcp_x, wy - 0.08, z)
        pip = (mcp_x, mcp[1] - length * 0.40, z)
        dip = (mcp_x, mcp[1] - length * 0.70, z)
        tip = (mcp_x, mcp[1] - length, z)
        return mcp, pip, dip, tip

    points[WRIST] = (wx, wy, 0.0)
    thumb_x = wx - side * 0.12
    points[THUMB_CMC] = (thumb_x, wy - 0.03, z)
    points[THUMB_MCP] = (thumb_x - side * 0.02, wy - 0.10, z)
    points[THUMB_IP] = (thumb_x - side * 0.03, wy - 0.16, z)
    points[THUMB_TIP] = (thumb_x - side * 0.04, wy - 0.22, z)

    index, middle, ring, pinky = (
        chain(wx + side * spread, finger_len * scale)
        for spread, scale in zip(_FINGER_SPREAD, (1.0, 1.05, 0.98, 0.82), strict=True)
    )
    points[INDEX_MCP], points[INDEX_PIP], points[INDEX_DIP], points[INDEX_TIP] = index
    points[MIDDLE_MCP], points[MIDDLE_PIP], points[MIDDLE_DIP], points[MIDDLE_TIP] = middle
    points[RING_MCP], points[RING_PIP], points[RING_DIP], points[RING_TIP] = ring
    points[PINKY_MCP], points[PINKY_PIP], points[PINKY_DIP], points[PINKY_TIP] = pinky
    return points


def curled_index(base: list[Landmark]) -> list[Landmark]:
    """Bend index ~90° at PIP (distal bone goes sideways in the image plane)."""
    points = list(base)
    pip = points[INDEX_PIP]
    mcp = points[INDEX_MCP]
    bone = abs(pip[1] - mcp[1])
    direction = 1.0 if pip[0] >= 0.5 else -1.0
    points[INDEX_DIP] = (pip[0] + direction * bone, pip[1], pip[2])
    points[INDEX_TIP] = (pip[0] + direction * bone * 1.6, pip[1], pip[2])
    return points


def selfie_mirror_hand(points: list[Landmark]) -> list[Landmark]:
    return mirror_landmarks_x(points)
