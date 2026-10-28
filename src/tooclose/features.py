"""Landmarks → flat feature dict.

Pure function: no image pixels, no MediaPipe, no I/O. Inputs are one hand's
21 landmarks plus frame size, camera FOV, client-supplied sharpness, and the
**physical** handedness label from Phase 1.

Handedness convention
---------------------
``handedness`` is the user's physical hand. The selfie-mirror flip already
happened in ``hand_tracking.py``. This module must not flip the label again.
It *does* negate the raw palm-normal for the right hand so "pads facing the
camera" has the same sign for Left and Right (see ``docs/feature_math.md`` §5).

Vector math is written with tuples and ``math`` so it can be ported to
TypeScript without NumPy.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from tooclose.hand_tracking import (
    FINGER_DIP,
    FINGERTIPS,
    INDEX_MCP,
    INDEX_PIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    PINKY_MCP,
    PINKY_PIP,
    RING_MCP,
    RING_PIP,
    THUMB_IP,
    THUMB_MCP,
    WRIST,
    mirror_landmarks_x,
)

Vec3 = tuple[float, float, float]

# Default anatomy — keep in sync with config.yaml.
DEFAULT_PALM_WIDTH_CM = 8.5
DEFAULT_FINGERTIP_WIDTH_MM = 15.0

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# (mcp, mid, distal) for curl: thumb uses IP/TIP; others use PIP/DIP.
FINGER_CURL_JOINTS: dict[str, tuple[int, int, int]] = {
    "thumb": (THUMB_MCP, THUMB_IP, FINGERTIPS["thumb"]),
    "index": (INDEX_MCP, INDEX_PIP, FINGER_DIP["index"]),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, FINGER_DIP["middle"]),
    "ring": (RING_MCP, RING_PIP, FINGER_DIP["ring"]),
    "pinky": (PINKY_MCP, PINKY_PIP, FINGER_DIP["pinky"]),
}

_EPS = 1e-12


def _as_xyz(point: Sequence[float]) -> Vec3:
    x = float(point[0])
    y = float(point[1]) if len(point) > 1 else 0.0
    z = float(point[2]) if len(point) > 2 else 0.0
    return (x, y, z)


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    """Right-handed cross product. Same formula the TS port should use."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: Vec3) -> Vec3:
    length = _norm(a)
    if length < _EPS:
        return (0.0, 0.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def _scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _angle_deg(a: Vec3, b: Vec3) -> float | None:
    na = _normalize(a)
    nb = _normalize(b)
    if _norm(na) < _EPS or _norm(nb) < _EPS:
        return None
    cosine = _clamp(_dot(na, nb), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def pixel_xy(landmark: Sequence[float], frame_width: int, frame_height: int) -> tuple[float, float]:
    """MediaPipe (x, y) → pixel coordinates in the processed frame."""
    xyz = _as_xyz(landmark)
    return (xyz[0] * frame_width, xyz[1] * frame_height)


def to_camera_xyz(landmark: Sequence[float], frame_width: int, frame_height: int) -> Vec3:
    """Aspect-corrected 3D point in pixels. Z uses the MediaPipe x-scale."""
    xyz = _as_xyz(landmark)
    return (xyz[0] * frame_width, xyz[1] * frame_height, xyz[2] * frame_width)


def focal_length_px(frame_width: int, hfov_deg: float) -> float:
    """Pinhole focal length in pixels from horizontal FOV."""
    hfov_rad = math.radians(hfov_deg)
    denom = 2.0 * math.tan(hfov_rad / 2.0)
    if denom < _EPS:
        return float("inf")
    return frame_width / denom


def pinhole_distance_cm(palm_width_px: float, frame_width: int, hfov_deg: float, palm_width_cm: float) -> float | None:
    """Distance of the MCP plane from the camera, centimetres."""
    if palm_width_px < 1.0 or palm_width_cm <= 0:
        return None
    f_px = focal_length_px(frame_width, hfov_deg)
    return (palm_width_cm * f_px) / palm_width_px


def ppi_from_distance(distance_cm: float | None, frame_width: int, hfov_deg: float) -> float | None:
    """Pixels per inch at ``distance_cm`` (equivalent to a 15 mm pad / 25.4 mm)."""
    if distance_cm is None or distance_cm <= 0:
        return None
    f_px = focal_length_px(frame_width, hfov_deg)
    return 2.54 * f_px / distance_cm


def raw_palm_normal(landmarks: Sequence[Sequence[float]], frame_width: int, frame_height: int) -> Vec3:
    """``(index_mcp - wrist) × (pinky_mcp - wrist)`` before the handedness sign fix."""
    wrist = to_camera_xyz(landmarks[WRIST], frame_width, frame_height)
    index_mcp = to_camera_xyz(landmarks[INDEX_MCP], frame_width, frame_height)
    pinky_mcp = to_camera_xyz(landmarks[PINKY_MCP], frame_width, frame_height)
    return _cross(_sub(index_mcp, wrist), _sub(pinky_mcp, wrist))


def palmar_normal(
    landmarks: Sequence[Sequence[float]],
    frame_width: int,
    frame_height: int,
    handedness: str,
) -> Vec3:
    """Unit vector out of the finger pads.

    The raw cross product points palmar on a left hand and dorsal on a right
    hand when the pads face the camera. Negate for the physical right hand.
    """
    raw = raw_palm_normal(landmarks, frame_width, frame_height)
    physical = handedness.strip().capitalize()
    if physical == "Right":
        raw = _scale(raw, -1.0)
    return _normalize(raw)


def pad_facing_metrics(n_palm: Vec3) -> tuple[float, float, bool]:
    """Return (cos, angle_deg, pads_facing) vs the toward-camera axis (0, 0, -1)."""
    toward_camera: Vec3 = (0.0, 0.0, -1.0)
    if _norm(n_palm) < _EPS:
        return (0.0, 90.0, False)
    cosine = _clamp(_dot(n_palm, toward_camera), -1.0, 1.0)
    angle = math.degrees(math.acos(cosine))
    return (cosine, angle, cosine > 0.0)


def finger_curl_deg(
    landmarks: Sequence[Sequence[float]],
    finger: str,
    frame_width: int,
    frame_height: int,
) -> float | None:
    """0° ≈ fully extended, ~90° ≈ right-angle bend at the middle joint."""
    mcp_i, mid_i, distal_i = FINGER_CURL_JOINTS[finger]
    mcp = to_camera_xyz(landmarks[mcp_i], frame_width, frame_height)
    mid = to_camera_xyz(landmarks[mid_i], frame_width, frame_height)
    distal = to_camera_xyz(landmarks[distal_i], frame_width, frame_height)
    return _angle_deg(_sub(mid, mcp), _sub(distal, mid))


def _empty_features(
    *,
    handedness: str,
    confidence: float,
    frame_width: int,
    frame_height: int,
    hfov_deg: float,
    mirrored: bool,
    camera_type: str,
) -> dict[str, Any]:
    physical = handedness.strip().capitalize()
    row: dict[str, Any] = {
        "valid": False,
        "handedness_right": 1.0 if physical == "Right" else 0.0,
        "confidence": float(confidence),
        "frame_width": float(frame_width),
        "frame_height": float(frame_height),
        "hfov_deg": float(hfov_deg),
        "focal_length_px": focal_length_px(frame_width, hfov_deg) if hfov_deg > 0 else None,
        "palm_width_px": None,
        "palm_width_norm": None,
        "distance_cm": None,
        "ppi": None,
        "palm_normal_x": None,
        "palm_normal_y": None,
        "palm_normal_z": None,
        "pad_facing_cos": None,
        "pad_facing_angle_deg": None,
        "pads_facing_camera": 0.0,
        "mean_dip_tip_norm": None,
        "max_dip_tip_norm": None,
        "mean_sharpness": 0.0,
        "mirrored": 1.0 if mirrored else 0.0,
        "camera_type_phone": 1.0 if camera_type == "phone" else 0.0,
    }
    for name in FINGER_NAMES:
        row[f"{name}_dip_tip_px"] = None
        row[f"{name}_dip_tip_norm"] = None
        row[f"{name}_curl_deg"] = None
        row[f"{name}_sharpness"] = 0.0
    return row


def extract_features(
    landmarks: Sequence[Sequence[float]],
    *,
    handedness: str,
    confidence: float,
    frame_width: int,
    frame_height: int,
    hfov_deg: float,
    sharpness: Mapping[str, float] | None = None,
    palm_width_cm: float | None = None,
    fingertip_width_mm: float | None = None,
    mirrored: bool = True,
    camera_type: str = "webcam",
) -> dict[str, Any]:
    """Convert one hand's landmarks into a flat JSON-serialisable feature dict.

    ``fingertip_width_mm`` is accepted so callers can document the 15 mm
    assumption; ppi is computed from distance and does not need the pad width
    explicitly (see ``docs/feature_math.md`` §3).
    """
    palm_cm = DEFAULT_PALM_WIDTH_CM if palm_width_cm is None else palm_width_cm
    # Kept for the public signature / TS port; algebraically absorbed into ppi_from_distance.
    _ = DEFAULT_FINGERTIP_WIDTH_MM if fingertip_width_mm is None else fingertip_width_mm
    sharp = sharpness or {}

    if frame_width <= 0 or frame_height <= 0 or len(landmarks) < 21:
        return _empty_features(
            handedness=handedness,
            confidence=confidence,
            frame_width=max(frame_width, 0),
            frame_height=max(frame_height, 0),
            hfov_deg=hfov_deg,
            mirrored=mirrored,
            camera_type=camera_type,
        )

    row = _empty_features(
        handedness=handedness,
        confidence=confidence,
        frame_width=frame_width,
        frame_height=frame_height,
        hfov_deg=hfov_deg,
        mirrored=mirrored,
        camera_type=camera_type,
    )

    index_xy = pixel_xy(landmarks[INDEX_MCP], frame_width, frame_height)
    pinky_xy = pixel_xy(landmarks[PINKY_MCP], frame_width, frame_height)
    palm_width_px = math.hypot(index_xy[0] - pinky_xy[0], index_xy[1] - pinky_xy[1])
    row["palm_width_px"] = palm_width_px
    row["palm_width_norm"] = palm_width_px / frame_width

    distance = pinhole_distance_cm(palm_width_px, frame_width, hfov_deg, palm_cm)
    row["distance_cm"] = distance
    row["ppi"] = ppi_from_distance(distance, frame_width, hfov_deg)
    row["valid"] = distance is not None

    # Pixel lengths are mirror-invariant. Palm winding is not: un-flip x so the
    # left/right chirality correction matches the unmirrored camera frame.
    geom = landmarks
    if mirrored:
        geom = mirror_landmarks_x([_as_xyz(p) for p in landmarks])

    n_palm = palmar_normal(geom, frame_width, frame_height, handedness)
    row["palm_normal_x"], row["palm_normal_y"], row["palm_normal_z"] = n_palm
    cosine, angle, facing = pad_facing_metrics(n_palm)
    row["pad_facing_cos"] = cosine
    row["pad_facing_angle_deg"] = angle
    row["pads_facing_camera"] = 1.0 if facing else 0.0

    dip_norms: list[float] = []
    sharp_vals: list[float] = []
    for name in FINGER_NAMES:
        tip_i = FINGERTIPS[name]
        dip_i = FINGER_DIP[name]
        tip_xy = pixel_xy(landmarks[tip_i], frame_width, frame_height)
        dip_xy = pixel_xy(landmarks[dip_i], frame_width, frame_height)
        dip_tip_px = math.hypot(tip_xy[0] - dip_xy[0], tip_xy[1] - dip_xy[1])
        dip_tip_norm = dip_tip_px / frame_width
        row[f"{name}_dip_tip_px"] = dip_tip_px
        row[f"{name}_dip_tip_norm"] = dip_tip_norm
        dip_norms.append(dip_tip_norm)
        row[f"{name}_curl_deg"] = finger_curl_deg(landmarks, name, frame_width, frame_height)
        value = float(sharp.get(name, 0.0) or 0.0)
        row[f"{name}_sharpness"] = value
        sharp_vals.append(value)

    row["mean_dip_tip_norm"] = sum(dip_norms) / len(dip_norms)
    row["max_dip_tip_norm"] = max(dip_norms)
    row["mean_sharpness"] = sum(sharp_vals) / len(sharp_vals)
    return row
