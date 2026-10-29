"""Synthetic landmark generator through a pinhole camera model. Phase 3.

Simulates 3D hands projected to 2D image coordinates across distances,
viewing angles, left/right hands, camera resolutions, and FOVs.
Extracts features using ``tooclose.features.extract_features``, adds sensor
noise, and generates structured DataFrames with subject/session groupings.

Marked with SYNTHETIC on all outputs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tooclose.baseline import ppi_to_risk, risk_to_state
from tooclose.config import AppConfig, load_config
from tooclose.features import FINGER_NAMES, extract_features, focal_length_px
from tooclose.hand_tracking import (
    FINGER_DIP,
    FINGERTIPS,
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
)

# Reference 3D hand canonical coordinates in centimeters (palm width ~8.5 cm)
# Wrist at origin (0, 0, 0), fingers pointing +Y, thumb towards -X for Right hand,
# palm normal pointing dorsal (+Z) for Right hand, so pads face -Z (towards camera).
_FINGER_SPREAD_CM = (-1.4, 0.0, 1.4, 2.7)  # index, middle, ring, pinky relative to center


def create_canonical_hand_3d(handedness: str = "Right") -> np.ndarray:
    """Return canonical 3D coordinates (21, 3) in centimeters.

    Convention:
    - Right hand: thumb is -X, pinky is +X.
    - Left hand: thumb is +X, pinky is -X.
    - Wrist at (0, 0, 0).
    - Fingers extend upwards in +Y.
    - Palmar side faces -Z (toward camera when camera looks along +Z).
    """
    pts = np.zeros((21, 3), dtype=np.float64)
    side = 1.0 if handedness.capitalize() == "Right" else -1.0

    pts[WRIST] = [0.0, 0.0, 0.0]

    # Thumb
    thumb_x = -2.8 * side
    pts[THUMB_CMC] = [thumb_x, 1.5, -0.4]
    pts[THUMB_MCP] = [thumb_x - 0.7 * side, 3.2, -0.8]
    pts[THUMB_IP] = [thumb_x - 1.2 * side, 4.8, -1.2]
    pts[THUMB_TIP] = [thumb_x - 1.6 * side, 6.2, -1.6]

    # 4 fingers: mcp, pip, dip, tip
    finger_joints = [
        (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, _FINGER_SPREAD_CM[0] * side, 7.8),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, _FINGER_SPREAD_CM[1] * side, 8.4),
        (RING_MCP, RING_PIP, RING_DIP, RING_TIP, _FINGER_SPREAD_CM[2] * side, 7.7),
        (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, _FINGER_SPREAD_CM[3] * side, 6.2),
    ]

    for mcp_idx, pip_idx, dip_idx, tip_idx, x_offset, length in finger_joints:
        pts[mcp_idx] = [x_offset, 4.0, 0.0]
        pts[pip_idx] = [x_offset, 4.0 + length * 0.45, 0.0]
        pts[dip_idx] = [x_offset, 4.0 + length * 0.75, 0.0]
        pts[tip_idx] = [x_offset, 4.0 + length * 1.00, 0.0]

    return pts


def rotate_hand_3d(
    pts_cm: np.ndarray,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """Rotate 3D points around X (pitch), Y (yaw), and Z (roll) axes."""
    rx = math.radians(pitch_deg)
    ry = math.radians(yaw_deg)
    rz = math.radians(roll_deg)

    # Rotation matrices
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(rx), -math.sin(rx)],
        [0.0, math.sin(rx), math.cos(rx)],
    ])
    Ry = np.array([
        [math.cos(ry), 0.0, math.sin(ry)],
        [0.0, 1.0, 0.0],
        [-math.sin(ry), 0.0, math.cos(ry)],
    ])
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0.0],
        [math.sin(rz), math.cos(rz), 0.0],
        [0.0, 0.0, 1.0],
    ])

    R = Rz @ Ry @ Rx
    return pts_cm @ R.T


def project_to_mediapipe(
    pts_cm: np.ndarray,
    distance_cm: float,
    frame_width: int,
    frame_height: int,
    hfov_deg: float,
    center_offset_x_cm: float = 0.0,
    center_offset_y_cm: float = 0.0,
) -> list[list[float]]:
    """Project 3D cm points to MediaPipe normalized coordinates [x, y, z].

    Camera is looking along +Z. The MCP/wrist plane is around Z = distance_cm.
    Image plane center is (0.5, 0.5).
    MediaPipe z is relative to wrist in units roughly matching image width.
    """
    f_px = focal_length_px(frame_width, hfov_deg)

    # Shift hand to distance_cm
    pts_in_cam = pts_cm.copy()
    pts_in_cam[:, 0] += center_offset_x_cm
    pts_in_cam[:, 1] += center_offset_y_cm
    # Note: MediaPipe Y increases downward on image, but hand extends up (+Y).
    # Flip Y so hand finger tips are higher up on screen (smaller y).
    pts_in_cam[:, 1] = -pts_in_cam[:, 1]
    pts_in_cam[:, 2] += distance_cm

    wrist_z = pts_in_cam[WRIST, 2]

    # Perspective projection
    # X_px = f_px * X_cm / Z_cm + frame_width / 2
    # Y_px = f_px * Y_cm / Z_cm + frame_height / 2
    X_px = f_px * pts_in_cam[:, 0] / pts_in_cam[:, 2] + (frame_width / 2.0)
    Y_px = f_px * pts_in_cam[:, 1] / pts_in_cam[:, 2] + (frame_height / 2.0)

    # MediaPipe depth z is wrist-relative, scaled roughly by frame_width:
    # Z_px = f_px * (Z_cm - wrist_z) / wrist_z
    # z_norm = Z_px / frame_width
    z_norm = (f_px * (pts_in_cam[:, 2] - wrist_z) / wrist_z) / frame_width

    x_norm = X_px / frame_width
    y_norm = Y_px / frame_height

    landmarks: list[list[float]] = []
    for x, y, z in zip(x_norm, y_norm, z_norm):
        landmarks.append([float(x), float(y), float(z)])
    return landmarks


def simulate_hand_frame(
    *,
    distance_cm: float,
    handedness: str,
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float,
    frame_width: int,
    frame_height: int,
    hfov_deg: float,
    camera_type: str,
    rng: np.random.Generator,
    landmark_noise_std: float = 0.003,
) -> tuple[list[list[float]], dict[str, float], float]:
    """Generate one frame's landmarks, simulated sharpness, and confidence."""
    hand_3d = create_canonical_hand_3d(handedness)
    rotated = rotate_hand_3d(hand_3d, pitch_deg, yaw_deg, roll_deg)

    # Random slight translation so hand isn't perfectly dead center
    offset_x = float(rng.uniform(-2.5, 2.5))
    offset_y = float(rng.uniform(-2.0, 2.0))

    landmarks = project_to_mediapipe(
        rotated,
        distance_cm=distance_cm,
        frame_width=frame_width,
        frame_height=frame_height,
        hfov_deg=hfov_deg,
        center_offset_x_cm=offset_x,
        center_offset_y_cm=offset_y,
    )

    # Add realistic sensor noise to landmarks
    for lm in landmarks:
        lm[0] += float(rng.normal(0, landmark_noise_std))
        lm[1] += float(rng.normal(0, landmark_noise_std))
        lm[2] += float(rng.normal(0, landmark_noise_std * 1.5))

    # Sharpness: closer and higher-resolution cameras generally produce sharper crops,
    # but with motion/focus blur noise.
    # Base sharpness scales inversely with distance and with resolution.
    base_sharpness = (500.0 / max(distance_cm, 10.0)) * (frame_width / 1280.0)
    sharpness: dict[str, float] = {}
    for name in FINGER_NAMES:
        noise = float(rng.normal(0, base_sharpness * 0.15))
        sharpness[name] = max(0.0, base_sharpness + noise)

    confidence = float(np.clip(rng.normal(0.92, 0.05), 0.5, 0.99))
    return landmarks, sharpness, confidence


def compute_synthetic_label(
    distance_cm: float,
    focal_px: float,
    pad_facing_cos: float,
    mean_sharpness: float,
    rng: np.random.Generator,
    ppi_low: float = 150.0,
    ppi_high: float = 400.0,
    warning_threshold: float = 0.4,
    danger_threshold: float = 0.7,
    noise_std: float = 0.04,
) -> tuple[float, str, float]:
    """Compute ground-truth label using true physics + pad angle + sharpness + noise.

    Returns:
        (true_ppi, risk_label, state_label)
    """
    true_ppi = 2.54 * focal_px / max(distance_cm, 1.0)
    base_risk = ppi_to_risk(true_ppi, ppi_low=ppi_low, ppi_high=ppi_high)

    # If pads are angled far away from camera, print capture risk decreases
    facing_factor = max(0.0, pad_facing_cos) if pad_facing_cos is not None else 1.0
    # Moderate facing factor influence (physics alone would oversimplify)
    adjusted_risk = base_risk * (0.3 + 0.7 * facing_factor)

    # Add label noise so model isn't just an exact algebraic mirror of 1 variable
    noisy_risk = float(np.clip(adjusted_risk + rng.normal(0, noise_std), 0.0, 1.0))
    state = risk_to_state(noisy_risk, warning_threshold=warning_threshold, danger_threshold=danger_threshold)
    return true_ppi, noisy_risk, state


def generate_synthetic_dataset(
    *,
    config: AppConfig | None = None,
    n_subjects: int | None = None,
    sessions_per_subject: int | None = None,
    frames_per_session: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a multi-subject, multi-session synthetic hand dataset.

    Every row is marked with dataset_type: 'SYNTHETIC'.
    Includes subject_id and session_id for GroupKFold cross-validation.
    """
    cfg = config or load_config()
    s = cfg.seed if seed is None else seed
    rng = np.random.Generator(np.random.PCG64(s))

    num_subjects = n_subjects or cfg.training.synthetic_subjects
    num_sessions = sessions_per_subject or cfg.training.synthetic_sessions_per_subject
    num_frames = frames_per_session or cfg.training.synthetic_frames_per_session

    ppi_low = cfg.risk.ppi_low
    ppi_high = cfg.risk.ppi_high
    warn_th = cfg.risk.warning_threshold
    dang_th = cfg.risk.danger_threshold

    # Camera configurations: 720p webcam, 1080p webcam, 4K phone, 1080p phone
    camera_presets = [
        {"name": "webcam", "type": "webcam", "w": 1280, "h": 720, "hfov": 70.0},
        {"name": "webcam_1080p", "type": "webcam", "w": 1920, "h": 1080, "hfov": 78.0},
        {"name": "phone_1080p", "type": "phone", "w": 1920, "h": 1080, "hfov": 75.0},
        {"name": "phone_4k", "type": "phone", "w": 3840, "h": 2160, "hfov": 75.0},
    ]

    records: list[dict[str, Any]] = []

    for subj_idx in range(num_subjects):
        subject_id = f"SYNTHETIC_SUBJ_{subj_idx + 1:03d}"
        # Subject anatomical variation: palm width ± 1.0 cm around mean 8.5 cm
        subj_palm_cm = float(np.clip(rng.normal(cfg.anatomy.palm_width_cm, 0.4), 7.0, 10.0))

        for sess_idx in range(num_sessions):
            session_id = f"SYNTHETIC_SESS_{subj_idx + 1:03d}_{sess_idx + 1:02d}"
            # Camera choice fixed per session
            cam_preset = camera_presets[int(rng.integers(0, len(camera_presets)))]
            w = cam_preset["w"]
            h = cam_preset["h"]
            hfov = cam_preset["hfov"]
            cam_type = cam_preset["type"]
            f_px = focal_length_px(w, hfov)

            # Handedness preference per session (alternating or mixed)
            primary_hand = "Right" if rng.random() > 0.4 else "Left"

            for frame_idx in range(num_frames):
                # Sample distance uniformly across very close (15cm) to far (80cm)
                distance_cm = float(rng.uniform(15.0, 80.0))

                # Slight hand pitch/yaw/roll variations
                pitch = float(rng.normal(0, 15.0))
                yaw = float(rng.normal(0, 20.0))
                roll = float(rng.normal(0, 10.0))

                # Occasional opposite hand
                hand = primary_hand if rng.random() > 0.15 else ("Left" if primary_hand == "Right" else "Right")

                landmarks, sharpness, confidence = simulate_hand_frame(
                    distance_cm=distance_cm,
                    handedness=hand,
                    pitch_deg=pitch,
                    yaw_deg=yaw,
                    roll_deg=roll,
                    frame_width=w,
                    frame_height=h,
                    hfov_deg=hfov,
                    camera_type=cam_type,
                    rng=rng,
                )

                # Extract features using the production feature extractor
                features = extract_features(
                    landmarks,
                    handedness=hand,
                    confidence=confidence,
                    frame_width=w,
                    frame_height=h,
                    hfov_deg=hfov,
                    sharpness=sharpness,
                    palm_width_cm=subj_palm_cm,
                    mirrored=False,
                    camera_type=cam_type,
                )

                # Ground truth physics label with realistic variations
                pad_cos = features.get("pad_facing_cos") or 1.0
                mean_sharp = features.get("mean_sharpness") or 0.0

                true_ppi, risk_label, state_label = compute_synthetic_label(
                    distance_cm=distance_cm,
                    focal_px=f_px,
                    pad_facing_cos=pad_cos,
                    mean_sharpness=mean_sharp,
                    rng=rng,
                    ppi_low=ppi_low,
                    ppi_high=ppi_high,
                    warning_threshold=warn_th,
                    danger_threshold=dang_th,
                )

                row: dict[str, Any] = {
                    "dataset_type": "SYNTHETIC",
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "frame_idx": frame_idx,
                    "true_distance_cm": distance_cm,
                    "true_ppi": true_ppi,
                    "risk_label": risk_label,
                    "state_label": state_label,
                    **features,
                }
                records.append(row)

    df = pd.DataFrame(records)
    return df


def save_synthetic_dataset(
    df: pd.DataFrame,
    output_path: Path | str = "data/synthetic_dataset.parquet",
) -> Path:
    """Save synthetic dataset to Parquet or CSV. Path and data marked SYNTHETIC."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    return out
