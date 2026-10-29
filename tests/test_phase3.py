"""Unit tests for Phase 3: baseline risk model and synthetic generator."""

from __future__ import annotations

import math
import numpy as np
import pytest

from tooclose.baseline import (
    baseline_predict,
    ppi_to_risk,
    risk_to_state,
)
from tooclose.config import load_config
from tooclose.labeling import label_row
from tooclose.synthetic import (
    create_canonical_hand_3d,
    generate_synthetic_dataset,
    project_to_mediapipe,
    rotate_hand_3d,
    simulate_hand_frame,
)


def test_baseline_ppi_to_risk_monotonic() -> None:
    """Risk must be 0 below ppi_low, linear to 1 at ppi_high, and strictly monotonic in between."""
    ppi_low = 150.0
    ppi_high = 400.0

    # Below or at ppi_low
    assert ppi_to_risk(None) == 0.0
    assert ppi_to_risk(float("nan")) == 0.0
    assert ppi_to_risk(0.0, ppi_low, ppi_high) == 0.0
    assert ppi_to_risk(100.0, ppi_low, ppi_high) == 0.0
    assert ppi_to_risk(150.0, ppi_low, ppi_high) == 0.0

    # Above or at ppi_high
    assert ppi_to_risk(400.0, ppi_low, ppi_high) == 1.0
    assert ppi_to_risk(600.0, ppi_low, ppi_high) == 1.0

    # Midpoints
    mid = 275.0  # (150 + 400) / 2
    assert pytest.approx(ppi_to_risk(mid, ppi_low, ppi_high), 1e-4) == 0.5

    # Strictly monotonic test
    ppis = [150.0, 180.0, 220.0, 275.0, 330.0, 390.0, 400.0]
    risks = [ppi_to_risk(p, ppi_low, ppi_high) for p in ppis]
    for i in range(len(risks) - 1):
        assert risks[i] <= risks[i + 1]
        if ppis[i] < ppis[i + 1]:
            assert risks[i] < risks[i + 1]


def test_baseline_states() -> None:
    assert risk_to_state(0.0) == "safe"
    assert risk_to_state(0.39) == "safe"
    assert risk_to_state(0.40) == "warning"
    assert risk_to_state(0.69) == "warning"
    assert risk_to_state(0.70) == "danger"
    assert risk_to_state(1.0) == "danger"

    risk, state = baseline_predict(500.0)
    assert risk == 1.0
    assert state == "danger"


def test_label_row() -> None:
    feat = {"ppi": 275.0, "other": 123}
    labeled = label_row(feat)
    assert pytest.approx(labeled["risk_label"], 1e-4) == 0.5
    assert labeled["state_label"] == "warning"
    assert labeled["other"] == 123


def test_canonical_hand_geometry() -> None:
    right = create_canonical_hand_3d("Right")
    left = create_canonical_hand_3d("Left")
    assert right.shape == (21, 3)
    assert left.shape == (21, 3)

    # Right hand thumb x < pinky x; Left hand thumb x > pinky x
    assert right[4, 0] < right[20, 0]  # THUMB_TIP vs PINKY_TIP
    assert left[4, 0] > left[20, 0]

    # Rotation should preserve distances between joints
    d0 = np.linalg.norm(right[0] - right[9])
    rotated = rotate_hand_3d(right, pitch_deg=20.0, yaw_deg=-15.0, roll_deg=10.0)
    d1 = np.linalg.norm(rotated[0] - rotated[9])
    assert pytest.approx(d0, 1e-4) == d1


def test_synthetic_hand_features_valid_for_full_hands() -> None:
    """Full hands simulated at reasonable distances must produce valid features."""
    rng = np.random.Generator(np.random.PCG64(42))
    landmarks, sharpness, confidence = simulate_hand_frame(
        distance_cm=35.0,
        handedness="Right",
        pitch_deg=0.0,
        yaw_deg=0.0,
        roll_deg=0.0,
        frame_width=1280,
        frame_height=720,
        hfov_deg=70.0,
        camera_type="webcam",
        rng=rng,
    )
    assert len(landmarks) == 21
    for pt in landmarks:
        assert len(pt) == 3
        # Should be within image normalized bounds
        assert 0.0 <= pt[0] <= 1.0
        assert 0.0 <= pt[1] <= 1.0


def test_synthetic_dataset_deterministic_with_same_seed() -> None:
    """Same seed must produce bit-for-bit identical DataFrames."""
    df1 = generate_synthetic_dataset(
        n_subjects=2,
        sessions_per_subject=2,
        frames_per_session=5,
        seed=123,
    )
    df2 = generate_synthetic_dataset(
        n_subjects=2,
        sessions_per_subject=2,
        frames_per_session=5,
        seed=123,
    )
    assert len(df1) == 20
    assert len(df2) == 20
    assert df1.equals(df2)


def test_synthetic_dataset_metadata_and_grouping() -> None:
    """Dataset rows must have SYNTHETIC mark, valid: True, and subject_id/session_id."""
    df = generate_synthetic_dataset(
        n_subjects=3,
        sessions_per_subject=2,
        frames_per_session=4,
        seed=42,
    )
    assert len(df) == 24
    assert (df["dataset_type"] == "SYNTHETIC").all()
    assert (df["valid"] == True).all()

    # Must contain subject_id and session_id
    assert "subject_id" in df.columns
    assert "session_id" in df.columns
    assert df["subject_id"].nunique() == 3
    assert df["session_id"].nunique() == 6

    # Labels must be in range
    assert (df["risk_label"] >= 0.0).all()
    assert (df["risk_label"] <= 1.0).all()
    assert set(df["state_label"].unique()).issubset({"safe", "warning", "danger"})

