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
from tooclose.features import extract_features, focal_length_px
from tooclose.labeling import label_row
from tooclose.hand_tracking import (
    INDEX_DIP,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_TIP,
    PINKY_DIP,
    PINKY_MCP,
    PINKY_TIP,
    WRIST,
)
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


def test_label_row_physics_factors() -> None:
    """Label differs from pure baseline when sharpness is low or pads face away."""
    base_ppi = 275.0  # baseline risk is 0.5
    feat_nominal = {"ppi": base_ppi}
    # With missing features and noise_sigma=0, factors default to 1.0 -> 0.5
    nominal = label_row(feat_nominal, noise_sigma=0.0)
    assert pytest.approx(nominal["risk_label"], 1e-4) == 0.5
    assert nominal["state_label"] == "warning"

    # 1. Low sharpness reduces risk
    feat_blurry = {"ppi": base_ppi, "mean_sharpness": 10.0}  # 10 / 100 = 0.1 -> clamped to 0.3
    blurry = label_row(feat_blurry, noise_sigma=0.0, sharpness_ref=100.0)
    assert blurry["risk_label"] < nominal["risk_label"]
    assert pytest.approx(blurry["risk_label"], 1e-4) == 0.5 * 0.3

    # 2. Turned away pads reduce risk
    feat_turned = {"ppi": base_ppi, "pad_facing_cos": 0.0}  # factor = 0.4 + 0.6 * 0 = 0.4
    turned = label_row(feat_turned, noise_sigma=0.0)
    assert turned["risk_label"] < nominal["risk_label"]
    assert pytest.approx(turned["risk_label"], 1e-4) == 0.5 * 0.4

    # 3. Both low sharpness and turned away
    feat_both = {"ppi": base_ppi, "mean_sharpness": 10.0, "pad_facing_cos": 0.0}
    both = label_row(feat_both, noise_sigma=0.0, sharpness_ref=100.0)
    assert pytest.approx(both["risk_label"], 1e-4) == 0.5 * 0.3 * 0.4


def test_label_row_noise_seed_and_bounds() -> None:
    """Same seed gives identical labels; labels stay strictly within [0, 1]."""
    feat = {"ppi": 380.0, "mean_sharpness": 90.0, "pad_facing_cos": 0.8}
    row1 = label_row(feat, noise_sigma=0.1, rng=42)
    row2 = label_row(feat, noise_sigma=0.1, rng=42)
    assert row1["risk_label"] == row2["risk_label"]

    # Bound check across extreme inputs and noises
    rng = np.random.Generator(np.random.PCG64(999))
    for ppi in [50.0, 150.0, 275.0, 400.0, 1000.0]:
        for sharp in [0.0, 50.0, 200.0]:
            for cos in [-1.0, 0.0, 0.5, 1.0]:
                labeled = label_row(
                    {"ppi": ppi, "mean_sharpness": sharp, "pad_facing_cos": cos},
                    noise_sigma=0.2,
                    rng=rng,
                )
                assert 0.0 <= labeled["risk_label"] <= 1.0
                assert labeled["state_label"] in ("safe", "warning", "danger")


def test_canonical_hand_geometry() -> None:
    config = load_config()
    target_palm_cm = config.anatomy.palm_width_cm
    right = create_canonical_hand_3d("Right", palm_width_cm=target_palm_cm)
    left = create_canonical_hand_3d("Left", palm_width_cm=target_palm_cm)
    assert right.shape == (21, 3)
    assert left.shape == (21, 3)

    # Check exact palm width between INDEX_MCP and PINKY_MCP
    palm_width_right = float(np.linalg.norm(right[INDEX_MCP] - right[PINKY_MCP]))
    assert pytest.approx(palm_width_right, rel=1e-4) == target_palm_cm

    # Check realistic anthropometric scale
    wrist_to_middle = float(np.linalg.norm(right[WRIST] - right[MIDDLE_TIP]))
    assert 17.0 <= wrist_to_middle <= 20.0  # ~18.3 cm

    index_dip_tip = float(np.linalg.norm(right[INDEX_DIP] - right[INDEX_TIP]))
    middle_dip_tip = float(np.linalg.norm(right[MIDDLE_DIP] - right[MIDDLE_TIP]))
    pinky_dip_tip = float(np.linalg.norm(right[PINKY_DIP] - right[PINKY_TIP]))
    assert 1.8 <= index_dip_tip <= 2.5
    assert 2.0 <= middle_dip_tip <= 2.6
    assert 1.6 <= pinky_dip_tip <= 2.2

    # Right hand thumb x < pinky x; Left hand thumb x > pinky x
    assert right[4, 0] < right[20, 0]  # THUMB_TIP vs PINKY_TIP
    assert left[4, 0] > left[20, 0]

    # Rotation should preserve distances between joints
    d0 = np.linalg.norm(right[0] - right[9])
    rotated = rotate_hand_3d(right, pitch_deg=20.0, yaw_deg=-15.0, roll_deg=10.0)
    d1 = np.linalg.norm(rotated[0] - rotated[9])
    assert pytest.approx(d0, 1e-4) == d1


def test_zero_noise_distance_and_ppi_across_presets() -> None:
    """With zero noise and 3 distances (15, 30, 60 cm) across 720p, 1080p, 4K:

    Estimated distance is within 3% of true, and estimated ppi within 5% of true_ppi.
    """
    config = load_config()
    palm_cm = config.anatomy.palm_width_cm
    hand = create_canonical_hand_3d("Right", palm_width_cm=palm_cm)

    presets = [
        {"name": "720p", "w": 1280, "h": 720, "hfov": 70.0},
        {"name": "1080p", "w": 1920, "h": 1080, "hfov": 78.0},
        {"name": "4K", "w": 3840, "h": 2160, "hfov": 75.0},
    ]
    test_distances = [15.0, 30.0, 60.0]

    for preset in presets:
        w, h, hfov = preset["w"], preset["h"], preset["hfov"]
        f_px = focal_length_px(w, hfov)
        for true_dist in test_distances:
            landmarks = project_to_mediapipe(
                hand,
                distance_cm=true_dist,
                frame_width=w,
                frame_height=h,
                hfov_deg=hfov,
            )
            features = extract_features(
                landmarks,
                handedness="Right",
                confidence=1.0,
                frame_width=w,
                frame_height=h,
                hfov_deg=hfov,
                palm_width_cm=palm_cm,
                mirrored=False,
            )
            est_dist = features["distance_cm"]
            est_ppi = features["ppi"]
            true_ppi = 2.54 * f_px / true_dist

            assert est_dist is not None
            assert est_ppi is not None
            dist_err = abs(est_dist - true_dist) / true_dist
            ppi_err = abs(est_ppi - true_ppi) / true_ppi

            assert dist_err < 0.03, f"Distance error {dist_err:.2%} exceeded 3% on {preset['name']} at {true_dist}cm"
            assert ppi_err < 0.05, f"PPI error {ppi_err:.2%} exceeded 5% on {preset['name']} at {true_dist}cm"


def test_implied_palm_width_equals_configured_value_without_noise() -> None:
    """Implied palm width in cm equals configured anatomy palm width within 2% when noise is off."""
    config = load_config()
    cfg_palm_cm = config.anatomy.palm_width_cm
    hand = create_canonical_hand_3d("Right", palm_width_cm=cfg_palm_cm)

    w, h, hfov = 1280, 720, 70.0
    f_px = focal_length_px(w, hfov)

    for true_dist in [15.0, 30.0, 50.0]:
        landmarks = project_to_mediapipe(
            hand,
            distance_cm=true_dist,
            frame_width=w,
            frame_height=h,
            hfov_deg=hfov,
        )
        features = extract_features(
            landmarks,
            handedness="Right",
            confidence=1.0,
            frame_width=w,
            frame_height=h,
            hfov_deg=hfov,
            palm_width_cm=cfg_palm_cm,
            mirrored=False,
        )
        palm_px = features["palm_width_px"]
        assert palm_px is not None
        # Implied palm width in cm from observed pixels and true distance:
        implied_palm_cm = (palm_px * true_dist) / f_px
        err = abs(implied_palm_cm - cfg_palm_cm) / cfg_palm_cm
        assert err < 0.02, f"Implied palm width error {err:.2%} exceeded 2% at {true_dist}cm"


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

