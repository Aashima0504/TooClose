"""Pinhole distance, ppi, palm normal, curl, and partial-hand handling."""

from __future__ import annotations

import math

import pytest

from tests.hand_factory import curled_index, open_palm, selfie_mirror_hand
from tooclose.features import (
    extract_features,
    focal_length_px,
    pad_facing_metrics,
    palmar_normal,
    pinhole_distance_cm,
    ppi_from_distance,
)
from tooclose.hand_tracking import flip_handedness

FRAME_W = 1280
FRAME_H = 720
HFOV = 70.0
PALM_CM = 8.5


def _extract(landmarks, handedness: str, **kwargs):
    return extract_features(
        landmarks,
        handedness=handedness,
        confidence=0.9,
        frame_width=kwargs.get("frame_width", FRAME_W),
        frame_height=kwargs.get("frame_height", FRAME_H),
        hfov_deg=kwargs.get("hfov_deg", HFOV),
        sharpness={"thumb": 1.0, "index": 2.0, "middle": 3.0, "ring": 4.0, "pinky": 5.0},
        palm_width_cm=PALM_CM,
        mirrored=kwargs.get("mirrored", False),
        camera_type=kwargs.get("camera_type", "webcam"),
    )


def test_focal_length_matches_pinhole_formula() -> None:
    expected = FRAME_W / (2.0 * math.tan(math.radians(HFOV) / 2.0))
    assert focal_length_px(FRAME_W, HFOV) == expected


def test_pinhole_distance_round_trip() -> None:
    """If the palm spans the pixels implied by 30 cm, recover ~30 cm."""
    f_px = focal_length_px(FRAME_W, HFOV)
    distance_cm = 30.0
    palm_px = (PALM_CM * f_px) / distance_cm
    recovered = pinhole_distance_cm(palm_px, FRAME_W, HFOV, PALM_CM)
    assert recovered is not None
    assert recovered == pytest.approx(distance_cm)


def test_ppi_equivalent_forms() -> None:
    distance_cm = 25.0
    f_px = focal_length_px(FRAME_W, HFOV)
    ppi = ppi_from_distance(distance_cm, FRAME_W, HFOV)
    assert ppi is not None
    fingertip_width_mm = 15.0
    fingertip_px = (fingertip_width_mm / 10.0) * f_px / distance_cm
    ppi_from_pad = fingertip_px * 25.4 / fingertip_width_mm
    assert abs(ppi - ppi_from_pad) < 1e-9
    assert abs(ppi - 2.54 * f_px / distance_cm) < 1e-9


def test_same_distance_higher_resolution_higher_ppi() -> None:
    """Camera-dependence: 4K vs 720p at the same FOV and pose."""
    hand = open_palm("Right")
    low = _extract(hand, "Right", frame_width=1280, frame_height=720)
    high = _extract(hand, "Right", frame_width=3840, frame_height=2160)
    assert low["distance_cm"] is not None and high["distance_cm"] is not None
    assert abs(low["distance_cm"] - high["distance_cm"]) / low["distance_cm"] < 0.02
    assert high["ppi"] > low["ppi"] * 2.5
    assert abs(high["ppi"] / low["ppi"] - 3840 / 1280) < 0.05


def test_right_palm_facing_camera() -> None:
    hand = open_palm("Right", palm_facing=True)
    feats = _extract(hand, "Right")
    assert feats["valid"] is True
    assert feats["pads_facing_camera"] == 1.0
    assert feats["pad_facing_angle_deg"] < 35.0
    n = palmar_normal(hand, FRAME_W, FRAME_H, "Right")
    cosine, _angle, facing = pad_facing_metrics(n)
    assert facing is True
    assert cosine > 0.5


def test_right_dorsal_not_facing() -> None:
    hand = open_palm("Right", palm_facing=False)
    feats = _extract(hand, "Right")
    assert feats["pads_facing_camera"] == 0.0
    assert feats["pad_facing_angle_deg"] > 90.0


def test_left_and_right_palm_facing_agree() -> None:
    left = _extract(open_palm("Left", palm_facing=True), "Left")
    right = _extract(open_palm("Right", palm_facing=True), "Right")
    assert left["pads_facing_camera"] == 1.0
    assert right["pads_facing_camera"] == 1.0


def test_selfie_mirror_keeps_physical_pad_facing() -> None:
    """Flip x like a selfie preview; physical Right must still read as pads-facing."""
    original = open_palm("Right", palm_facing=True)
    mirrored = selfie_mirror_hand(original)
    feats = _extract(mirrored, "Right", mirrored=True)
    assert feats["pads_facing_camera"] == 1.0
    assert feats["mirrored"] == 1.0
    # Without the un-flip, winding would invert this.
    naive = _extract(mirrored, "Right", mirrored=False)
    assert naive["pads_facing_camera"] == 0.0


def test_flip_handedness_is_involution() -> None:
    assert flip_handedness(flip_handedness("Left")) == "Left"
    assert flip_handedness(flip_handedness("Right")) == "Right"


def test_curl_increases_when_index_folds() -> None:
    open_hand = open_palm("Right")
    folded = curled_index(open_hand)
    open_feats = _extract(open_hand, "Right")
    folded_feats = _extract(folded, "Right")
    assert open_feats["index_curl_deg"] < 20.0
    assert folded_feats["index_curl_deg"] > 70.0


def test_dip_tip_normalized_by_frame_width() -> None:
    hand = open_palm("Right")
    feats = _extract(hand, "Right")
    assert feats["index_dip_tip_px"] > 0
    assert abs(feats["index_dip_tip_norm"] - feats["index_dip_tip_px"] / FRAME_W) < 1e-12
    assert feats["mean_dip_tip_norm"] == sum(
        feats[f"{n}_dip_tip_norm"] for n in ("thumb", "index", "middle", "ring", "pinky")
    ) / 5.0


def test_client_sharpness_passed_through() -> None:
    feats = _extract(open_palm("Right"), "Right")
    assert feats["index_sharpness"] == 2.0
    assert feats["mean_sharpness"] == 3.0


def test_partial_hand_is_invalid_not_an_exception() -> None:
    feats = _extract([(0.1, 0.2, 0.0)] * 5, "Right")
    assert feats["valid"] is False
    assert feats["distance_cm"] is None
    assert feats["ppi"] is None
    assert feats["index_dip_tip_px"] is None


def test_zero_palm_width_is_invalid() -> None:
    assert pinhole_distance_cm(0.0, FRAME_W, HFOV, PALM_CM) is None
    assert ppi_from_distance(None, FRAME_W, HFOV) is None


def test_phone_flag() -> None:
    feats = _extract(open_palm("Right"), "Right", camera_type="phone")
    assert feats["camera_type_phone"] == 1.0
    assert feats["hfov_deg"] == HFOV
