"""Config loads and handedness flip helper — Phase 1 smoke tests."""

from __future__ import annotations

from tooclose.config import load_config
from tooclose.hand_tracking import flip_handedness


def test_config_loads() -> None:
    config = load_config()
    assert config.server.host == "127.0.0.1"
    assert config.mediapipe.num_landmarks == 21
    assert config.anatomy.palm_width_cm == 8.5
    assert config.sharpness.resize == 64
    assert config.sharpness.laplacian_ksize == 3


def test_flip_handedness_selfie_convention() -> None:
    assert flip_handedness("Left") == "Right"
    assert flip_handedness("Right") == "Left"
    assert flip_handedness("left") == "Right"
