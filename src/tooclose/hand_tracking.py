"""MediaPipe Hands wrapper for collection, debugging, and tests.

Handedness and the mirrored selfie view
---------------------------------------
Webcams used as "selfie" cameras typically display a horizontally mirrored
preview. TooClose follows that convention when ``mediapipe.selfie_mirror`` is
true (the default):

1. The BGR frame is flipped left-right before detection so overlays match the
   preview the user sees.
2. MediaPipe's handedness classifier is appearance-based. A mirrored right
   hand looks like a left hand, so the raw label is inverted.
3. The ``handedness`` field on :class:`HandResult` is therefore the user's
   **physical** hand (``Right`` means the right hand). ``raw_handedness``
   preserves MediaPipe's label before the flip.

When ``selfie_mirror`` is false, frames are processed as captured and labels
are left unchanged (camera / "other person" view).

Landmarks are MediaPipe-normalized (x, y in 0–1 of the processed frame, z
roughly in image-widths). After a mirror flip, x is already expressed in the
mirrored image, so clients that draw on a mirrored canvas can use them as-is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from tooclose.config import AppConfig, load_config

Handedness = Literal["Left", "Right"]

# MediaPipe Hands landmark indices (21 points).
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERTIPS = {
    "thumb": THUMB_TIP,
    "index": INDEX_TIP,
    "middle": MIDDLE_TIP,
    "ring": RING_TIP,
    "pinky": PINKY_TIP,
}

FINGER_DIP = {
    "thumb": THUMB_IP,  # thumb has IP rather than DIP; used as the distal joint
    "index": INDEX_DIP,
    "middle": MIDDLE_DIP,
    "ring": RING_DIP,
    "pinky": PINKY_DIP,
}

CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)


def flip_handedness(label: str) -> Handedness:
    """Invert Left/Right. Used after mirroring the selfie frame.

    Unknown labels default to ``Right`` after a warning-free pass-through of
    the two canonical values only.
    """
    normalized = label.strip().capitalize()
    if normalized == "Left":
        return "Right"
    if normalized == "Right":
        return "Left"
    raise ValueError(f"Unknown handedness label: {label!r}")


def mirror_landmarks_x(
    landmarks: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Flip normalized landmark x-coordinates (0–1) for a mirrored frame."""
    return [(1.0 - x, y, z) for x, y, z in landmarks]


@dataclass(frozen=True)
class HandResult:
    """One detected hand in a single frame."""

    landmarks: list[tuple[float, float, float]]
    handedness: Handedness
    confidence: float
    raw_handedness: str
    mirrored: bool


class HandTracker:
    """Thin wrapper around MediaPipe Tasks HandLandmarker (VIDEO mode)."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self._landmarker = None
        self._timestamp_ms = 0

    def _ensure_landmarker(self) -> None:
        if self._landmarker is not None:
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "mediapipe is required for live tracking. Run `make setup`."
            ) from exc

        model_path = self.config.resolve(self.config.mediapipe.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. Run `make model`."
            )

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.mediapipe.max_hands,
            min_hand_detection_confidence=self.config.mediapipe.min_detection_confidence,
            min_hand_presence_confidence=self.config.mediapipe.min_detection_confidence,
            min_tracking_confidence=self.config.mediapipe.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def process_bgr(
        self,
        frame_bgr: np.ndarray,
        *,
        timestamp_ms: int | None = None,
        mirror: bool | None = None,
    ) -> tuple[np.ndarray, list[HandResult]]:
        """Detect hands on a BGR OpenCV frame.

        Returns the (possibly mirrored) frame that landmarks are expressed in,
        plus a list of :class:`HandResult`. Never writes the frame to disk.
        """
        self._ensure_landmarker()
        apply_mirror = self.config.mediapipe.selfie_mirror if mirror is None else mirror

        processed = frame_bgr
        if apply_mirror:
            processed = np.ascontiguousarray(frame_bgr[:, ::-1, :])

        rgb = processed[:, :, ::-1].copy()
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        if timestamp_ms is None:
            self._timestamp_ms += 33
            timestamp_ms = self._timestamp_ms
        else:
            self._timestamp_ms = timestamp_ms

        detection = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        hands = self._parse_result(detection, mirrored=apply_mirror)
        return processed, hands

    def _parse_result(self, detection: object, *, mirrored: bool) -> list[HandResult]:
        hands: list[HandResult] = []
        landmarks_list = getattr(detection, "hand_landmarks", None) or []
        handedness_list = getattr(detection, "handedness", None) or []

        for index, landmark_list in enumerate(landmarks_list):
            points: list[tuple[float, float, float]] = [
                (float(lm.x), float(lm.y), float(lm.z)) for lm in landmark_list
            ]
            raw_label = "Right"
            score = 0.0
            if index < len(handedness_list) and handedness_list[index]:
                category = handedness_list[index][0]
                raw_label = str(getattr(category, "category_name", "Right"))
                score = float(getattr(category, "score", 0.0))

            physical: Handedness
            if mirrored:
                try:
                    physical = flip_handedness(raw_label)
                except ValueError:
                    physical = "Right"
            else:
                physical = "Left" if raw_label.strip().lower() == "left" else "Right"

            hands.append(
                HandResult(
                    landmarks=points,
                    handedness=physical,
                    confidence=score,
                    raw_handedness=raw_label,
                    mirrored=mirrored,
                )
            )
        return hands
