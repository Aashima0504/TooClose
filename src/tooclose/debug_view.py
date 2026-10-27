"""Minimal OpenCV landmark viewer for Phase 1 debugging.

Usage::

    make debug
    # or: .venv/bin/python -m tooclose.debug_view

Keys: q quit, m toggle mirror. No frames are written to disk.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from tooclose.config import load_config
from tooclose.hand_tracking import CONNECTIONS, FINGERTIPS, HandTracker


def _draw_hands(frame_bgr: np.ndarray, hands: list) -> None:
    height, width = frame_bgr.shape[:2]
    for hand in hands:
        points = [
            (int(x * width), int(y * height)) for x, y, _z in hand.landmarks
        ]
        for start, end in CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame_bgr, points[start], points[end], (80, 200, 255), 2)
        for _name, idx in FINGERTIPS.items():
            if idx < len(points):
                cv2.circle(frame_bgr, points[idx], 6, (0, 220, 120), -1)
        for x_px, y_px in points:
            cv2.circle(frame_bgr, (x_px, y_px), 3, (255, 255, 255), -1)
        wrist = points[0] if points else (10, 30)
        label = f"{hand.handedness} ({hand.confidence:.2f})"
        cv2.putText(
            frame_bgr,
            label,
            (wrist[0] + 8, max(24, wrist[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TooClose live landmark debugger")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--no-mirror", action="store_true", help="Disable selfie mirror")
    args = parser.parse_args(argv)

    config = load_config()
    width = args.width or config.camera.default_width
    height = args.height or config.camera.default_height
    mirror = not args.no_mirror

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(
            "Could not open the camera. On macOS, grant Camera permission to "
            "Terminal/iTerm/your IDE, then retry. No frames are saved."
        )
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    tracker = HandTracker(config)
    window = "TooClose debug (q quit, m mirror)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    prev = time.perf_counter()
    fps = 0.0
    print("TooClose debug view — processing locally, frames are not saved.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera frame grab failed.")
                break
            processed, hands = tracker.process_bgr(frame, mirror=mirror)
            _draw_hands(processed, hands)

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            banner = (
                f"FPS {fps:4.1f}  hands {len(hands)}  "
                f"mirror {'on' if mirror else 'off'}  "
                "local only — no frames saved"
            )
            cv2.putText(
                processed,
                banner,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 255, 180),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, processed)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                mirror = not mirror
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
