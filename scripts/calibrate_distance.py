#!/usr/bin/env python3
"""Live pinhole-distance readout for manual ruler checks.

Hold your palm facing the camera and a ruler at **20 / 30 / 40 cm**. Compare
the printed ``distance_cm`` to the ruler. Nothing is saved to disk.

Usage::

    make calibrate
    # or: .venv/bin/python scripts/calibrate_distance.py
"""

from __future__ import annotations

import argparse
import time

import cv2

from tooclose.config import load_config
from tooclose.features import extract_features
from tooclose.hand_tracking import FINGER_DIP, FINGERTIPS, HandTracker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TooClose live distance calibration")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hfov", type=float, default=None, help="Override horizontal FOV (deg)")
    parser.add_argument("--palm-cm", type=float, default=None, help="Override palm width (cm)")
    args = parser.parse_args(argv)

    config = load_config()
    hfov = args.hfov if args.hfov is not None else config.camera.default_hfov_deg
    palm_cm = args.palm_cm if args.palm_cm is not None else config.anatomy.palm_width_cm

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open the camera. Grant Camera permission, then retry. No frames are saved.")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.default_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.default_height)

    tracker = HandTracker(config)
    print(
        "TooClose distance calibration — local only, no frames saved.\n"
        f"Assuming palm width {palm_cm:.1f} cm, hfov {hfov:.1f}°.\n"
        "Hold a ruler at 20 / 30 / 40 cm with your palm facing the camera.\n"
        "Keys: q quit, m toggle mirror.\n"
    )
    mirror = config.mediapipe.selfie_mirror
    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera frame grab failed.")
                break
            processed, hands = tracker.process_bgr(frame, mirror=mirror)
            height, width = processed.shape[:2]
            now = time.time()
            if now - last_print >= 0.4:
                last_print = now
                if not hands:
                    print("  no hand  | hold your palm toward the camera at 20/30/40 cm")
                else:
                    for hand in hands:
                        feats = extract_features(
                            hand.landmarks,
                            handedness=hand.handedness,
                            confidence=hand.confidence,
                            frame_width=width,
                            frame_height=height,
                            hfov_deg=hfov,
                            sharpness={name: 0.0 for name in FINGERTIPS},
                            palm_width_cm=palm_cm,
                            mirrored=hand.mirrored,
                        )
                        dist = feats["distance_cm"]
                        ppi = feats["ppi"]
                        dist_s = f"{dist:6.1f} cm" if dist is not None else "   n/a"
                        ppi_s = f"{ppi:6.0f}" if ppi is not None else "   n/a"
                        print(
                            f"  {hand.handedness:5}  distance={dist_s}  ppi={ppi_s}  "
                            f"palm_px={feats['palm_width_px']:.0f}  "
                            f"pads={'yes' if feats['pads_facing_camera'] else 'no '}  "
                            f"conf={hand.confidence:.2f}"
                        )
            vis = processed.copy()
            if hands:
                hand = hands[0]
                feats = extract_features(
                    hand.landmarks,
                    handedness=hand.handedness,
                    confidence=hand.confidence,
                    frame_width=width,
                    frame_height=height,
                    hfov_deg=hfov,
                    palm_width_cm=palm_cm,
                    mirrored=hand.mirrored,
                )
                dist = feats["distance_cm"]
                label = (
                    f"{hand.handedness}  {dist:.0f} cm  ppi {feats['ppi']:.0f}"
                    if dist is not None
                    else "measuring..."
                )
                cv2.putText(
                    vis,
                    label,
                    (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (40, 255, 180),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    "ruler check: 20 / 30 / 40 cm   (q quit)",
                    (16, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                for name, idx in FINGERTIPS.items():
                    x, y, _z = hand.landmarks[idx]
                    cv2.circle(vis, (int(x * width), int(y * height)), 5, (0, 220, 120), -1)
                    _ = FINGER_DIP
            else:
                cv2.putText(
                    vis,
                    "No hand — palm toward camera",
                    (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("TooClose calibrate (q quit, m mirror)", vis)
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
