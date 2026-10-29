"""Label sources: physics-based ppi risk, optional NFIQ 2 oracle. Phase 3/4.

Physics labeling combines:
1. Pinhole PPI risk (same formula as baseline).
2. Sharpness factor: blurry fingertips reduce risk of viable print capture.
3. Pad-facing factor: pads angled away reduce risk of usable ridge detail.
4. Gaussian label noise seeded for reproducibility.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from tooclose.baseline import ppi_to_risk, risk_to_state

LabelSource = Literal["physics", "nfiq"]

DEFAULT_PPI_LOW = 150.0
DEFAULT_PPI_HIGH = 400.0
DEFAULT_WARNING_THRESHOLD = 0.4
DEFAULT_DANGER_THRESHOLD = 0.7
DEFAULT_SHARPNESS_REF = 100.0
DEFAULT_NOISE_SIGMA = 0.05


def compute_sharpness_factor(
    features: dict[str, Any],
    sharpness_ref: float = DEFAULT_SHARPNESS_REF,
) -> float:
    """Blurry fingertips lower the capture risk.

    factor = clip(sharpness / sharpness_ref, 0.3, 1.0)
    If sharpness is missing or invalid, defaults to 1.0.
    """
    if sharpness_ref <= 0.0:
        return 1.0
    val = features.get("mean_sharpness")
    if val is None or math.isnan(val):
        return 1.0
    return float(np.clip(float(val) / sharpness_ref, 0.3, 1.0))


def compute_pad_facing_factor(features: dict[str, Any]) -> float:
    """Pads turned away lower the capture risk.

    factor = 0.4 + 0.6 * pad_facing_score
    pad_facing_score is max(0.0, pad_facing_cos).
    If pad_facing_cos is missing or invalid, defaults to 1.0.
    """
    cos = features.get("pad_facing_cos")
    if cos is None or math.isnan(cos):
        return 1.0
    pad_score = max(0.0, float(cos))
    factor = 0.4 + 0.6 * pad_score
    return float(np.clip(factor, 0.0, 1.0))


def label_row(
    features: dict[str, Any],
    *,
    source: LabelSource = "physics",
    ppi_low: float = DEFAULT_PPI_LOW,
    ppi_high: float = DEFAULT_PPI_HIGH,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    danger_threshold: float = DEFAULT_DANGER_THRESHOLD,
    sharpness_ref: float = DEFAULT_SHARPNESS_REF,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    rng: np.random.Generator | int | None = None,
) -> dict[str, Any]:
    """Attach a continuous risk label and a 3-class state to a feature row.

    Combines baseline PPI risk with sharpness factor, pad-facing factor, and
    Gaussian label noise, ensuring the machine learning target differs from a pure
    single-feature function of PPI.
    """
    if source == "nfiq":
        raise NotImplementedError("NFIQ 2 labeling oracle is not configured.")

    ppi = features.get("ppi")
    base_risk = ppi_to_risk(ppi, ppi_low=ppi_low, ppi_high=ppi_high)

    sharp_factor = compute_sharpness_factor(features, sharpness_ref=sharpness_ref)
    pad_factor = compute_pad_facing_factor(features)

    combined_risk = base_risk * sharp_factor * pad_factor

    # Gaussian label noise
    noise = 0.0
    if noise_sigma > 0.0:
        if isinstance(rng, np.random.Generator):
            gen = rng
        elif isinstance(rng, int):
            gen = np.random.Generator(np.random.PCG64(rng))
        else:
            gen = np.random.default_rng()
        noise = float(gen.normal(0.0, noise_sigma))

    final_risk = float(np.clip(combined_risk + noise, 0.0, 1.0))
    state = risk_to_state(
        final_risk,
        warning_threshold=warning_threshold,
        danger_threshold=danger_threshold,
    )

    result = dict(features)
    result["risk_label"] = final_risk
    result["state_label"] = state
    return result
