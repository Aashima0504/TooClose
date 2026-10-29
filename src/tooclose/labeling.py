"""Label sources: physics-based ppi risk, optional NFIQ 2 oracle. Phase 3/4."""

from __future__ import annotations

from typing import Any, Literal

from tooclose.baseline import baseline_predict, ppi_to_risk, risk_to_state

LabelSource = Literal["physics", "nfiq"]


def label_row(
    features: dict[str, Any],
    *,
    source: LabelSource = "physics",
    ppi_low: float = 150.0,
    ppi_high: float = 400.0,
    warning_threshold: float = 0.4,
    danger_threshold: float = 0.7,
) -> dict[str, Any]:
    """Attach a continuous risk label and a 3-class state to a feature row."""
    if source == "nfiq":
        raise NotImplementedError("NFIQ 2 labeling oracle is not configured.")

    ppi = features.get("ppi")
    risk, state = baseline_predict(
        ppi,
        ppi_low=ppi_low,
        ppi_high=ppi_high,
        warning_threshold=warning_threshold,
        danger_threshold=danger_threshold,
    )
    result = dict(features)
    result["risk_label"] = risk
    result["state_label"] = state
    return result
