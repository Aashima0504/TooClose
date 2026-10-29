"""Rule-based ppi risk model. Pure function, easy to port to TypeScript."""

from __future__ import annotations

import math
from typing import Literal

RiskState = Literal["safe", "warning", "danger"]

DEFAULT_PPI_LOW = 150.0
DEFAULT_PPI_HIGH = 400.0
DEFAULT_WARNING_THRESHOLD = 0.4
DEFAULT_DANGER_THRESHOLD = 0.7


def ppi_to_risk(
    ppi: float | None,
    ppi_low: float = DEFAULT_PPI_LOW,
    ppi_high: float = DEFAULT_PPI_HIGH,
) -> float:
    """Map estimated fingertip ppi to a continuous risk in [0, 1].

    - ppi is None or <= ppi_low: risk = 0.0
    - ppi >= ppi_high: risk = 1.0
    - in between: linear interpolation from 0.0 to 1.0
    """
    if ppi is None or math.isnan(ppi) or ppi <= ppi_low:
        return 0.0
    if ppi >= ppi_high:
        return 1.0
    if ppi_high <= ppi_low:
        return 1.0
    return (ppi - ppi_low) / (ppi_high - ppi_low)


def risk_to_state(
    risk: float,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    danger_threshold: float = DEFAULT_DANGER_THRESHOLD,
) -> RiskState:
    """Map continuous risk to discrete state without hysteresis."""
    if risk >= danger_threshold:
        return "danger"
    if risk >= warning_threshold:
        return "warning"
    return "safe"


def baseline_predict(
    ppi: float | None,
    *,
    ppi_low: float = DEFAULT_PPI_LOW,
    ppi_high: float = DEFAULT_PPI_HIGH,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    danger_threshold: float = DEFAULT_DANGER_THRESHOLD,
) -> tuple[float, RiskState]:
    """Compute risk score and categorical state from ppi."""
    risk = ppi_to_risk(ppi, ppi_low=ppi_low, ppi_high=ppi_high)
    state = risk_to_state(risk, warning_threshold=warning_threshold, danger_threshold=danger_threshold)
    return risk, state
