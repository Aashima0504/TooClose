"""EMA smoothing and hysteresis for risk state. Phase 7."""

from __future__ import annotations

from typing import Literal

RiskState = Literal["safe", "warning", "danger"]


class RiskSmoother:
    """Per-connection EMA + hysteresis. Implemented in Phase 7."""

    def update(self, risk: float) -> tuple[float, RiskState]:
        raise NotImplementedError("Phase 7: smoothing.py")
