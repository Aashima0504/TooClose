"""Rule-based ppi risk model. Implemented in Phase 3."""

from __future__ import annotations


def ppi_to_risk(ppi: float, ppi_low: float = 150.0, ppi_high: float = 400.0) -> float:
    """Map estimated fingertip ppi to a continuous risk in [0, 1]."""
    raise NotImplementedError("Phase 3: baseline.py")
