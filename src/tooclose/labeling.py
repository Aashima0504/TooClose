"""Label sources: physics-based ppi risk, optional NFIQ 2 oracle. Phase 3/4."""

from __future__ import annotations

from typing import Any, Literal

LabelSource = Literal["physics", "nfiq"]


def label_row(features: dict[str, Any], *, source: LabelSource = "physics") -> dict[str, Any]:
    """Attach a continuous risk label and a 3-class state to a feature row."""
    raise NotImplementedError("Phase 3: labeling.py")
