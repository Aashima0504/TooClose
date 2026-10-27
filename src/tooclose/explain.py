"""SHAP explanations and plain-language advice. Phase 6."""

from __future__ import annotations

from typing import Any


def explain_prediction(model: Any, features: dict[str, float]) -> dict[str, Any]:
    """Return top-k SHAP reasons plus advice derived from the pinhole distance."""
    raise NotImplementedError("Phase 6: explain.py")
