"""Metrics for Remaining Useful Life predictions.

Every model in the project, ML and DL, is scored through this module so the
comparison stays fair.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error, in cycles."""
    error = _errors(y_true, y_pred)
    return float(np.sqrt(np.mean(error**2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error, in cycles."""
    return float(np.mean(np.abs(_errors(y_true, y_pred))))


def _errors(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Prediction errors ``y_pred - y_true``: negative is early, positive late."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"got {true.size} targets and {pred.size} predictions")
    return pred - true
