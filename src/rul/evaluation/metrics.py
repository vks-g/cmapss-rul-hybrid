"""Metrics for Remaining Useful Life predictions.

Every model in the project, ML and DL, is scored through this module so the
comparison stays fair.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray

# Degradation stage from the true RUL: late (close to failure) is the hardest
# and most valuable part of the curve to get right.
STAGE_BINS = (50.0, 100.0)
STAGES = ("early", "mid", "late")


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


def nasa_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """PHM08 asymmetric score: lower is better, 0 is perfect.

    Late predictions (the engine fails sooner than predicted) are penalised
    harder than early ones, because a missed failure costs more than an early
    maintenance stop.
    """
    error = _errors(y_true, y_pred)
    scale = np.where(error < 0, 13.0, 10.0)
    return float(np.sum(np.exp(np.abs(error) / scale) - 1))


def evaluate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    stage_bins: tuple[float, float] = STAGE_BINS,
) -> dict:
    """Score predictions overall and per degradation stage.

    Args:
        y_true: True RUL values, in cycles.
        y_pred: Predicted RUL values, in cycles.
        stage_bins: ``(late_max, mid_max)`` boundaries on the true RUL. With the
            default, ``RUL <= 50`` is late, ``50 < RUL <= 100`` is mid and
            anything above is early.

    Returns:
        A JSON-serialisable dict with the overall ``n``, ``rmse``, ``mae`` and
        ``nasa_score``, and the same figures per stage under ``by_stage``.
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    _errors(true, pred)  # validates the shapes

    late_max, mid_max = stage_bins
    masks = {
        "early": true > mid_max,
        "mid": (true > late_max) & (true <= mid_max),
        "late": true <= late_max,
    }
    return {
        **_scores(true, pred),
        "stage_bins": [float(late_max), float(mid_max)],
        "by_stage": {stage: _scores(true[m], pred[m]) for stage, m in masks.items()},
    }


def _scores(true: np.ndarray, pred: np.ndarray) -> dict:
    """Overall scores for one group of engines, or empty scores if it has none."""
    if true.size == 0:
        return {"n": 0, "rmse": None, "mae": None, "nasa_score": None}
    return {
        "n": int(true.size),
        "rmse": rmse(true, pred),
        "mae": mae(true, pred),
        "nasa_score": nasa_score(true, pred),
    }
