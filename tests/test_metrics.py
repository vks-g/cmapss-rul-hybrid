"""Tests for rul.evaluation.metrics."""

import math

import pytest

from rul.evaluation.metrics import mae, rmse


def test_rmse_and_mae_on_hand_checked_errors():
    # errors (pred - true): +2, -2, +3  ->  mae = 7/3, rmse = sqrt(17/3)
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]

    assert mae(y_true, y_pred) == pytest.approx(7 / 3)
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(17 / 3))


def test_perfect_predictions_score_zero():
    assert mae([5, 90], [5, 90]) == 0.0
    assert rmse([5, 90], [5, 90]) == 0.0


@pytest.mark.parametrize("metric", [rmse, mae])
def test_mismatched_lengths_are_rejected(metric):
    with pytest.raises(ValueError, match="3 targets"):
        metric([1, 2, 3], [1, 2])
