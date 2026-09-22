"""Tests for rul.evaluation.metrics."""

import math

import pytest

from rul.evaluation.metrics import mae, nasa_score, rmse


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


def test_nasa_score_penalises_late_predictions_more_than_early_ones():
    # A prediction 10 cycles early scores exp(10/13) - 1; 10 cycles late scores
    # exp(10/10) - 1. Predicting failure too late is the expensive mistake.
    early = nasa_score([50], [40])
    late = nasa_score([50], [60])

    assert early == pytest.approx(math.exp(10 / 13) - 1)
    assert late == pytest.approx(math.e - 1)
    assert late > early


def test_nasa_score_sums_over_engines_and_is_zero_when_exact():
    assert nasa_score([30, 30], [30, 30]) == 0.0
    assert nasa_score([50, 50], [40, 60]) == pytest.approx(
        (math.exp(10 / 13) - 1) + (math.e - 1)
    )
