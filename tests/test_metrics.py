"""Tests for rul.evaluation.metrics."""

import json
import math

import pytest

from rul.evaluation.metrics import evaluate, mae, nasa_score, rmse


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


def test_evaluate_reports_overall_metrics():
    result = evaluate([10, 20, 30], [12, 18, 33])

    assert result["n"] == 3
    assert result["mae"] == pytest.approx(7 / 3)
    assert result["rmse"] == pytest.approx(math.sqrt(17 / 3))
    assert result["nasa_score"] == pytest.approx(
        (math.exp(2 / 10) - 1) + (math.exp(2 / 13) - 1) + (math.exp(3 / 10) - 1)
    )


def test_evaluate_buckets_errors_by_degradation_stage():
    # true RUL 150 (early), 75 (mid), 20 (late); absolute errors 10, 5, 4
    result = evaluate([150, 75, 20], [160, 70, 24])

    by_stage = result["by_stage"]
    assert [by_stage[s]["n"] for s in ("early", "mid", "late")] == [1, 1, 1]
    assert by_stage["early"]["mae"] == pytest.approx(10)
    assert by_stage["mid"]["mae"] == pytest.approx(5)
    assert by_stage["late"]["mae"] == pytest.approx(4)


def test_stage_boundaries_are_inclusive_at_the_top_of_each_bin():
    # Defaults: late <= 50 < mid <= 100 < early
    result = evaluate([50, 100, 101], [50, 100, 101])

    assert [result["by_stage"][s]["n"] for s in ("late", "mid", "early")] == [1, 1, 1]


def test_stage_bins_are_configurable():
    # With late <= 30 < mid <= 60, a true RUL of 50 is a mid-stage engine.
    result = evaluate([50], [55], stage_bins=(30, 60))

    assert result["by_stage"]["mid"]["n"] == 1
    assert result["by_stage"]["late"]["n"] == 0
    assert result["stage_bins"] == [30.0, 60.0]


def test_result_can_be_written_to_a_metrics_file(tmp_path):
    # results/metrics/ holds plain JSON, so no numpy types may leak into the dict.
    result = evaluate([150, 20], [140, 25])

    path = tmp_path / "run.json"
    path.write_text(json.dumps(result))

    assert json.loads(path.read_text())["by_stage"]["mid"] == {
        "n": 0,
        "rmse": None,
        "mae": None,
        "nasa_score": None,
    }
