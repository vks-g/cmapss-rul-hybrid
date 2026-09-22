"""Tests for rul.evaluation.plots."""

import matplotlib

matplotlib.use("Agg")  # render off-screen; must run before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from rul.evaluation.metrics import evaluate  # noqa: E402
from rul.evaluation.plots import (  # noqa: E402
    SERIES,
    plot_pred_vs_true,
    plot_stage_errors,
    plot_trajectory,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_pred_vs_true_draws_one_point_per_engine_and_a_perfect_prediction_line():
    ax = plot_pred_vs_true([10, 50, 120], [15, 40, 130])

    assert ax.collections[0].get_offsets().tolist() == [[10, 15], [50, 40], [120, 130]]
    xs, ys = ax.lines[0].get_data()
    assert list(xs) == list(ys)  # the y = x reference
    assert xs[0] == 0 and xs[-1] >= 130  # and it spans every point
    assert "true" in ax.get_xlabel().lower()
    assert "predicted" in ax.get_ylabel().lower()


def test_pred_vs_true_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="3 targets"):
        plot_pred_vs_true([10, 50, 120], [15, 40])


def test_trajectory_draws_true_and_predicted_rul_over_the_engines_cycles():
    ax = plot_trajectory([1, 2, 3], [114, 113, 112], [100, 105, 111])

    lines = {line.get_label(): line.get_data() for line in ax.get_lines()}
    assert [list(v) for v in lines["True RUL"]] == [[1, 2, 3], [114, 113, 112]]
    assert [list(v) for v in lines["Predicted RUL"]] == [[1, 2, 3], [100, 105, 111]]
    assert ax.get_legend() is not None
    assert "cycle" in ax.get_xlabel().lower()


def test_trajectory_rejects_cycles_that_do_not_match_the_targets():
    with pytest.raises(ValueError, match="2 cycles for 3 targets"):
        plot_trajectory([1, 2], [114, 113, 112], [100, 105, 111])


# One engine per stage (true RUL 150 / 75 / 20), so each stage's RMSE is just
# that engine's absolute error.
RF = evaluate([150, 75, 20], [160, 70, 24])  # errors 10, 5, 4
XGB = evaluate([150, 75, 20], [150, 80, 30])  # errors 0, 5, 10


def test_stage_errors_draw_one_bar_group_per_stage_and_one_colour_per_model():
    ax = plot_stage_errors({"Random Forest": RF, "XGBoost": XGB})

    rf_bars, xgb_bars = ax.containers
    assert [b.get_height() for b in rf_bars] == [10, 5, 4]
    assert [b.get_height() for b in xgb_bars] == [0, 5, 10]
    assert [rf_bars.get_label(), xgb_bars.get_label()] == ["Random Forest", "XGBoost"]
    assert [t.get_text().split()[0] for t in ax.get_xticklabels()] == ["Early", "Mid", "Late"]
    # Colour follows the model's position: first model slot 1, second slot 2.
    assert rf_bars[0].get_facecolor() == to_rgba(SERIES[0])
    assert xgb_bars[0].get_facecolor() == to_rgba(SERIES[1])
    assert ax.get_legend() is not None


def test_stage_errors_can_show_mae_instead_of_rmse():
    rf = evaluate([150, 150, 75], [160, 150, 70])  # early errors 10 and 0: RMSE 7.07, MAE 5

    ax = plot_stage_errors({"Random Forest": rf}, metric="mae")

    assert [b.get_height() for b in ax.containers[0]][:2] == [5, 5]
    assert "MAE" in ax.get_ylabel()
