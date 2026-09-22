"""Tests for rul.evaluation.plots."""

import matplotlib

matplotlib.use("Agg")  # render off-screen; must run before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from rul.evaluation.plots import plot_pred_vs_true  # noqa: E402


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
