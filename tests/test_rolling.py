"""Causal, per-engine degradation features."""

import pandas as pd
import pytest

from rul.features.rolling import add_rolling_features


def test_rolling_mean_uses_only_past_cycles_of_same_engine():
    frame = pd.DataFrame(
        {
            "unit": [1, 2, 1, 2, 1],
            "cycle": [1, 1, 2, 2, 3],
            "s_1": [1.0, 10.0, 3.0, 12.0, 100.0],
        }
    )

    result = add_rolling_features(frame, sensors=["s_1"], windows=[2])

    assert result["s_1_mean_2"].tolist() == [1.0, 10.0, 2.0, 11.0, 51.5]
    assert result.index.equals(frame.index)


def test_rolling_slope_uses_trailing_cycles_only():
    frame = pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2],
            "s_1": [1.0, 3.0, 5.0, 100.0, 10.0, 11.0],
        }
    )

    result = add_rolling_features(frame, sensors=["s_1"], windows=[3])

    assert result["s_1_slope_3"].tolist() == [0.0, 2.0, 2.0, 48.5, 0.0, 1.0]


def test_health_indicator_compares_recent_mean_with_first_engine_cycle():
    frame = pd.DataFrame(
        {"unit": [1, 1, 2, 1, 2], "cycle": [1, 2, 1, 3, 2], "s_1": [1, 3, 10, 5, 11]}
    )

    result = add_rolling_features(frame, sensors=["s_1"], windows=[2])

    assert result["s_1_health_2"].tolist() == [0.0, 1.0, 0.0, 3.0, 0.5]
    assert result["cycle"].tolist() == frame["cycle"].tolist()


@pytest.mark.parametrize("windows", [[], [0], [-2], [2, 2], [1.5]])
def test_rolling_windows_must_be_distinct_positive_integers(windows):
    frame = pd.DataFrame({"unit": [1, 1], "cycle": [1, 2], "s_1": [1.0, 2.0]})

    with pytest.raises(ValueError, match="windows"):
        add_rolling_features(frame, sensors=["s_1"], windows=windows)
