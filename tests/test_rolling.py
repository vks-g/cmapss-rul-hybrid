"""Causal, per-engine degradation features."""

import pandas as pd

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
