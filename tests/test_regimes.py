"""Tests for training-fitted operating regimes."""

import pandas as pd
import pytest

from rul.data.load import SENSOR_COLS
from rul.features.regimes import RegimeNormalizer


def cycles(rows):
    """Build a small C-MAPSS-shaped frame from (unit, cycle, setting, sensor)."""
    records = []
    for unit, cycle, setting, sensor in rows:
        record = {"unit": unit, "cycle": cycle, "op_1": setting, "op_2": 0, "op_3": 0}
        record.update({column: 5.0 for column in SENSOR_COLS})
        record["s_1"] = sensor
        records.append(record)
    return pd.DataFrame(records)


def test_six_regimes_are_detected_from_six_operating_conditions():
    train = cycles(
        [
            (unit, 1, setting, sensor)
            for setting in range(6)
            for unit, sensor in ((setting * 2 + 1, 0), (setting * 2 + 2, 2))
        ]
    )

    normalizer = RegimeNormalizer(n_regimes=6, random_state=42).fit(train)
    normalized = normalizer.transform(train)
    held_out = normalizer.transform(cycles([(13, 1, 4, 5)]))

    assert normalized["regime"].nunique() == 6
    assert held_out["regime"].iloc[0] == normalized["regime"].iloc[8]


def test_test_rows_use_training_sensor_statistics_within_their_regime():
    train = cycles([(1, 1, 0, 0), (2, 1, 0, 2), (3, 1, 10, 100), (4, 1, 10, 104)])
    test = cycles([(5, 1, 0, 3), (6, 1, 10, 108)])

    normalizer = RegimeNormalizer(n_regimes=2, random_state=42).fit(train)
    transformed_train = normalizer.transform(train)
    transformed_test = normalizer.transform(test)

    assert transformed_train["s_1"].tolist() == pytest.approx([-1, 1, -1, 1])
    assert transformed_test["s_1"].tolist() == pytest.approx([2, 3])
    assert transformed_test["s_2"].tolist() == [0, 0]
