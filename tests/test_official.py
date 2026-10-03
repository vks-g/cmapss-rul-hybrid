"""Tests for rul.evaluation.official."""

import pandas as pd

from rul.evaluation.official import predict_test_cycles


def cycles(spec: dict) -> pd.DataFrame:
    """Frame with one row per (unit, cycle): {unit: [cycles in row order]}."""
    rows = [(unit, c) for unit, cs in spec.items() for c in cs]
    return pd.DataFrame(rows, columns=["unit", "cycle"])


def rul_file(values: dict) -> pd.Series:
    """True RUL at the last test cycle, shaped like load_subset's rul_test."""
    return pd.Series(values, name="rul").rename_axis("unit")


class CountdownModel:
    """Predicts 120 - cycle and remembers the frame it was asked about."""

    def predict(self, frame: pd.DataFrame) -> list[float]:
        self.seen = frame
        return (120 - frame["cycle"]).tolist()


def test_every_test_cycle_gets_its_prediction_next_to_its_true_rul():
    # Engine 1 stops at cycle 3 with 112 cycles left; engine 2 at cycle 2 with 98.
    test = cycles({1: [1, 2, 3], 2: [1, 2]})

    result = predict_test_cycles(CountdownModel(), test, rul_file({1: 112, 2: 98}))

    assert result.values.tolist() == [
        [1, 1, 114, 119],
        [1, 2, 113, 118],
        [1, 3, 112, 117],
        [2, 1, 99, 119],
        [2, 2, 98, 118],
    ]
    assert list(result.columns) == ["unit", "cycle", "rul", "prediction"]


def test_the_model_never_sees_the_true_rul():
    model = CountdownModel()

    predict_test_cycles(model, cycles({1: [1, 2, 3]}), rul_file({1: 112}))

    assert "rul" not in model.seen.columns


def test_the_model_sees_each_engines_whole_history_not_just_its_last_cycle():
    # Rolling features at the last cycle need the cycles before it.
    model = CountdownModel()

    predict_test_cycles(model, cycles({1: [1, 2, 3], 2: [1, 2]}), rul_file({1: 112, 2: 98}))

    assert model.seen[["unit", "cycle"]].values.tolist() == [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2]]
