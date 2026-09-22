"""Tests for rul.data.labels."""

import pandas as pd

from rul.data.labels import add_train_rul


def cycles(spec: dict) -> pd.DataFrame:
    """Frame with one row per (unit, cycle): {unit: [cycles in row order]}."""
    rows = [(unit, c) for unit, cs in spec.items() for c in cs]
    return pd.DataFrame(rows, columns=["unit", "cycle"])


def test_train_rul_counts_cycles_left_until_each_engines_last_cycle():
    train = cycles({1: [1, 2, 3], 2: [1, 2]})

    labelled = add_train_rul(train)

    assert labelled["rul"].tolist() == [2, 1, 0, 1, 0]


def test_train_rul_does_not_depend_on_row_order():
    train = cycles({1: [3, 1, 2]})

    assert add_train_rul(train)["rul"].tolist() == [0, 2, 1]


def test_train_rul_leaves_the_input_frame_untouched():
    train = cycles({1: [1, 2]})

    add_train_rul(train)

    assert "rul" not in train.columns
