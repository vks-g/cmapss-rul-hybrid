"""Tests for rul.data.labels."""

import pandas as pd
import pytest

from rul.data.labels import add_test_rul, add_train_rul


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


def test_train_rul_cap_flattens_the_early_part_of_the_curve():
    # Piecewise-linear target: constant at the cap, then falling to 0.
    train = cycles({1: [1, 2, 3, 4, 5]})

    assert add_train_rul(train, cap=2)["rul"].tolist() == [2, 2, 2, 1, 0]
    assert add_train_rul(train)["rul"].tolist() == [4, 3, 2, 1, 0]


@pytest.mark.parametrize("bad_cap", [0, -5])
def test_non_positive_cap_is_rejected(bad_cap):
    with pytest.raises(ValueError, match="cap"):
        add_train_rul(cycles({1: [1, 2]}), cap=bad_cap)


def rul_file(values: dict) -> pd.Series:
    """True RUL at the last test cycle, shaped like load_subset's rul_test."""
    return pd.Series(values, name="rul").rename_axis("unit")


def test_test_rul_counts_back_from_the_true_rul_at_the_last_cycle():
    # Engine 1 stops at cycle 3 with 112 cycles left, so cycle 1 had 114.
    test = cycles({1: [1, 2, 3], 2: [1, 2]})

    labelled = add_test_rul(test, rul_file({1: 112, 2: 98}))

    assert labelled["rul"].tolist() == [114, 113, 112, 99, 98]


def test_test_rul_can_be_capped_like_the_training_target():
    test = cycles({1: [1, 2, 3], 2: [1, 2]})

    labelled = add_test_rul(test, rul_file({1: 112, 2: 98}), cap=100)

    assert labelled["rul"].tolist() == [100, 100, 100, 99, 98]
