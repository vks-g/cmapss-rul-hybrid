"""Training-only selection of useful sensor columns."""

import pandas as pd
import pytest

from rul.features.cleaning import select_sensors


def test_select_sensors_drops_only_training_constants():
    train = pd.DataFrame({"s_1": [5.0, 5.0, 5.0], "s_2": [1.0, 2.0, 3.0]})

    assert select_sensors(train, ["s_1", "s_2"]) == ["s_2"]


def test_select_sensors_can_drop_nearly_flat_training_signals():
    train = pd.DataFrame(
        {
            "s_1": [999.99, 1000.0, 1000.01],
            "s_2": [1.0, 2.0, 3.0],
        }
    )

    assert select_sensors(train, ["s_1", "s_2"], min_relative_std=0.001) == ["s_2"]


@pytest.mark.parametrize("cutoff", [-0.1, float("nan"), float("inf")])
def test_select_sensors_rejects_invalid_cutoff(cutoff):
    train = pd.DataFrame({"s_1": [1.0, 2.0]})

    with pytest.raises(ValueError, match="min_relative_std"):
        select_sensors(train, ["s_1"], min_relative_std=cutoff)


def test_select_sensors_rejects_training_fold_without_useful_sensors():
    train = pd.DataFrame({"s_1": [5.0, 5.0]})

    with pytest.raises(ValueError, match="No sensors"):
        select_sensors(train, ["s_1"])


def test_select_sensors_drops_float_constant_despite_roundoff():
    train = pd.DataFrame(
        {"s_1": [14.62] * 7, "s_2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
    )

    assert select_sensors(train, ["s_1", "s_2"]) == ["s_2"]
