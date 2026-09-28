"""Training-only selection of useful sensor columns."""

import pandas as pd

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
