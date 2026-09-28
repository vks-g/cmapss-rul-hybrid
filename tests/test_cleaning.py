"""Training-only selection of useful sensor columns."""

import pandas as pd

from rul.features.cleaning import select_sensors


def test_select_sensors_drops_only_training_constants():
    train = pd.DataFrame({"s_1": [5.0, 5.0, 5.0], "s_2": [1.0, 2.0, 3.0]})

    assert select_sensors(train, ["s_1", "s_2"]) == ["s_2"]
