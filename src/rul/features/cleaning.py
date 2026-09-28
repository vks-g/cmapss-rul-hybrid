"""Select sensors using training rows only."""

from collections.abc import Sequence

import pandas as pd

from rul.data.load import SENSOR_COLS


def select_sensors(
    train: pd.DataFrame,
    sensors: Sequence[str] = SENSOR_COLS,
    min_relative_std: float = 0.0,
) -> list[str]:
    """Return training sensors with relative standard deviation above the cutoff."""
    return [
        sensor
        for sensor in sensors
        if train[sensor].std(ddof=0) / max(abs(train[sensor].mean()), 1.0)
        > min_relative_std
    ]
