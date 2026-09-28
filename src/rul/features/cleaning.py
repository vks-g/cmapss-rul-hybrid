"""Select sensors using training rows only."""

from collections.abc import Sequence
from math import isfinite

import pandas as pd

from rul.data.load import SENSOR_COLS


def select_sensors(
    train: pd.DataFrame,
    sensors: Sequence[str] = SENSOR_COLS,
    min_relative_std: float = 0.0,
) -> list[str]:
    """Return training sensors with relative standard deviation above the cutoff."""
    if not isfinite(min_relative_std) or min_relative_std < 0:
        raise ValueError("min_relative_std must be finite and nonnegative")
    kept = [
        sensor
        for sensor in sensors
        if train[sensor].nunique(dropna=False) > 1
        and train[sensor].std(ddof=0) / max(abs(train[sensor].mean()), 1.0)
        > min_relative_std
    ]
    if not kept:
        raise ValueError("No sensors exceeded the training variability cutoff")
    return kept
