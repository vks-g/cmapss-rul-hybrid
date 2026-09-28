"""Select sensors using training rows only."""

from collections.abc import Sequence

import pandas as pd

from rul.data.load import SENSOR_COLS


def select_sensors(
    train: pd.DataFrame, sensors: Sequence[str] = SENSOR_COLS
) -> list[str]:
    """Return sensors that vary in the supplied training fold."""
    return [sensor for sensor in sensors if train[sensor].nunique(dropna=False) > 1]
