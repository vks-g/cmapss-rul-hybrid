"""Causal degradation features computed within each engine."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rul.data.load import SENSOR_COLS


def add_rolling_features(
    frame: pd.DataFrame,
    sensors: Sequence[str] = SENSOR_COLS,
    windows: Sequence[int] = (5, 10, 20),
) -> pd.DataFrame:
    """Add trailing sensor means in cycle order, without mixing engines."""
    order = np.lexsort(
        (np.arange(len(frame)), frame["cycle"].to_numpy(), frame["unit"].to_numpy())
    )
    result = frame.iloc[order].copy()
    for window in windows:
        for sensor in sensors:
            result[f"{sensor}_mean_{window}"] = result.groupby("unit", sort=False)[
                sensor
            ].transform(lambda values: values.rolling(window, min_periods=1).mean())
    return result.iloc[np.argsort(order)]
