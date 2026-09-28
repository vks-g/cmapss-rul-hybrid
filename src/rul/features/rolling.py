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
    """Add trailing sensor means and slopes without mixing engines."""
    order = np.lexsort(
        (np.arange(len(frame)), frame["cycle"].to_numpy(), frame["unit"].to_numpy())
    )
    result = frame.iloc[order].copy()
    for window in windows:
        def rolling_sum(values: pd.Series) -> np.ndarray:
            return values.groupby(result["unit"], sort=False).transform(
                lambda group: group.rolling(window, min_periods=1).sum()
            ).to_numpy()

        cycle = result["cycle"].astype(float)
        count = result.groupby("unit", sort=False).cumcount().add(1).clip(upper=window)
        sum_cycle = rolling_sum(cycle)
        sum_cycle_sq = rolling_sum(cycle * cycle)
        denominator = count.to_numpy() * sum_cycle_sq - sum_cycle**2
        for sensor in sensors:
            result[f"{sensor}_mean_{window}"] = result.groupby("unit", sort=False)[
                sensor
            ].transform(lambda values: values.rolling(window, min_periods=1).mean())
            values = result[sensor]
            numerator = (
                count.to_numpy() * rolling_sum(cycle * values)
                - sum_cycle * rolling_sum(values)
            )
            result[f"{sensor}_slope_{window}"] = np.divide(
                numerator,
                denominator,
                out=np.zeros(len(result), dtype=float),
                where=denominator != 0,
            )
    return result.iloc[np.argsort(order)]
