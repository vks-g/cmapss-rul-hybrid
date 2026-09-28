"""Causal degradation features computed within each engine."""

from collections.abc import Sequence
from numbers import Integral

import numpy as np
import pandas as pd

from rul.data.load import SENSOR_COLS


def add_rolling_features(
    frame: pd.DataFrame,
    sensors: Sequence[str] = SENSOR_COLS,
    windows: Sequence[int] = (5, 10, 20),
) -> pd.DataFrame:
    """Add trailing means, slopes, and change from each engine's first cycle."""
    if (
        not windows
        or any(
            not isinstance(window, Integral) or isinstance(window, bool) or window <= 0
            for window in windows
        )
        or len(set(windows)) != len(windows)
    ):
        raise ValueError("windows must contain distinct positive integers")
    new_columns = {
        f"{sensor}_{feature}_{window}"
        for sensor in sensors
        for window in windows
        for feature in ("mean", "slope", "health")
    }
    collisions = new_columns.intersection(frame.columns)
    if collisions:
        raise ValueError(f"Rolling feature columns already exist: {sorted(collisions)}")
    if frame[list(sensors)].isna().to_numpy().any():
        raise ValueError("Input contains missing sensor readings")
    order = np.lexsort(
        (np.arange(len(frame)), frame["cycle"].to_numpy(), frame["unit"].to_numpy())
    )
    result = frame.iloc[order].copy()
    feature_columns: dict[str, np.ndarray] = {}
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
            recent_mean = result.groupby("unit", sort=False)[
                sensor
            ].transform(lambda values: values.rolling(window, min_periods=1).mean()).to_numpy()
            initial = result.groupby("unit", sort=False)[sensor].transform("first").to_numpy()
            feature_columns[f"{sensor}_mean_{window}"] = recent_mean
            feature_columns[f"{sensor}_health_{window}"] = recent_mean - initial
            values = result[sensor]
            numerator = (
                count.to_numpy() * rolling_sum(cycle * values)
                - sum_cycle * rolling_sum(values)
            )
            feature_columns[f"{sensor}_slope_{window}"] = np.divide(
                numerator,
                denominator,
                out=np.zeros(len(result), dtype=float),
                where=denominator != 0,
            )
    result = pd.concat([result, pd.DataFrame(feature_columns, index=result.index)], axis=1)
    return result.iloc[np.argsort(order)]
