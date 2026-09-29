"""Assemble final, training-fitted C-MAPSS feature matrices."""

from collections.abc import Sequence
from typing import NamedTuple

import pandas as pd

from rul.data.labels import add_train_rul
from rul.data.load import SETTING_COLS
from rul.features.cleaning import select_sensors
from rul.features.regimes import RegimeNormalizer
from rul.features.rolling import add_rolling_features


class FeatureMatrices(NamedTuple):
    """Aligned train/test matrices and their permitted predictor columns."""

    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    sensors: list[str]


def build_feature_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    n_regimes: int,
    windows: Sequence[int] = (5, 10, 20),
    seed: int = 42,
) -> FeatureMatrices:
    """Fit preprocessing on official training rows and transform both sets."""
    sensors = select_sensors(train)
    normalizer = RegimeNormalizer(n_regimes=n_regimes, random_state=seed).fit(train)
    train_scaled = normalizer.transform(add_train_rul(train))
    test_scaled = normalizer.transform(test)
    train_rolled = add_rolling_features(train_scaled, sensors=sensors, windows=windows)
    test_rolled = add_rolling_features(test_scaled, sensors=sensors, windows=windows)
    feature_columns = ["cycle", *SETTING_COLS, "regime", *sensors]
    feature_columns += [
        f"{sensor}_{feature}_{window}"
        for window in windows
        for sensor in sensors
        for feature in ("mean", "slope", "health")
    ]
    return FeatureMatrices(
        train_rolled.loc[:, ["unit", *feature_columns, "rul"]],
        test_rolled.loc[:, ["unit", *feature_columns]],
        feature_columns,
        sensors,
    )
