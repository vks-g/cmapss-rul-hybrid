"""Fold-fitted feature pipelines for the raw/engineered ML comparison."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from rul.data.load import SENSOR_COLS, SETTING_COLS
from rul.features.cleaning import select_sensors
from rul.features.regimes import RegimeNormalizer
from rul.features.rolling import add_rolling_features


class EngineFeatures(TransformerMixin, BaseEstimator):
    """Keep engine metadata for causal transforms, then return predictors only."""

    def __init__(self, engineered=False, n_regimes=1, windows=(5, 10, 20), seed=42):
        self.engineered = engineered
        self.n_regimes = n_regimes
        self.windows = windows
        self.seed = seed

    def fit(self, X: pd.DataFrame, y=None):
        """Learn selected sensors and regime statistics from this training fold."""
        if self.engineered:
            self.sensors_ = select_sensors(X)
            self.normalizer_ = RegimeNormalizer(self.n_regimes, self.seed).fit(X)
            self.feature_names_ = ["cycle", *SETTING_COLS, "regime", *self.sensors_]
            self.feature_names_ += [
                f"{sensor}_{feature}_{window}"
                for window in self.windows
                for sensor in self.sensors_
                for feature in ("mean", "slope", "health")
            ]
        else:
            self.feature_names_ = ["cycle", *SETTING_COLS, *SENSOR_COLS]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform each engine using training statistics and observed history."""
        check_is_fitted(self, "feature_names_")
        if self.engineered:
            X = add_rolling_features(
                self.normalizer_.transform(X), self.sensors_, self.windows
            )
        return X.loc[:, self.feature_names_].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the predictor names in their model-input order."""
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)
