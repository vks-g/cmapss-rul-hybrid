"""Training-fitted operating regimes and within-regime sensor normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from rul.data.load import SENSOR_COLS, SETTING_COLS


class RegimeNormalizer:
    """Cluster operating settings and z-score sensors within each training regime."""

    def __init__(self, n_regimes: int, random_state: int = 42) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state

    def fit(self, train: pd.DataFrame) -> RegimeNormalizer:
        """Fit regime centers and sensor statistics using training rows only."""
        if train[SETTING_COLS].drop_duplicates().shape[0] < self.n_regimes:
            raise ValueError("Fewer distinct operating settings than requested regimes")
        self.kmeans_ = KMeans(
            n_clusters=self.n_regimes, random_state=self.random_state, n_init=10
        ).fit(train[SETTING_COLS])
        labels = self.kmeans_.labels_
        self.scalers_ = {
            int(label): StandardScaler().fit(train.loc[labels == label, SENSOR_COLS])
            for label in np.unique(labels)
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Assign learned regimes and scale sensors with training statistics."""
        if not hasattr(self, "kmeans_") or not hasattr(self, "scalers_"):
            raise NotFittedError("Fit RegimeNormalizer on training rows before transform")
        labels = self.kmeans_.predict(frame[SETTING_COLS])
        scaled = np.empty((len(frame), len(SENSOR_COLS)), dtype=float)
        for label in np.unique(labels):
            mask = labels == label
            scaled[mask] = self.scalers_[int(label)].transform(frame.loc[mask, SENSOR_COLS])
        result = frame.copy()
        result[SENSOR_COLS] = scaled
        result["regime"] = labels
        return result
