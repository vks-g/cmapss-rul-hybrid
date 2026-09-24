"""Detect operating regimes from training operating settings."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans

from rul.data.load import SETTING_COLS


class RegimeNormalizer:
    """Assign operating regimes learned from training rows."""

    def __init__(self, n_regimes: int, random_state: int = 42) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state

    def fit(self, train: pd.DataFrame) -> RegimeNormalizer:
        """Fit regime centers using training operating settings only."""
        self.kmeans_ = KMeans(
            n_clusters=self.n_regimes, random_state=self.random_state, n_init=10
        ).fit(train[SETTING_COLS])
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Assign regimes using centers fitted on training rows."""
        labels = self.kmeans_.predict(frame[SETTING_COLS])
        result = frame.copy()
        result["regime"] = labels
        return result
