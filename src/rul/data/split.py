"""Reproducible train/validation splits that keep each engine intact."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def group_kfold_splits(
    data: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield seeded GroupKFold row indices, grouping all cycles by ``unit``."""
    groups = data["unit"]
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(data, groups=groups):
        assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[val_idx])
        yield train_idx, val_idx


def holdout_split(
    data: pd.DataFrame, validation_size: float | int = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return one seeded engine-level train/validation split as row indices."""
    groups = data["unit"]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_size, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(data, groups=groups))
    assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[val_idx])
    return train_idx, val_idx
