"""Engine-level split contracts shared by all models."""

import pandas as pd

from rul.data.split import group_kfold_splits, holdout_split


def sample_cycles():
    return pd.DataFrame(
        {"unit": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], "cycle": [1, 2] * 6}
    )


def test_group_kfold_keeps_engines_together_and_validates_each_once():
    data = sample_cycles()

    folds = list(group_kfold_splits(data, n_splits=3, seed=42))

    assert len(folds) == 3
    validation_units = []
    for train_idx, val_idx in folds:
        train_units = set(data.iloc[train_idx]["unit"])
        val_units = set(data.iloc[val_idx]["unit"])
        assert train_units.isdisjoint(val_units)
        validation_units.extend(val_units)
    assert sorted(validation_units) == [1, 2, 3, 4, 5, 6]


def test_group_kfold_seed_reproduces_folds():
    data = sample_cycles()

    first = [
        tuple(data.iloc[val_idx]["unit"].unique())
        for _, val_idx in group_kfold_splits(data, 3, seed=42)
    ]
    again = [
        tuple(data.iloc[val_idx]["unit"].unique())
        for _, val_idx in group_kfold_splits(data, 3, seed=42)
    ]

    assert first == again


def test_holdout_split_keeps_engines_together_and_is_repeatable():
    data = sample_cycles()

    train_idx, val_idx = holdout_split(data, validation_size=2, seed=42)
    _, repeated_val_idx = holdout_split(data, validation_size=2, seed=42)

    assert len(set(data.iloc[val_idx]["unit"])) == 2
    assert set(data.iloc[train_idx]["unit"]).isdisjoint(set(data.iloc[val_idx]["unit"]))
    assert val_idx.tolist() == repeated_val_idx.tolist()
