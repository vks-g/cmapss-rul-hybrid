"""Load the NASA C-MAPSS turbofan degradation files into DataFrames.

Each subset (FD001-FD004) ships as three whitespace-separated text files:
``train_FDxxx.txt`` (engines run to failure), ``test_FDxxx.txt`` (engines cut
off some time before failure) and ``RUL_FDxxx.txt`` (the true remaining useful
life at the last recorded cycle of each test engine).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

SETTING_COLS = [f"op_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]

# src/rul/data/load.py -> repo root is three folders up from this file's folder.
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "CMAPSSData"


class CMAPSSData(NamedTuple):
    """One C-MAPSS subset. Unpacks as ``train, test, rul_test = ...``."""

    train: pd.DataFrame
    test: pd.DataFrame
    rul_test: pd.Series | None


def load_subset(subset: str, data_dir: str | Path = DATA_DIR) -> CMAPSSData:
    """Load the train and test files of one C-MAPSS subset.

    Args:
        subset: Subset name, e.g. ``"FD001"``.
        data_dir: Folder holding the raw ``.txt`` files.

    Returns:
        ``CMAPSSData(train, test, rul_test)``. ``train`` and ``test`` have one
        row per engine cycle, with the columns in :data:`COLUMNS`.
    """
    data_dir = Path(data_dir)
    train = _read_cycles(data_dir / f"train_{subset}.txt")
    test = _read_cycles(data_dir / f"test_{subset}.txt")
    return CMAPSSData(train, test, None)


def _read_cycles(path: Path) -> pd.DataFrame:
    """Read one train/test file: 26 space-separated columns, no header."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return df.astype({"unit": "int64", "cycle": "int64"})
