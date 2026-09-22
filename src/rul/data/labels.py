"""Remaining Useful Life (RUL) targets for C-MAPSS engines.

Training engines run until failure, so the RUL at any cycle is simply how many
cycles that engine still had left. Test engines stop early; their true RUL at
the last recorded cycle comes from ``RUL_FDxxx.txt``.
"""

from __future__ import annotations

import pandas as pd


def add_train_rul(train: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``train`` with an ``rul`` column.

    ``rul = (last cycle of that engine) - cycle``, so every engine ends at 0.
    """
    last_cycle = train.groupby("unit")["cycle"].transform("max")
    return train.assign(rul=last_cycle - train["cycle"])
