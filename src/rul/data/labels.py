"""Remaining Useful Life (RUL) targets for C-MAPSS engines.

Training engines run until failure, so the RUL at any cycle is simply how many
cycles that engine still had left. Test engines stop early; their true RUL at
the last recorded cycle comes from ``RUL_FDxxx.txt``.

Uncapped RUL is the project's primary target. ``cap`` produces the common
piecewise-linear target (flat at ``cap`` while the engine is still healthy),
which we only use as an ablation.
"""

from __future__ import annotations

import pandas as pd


def add_train_rul(train: pd.DataFrame, cap: float | None = None) -> pd.DataFrame:
    """Return a copy of ``train`` with an ``rul`` column.

    ``rul = (last cycle of that engine) - cycle``, so every engine ends at 0.

    Args:
        train: Cycles of run-to-failure engines, with ``unit`` and ``cycle``.
        cap: Optional upper limit on the RUL; ``None`` keeps it uncapped.
    """
    last_cycle = train.groupby("unit")["cycle"].transform("max")
    return train.assign(rul=_apply_cap(last_cycle - train["cycle"], cap))


def _apply_cap(rul: pd.Series, cap: float | None) -> pd.Series:
    if cap is None:
        return rul
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    return rul.clip(upper=cap)
