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


def add_test_rul(
    test: pd.DataFrame, rul_test: pd.Series, cap: float | None = None
) -> pd.DataFrame:
    """Return a copy of ``test`` with the true ``rul`` at every recorded cycle.

    ``RUL_FDxxx.txt`` gives the RUL at each engine's last recorded cycle, so an
    earlier cycle had that many cycles left plus the cycles still to come:
    ``rul = rul_test[unit] + (last cycle - cycle)``.

    Args:
        test: Cycles of the test engines, with ``unit`` and ``cycle``.
        rul_test: True RUL at the last cycle, indexed by ``unit`` (as returned
            by :func:`rul.data.load.load_subset`).
        cap: Optional upper limit on the RUL; ``None`` keeps it uncapped.
    """
    last_cycle = test.groupby("unit")["cycle"].transform("max")
    missing = sorted(set(test["unit"]) - set(rul_test.index))
    if missing:
        raise ValueError(f"no true RUL for test engine(s) {missing}")
    rul_at_last = test["unit"].map(rul_test)
    rul = rul_at_last + (last_cycle - test["cycle"])
    return test.assign(rul=_apply_cap(rul, cap))
