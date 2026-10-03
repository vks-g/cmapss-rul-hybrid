"""Scoring on the official C-MAPSS test set.

Test engines stop before failure, and ``RUL_FDxxx.txt`` gives the true RUL at
each engine's last recorded cycle only. A model therefore predicts from the
engine's whole recorded history (so rolling features have their past), and
the engine is scored once, at that last cycle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rul.data.labels import add_test_rul, last_cycles
from rul.evaluation.metrics import evaluate


def predict_test_cycles(model, test: pd.DataFrame, rul_test: pd.Series) -> pd.DataFrame:
    """Predict RUL at every recorded test cycle, next to the true RUL.

    Args:
        model: Fitted estimator whose ``predict`` takes a frame of test cycles.
        test: Test cycles as returned by :func:`rul.data.load.load_subset`.
        rul_test: True RUL at each engine's last cycle, indexed by ``unit``.

    Returns:
        ``unit``, ``cycle``, ``rul`` and ``prediction``, one row per test cycle
        in the order of ``test``.
    """
    labelled = add_test_rul(test, rul_test)
    prediction = np.asarray(model.predict(test), dtype=float)
    return labelled.loc[:, ["unit", "cycle", "rul"]].assign(prediction=prediction)


def score_last_cycle(cycle_predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Score each test engine once, at its last recorded cycle.

    Args:
        cycle_predictions: Output of :func:`predict_test_cycles`.

    Returns:
        The last-cycle rows (one per engine, sorted by ``unit``) and their
        :func:`rul.evaluation.metrics.evaluate` scores.
    """
    last = last_cycles(cycle_predictions)
    return last, evaluate(last["rul"], last["prediction"])
