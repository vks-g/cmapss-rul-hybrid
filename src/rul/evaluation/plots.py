"""Evaluation plots shared by every model, sized and styled for the IEEE report.

Each function draws on an existing ``ax`` when given one (so notebooks can build
subplot grids), otherwise on a new figure, and returns the Axes.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

ArrayLike = Sequence[float] | np.ndarray

# Figures go into a white-page PDF, so the surface is plain white.
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Categorical slots in a fixed, colour-blind-checked order. A model keeps its
# slot in every figure, so "XGBoost is orange" holds across the whole report.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


def plot_pred_vs_true(
    y_true: ArrayLike, y_pred: ArrayLike, ax: Axes | None = None, title: str | None = None
) -> Axes:
    """Scatter of predicted against true RUL, one point per engine.

    Points on the ``y = x`` line are perfect. Above it the model
    predicts too much life left (late, the costly side); below it, too little.
    """
    true, pred = _paired(y_true, y_pred)
    ax = _axes(ax)
    top = float(max(true.max(), pred.max())) * 1.05

    ax.plot([0, top], [0, top], color=INK_MUTED, linewidth=1, zorder=2)
    ax.annotate("perfect prediction", (top, top), xytext=(-4, -12), textcoords="offset points",
                ha="right", va="top", fontsize=8, color=INK_MUTED)
    ax.scatter(true, pred, s=36, color=SERIES[0], edgecolors=SURFACE, linewidths=1, zorder=3)

    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    _title(ax, title)
    return ax


def _paired(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"got {true.size} targets and {pred.size} predictions")
    return true, pred


def _axes(ax: Axes | None) -> Axes:
    """Return ``ax``, or a new styled figure's Axes, with the shared chart chrome."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    return ax


def _title(ax: Axes, title: str | None) -> None:
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
