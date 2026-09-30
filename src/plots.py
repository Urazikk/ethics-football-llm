"""Graphiques du projet (matplotlib), palette sobre et cohérente."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GREY,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "font.size": 10, "axes.axisbelow": True,
})


def plot_premium(table: pd.DataFrame, title: str, ax=None):
    """Barres horizontales de la prime inexpliquée (en %), avec IC 95 %."""
    t = table.sort_values("premium_pct")
    ax = ax or plt.subplots(figsize=(7, 0.35 * len(t) + 1.2))[1]
    err = None
    if {"premium_low_pct", "premium_high_pct"} <= set(t.columns):
        err = [t["premium_pct"] - t["premium_low_pct"], t["premium_high_pct"] - t["premium_pct"]]
    ax.barh(t.index.astype(str), t["premium_pct"], color=BLUE, height=0.6, xerr=err,
            error_kw={"ecolor": GREY, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel("Prime de valeur à performance égale (%)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    return ax


def plot_selection_rates(before: pd.Series, after: pd.Series, title: str, labels=("Avant", "Après"), ax=None):
    """Taux de sélection par groupe, avant / après réduction du biais."""
    idx = before.index
    x = np.arange(len(idx))
    ax = ax or plt.subplots(figsize=(7, 3.5))[1]
    ax.bar(x - 0.2, before.values * 100, width=0.38, color=BLUE, label=labels[0])
    ax.bar(x + 0.2, after.reindex(idx).values * 100, width=0.38, color=ORANGE, label=labels[1])
    ax.set_xticks(x, idx, rotation=0)
    ax.set_ylabel("Taux de short-list (%)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    return ax
