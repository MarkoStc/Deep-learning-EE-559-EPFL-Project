"""Matplotlib figures for training curves and evaluation reports (headless backend)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

LABELS = ("safe", "hate")
DPI = 150


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def confusion_matrix(cm: Sequence[Sequence[int]], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        np.asarray(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
    )
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    return _save(fig, path)


def grouped_bars(
    groups: Sequence[str],
    series: Mapping[str, Sequence[float]],
    title: str,
    ylabel: str,
    path: Path,
) -> Path:
    """One bar per (group, series); used for per-action ASR and subgroup F1."""
    x = np.arange(len(groups))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.2), 5))
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set(ylabel=ylabel, title=title, ylim=(0, 1.05))
    ax.legend()
    return _save(fig, path)


def line_panels(
    x: Sequence[float],
    panels: Mapping[str, Mapping[str, Sequence[float]]],
    xlabel: str,
    path: Path,
) -> Path:
    """One subplot per panel title, one line per series inside it."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4), squeeze=False)
    for ax, (title, series) in zip(axes[0], panels.items(), strict=True):
        for name, values in series.items():
            ax.plot(x, values, marker="o" if len(x) < 15 else None, label=name)
        ax.set(xlabel=xlabel, title=title)
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend(fontsize=7)
    return _save(fig, path)
