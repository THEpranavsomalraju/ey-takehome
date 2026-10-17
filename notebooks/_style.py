"""Shared chart style for the notebooks: class colors, chrome, and report loading.

Class colors follow a validated categorical palette (fixed order, colorblind
checked: adjacent CVD delta E >= 9.1 light); the dashboard reuses the same
order so a class keeps its color everywhere. Three light colors sit below 3:1
contrast, so every chart carries a legend and its table is shown beside it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from ledgerclf.config import get_settings
from ledgerclf.labels import CANONICAL_CLASSES
from ledgerclf.reports import load_report

CLASS_COLORS = dict(
    zip(
        CANONICAL_CLASSES,
        ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
        strict=True,
    )
)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SEQUENTIAL = "Blues"


def apply_style() -> None:
    """Set recessive axes, hairline grid, thin marks, and the light chart surface."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "figure.dpi": 110,
            "figure.figsize": (9, 4),
            "font.family": "sans-serif",
        }
    )


def report(file: str) -> pd.DataFrame | dict:
    """Load one report through the package (paths resolved from the project root)."""
    return load_report(get_settings(), file)


def class_colors(classes) -> list[str]:
    """Return the fixed color of each class, in the given order."""
    return [CLASS_COLORS[c] for c in classes]
