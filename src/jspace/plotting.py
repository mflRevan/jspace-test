"""Shared figure style (thin marks, recessive grid, fixed categorical order)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated palette, light mode); the first
# three are safe for all-pairs comparisons.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#8a8986"
BAND = "#2a78d6"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelcolor": "#52514e",
            "axes.edgecolor": "#c9c8c3",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#ebeae6",
            "grid.linewidth": 0.6,
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "lines.linewidth": 2.0,
            "lines.markersize": 4,
            "legend.frameon": False,
            "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        }
    )


def shade_band(ax, band: tuple[int, int], label: str | None = "workspace band") -> None:
    ax.axvspan(band[0] - 0.5, band[1] + 0.5, color=BAND, alpha=0.07, lw=0, label=label)
