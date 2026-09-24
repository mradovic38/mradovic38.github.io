"""Shared style for the blog figures (English versions of the paper's figures).

Figures are saved as PNG to images/f-divergences/ at the root of the website repo.
Run all of them with:  python scripts_en/make_all.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# <repo>/images/f-divergences (this file lives in <repo>/_drafts/f-divergences/scripts_en/)
OUT = Path(__file__).resolve().parents[3] / "images" / "f-divergences"

# Same palette as the paper.
BLUE = "#2b5d8a"
RED = "#b03a2e"
GREEN = "#3d7a4f"
ORANGE = "#c87a20"
PURPLE = "#6b4a8a"
GRAY = "#6e6e6e"

COLORS = [BLUE, RED, GREEN, ORANGE, PURPLE]


def setup() -> None:
    """Global settings, tuned for figures shown ~770px wide on the blog."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "none",
            "legend.framealpha": 0.9,
            "figure.constrained_layout.use": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


def save(fig, name: str) -> None:
    """Save the figure as images/f-divergences/<name>.png."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"saved: {path}")
