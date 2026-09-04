"""Shared publication style and deterministic exports for appendix figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK = "#25364A"
MUTED = "#637487"
BLUE = "#326A91"
TEAL = "#25878A"
ORANGE = "#C57731"
GRAY = "#8A96A3"
LIGHT = "#F1F5F7"
GRID = "#DEE5EB"


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
        "savefig.facecolor": "white",
    })


def save_figure(fig, name):
    (ROOT / "figures").mkdir(exist_ok=True)
    preview = ROOT / "tmp" / "appendix_figures"
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        ROOT / "figures" / f"{name}.pdf",
        bbox_inches="tight", pad_inches=0.06,
        metadata={"CreationDate": None, "ModDate": None, "Creator": "Matplotlib"},
    )
    fig.savefig(preview / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
