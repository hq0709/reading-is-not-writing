"""Shared publication style and deterministic exports for every figure of the paper.

The style is the MedVIGIL journal house style (serif type matching the body face, a full box frame with a
light solid grid, boxed legends, "(a)  ..." panel titles, PDF with Type 42 fonts) with a Morandi palette:
muted, dusty, low-saturation tones ordered by lightness. The constant names of the original appendix style
are kept so the plotting scripts keep working; their values now map onto the palette
(BLUE -> haze, TEAL -> sage, ORANGE -> terracotta, GRAY -> warm grey, LIGHT -> paper tint).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]

# Morandi palette (eight chromatic slots, ordered by lightness) and warm neutrals
ROSE = "#C08585"
SAGE = "#8FA68E"
HAZE = "#7D9AB2"
OAT = "#C6A87C"
LILAC = "#9B8EA9"
MUSTARD = "#C9B36A"
TERRACOTTA = "#B5766A"
SLATE = "#6F7C8A"
CHARCOAL, GREY, STONE = "#5C5B58", "#B8B4AE", "#D6D2CC"
RED, GREEN = TERRACOTTA, "#6E8B6B"
BAND = "#EEF2EC"
SLOTS = [ROSE, SAGE, HAZE, OAT, LILAC, MUSTARD, TERRACOTTA, SLATE]

# legacy names used by the appendix scripts
INK = "#3a3a3a"
MUTED = "#6a6a6a"
CHANCE = "#9a9a9a"
BLUE = HAZE
TEAL = SAGE
ORANGE = TERRACOTTA
GRAY = GREY
LIGHT = "#F5F1EC"
GRID = "#e6e4e1"

DOUBLE, SINGLE = 7.2, 3.5
LEG = dict(fontsize=6.3, handlelength=1.4, borderpad=0.35, labelspacing=0.3, handletextpad=0.4, columnspacing=1.0)

RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": "#7a7a7a", "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.color": "#444444", "ytick.color": "#444444", "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "#8a8a8a", "legend.fancybox": False,
    "legend.borderpad": 0.4, "legend.handlelength": 2.2, "legend.handletextpad": 0.5,
    "lines.linewidth": 1.5, "lines.markersize": 5.5, "lines.markeredgewidth": 0.9, "lines.markeredgecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
}


def apply_style():
    plt.rcParams.update(RC)


def panel_title(ax, letter, text):
    ax.set_title(f"({letter})  {text}", loc="center", fontsize=9, pad=5)


def series_handle(color, marker, label, ls="--"):
    return Line2D([], [], color=color, marker=marker, ls=ls, lw=1.5, ms=5.5, mec="white", mew=0.9, label=label)


def boxed_legend(fig_or_ax, handles, **kw):
    kw.setdefault("frameon", True)
    kw.setdefault("edgecolor", "#8a8a8a")
    return fig_or_ax.legend(handles=handles, **kw)


def ref_line(ax, y=None, x=None, color=RED, label=None, ls="--", lw=1.0, where="right", pad=1.0):
    if y is not None:
        ax.axhline(y, color=color, ls=ls, lw=lw, zorder=2)
        if label:
            xr = ax.get_xlim()[1] if where == "right" else ax.get_xlim()[0]
            ax.annotate(label, (xr, y), xytext=(-3 if where == "right" else 3, pad), textcoords="offset points",
                        ha="right" if where == "right" else "left", va="bottom", fontsize=7.5, color=color)
    if x is not None:
        ax.axvline(x, color=color, ls=ls, lw=lw, zorder=2)
        if label:
            ax.annotate(label, (x, ax.get_ylim()[1]), xytext=(3, -3), textcoords="offset points",
                        ha="left", va="top", fontsize=7.5, color=color)


def chance_line(ax, y=None, x=None, label=None, where="right"):
    if y is not None:
        ax.axhline(y, color=CHANCE, lw=0.8, zorder=2)
        if label:
            xr = ax.get_xlim()[1] if where == "right" else ax.get_xlim()[0]
            ax.annotate(label, (xr, y), xytext=(-3 if where == "right" else 3, 2), textcoords="offset points",
                        ha="right" if where == "right" else "left", va="bottom", fontsize=6.2, color=MUTED)
    if x is not None:
        ax.axvline(x, color=CHANCE, lw=0.8, zorder=2)


def save_figure(fig, name):
    (ROOT / "figures").mkdir(exist_ok=True)
    preview = ROOT / "tmp" / "appendix_figures"
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        ROOT / "figures" / f"{name}.pdf",
        bbox_inches="tight", pad_inches=0.06,
        metadata={"CreationDate": None, "ModDate": None, "Creator": "Matplotlib"},
    )
    fig.savefig(preview / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
