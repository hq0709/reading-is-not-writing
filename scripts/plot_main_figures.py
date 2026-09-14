"""Render the main-text figures 1-3 from the repository data in the house style.

fig1_framework            schematic of reading, answering and writing at one consumed visual block
fig2_dose_responses       dose-response curves of the three original cells (data/accepted_results.json: cells)
fig3_direction_specificity locked-dose effects of the Effusion direction, the five clinical alternatives,
                          the sham and the random group, plus the ownership contrast
                          (data/accepted_results.json: specificity)
Run with the paper's figure environment; numbers are copied from the data files, never recomputed.
"""

import json

from appendix_style import (
    BAND, CHANCE, CHARCOAL, GREEN, GREY, HAZE, INK, LEG, LILAC, MUTED, OAT, RED, ROOT, ROSE, SAGE, SLATE,
    STONE, TERRACOTTA, apply_style, boxed_legend, chance_line, panel_title, ref_line, save_figure,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np


def read():
    with (ROOT / "data" / "accepted_results.json").open(encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------- fig2
def dose_responses(results):
    cells = {c["key"]: c for c in results["cells"]}
    order = [("llava_effusion", "LLaVA-1.5 Effusion"), ("llava_edema", "LLaVA-1.5 Edema"),
             ("qwen_effusion", "Qwen2.5-VL Effusion")]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.15), sharey=True, constrained_layout=True)
    for k, (ax, (key, title)) in enumerate(zip(axes, order)):
        cell = cells[key]
        alphas = np.array([d["alpha"] for d in cell["dose"]])
        change = np.array([d["raw_change"] for d in cell["dose"]])
        ctrl = {float(a): v for a, v in cell["controls"].items()}
        ca = np.array(sorted(ctrl))
        lo = np.array([ctrl[a]["random_p05"] for a in ca])
        hi = np.array([ctrl[a]["random_p95"] for a in ca])
        sham = np.array([ctrl[a]["sham_effect"] for a in ca])
        ax.fill_between(ca, lo, hi, color=STONE, alpha=0.6, lw=0, zorder=1)
        ax.plot(alphas, change, ls="--", marker="o", color=HAZE, ms=4.5, zorder=3)
        ax.plot(ca, sham, ls="none", marker="x", color=TERRACOTTA, ms=5.5, mew=1.2, mec=TERRACOTTA, zorder=4)
        chance_line(ax, y=0)
        ax.axvline(0, color=CHANCE, lw=0.6, ls=":")
        ax.set_xlim(-1.08, 1.08)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlabel(r"relative-token dose $\alpha$")
        panel_title(ax, "abc"[k], title)
    axes[0].set_ylabel(r"change in mean $P(\mathrm{yes})$")
    axes[0].set_ylim(-0.32, 0.32)
    handles = [Line2D([], [], ls="--", marker="o", color=HAZE, ms=4.5, mec="white", label="probe normal"),
               Patch(facecolor="#E9E5DF", edgecolor=GREY, lw=0.5, label="random 5–95%"),
               Line2D([], [], ls="none", marker="x", color=TERRACOTTA, mec=TERRACOTTA, ms=5.5, mew=1.2, label="sham")]
    boxed_legend(axes[2], handles, loc="upper left", **LEG)
    save_figure(fig, "fig2_dose_responses")


# ------------------------------------------------------------------------------------------- fig3
def direction_specificity(results):
    spec = results["specificity"]
    eff = spec["direction_effects"]
    randoms = np.array([v for k, v in eff.items() if k.startswith("random")])
    alt = [("unrelated_Atelectasis", "Atel."), ("unrelated_Cardiomegaly", "Card."), ("unrelated_Mass", "Mass"),
           ("unrelated_Nodule", "Nodule"), ("unrelated_Pneumothorax", "Pneumo.")]
    fig = plt.figure(figsize=(5.5, 2.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0])
    ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])

    # (a) effects at the locked dose
    xs = {"Random": 0, "Effusion": 1, "Sham": 2}
    for i, (_, lab) in enumerate(alt):
        xs[lab] = 3 + i
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.14, 0.14, size=len(randoms))
    ax.plot(jitter, randoms, ls="none", marker="o", ms=3.6, color=GREY, mec="white", mew=0.5, zorder=3)
    ax.hlines(spec["random_effect_p95"], -0.28, 0.28, color=RED, ls="--", lw=1.0, zorder=4)
    ax.annotate("p95", (0.30, spec["random_effect_p95"]), fontsize=7, color=RED, va="center", ha="left")
    ax.plot(1, eff["concept"], ls="none", marker="o", ms=8, color=HAZE, mec="white", zorder=5)
    ax.plot(2, eff["sham"], ls="none", marker="X", ms=8, color=LILAC, mec="white", zorder=5)
    for i, (key, lab) in enumerate(alt):
        col = TERRACOTTA if key == spec["primary"]["maximum_unrelated_direction"] else SAGE
        ax.plot(3 + i, eff[key], ls="none", marker="D", ms=7, color=col, mec="white", zorder=5)
    chance_line(ax, y=0)
    ax.set_xticks(range(8)); ax.set_xticklabels(list(xs), rotation=30, ha="right")
    ax.set_xlim(-0.6, 7.6)
    ax.set_ylabel(r"change in mean $P(\mathrm{yes})$")
    panel_title(ax, "a", rf"locked dose $\alpha=+{spec['alpha']:.2f}$, {spec['n_eval_patients']} new patients")
    handles = [Line2D([], [], ls="none", marker="o", color=GREY, ms=3.6, mec="white", label="20 random directions"),
               Line2D([], [], color=RED, ls="--", lw=1.0, label="random p95"),
               Line2D([], [], ls="none", marker="o", color=HAZE, ms=6, mec="white", label="Effusion direction"),
               Line2D([], [], ls="none", marker="X", color=LILAC, ms=6, mec="white", label="sham"),
               Line2D([], [], ls="none", marker="D", color=SAGE, ms=5.5, mec="white", label="clinical alternatives"),
               Line2D([], [], ls="none", marker="D", color=TERRACOTTA, ms=5.5, mec="white", label="strongest alternative")]
    boxed_legend(ax, handles, loc="upper left", ncol=2, fontsize=6.6, handlelength=1.2, borderpad=0.35, labelspacing=0.3, handletextpad=0.4, columnspacing=0.8)

    # (b) ownership contrast
    p = spec["primary"]
    lo, hi = p["ci95"]
    bx.errorbar(p["margin"], 0, xerr=[[p["margin"] - lo], [hi - p["margin"]]], fmt="o", color=TERRACOTTA,
                ms=6, mec="white", ecolor="#555555", elinewidth=0.7, capsize=2.5, zorder=4)
    bx.axvline(0, color=CHANCE, lw=0.8)
    bx.plot([p["one_sided_lower_95"]], [0], marker="|", ms=9, color=TERRACOTTA, mew=1.2, ls="none", zorder=5)
    bx.set_yticks([0]); bx.set_yticklabels([r"$O_{\mathrm{Effusion}}$"])
    bx.set_ylim(-1, 1)
    bx.set_xlim(-0.095, 0.015)
    bx.set_xticks([-0.08, -0.04, 0])
    bx.set_xlabel("Effusion ownership contrast")
    panel_title(bx, "b", "95% patient bootstrap")
    bx.annotate(f"{p['margin']:.4f}  [{lo:.4f}, {hi:.4f}]", (p["margin"], 0), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=7, color=TERRACOTTA, fontweight="bold")
    save_figure(fig, "fig3_direction_specificity")


# ------------------------------------------------------------------------------------------- fig1
def box(ax, x, y, w, h, title, body, color, title_size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="white", edgecolor=color, linewidth=1.1, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.30), w, 0.30, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=color, edgecolor=color, linewidth=1.1, zorder=3))
    ax.text(x + w / 2, y + h - 0.15, title, ha="center", va="center", fontsize=title_size, color="white",
            fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + (h - 0.30) / 2, body, ha="center", va="center", fontsize=7.4, color=INK,
            linespacing=1.35, zorder=4)


def arrow(ax, p, q, color=MUTED, style="-|>", lw=1.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, linewidth=lw, color=color,
                                 shrinkA=2, shrinkB=2, zorder=5))


def framework():
    fig = plt.figure(figsize=(7.2, 3.15))
    ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, 7.2); ax.set_ylim(0, 3.15); ax.set_axis_off(); ax.grid(False)

    # top row: the pipeline at one consumed block
    top_y, h = 1.85, 1.15
    box(ax, 0.12, top_y, 1.45, h, "consumed visual block",
        "tokens the language\nmodel reads, $h_{it}$\n(final block, pooled $h_i$)", SLATE)
    box(ax, 1.85, top_y, 1.45, h, "read",
        "probe on pooled tokens\n$\\rightarrow$ direction $\\hat{w}_c$\n$+$ 20 random-label controls", HAZE)
    box(ax, 3.58, top_y, 1.55, h, "write",
        "$h' = h + \\alpha\\,\\|h_{t}\\|\\,\\hat{w}_c$\nsame tokens, fixed dose\n$\\alpha = +0.25$", TERRACOTTA)
    box(ax, 5.41, top_y, 1.67, h, "answer",
        "$P(\\mathrm{yes}\\mid q)$ at the first\nanswer position (fp32 logits)\n$W_{q,d}$: paired change", SAGE)
    for x0, x1 in ((1.57, 1.85), (3.30, 3.58), (5.13, 5.41)):
        arrow(ax, (x0, top_y + h / 2), (x1, top_y + h / 2), color=CHARCOAL, lw=1.2)

    # bottom row: comparison and the seed numbers
    bot_y, hb = 0.12, 1.48
    ax.add_patch(FancyBboxPatch((0.12, bot_y), 4.05, hb, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#FBFAF8", edgecolor="#8a8a8a", linewidth=0.8, zorder=1))
    ax.text(0.24, bot_y + hb - 0.14, "compare on the same rows, dose and endpoint", fontsize=8.5,
            fontweight="bold", color=INK, va="top")
    rows = [("concept write", "$W_{q,q}$", HAZE), ("five competing clinical directions", "$\\max_{d\\neq q} W_{q,d}$", TERRACOTTA),
            ("119 random directions", "95th percentile", GREY), ("coordinate-permutation sham", "$|\\mathrm{sham}|$", LILAC)]
    for i, (name, stat, col) in enumerate(rows):
        yy = bot_y + hb - 0.40 - i * 0.21
        ax.plot(0.32, yy, marker="s", ms=5, color=col, mec="white", ls="none", zorder=3)
        ax.text(0.45, yy, name, fontsize=7.4, color=INK, va="center")
        ax.text(2.55, yy, stat, fontsize=7.4, color=MUTED, va="center")
    ax.text(0.24, bot_y + 0.09, "owned: $O_q = W_{q,q}-\\max_{d\\neq q}W_{q,d} > 0$ (simultaneous max-$T$ bounds), "
            "$W_{q,q} >$ random p95 and $|\\mathrm{sham}|$", fontsize=7.0, color=INK, va="bottom")
    arrow(ax, (6.25, top_y), (6.25, bot_y + hb + 0.02), color=CHARCOAL, lw=1.2)
    arrow(ax, (2.1, top_y), (2.1, bot_y + hb + 0.02), color=CHARCOAL, lw=1.2)

    # seed inset: Effusion question in Qwen2.5-VL-7B on NIH
    ins = fig.add_axes((0.632, 0.10, 0.335, 0.33))
    ins.grid(True, axis="y")
    labels = ["Effusion\nwrite", "Nodule\nwrite", "sham"]
    vals = [0.190, 0.252, 0.017]
    cols = [HAZE, TERRACOTTA, LILAC]
    bars = ins.bar(range(3), vals, color=cols, width=0.62, edgecolor="none", zorder=3)
    bars[1].set_edgecolor(CHARCOAL); bars[1].set_linewidth(1.2)
    ref_line(ins, y=0.137, label="random p95", where="right")
    for b, v in zip(bars, vals):
        ins.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                     ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=INK)
    ins.set_xticks(range(3)); ins.set_xticklabels(labels, fontsize=6.5)
    ins.set_ylim(0, 0.31); ins.set_yticks([0, 0.1, 0.2, 0.3]); ins.tick_params(axis="y", labelsize=6.5)
    ins.set_title("seed: Effusion question, Qwen2.5-VL-7B, NIH", fontsize=7.2, pad=3)
    ins.set_ylabel(r"change in $P(\mathrm{yes})$", fontsize=6.8)
    save_figure(fig, "fig1_framework")


def main():
    apply_style()
    results = read()
    dose_responses(results)
    direction_specificity(results)
    framework()


if __name__ == "__main__":
    main()
