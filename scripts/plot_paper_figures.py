"""Every figure of the paper, generated from the packaged campaign statistics and the seed-study data.

Size system: each figure is authored at the ICLR text width (5.5 in) with a height chosen so that no panel is
squashed (1x3 rows 2.6 in, 2x2 grids 4.8 in, 2x3 grids 4.4 in) and is included with width=\textwidth, so nothing is
rescaled. Style: figstyle (house typography, Morandi palette for categorical series); heat maps use a perceptual
diverging map (RdBu_r, white at zero) with every value printed. Numbers are copied from the data files, never
recomputed beyond means, medians, and quantiles stated in the captions.

Main text
  fig1_framework        read / write / answer schematic with the seed inset
  fig2_overview         (a) graded fractions per dataset, (b) model landscape, (c) dose response, (d) rank ECDF
  fig3_write_matrices   6x6 write matrices W_{q,d} for three checkpoints on NIH (top) and COCO (bottom)
  fig4_same_write       same write, different reader: Gemma 3 4B/12B/27B and MedGemma 4B/27B per concept
  fig5_ladders          size ladders: Qwen2.5-VL, Qwen3-VL (NIH), Lingshu (CheXpert)
Appendix
  figA1/A2/A3_write_matrices_{nih,chexpert,coco}   the write matrix of every checkpoint
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fs                                   # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.colors import TwoSlopeNorm              # noqa: E402
from matplotlib.lines import Line2D                     # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle   # noqa: E402
import numpy as np                                      # noqa: E402

ROOT = HERE.parent
FIG = ROOT / "figures"
RUNS = Path("/rodata/azradonc_dev/m253405/cf-transfer/runs")
DATASETS = ("nih", "chexpert", "coco")
DS_SHORT = {"nih": "NIH", "chexpert": "CheXpert", "coco": "COCO"}
NAMES = {"q25-3": "Qwen2.5-VL-3B", "q25-7": "Qwen2.5-VL-7B", "q25-32": "Qwen2.5-VL-32B", "q25-72": "Qwen2.5-VL-72B",
         "q3-4": "Qwen3-VL-4B", "q3-8": "Qwen3-VL-8B", "q3-32": "Qwen3-VL-32B",
         "iv35-8": "InternVL3.5-8B", "iv35-14": "InternVL3.5-14B", "iv35-38": "InternVL3.5-38B",
         "gemma3-4": "Gemma 3 4B", "gemma3-12": "Gemma 3 12B", "gemma3-27": "Gemma 3 27B",
         "medgemma-4": "MedGemma 4B", "medgemma-27": "MedGemma 27B",
         "lingshu-7": "Lingshu 7B", "lingshu-32": "Lingshu 32B",
         "llava15-7": "LLaVA-1.5-7B", "llava15-13": "LLaVA-1.5-13B", "llavamed-7": "LLaVA-Med 7B",
         "llama32-11": "Llama 3.2 Vision 11B", "llama32-90": "Llama 3.2 Vision 90B"}
SHORT = {"q25-3": "Q2.5-3B", "q25-7": "Q2.5-7B", "q25-32": "Q2.5-32B", "q25-72": "Q2.5-72B", "q3-4": "Q3-4B", "q3-8": "Q3-8B",
         "q3-32": "Q3-32B", "iv35-8": "IVL-8B", "iv35-14": "IVL-14B", "iv35-38": "IVL-38B", "gemma3-4": "G3-4B",
         "gemma3-12": "G3-12B", "gemma3-27": "G3-27B", "medgemma-4": "MedG-4B", "medgemma-27": "MedG-27B",
         "lingshu-7": "Ling-7B", "lingshu-32": "Ling-32B", "llava15-7": "LLaVA-7B", "llava15-13": "LLaVA-13B",
         "llavamed-7": "LLaVA-Med", "llama32-11": "Llama-11B", "llama32-90": "Llama-90B"}
ORDER = list(NAMES)
CONCEPT_COLOR = {"Effusion": fs.ROSE, "Atelectasis": fs.SAGE, "Pneumothorax": fs.HAZE, "Cardiomegaly": fs.OAT, "Mass": fs.LILAC,
                 "Nodule": fs.SLATE, "Consolidation": fs.MUSTARD, "Edema": fs.TERRACOTTA}
DS_COL, DS_MK, DS_LAB = fs.DATASET_COLOR, fs.DATASET_MARKER, fs.DATASET_LABEL
HEATMAP = "RdBu_r"
SHORT_CONCEPT = {"Effusion": "Effus.", "Atelectasis": "Atel.", "Pneumothorax": "Pneu.", "Cardiomegaly": "Card.",
                 "Mass": "Mass", "Nodule": "Nodule", "Consolidation": "Cons.", "Edema": "Edema",
                 "person": "person", "dog": "dog", "car": "car", "chair": "chair", "bottle": "bottle", "bicycle": "bicycle"}


# ============================================================================================== data
def load_blocks():
    out = {}
    for mk in ORDER:
        for ds in DATASETS:
            s = RUNS / mk / ds / "summary.json"
            if not s.exists():
                continue
            j = json.loads(s.read_text())
            if not (j.get("core") or {}).get("per_question"):
                continue
            r = RUNS / mk / ds / "run.json"
            out[(mk, ds)] = {"s": j, "run": json.loads(r.read_text()) if r.exists() else {}}
    return out


def owned(v):
    return bool(v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage")


def owned_set(s):
    return {c for c, v in s["core"]["per_question"].items() if owned(v)}


def wmatrix(s):
    """(concepts, W) with W[q, d] = W_{q,d} in the question order of the summary."""
    concepts = list(s["core"]["per_question"])
    W = np.array([[s["core"]["W"][q].get(f"concept:{d}", np.nan) for d in concepts] for q in concepts])
    return concepts, W


def dose_cache():
    p = RUNS / "figures" / "dose_curves.json"
    return json.loads(p.read_text()) if p.exists() else {}


def panel(ax, letter, text):
    fs.panel_title(ax, letter, text)


# ============================================================================================== fig1
def fig1_framework(blocks):
    f = plt.figure(figsize=(fs.WIDTH, fs.H1))
    ax = f.add_axes((0, 0, 1, 1)); ax.set_xlim(0, 5.5); ax.set_ylim(0, 2.4); ax.set_axis_off(); ax.grid(False)

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.05", facecolor="white",
                                    edgecolor=color, linewidth=1.0, zorder=2))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.24), w, 0.24, boxstyle="round,pad=0.015,rounding_size=0.05",
                                    facecolor=color, edgecolor=color, linewidth=1.0, zorder=3))
        ax.text(x + w / 2, y + h - 0.12, title, ha="center", va="center", fontsize=8.5, color="white", fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + (h - 0.24) / 2, body, ha="center", va="center", fontsize=8, color=fs.INK, linespacing=1.3, zorder=4)

    def arrow(p, q, lw=1.1, color=fs.CHARCOAL):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, linewidth=lw, color=color, shrinkA=1.5, shrinkB=1.5, zorder=5))

    y0, h = 1.42, 0.86
    xs, w, gap = [0.08, 1.44, 2.80, 4.16], 1.26, 0.10
    box(xs[0], y0, w, h, "consumed visual block", "tokens the LM reads, $h_{it}$\n(final block; pooled $h_i$)", fs.SLATE)
    box(xs[1], y0, w, h, "read", "logistic probe on $h_i$\n$\\rightarrow$ direction $\\hat{w}_c$\n$+$20 random-label controls", fs.HAZE)
    box(xs[2], y0, w, h, "write", "$h' = h + \\alpha\\,\\|h_t\\|\\,\\hat{w}_c$\nsame tokens, $\\alpha=+0.25$", fs.TERRACOTTA)
    box(xs[3], y0, w, h, "answer", "$P(\\mathrm{yes}\\mid q)$, fp32 logits\n$W_{q,d}$: paired change", fs.SAGE)
    for x0 in xs[:-1]:
        arrow((x0 + w + 0.01, y0 + h / 2), (x0 + w + gap - 0.01, y0 + h / 2))

    bx, by, bw, bh = 0.08, 0.10, 3.00, 1.16
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.015,rounding_size=0.05", facecolor="#FBFAF8",
                                edgecolor="#8a8a8a", linewidth=0.8, zorder=1))
    ax.text(bx + 0.10, by + bh - 0.10, "compare on the same rows, dose, and endpoint", fontsize=8.5, fontweight="bold", color=fs.INK, va="top")
    rows = [("concept write", "$W_{q,q}$", fs.HAZE), ("five competing clinical writes", "$\\max_{d\\neq q}W_{q,d}$", fs.TERRACOTTA),
            ("119 random directions", "95th percentile", fs.GREY), ("coordinate-permutation sham", "$|\\mathrm{sham}|$", fs.LILAC)]
    for i, (name, stat, col) in enumerate(rows):
        yy = by + bh - 0.30 - i * 0.165
        ax.plot(bx + 0.17, yy, marker="s", ms=4.6, color=col, mec="white", ls="none", zorder=3)
        ax.text(bx + 0.28, yy, name, fontsize=8, color=fs.INK, va="center")
        ax.text(bx + 1.95, yy, stat, fontsize=8, color=fs.MUTED, va="center")
    ax.text(bx + 0.10, by + 0.06, "owned: $O_q=W_{q,q}-\\max_{d\\neq q}W_{q,d}>0$ (max-$T$ bounds), $W_{q,q}>$ p95, $|$sham$|$",
            fontsize=7.6, color=fs.INK, va="bottom")
    arrow((xs[1] + 0.35, y0), (xs[1] + 0.35, by + bh + 0.02))
    arrow((xs[3] + 0.63, y0), (xs[3] + 0.63, by + bh + 0.02))

    ins = f.add_axes((0.655, 0.175, 0.32, 0.31)); ins.grid(True, axis="y")
    vals, labels, cols = [0.190, 0.252, 0.017], ["Effusion\nwrite", "Nodule\nwrite", "sham"], [fs.HAZE, fs.TERRACOTTA, fs.LILAC]
    bars = ins.bar(range(3), vals, color=cols, width=0.62, edgecolor="none", zorder=3)
    bars[1].set_edgecolor(fs.CHARCOAL); bars[1].set_linewidth(1.1)
    ins.set_ylim(0, 0.31); ins.set_yticks([0, 0.1, 0.2, 0.3]); ins.tick_params(labelsize=7.5)
    fs.ref_line(ins, y=0.137, label="random p95", where="right")
    for b, v in zip(bars, vals):
        ins.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=fs.INK)
    ins.set_xticks(range(3)); ins.set_xticklabels(labels, fontsize=7.5)
    ins.set_title("seed: Effusion question, Qwen2.5-VL-7B, NIH", fontsize=8, pad=3)
    ins.set_ylabel(r"$\Delta P(\mathrm{yes})$", fontsize=8)
    fs.save(f, FIG / "fig1_framework")


# ============================================================================================== fig2
def grade_cells(blocks, ds):
    """Per concept cell of a dataset: (readable, answerable, owned) flags; readable over every probe-graded cell."""
    read, ans, own = [], [], []
    for (mk, d), b in blocks.items():
        if d != ds:
            continue
        cal = b["s"].get("calibration", {}); core = b["s"]["core"]["per_question"]
        for c, v in core.items():
            cc = cal.get(c, {}) if isinstance(cal.get(c, {}), dict) else {}
            read.append(bool(cc.get("readable"))); ans.append(bool(cc.get("answer_capable"))); own.append(owned(v))
    # probe-only blocks (write matrix ineligible) still carry readability
    for mk in ORDER:
        s = RUNS / mk / ds / "summary.json"
        if s.exists() and (mk, ds) not in blocks:
            j = json.loads(s.read_text())
            for c, v in (j.get("calibration") or {}).items():
                if isinstance(v, dict):
                    read.append(bool(v.get("readable")))
    return np.array(read), np.array(ans), np.array(own)


def boot_frac(flags, n=2000, seed=0):
    rng = np.random.default_rng(seed); flags = np.asarray(flags, dtype=float)
    draws = rng.choice(flags, size=(n, len(flags)), replace=True).mean(axis=1)
    return flags.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def fig2_overview(blocks):
    f, axes = plt.subplots(2, 2, figsize=(fs.WIDTH, 4.8), constrained_layout=True)
    # (a) graded fractions
    ax = axes[0, 0]; grades = ["readable", "answerable", "owned"]; w = 0.26
    for k, ds in enumerate(DATASETS):
        read, ans, own = grade_cells(blocks, ds)
        for gi, flags in enumerate((read, ans, own)):
            m, lo, hi = boot_frac(flags)
            x = gi + (k - 1) * w
            ax.bar(x, m, w - 0.03, color=DS_COL[ds], edgecolor="none", zorder=3)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#555555", elinewidth=0.7, capsize=1.5, zorder=4)
            ax.annotate(f"{int(flags.sum())}/{len(flags)}", (x, hi), xytext=(0, 2.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.2, color=fs.INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(grades); ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("fraction of concept cells"); ax.grid(False, axis="x")
    fs.boxed_legend(ax, [Line2D([], [], marker=DS_MK[d], ms=6, color=DS_COL[d], ls="none", mec="white", label=DS_SHORT[d] if d != "coco" else "COCO (control)") for d in DATASETS],
                    loc="upper center", ncol=3, fontsize=6.8, columnspacing=0.9, handletextpad=0.3)
    panel(ax, "a", "graded outcomes per concept cell")

    # (b) model landscape
    ax = axes[0, 1]
    pts = []
    for (mk, ds), b in blocks.items():
        cal = b["s"].get("calibration", {}); core = b["s"]["core"]["per_question"]
        S = np.nanmean([cal[c]["selectivity"] for c in core if isinstance(cal.get(c), dict) and "selectivity" in cal[c]])
        O = np.nanmean([v["O_q"] for v in core.values() if v.get("O_q") is not None])
        pts.append((mk, ds, S, O))
        ax.plot(S, O, marker=DS_MK[ds], color=DS_COL[ds], ls="none", ms=5.5, mec="white", mew=0.7, zorder=3)
    # label only the checkpoints discussed in the text, with fixed offsets chosen to avoid collisions
    want = {("q25-7", "nih"): (4, -8, "left"), ("q25-72", "nih"): (4, 6, "left"), ("lingshu-32", "chexpert"): (4, -8, "left"),
            ("lingshu-32", "nih"): (-4, 3, "right"), ("gemma3-12", "nih"): (-4, -8, "right"), ("gemma3-12", "chexpert"): (4, -8, "left"),
            ("lingshu-7", "coco"): (4, 2, "left"), ("q25-7", "coco"): (4, -8, "left"), ("llavamed-7", "coco"): (4, -8, "left"), ("iv35-8", "coco"): (4, 3, "left")}
    for mk, ds, S, O in pts:
        if (mk, ds) in want:
            dx, dy, ha = want[(mk, ds)]
            ax.annotate(SHORT[mk] + ("" if ds == "coco" else f" ({DS_SHORT[ds][0]})"), (S, O), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6, color=fs.INK, zorder=5, ha=ha)
    ax.axhline(0, color=fs.CHANCE, ls="--", lw=0.8)
    ax.set_xlabel("mean readability $S$ over concepts"); ax.set_ylabel("mean ownership $O$ over concepts")
    ax.set_ylim(-0.5, 0.8); ax.set_xlim(-0.02, 0.34)
    ax.annotate("natural objects: written", (0.13, 0.71), fontsize=7.2, color=DS_COL["coco"])
    ax.annotate("chest findings: read, not written", (0.165, 0.13), fontsize=7.2, color=DS_COL["nih"])
    panel(ax, "b", "model landscape (N: NIH, C: CheXpert)")

    # (c) dose response
    ax = axes[1, 0]; curves = dose_cache()
    for ds in ("nih", "coco"):
        rows = [v for k, v in curves.items() if k.endswith("/" + ds) and v]
        alphas = sorted({float(a) for r in rows for a in r})
        M = np.array([[r.get(str(a), np.nan) for a in alphas] for r in rows])
        med, lo, hi = np.nanmedian(M, axis=0), np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0)
        ax.fill_between(alphas, lo, hi, color=DS_COL[ds], alpha=0.18, lw=0)
        ax.plot(alphas, med, marker=DS_MK[ds], color=DS_COL[ds], ls="-", lw=1.5, ms=5, mec="white", label=f"{DS_LAB[ds]} ({len(rows)} blocks)")
    ax.axhline(0, color=fs.CHANCE, ls="--", lw=0.8); ax.axvline(0, color=fs.CHANCE, lw=0.6)
    ax.set_xlabel(r"relative dose $\alpha$"); ax.set_ylabel("median ownership $O$ over questions")
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    fs.boxed_legend(ax, ax.get_legend_handles_labels()[0], loc="upper left", fontsize=7)
    panel(ax, "c", "dose response (median and IQR across blocks)")

    # (d) rank ECDF
    ax = axes[1, 1]
    for ds in DATASETS:
        ranks = np.array([v["rank_in_random_family"] for (mk, d), b in blocks.items() if d == ds
                          for v in b["s"]["core"]["per_question"].values() if v.get("rank_in_random_family")])
        xs = np.arange(1, 121); ecdf = [(ranks <= x).mean() for x in xs]
        ax.step(xs, ecdf, where="post", color=DS_COL[ds], lw=1.5, label=f"{DS_LAB[ds]} ({len(ranks)} cells)")
        r1 = (ranks == 1).mean()
        ax.annotate(f"rank 1: {100 * r1:.0f}%", (1.0, r1), xytext=(6, 0), textcoords="offset points", fontsize=6.8, color=DS_COL[ds], va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=6)
    ax.set_xscale("log"); ax.set_xlim(0.9, 125); ax.set_ylim(0, 1.02)
    ax.set_xlabel("rank of the concept write among 120 effects"); ax.set_ylabel("fraction of concept cells")
    fs.boxed_legend(ax, ax.get_legend_handles_labels()[0], loc="lower right", fontsize=7)
    panel(ax, "d", "is the concept write the strongest effect?")
    fs.save(f, FIG / "fig2_overview")


# ============================================================================================== fig3 / appendix matrices
def draw_matrix(ax, concepts, W, own, vlim, value_size=6.5, label_size=7, title=None, values=True):
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    ax.imshow(W, cmap=HEATMAP, norm=norm, aspect="equal", interpolation="nearest")
    n = len(concepts)
    for i in range(n):
        for j in range(n):
            v = W[i, j]
            if values and np.isfinite(v):
                col = "white" if abs(v) > 0.55 * vlim else fs.INK
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=value_size, color=col)
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black" if concepts[i] in own else "#777777",
                               linewidth=1.3 if concepts[i] in own else 0.6, zorder=4))
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", lw=0.8, zorder=3); ax.axvline(k - 0.5, color="white", lw=0.8, zorder=3)
    lab = [SHORT_CONCEPT.get(c, c) for c in concepts]
    ax.set_xticks(range(n)); ax.set_xticklabels(lab, rotation=45, ha="right", rotation_mode="anchor", fontsize=label_size)
    ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=label_size)
    ax.tick_params(length=0); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8.5, pad=4)


def fig3_write_matrices(blocks):
    picks = [("q25-7", "nih"), ("q25-72", "nih"), ("lingshu-32", "nih"), ("q25-7", "coco"), ("q3-32", "coco"), ("lingshu-32", "coco")]
    mats = [(mk, ds, *wmatrix(blocks[(mk, ds)]["s"]), owned_set(blocks[(mk, ds)]["s"])) for mk, ds in picks]
    vlim = float(np.nanmax([np.nanmax(np.abs(W)) for _, _, _, W, _ in mats]))
    f = plt.figure(figsize=(fs.WIDTH, 4.4))
    gs = f.add_gridspec(3, 3, height_ratios=[1, 1, 0.07], hspace=0.62, wspace=0.28, left=0.09, right=0.99, top=0.94, bottom=0.10)
    letters = "abcdef"
    for k, (mk, ds, concepts, W, own) in enumerate(mats):
        ax = f.add_subplot(gs[k // 3, k % 3])
        draw_matrix(ax, concepts, W, own, vlim, title=f"({letters[k]})  {NAMES[mk]}, {DS_SHORT[ds]}")
        if k % 3 == 0:
            ax.set_ylabel("question $q$", fontsize=8)
        if k // 3 == 1:
            ax.set_xlabel("written direction $d$", fontsize=8)
    cax = f.add_subplot(gs[2, :])
    sm = plt.cm.ScalarMappable(cmap=HEATMAP, norm=TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim))
    cb = f.colorbar(sm, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=7)
    cb.set_label(r"$W_{q,d}$: mean change in $P(\mathrm{yes}\mid q)$ under the write of $d$ (black outline: owned diagonal)", fontsize=7.5)
    f.savefig(FIG / "fig3_write_matrices.pdf"); f.savefig(FIG / "fig3_write_matrices.png", dpi=200); plt.close(f)


def matrices_all(blocks, ds, name):
    keys = [(mk, d) for (mk, d) in blocks if d == ds]
    mats = [(mk, *wmatrix(blocks[(mk, ds)]["s"]), owned_set(blocks[(mk, ds)]["s"])) for mk, _ in keys]
    vlim = float(np.nanmax([np.nanmax(np.abs(W)) for _, _, W, _ in mats]))
    ncol = 4; nrow = int(np.ceil(len(mats) / ncol)); height = 1.25 * nrow + 0.75
    f = plt.figure(figsize=(fs.WIDTH, height))
    gs = f.add_gridspec(nrow + 1, ncol, height_ratios=[1] * nrow + [0.05], hspace=0.42, wspace=0.30, left=0.09, right=0.99,
                        top=1 - 0.25 / height, bottom=0.5 / height)
    for k, (mk, concepts, W, own) in enumerate(mats):
        ax = f.add_subplot(gs[k // ncol, k % ncol])
        draw_matrix(ax, concepts, W, own, vlim, label_size=6.2, title=None, values=False)
        ax.set_title(NAMES[mk], fontsize=7.5, pad=3)
        if k % ncol:
            ax.set_yticklabels([])
        if k // ncol < nrow - 1:
            ax.set_xticklabels([])
    cax = f.add_subplot(gs[nrow, :])
    sm = plt.cm.ScalarMappable(cmap=HEATMAP, norm=TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim))
    cb = f.colorbar(sm, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=6.5)
    cb.set_label(r"$W_{q,d}$ (rows: question $q$; columns: written direction $d$; black outline: owned diagonal)", fontsize=7)
    f.savefig(FIG / f"{name}.pdf"); f.savefig(FIG / f"{name}.png", dpi=200); plt.close(f)


# ============================================================================================== fig4
def fig4_same_write(blocks):
    f, axes = plt.subplots(1, 3, figsize=(fs.WIDTH, 3.0), constrained_layout=True, sharey=True)
    fam = [("gemma3-4", fs.ROSE, "o", "Gemma 3 4B"), ("gemma3-12", fs.SAGE, "o", "Gemma 3 12B"), ("gemma3-27", fs.SLATE, "o", "Gemma 3 27B"),
           ("medgemma-4", fs.ROSE, "s", "MedGemma 4B"), ("medgemma-27", fs.SLATE, "s", "MedGemma 27B")]
    for ax, ds in zip(axes, DATASETS):
        concepts = None
        for mk, col, mk_shape, lab in fam:
            b = blocks.get((mk, ds))
            if not b:
                continue
            core = b["s"]["core"]["per_question"]; own = owned_set(b["s"])
            if concepts is None:
                concepts = list(core)
            off = -0.16 if mk_shape == "o" else 0.16
            xs = np.arange(len(concepts)) + off
            ys = [core[c]["O_q"] for c in concepts]
            for x, y, c in zip(xs, ys, concepts):
                ax.plot(x, y, marker=mk_shape, ms=5.2, color=col, mfc=col if c in own else "white", mec=col, mew=1.0, ls="none", zorder=4)
        # connectors per concept for the Gemma 3 triple and the MedGemma pair
        for shape, members, off in (("o", ["gemma3-4", "gemma3-12", "gemma3-27"], -0.16), ("s", ["medgemma-4", "medgemma-27"], 0.16)):
            for i, c in enumerate(concepts):
                ys = [blocks[(m, ds)]["s"]["core"]["per_question"][c]["O_q"] for m in members if (m, ds) in blocks]
                if len(ys) > 1:
                    ax.plot([i + off, i + off], [min(ys), max(ys)], color="#9a9a9a", lw=0.8, zorder=2)
        ax.axhline(0, color=fs.CHANCE, ls="--", lw=0.8)
        ax.set_xticks(range(len(concepts))); ax.set_xticklabels([SHORT_CONCEPT.get(c, c) for c in concepts], rotation=35, ha="right", rotation_mode="anchor", fontsize=7.5)
        ax.grid(False, axis="x"); ax.set_xlim(-0.6, len(concepts) - 0.4)
        panel(ax, "abc"[DATASETS.index(ds)], DS_LAB[ds])
    axes[0].set_ylabel("ownership $O$ (filled: owned)")
    handles = [Line2D([], [], marker=m, ms=5.2, color=c, ls="none", mec=c, mfc=c, label=l) for _, c, m, l in fam]
    f.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=7, frameon=True, edgecolor="#8a8a8a")
    fs.save(f, FIG / "fig4_same_write")


# ============================================================================================== fig5
def fig5_ladders(blocks):
    ladders = [("Qwen2.5-VL on NIH", ["q25-3", "q25-7", "q25-32", "q25-72"], "nih", ["3B", "7B", "32B", "72B"]),
               ("Qwen3-VL on NIH", ["q3-4", "q3-8", "q3-32"], "nih", ["4B", "8B", "32B"]),
               ("Lingshu on CheXpert", ["lingshu-7", "lingshu-32"], "chexpert", ["7B", "32B"])]
    f, axes = plt.subplots(1, 3, figsize=(fs.WIDTH, 2.9), constrained_layout=True)
    handles = {}
    for ax, (title, mks, ds, sizes), letter in zip(axes, ladders, "abc"):
        mks = [m for m in mks if (m, ds) in blocks]; sizes = sizes[:len(mks)]
        concepts = list(blocks[(mks[0], ds)]["s"]["core"]["per_question"])
        for ci, c in enumerate(concepts):
            col = CONCEPT_COLOR[c]
            ys = [blocks[(m, ds)]["s"]["core"]["per_question"][c]["O_q"] for m in mks]
            ow = [c in owned_set(blocks[(m, ds)]["s"]) for m in mks]
            ax.plot(range(len(mks)), ys, color=col, lw=1.3, ls="-", zorder=3)
            for x, y, o in zip(range(len(mks)), ys, ow):
                ax.plot(x, y, marker="o", ms=5, color=col, mfc=col if o else "white", mec=col, mew=1.0, ls="none", zorder=4)
            handles.setdefault(c, Line2D([], [], color=col, marker="o", ms=5, ls="-", lw=1.3, mec=col, label=c))
        ax.axhline(0, color=fs.CHANCE, ls="--", lw=0.8)
        ax.set_xticks(range(len(mks))); ax.set_xticklabels(sizes); ax.set_xlim(-0.3, len(mks) - 0.7)
        ax.grid(False, axis="x"); panel(ax, letter, title)
    axes[0].set_ylabel("ownership $O$ (filled: owned)")
    f.legend(handles=list(handles.values()), loc="outside lower center", ncol=4, fontsize=7, frameon=True, edgecolor="#8a8a8a",
             handlelength=1.6, columnspacing=0.9)
    fs.save(f, FIG / "fig5_ladders")


# ============================================================================================== main
KEEP = {"fig1_framework", "fig2_overview", "fig3_write_matrices", "fig4_same_write", "fig5_ladders",
        "figA1_write_matrices_nih", "figA2_write_matrices_chexpert", "figA3_write_matrices_coco"}


def main(only=None):
    fs.use_house_style()
    blocks = load_blocks()
    jobs = {"fig1_framework": lambda: fig1_framework(blocks), "fig2_overview": lambda: fig2_overview(blocks),
            "fig3_write_matrices": lambda: fig3_write_matrices(blocks), "fig4_same_write": lambda: fig4_same_write(blocks),
            "fig5_ladders": lambda: fig5_ladders(blocks),
            "figA1_write_matrices_nih": lambda: matrices_all(blocks, "nih", "figA1_write_matrices_nih"),
            "figA2_write_matrices_chexpert": lambda: matrices_all(blocks, "chexpert", "figA2_write_matrices_chexpert"),
            "figA3_write_matrices_coco": lambda: matrices_all(blocks, "coco", "figA3_write_matrices_coco")}
    for name, fn in jobs.items():
        if only and name not in only:
            continue
        fn(); print("ok", name)
    for p in FIG.iterdir():
        if p.suffix in (".pdf", ".png", ".svg") and p.stem not in KEEP:
            p.unlink(); print("removed", p.name)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
