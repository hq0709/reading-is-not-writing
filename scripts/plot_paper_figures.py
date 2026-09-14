"""Every figure of the paper, generated from the packaged campaign statistics and the seed-study data.

Size system: each figure is authored at the ICLR text width (5.5 in) and one of three fixed heights (2.4 / 4.4 /
6.2 in) and is included with width=\textwidth, so nothing is rescaled or squashed. Style: figstyle (MedVIGIL house
style, Morandi palette). Numbers are copied from the data files, never recomputed.

Main text
  fig1_framework            read / write / answer schematic with the seed inset
  fig2_landscape            (a-c) graded quantities per dataset, (d) model landscape, (e) owned counts chest vs COCO,
                            (f) Effusion ownership across the 36 chest blocks
  fig3_write_matrices       6x6 write matrices W_{q,d} for four checkpoints on NIH and COCO
  fig4_readers              (a-c) same write, different reader; (d-f) size ladders
  fig5_dose_reference       (a) dose response, (b) reference family, (c) rank of the concept write
  fig6_controls             (a) connector vs primary locus, (b) refit SD, (c) label-shift gap, (d) prompt contrasts
Appendix
  figA1_prompt_score, figA3_ownership_matrix, figA4_shared_alias, figA5_input_closure   seed-study figures
  figA7/A8/A9_write_matrices_{nih,chexpert,coco}   every checkpoint's write matrix
  figA10_controls_all, figA11_prompt_dependence     per-block controls
  figA12_seed_dose, figA13_seed_specificity         seed-study dose curves and locked-dose specificity
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fs                                   # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm   # noqa: E402
from matplotlib.lines import Line2D                     # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle   # noqa: E402
import matplotlib.patheffects as pe                     # noqa: E402
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
CONCEPT_COLORS = fs.SLOTS[:6]
DIVERGE = LinearSegmentedColormap.from_list("morandi_div", [fs.TERRACOTTA, "#E8D3CD", "#FBF9F6", "#D3D9DF", fs.SLATE], N=256)
DS_COL, DS_MK, DS_LAB = fs.DATASET_COLOR, fs.DATASET_MARKER, fs.DATASET_LABEL


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


def t3(b, key):
    v = b["s"].get("t3", {}).get(key)
    if isinstance(v, dict) and v.get("estimate") is not None:
        return v["estimate"], v.get("ci_low"), v.get("ci_high")
    return None


def seed():
    return json.loads((ROOT / "data" / "accepted_results.json").read_text()), json.loads((ROOT / "data" / "ownership_summary.json").read_text())


def fig(height, **kw):
    return plt.figure(figsize=(fs.WIDTH, height), constrained_layout=True, **kw)


def concept_labels(ax, concepts, axis="x", size=7):
    short = [c[:5] + "." if len(c) > 6 else c for c in concepts]
    if axis == "x":
        ax.set_xticks(range(len(concepts))); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=size, rotation_mode="anchor")
    else:
        ax.set_yticks(range(len(concepts))); ax.set_yticklabels(short, fontsize=size)


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
        ax.text(x + w / 2, y + (h - 0.24) / 2, body, ha="center", va="center", fontsize=7.6, color=fs.INK, linespacing=1.3, zorder=4)

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

    # comparison panel
    bx, by, bw, bh = 0.08, 0.10, 3.00, 1.16
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.015,rounding_size=0.05", facecolor="#FBFAF8",
                                edgecolor="#8a8a8a", linewidth=0.8, zorder=1))
    ax.text(bx + 0.10, by + bh - 0.10, "compare on the same rows, dose, and endpoint", fontsize=8.2, fontweight="bold", color=fs.INK, va="top")
    rows = [("concept write", "$W_{q,q}$", fs.HAZE), ("five competing clinical writes", "$\\max_{d\\neq q}W_{q,d}$", fs.TERRACOTTA),
            ("119 random directions", "95th percentile", fs.GREY), ("coordinate-permutation sham", "$|\\mathrm{sham}|$", fs.LILAC)]
    for i, (name, stat, col) in enumerate(rows):
        yy = by + bh - 0.30 - i * 0.165
        ax.plot(bx + 0.17, yy, marker="s", ms=4.6, color=col, mec="white", ls="none", zorder=3)
        ax.text(bx + 0.28, yy, name, fontsize=7.4, color=fs.INK, va="center")
        ax.text(bx + 1.95, yy, stat, fontsize=7.4, color=fs.MUTED, va="center")
    ax.text(bx + 0.10, by + 0.06, "owned: $O_q=W_{q,q}-\\max_{d\\neq q}W_{q,d}>0$ (max-$T$ bounds), $W_{q,q}>$ p95, $|$sham$|$",
            fontsize=7.0, color=fs.INK, va="bottom")
    arrow((xs[1] + 0.35, y0), (xs[1] + 0.35, by + bh + 0.02))
    arrow((xs[3] + 0.63, y0), (xs[3] + 0.63, by + bh + 0.02))

    # seed inset
    ins = f.add_axes((0.655, 0.175, 0.32, 0.31)); ins.grid(True, axis="y")
    vals, labels, cols = [0.190, 0.252, 0.017], ["Effusion\nwrite", "Nodule\nwrite", "sham"], [fs.HAZE, fs.TERRACOTTA, fs.LILAC]
    bars = ins.bar(range(3), vals, color=cols, width=0.62, edgecolor="none", zorder=3)
    bars[1].set_edgecolor(fs.CHARCOAL); bars[1].set_linewidth(1.1)
    ins.set_ylim(0, 0.31); ins.set_yticks([0, 0.1, 0.2, 0.3]); ins.tick_params(labelsize=7)
    fs.ref_line(ins, y=0.137, label="random p95", where="right")
    for b, v in zip(bars, vals):
        ins.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7, fontweight="bold", color=fs.INK)
    ins.set_xticks(range(3)); ins.set_xticklabels(labels, fontsize=7)
    ins.set_title("seed: Effusion question, Qwen2.5-VL-7B, NIH", fontsize=7.4, pad=3)
    ins.set_ylabel(r"$\Delta P(\mathrm{yes})$", fontsize=7.4)
    fs.save(f, FIG / "fig1_framework")


# ============================================================================================== fig2
def cell_table(blocks):
    rows = []
    for (mk, ds), b in blocks.items():
        cal, pq = b["s"].get("calibration", {}), b["s"]["core"]["per_question"]
        for c, v in pq.items():
            cc = cal.get(c, {}) if isinstance(cal.get(c, {}), dict) else {}
            rows.append(dict(mk=mk, ds=ds, c=c, S=cc.get("selectivity"), readable=bool(cc.get("readable")),
                             auroc=cc.get("answer_auroc"), capable=bool(cc.get("answer_capable")), O=v.get("O_q"),
                             own=owned(v), ref=bool(v.get("steering_reference")), rank=v.get("rank_in_random_family"),
                             ci=v.get("O_q_ci95_percentile")))
    return rows


def strip_box(ax, groups, color_of, y_key, threshold, flag_key, ylim, ylabel, title_letter, title):
    """groups: dict dataset -> list of rows; one strip+box per dataset at x = 0,1,2."""
    rng = np.random.default_rng(1)
    for i, ds in enumerate(DATASETS):
        vals = np.array([r[y_key] for r in groups[ds] if r[y_key] is not None and r[y_key] == r[y_key]], dtype=float)
        flags = np.array([r[flag_key] for r in groups[ds] if r[y_key] is not None and r[y_key] == r[y_key]], dtype=bool)
        if len(vals) == 0:
            continue
        x = i + rng.uniform(-0.2, 0.2, len(vals))
        ax.scatter(x[~flags], vals[~flags], s=7, facecolors="none", edgecolors=color_of[ds], linewidths=0.6, zorder=3, alpha=0.9)
        ax.scatter(x[flags], vals[flags], s=8, color=color_of[ds], edgecolors="white", linewidths=0.3, zorder=4)
        ax.boxplot([vals], positions=[i], widths=0.55, showfliers=False, patch_artist=False, zorder=2,
                   medianprops=dict(color=fs.CHARCOAL, lw=1.1), boxprops=dict(color=fs.CHARCOAL, lw=0.7),
                   whiskerprops=dict(color=fs.CHARCOAL, lw=0.7), capprops=dict(color=fs.CHARCOAL, lw=0.7))
        share = flags.mean()
        ax.annotate(f"{100 * share:.0f}%", (i, ylim[1]), xytext=(0, -1), textcoords="offset points", ha="center", va="top",
                    fontsize=7.4, fontweight="bold", color=color_of[ds], bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=6)
    ax.axhline(threshold, color=fs.RED, ls="--", lw=0.9, zorder=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["NIH", "CheX.", "COCO"], fontsize=7.5)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(*ylim); ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5)
    fs.panel_title(ax, title_letter, title)


def fig2_landscape(blocks):
    rows = cell_table(blocks)
    groups = {ds: [r for r in rows if r["ds"] == ds] for ds in DATASETS}
    f = fig(fs.H2)
    gs = f.add_gridspec(2, 3, height_ratios=[1, 1.05])
    a, b, c = (f.add_subplot(gs[0, k]) for k in range(3))
    strip_box(a, groups, DS_COL, "S", 0.0, "readable", (-0.12, 0.38), "probe selectivity $S$", "a", "readable")
    strip_box(b, groups, DS_COL, "auroc", 0.5, "capable", (0.35, 1.03), "clean-answer AUROC", "b", "answerable")
    strip_box(c, groups, DS_COL, "O", 0.0, "own", (-0.9, 0.95), "ownership $O_q$", "c", "owned")

    # (d) model landscape
    d = f.add_subplot(gs[1, 0])
    for ds in DATASETS:
        for mk in ORDER:
            rr = [r for r in groups[ds] if r["mk"] == mk and r["S"] is not None and r["O"] is not None]
            if not rr:
                continue
            xs, ys = np.mean([r["S"] for r in rr]), np.mean([r["O"] for r in rr])
            d.plot(xs, ys, marker=DS_MK[ds], ms=5, color=DS_COL[ds], mec="white", mew=0.5, ls="none", zorder=4)
            if (ds == "coco" and mk in ("lingshu-7", "llavamed-7")) or (ds != "coco" and (ys > 0.06 or ys < -0.2)):
                d.annotate(SHORT[mk], (xs, ys), xytext=(3, 1), textcoords="offset points", fontsize=5.8, color=fs.MUTED)
    d.axhline(0, color=fs.CHANCE, lw=0.8); d.axvline(0, color=fs.CHANCE, lw=0.8)
    d.set_xlabel("mean readability $S$ over concepts", fontsize=8); d.set_ylabel("mean ownership $O$", fontsize=8)
    d.tick_params(labelsize=7.5)
    fs.panel_title(d, "d", "model landscape")
    fs.boxed_legend(d, [Line2D([], [], marker=DS_MK[ds], color=DS_COL[ds], ls="none", ms=5, mec="white", label=DS_LAB[ds]) for ds in DATASETS],
                    loc="upper right", **fs.LEG)

    # (e) owned concepts chest vs COCO
    e = f.add_subplot(gs[1, 1])
    ent = []
    for mk in ORDER:
        chest = [r for r in rows if r["mk"] == mk and r["ds"] in ("nih", "chexpert")]
        coco = [r for r in rows if r["mk"] == mk and r["ds"] == "coco"]
        if not chest:
            continue
        ent.append((mk, np.mean([r["own"] for r in chest]), np.mean([r["own"] for r in coco]) if coco else np.nan))
    ent.sort(key=lambda t: (np.nan_to_num(t[2], nan=-1), t[1]))
    for i, (mk, ch, co) in enumerate(ent):
        if co == co:
            e.plot([ch, co], [i, i], color=fs.STONE, lw=1.4, zorder=2)
            e.plot(co, i, marker=DS_MK["coco"], color=DS_COL["coco"], ms=5, mec="white", ls="none", zorder=4)
        e.plot(ch, i, marker="o", color=fs.ROSE, ms=5, mec="white", ls="none", zorder=4)
    e.set_yticks(range(len(ent))); e.set_yticklabels([SHORT[m] for m, _, _ in ent], fontsize=6.2)
    e.set_xlim(-0.05, 1.05); e.set_xticks([0, 0.5, 1]); e.set_xticklabels(["0", "½", "all"], fontsize=7.5)
    e.set_xlabel("fraction of concepts owned", fontsize=8); e.tick_params(axis="y", length=0)
    fs.panel_title(e, "e", "owned: chest vs COCO")
    fs.boxed_legend(e, [Line2D([], [], marker="o", color=fs.ROSE, ls="none", ms=5, mec="white", label="chest"),
                        Line2D([], [], marker="^", color=fs.SLATE, ls="none", ms=5, mec="white", label="COCO")], loc="lower right", **fs.LEG)

    # (f) Effusion across chest blocks
    g = f.add_subplot(gs[1, 2])
    eff = sorted([r for r in rows if r["c"] == "Effusion" and r["ds"] != "coco" and r["O"] is not None], key=lambda r: r["O"])
    for i, r in enumerate(eff):
        lo, hi = (r["ci"] or [r["O"], r["O"]])
        g.plot([i, i], [lo, hi], color=DS_COL[r["ds"]], lw=1.0, zorder=3)
        g.plot(i, r["O"], marker=DS_MK[r["ds"]], ms=3.6, color=DS_COL[r["ds"]] if r["own"] else "white",
               mec=DS_COL[r["ds"]], mew=0.8, ls="none", zorder=4)
    g.axhline(0, color=fs.CHANCE, lw=0.8)
    seedix = [i for i, r in enumerate(eff) if r["mk"] == "q25-7" and r["ds"] == "nih"]
    if seedix:
        i = seedix[0]; g.annotate("Qwen2.5-VL-7B\n(seed)", (i, eff[i]["O"]), xytext=(8, -14), textcoords="offset points",
                                  fontsize=6.4, color=fs.INK, arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.6))
    for i, r in enumerate(eff):
        if r["own"] and r["O"] > 0.05:
            g.annotate(SHORT[r["mk"]], (i, r["O"]), xytext=(-2, 4), textcoords="offset points", fontsize=6.0, color=fs.MUTED, ha="right")
    g.set_xlim(-1, len(eff)); g.set_xticks([]); g.set_xlabel(f"{len(eff)} chest blocks, sorted", fontsize=8)
    g.set_ylabel("Effusion ownership $O$", fontsize=8); g.tick_params(axis="y", labelsize=7.5)
    n_own = sum(r["own"] for r in eff)
    g.annotate(f"owned in {n_own}/{len(eff)}", (0.03, 0.95), xycoords="axes fraction", fontsize=7, color=fs.INK, va="top")
    fs.panel_title(g, "f", "Effusion")
    fs.boxed_legend(g, [Line2D([], [], marker="o", color=fs.ROSE, ls="none", ms=4, mec="white", label="NIH"),
                        Line2D([], [], marker="s", color=fs.OAT, ls="none", ms=4, mec="white", label="CheXpert")], loc="lower right", **fs.LEG)
    fs.save(f, FIG / "fig2_landscape")


# ============================================================================================== fig3 / A7-A9
def draw_matrix(ax, concepts, W, own, vlim, labels_x=True, labels_y=True, title=None, mark_max=True, values=False):
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    im = ax.imshow(W, cmap=DIVERGE, norm=norm, aspect="equal")
    ax.set_xticks(range(len(concepts))); ax.set_yticks(range(len(concepts)))
    short = [c[:4] for c in concepts]
    ax.set_xticklabels(short if labels_x else [], rotation=45, ha="right", fontsize=6.4, rotation_mode="anchor")
    ax.set_yticklabels(short if labels_y else [], fontsize=6.4)
    ax.tick_params(length=0, pad=1.5); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    n = len(concepts)
    for i in range(n):
        r = Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=fs.CHARCOAL if concepts[i] in own else "white",
                      linewidth=1.3 if concepts[i] in own else 0.6, zorder=4)
        ax.add_patch(r)
    if mark_max:
        for q in range(n):
            row = W[q].copy(); row[q] = -np.inf
            j = int(np.nanargmax(row))
            off = 0.33 if values else 0.0
            ax.plot(j + off, q - off, marker="o", ms=2.2, color=fs.CHARCOAL, mec="none", ls="none", zorder=5)
    if values:
        for q in range(n):
            for dd in range(n):
                ax.text(dd, q, f"{W[q, dd]:.2f}", ha="center", va="center", fontsize=5.4, color=fs.INK)
    if title:
        ax.set_title(title, fontsize=7.4, pad=3)
    return im


def fig3_write_matrices(blocks):
    picks = [("q25-7", "nih"), ("q25-72", "nih"), ("lingshu-32", "nih"), ("gemma3-12", "nih"),
             ("q25-7", "coco"), ("q25-72", "coco"), ("lingshu-32", "coco"), ("gemma3-12", "coco")]
    f = fig(fs.H2)
    gs = f.add_gridspec(3, 4, height_ratios=[1, 1, 0.07])
    vlim = 0.9
    im = None
    for k, (mk, ds) in enumerate(picks):
        r, c = divmod(k, 4)
        ax = f.add_subplot(gs[r, c])
        note = ""
        if (mk, ds) not in blocks:
            mk, note = "q3-32", "$^\\dagger$"
        b = blocks[(mk, ds)]; concepts, W = wmatrix(b["s"]); own = owned_set(b["s"])
        im = draw_matrix(ax, concepts, W, own, vlim, labels_x=True, labels_y=(c == 0),
                         title=f"({'abcdefgh'[k]})  {NAMES[mk]}{note}")
        ax.text(0.98, 0.98, DS_SHORT[ds], transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color=fs.MUTED)
        if c == 0:
            ax.set_ylabel("question $q$", fontsize=7.5)
        if r == 1:
            ax.set_xlabel("direction $d$", fontsize=7.5)
    cax = f.add_subplot(gs[2, 1:3])
    cb = f.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("$W_{q,d}$: mean change in $P(\\mathrm{yes}\\mid q)$ under the write of $d$", fontsize=7.5)
    cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
    lax = f.add_subplot(gs[2, 0]); lax.set_axis_off()
    lax.legend(handles=[Patch(facecolor="none", edgecolor=fs.CHARCOAL, lw=1.3, label="owned diagonal"),
                        Line2D([], [], marker="o", color=fs.CHARCOAL, ls="none", ms=3, label="strongest competitor")],
               loc="center", frameon=False, **fs.LEG)
    fs.save(f, FIG / "fig3_write_matrices")


def matrices_all(blocks, ds, name):
    keys = [mk for mk in ORDER if (mk, ds) in blocks]
    ncol = 4; nrow = int(np.ceil(len(keys) / ncol))
    f = plt.figure(figsize=(fs.WIDTH, fs.H3), constrained_layout=True)
    gs = f.add_gridspec(nrow, ncol + 1, width_ratios=[1] * ncol + [0.07])
    vlim = 0.9; im = None
    for k, mk in enumerate(keys):
        r, c = divmod(k, ncol); ax = f.add_subplot(gs[r, c])
        b = blocks[(mk, ds)]; concepts, W = wmatrix(b["s"]); own = owned_set(b["s"])
        im = draw_matrix(ax, concepts, W, own, vlim, labels_x=(r == nrow - 1 or k + ncol >= len(keys)), labels_y=(c == 0),
                         title=f"{NAMES[mk]} ({len(own)} owned)")
    cax = f.add_subplot(gs[:, ncol]); cb = f.colorbar(im, cax=cax); cb.set_label("$W_{q,d}$", fontsize=8); cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    f.suptitle(f"{DS_LAB[ds]}: write matrices $W_{{q,d}}$ (rows $q$, columns $d$); outlined diagonal = owned, dot = strongest competitor", fontsize=8)
    fs.save(f, FIG / name)


# ============================================================================================== fig4
def bars_by_size(ax, blocks, mks, ds, letter, title, show_legend=True):
    concepts = None
    n = len(mks); w = 0.8 / n
    for k, mk in enumerate(mks):
        b = blocks.get((mk, ds))
        if not b:
            continue
        pq = b["s"]["core"]["per_question"]; concepts = list(pq)
        own = owned_set(b["s"])
        vals = [pq[c]["O_q"] for c in concepts]
        x = np.arange(len(concepts)) + (k - (n - 1) / 2) * w
        bars = ax.bar(x, vals, w, color=fs.SLOTS[k], edgecolor="none", zorder=3, label=NAMES[mk].replace("Gemma 3 ", "G3 ").replace("MedGemma ", "MedG "))
        for bb, c in zip(bars, concepts):
            if c in own:
                bb.set_edgecolor(fs.CHARCOAL); bb.set_linewidth(1.1)
    ax.axhline(0, color=fs.CHANCE, lw=0.8)
    if concepts:
        concept_labels(ax, concepts, "x", size=6.8)
    ax.tick_params(axis="y", labelsize=7.5); ax.grid(False, axis="x")
    fs.panel_title(ax, letter, title)
    if show_legend:
        fs.boxed_legend(ax, ax.get_legend_handles_labels()[0], loc="best", **fs.LEG)


def ladder(ax, blocks, series, letter, title, ylab=True, legend=True):
    """series: list of (label, [(mk, ds), ...]) drawn as x-groups; lines per concept."""
    xpos, xlabels, gi = [], [], 0
    handles = {}
    for label, mks in series:
        pts = [(mk, ds) for mk, ds in mks if (mk, ds) in blocks]
        xs = list(range(gi, gi + len(pts)))
        concepts = list(blocks[pts[0]]["s"]["core"]["per_question"])
        for ci, c in enumerate(concepts):
            ys = [blocks[p]["s"]["core"]["per_question"][c]["O_q"] for p in pts]
            ow = [c in owned_set(blocks[p]["s"]) for p in pts]
            ax.plot(xs, ys, ls="--", lw=1.1, color=CONCEPT_COLORS[ci], zorder=3)
            for xx, yy, o in zip(xs, ys, ow):
                ax.plot(xx, yy, marker="o", ms=4.2, color=CONCEPT_COLORS[ci] if o else "white", mec=CONCEPT_COLORS[ci], mew=0.9, ls="none", zorder=4)
            handles[c] = Line2D([], [], color=CONCEPT_COLORS[ci], marker="o", ls="--", lw=1.1, ms=4, mec="white", label=c)
        xpos += xs; xlabels += [SHORT[mk].split("-")[-1] for mk, _ in pts]
        ax.text(np.mean(xs), 0.02, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.6, color=fs.MUTED)
        gi += len(pts) + 1
    ax.axhline(0, color=fs.CHANCE, lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels(xlabels, fontsize=7); ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlim(-0.7, gi - 1.3)
    if ylab:
        ax.set_ylabel("ownership $O$ (filled = owned)", fontsize=8)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.55 * (hi - lo))          # headroom for the group labels and legend
    fs.panel_title(ax, letter, title)
    if legend:
        fs.boxed_legend(ax, list(handles.values()), loc="upper left", ncol=2, **fs.LEG)


def fig4_readers(blocks):
    f = fig(fs.H2); gs = f.add_gridspec(2, 3)
    top = [f.add_subplot(gs[0, k]) for k in range(3)]
    for k, ds in enumerate(DATASETS):
        bars_by_size(top[k], blocks, ["gemma3-4", "gemma3-12", "gemma3-27", "medgemma-4", "medgemma-27"], ds, "abc"[k],
                     f"same write, different reader: {DS_SHORT[ds]}" if k == 0 else DS_SHORT[ds], show_legend=(k == 2))
    top[0].set_ylabel("ownership $O$ (outlined = owned)", fontsize=8)
    ymin = min(a.get_ylim()[0] for a in top); ymax = max(a.get_ylim()[1] for a in top)
    for a in top:
        a.set_ylim(ymin, ymax)
    bot = [f.add_subplot(gs[1, k]) for k in range(3)]
    ladder(bot[0], blocks, [("Qwen2.5-VL", [("q25-3", "nih"), ("q25-7", "nih"), ("q25-32", "nih"), ("q25-72", "nih")])], "d", "size ladders on NIH", legend=True)
    ladder(bot[1], blocks, [("Qwen3-VL", [("q3-4", "nih"), ("q3-8", "nih"), ("q3-32", "nih")]),
                            ("InternVL3.5", [("iv35-8", "nih"), ("iv35-14", "nih")])], "e", "NIH", ylab=False, legend=False)
    ladder(bot[2], blocks, [("Lingshu", [("lingshu-7", "chexpert"), ("lingshu-32", "chexpert")])], "f", "CheXpert", ylab=False, legend=True)
    fs.save(f, FIG / "fig4_readers")


# ============================================================================================== fig5
def fig5_dose_reference(blocks):
    f = fig(fs.H1); gs = f.add_gridspec(1, 3)
    # (a) dose response
    a = f.add_subplot(gs[0])
    curves = json.loads((RUNS / "figures" / "dose_curves.json").read_text())
    alphas = sorted({float(x) for cv in curves.values() for x in cv})
    for ds in ("nih", "coco"):
        mat = []
        for key, cv in curves.items():
            if key.split("/")[1] != ds:
                continue
            ys = [cv.get(str(x), np.nan) for x in alphas]; mat.append(ys)
            a.plot(alphas, ys, color=DS_COL[ds], lw=0.6, alpha=0.35, zorder=2)
        mat = np.array(mat)
        med = np.nanmedian(mat, 0); q1, q3 = np.nanpercentile(mat, 25, 0), np.nanpercentile(mat, 75, 0)
        a.fill_between(alphas, q1, q3, color=DS_COL[ds], alpha=0.18, lw=0, zorder=3)
        a.plot(alphas, med, color=DS_COL[ds], lw=1.8, marker=DS_MK[ds], ms=4, mec="white", zorder=5, label=f"{DS_LAB[ds]} (n={len(mat)})")
    a.axhline(0, color=fs.CHANCE, lw=0.8); a.axvline(0, color=fs.CHANCE, lw=0.8)
    a.set_xlabel(r"relative dose $\alpha$", fontsize=8); a.set_ylabel("median ownership $O$ over questions", fontsize=8)
    a.tick_params(labelsize=7.5)
    fs.panel_title(a, "a", "dose response")
    fs.boxed_legend(a, a.get_legend_handles_labels()[0], loc="upper left", **fs.LEG)

    # (b) reference family
    b = f.add_subplot(gs[1])
    for (mk, ds, c), col, lab in ((("q25-7", "nih", "Effusion"), fs.ROSE, "NIH: Effusion"), (("q25-7", "coco", "person"), fs.SLATE, "COCO: person")):
        W = blocks[(mk, ds)]["s"]["core"]["W"][c]
        rand = np.array([v for k, v in W.items() if k.startswith("random:")])
        b.hist(rand, bins=24, color=col, alpha=0.55, zorder=3)
        b.axvline(W[f"concept:{c}"], color=col, lw=1.8, zorder=5)
        b.axvline(W[f"sham:{c}"], color=col, lw=1.2, ls=":", zorder=5)
        b.axvline(np.percentile(rand, 95), color=col, lw=0.9, ls="--", zorder=4)
    b.set_xlabel("$W$: mean change in $P(\\mathrm{yes})$", fontsize=8); b.set_ylabel("random directions (count)", fontsize=8)
    b.tick_params(labelsize=7.5)
    fs.panel_title(b, "b", "reference family")
    fs.boxed_legend(b, [Patch(facecolor=fs.ROSE, alpha=0.55, label="NIH Effusion (119 random)"), Patch(facecolor=fs.SLATE, alpha=0.55, label="COCO person (119 random)"),
                        Line2D([], [], color=fs.CHARCOAL, lw=1.8, label="concept write"), Line2D([], [], color=fs.CHARCOAL, lw=0.9, ls="--", label="random p95"),
                        Line2D([], [], color=fs.CHARCOAL, lw=1.2, ls=":", label="sham")], loc="upper right", **fs.LEG)

    # (c) rank of the concept write
    c = f.add_subplot(gs[2])
    rows = cell_table(blocks)
    for ds in DATASETS:
        rk = np.array(sorted(r["rank"] for r in rows if r["ds"] == ds and r["rank"] is not None))
        if len(rk) == 0:
            continue
        y = np.arange(1, len(rk) + 1) / len(rk)
        c.step(rk, y, where="post", color=DS_COL[ds], lw=1.5, label=f"{DS_LAB[ds]} (n={len(rk)})", zorder=3)
        c.text(0.04, 0.66 - 0.08 * DATASETS.index(ds), f"rank 1: {100 * np.mean(rk == 1):.0f}% of cells", transform=c.transAxes, fontsize=6.4, color=DS_COL[ds], va="top")
    c.set_xscale("log"); c.set_xlim(0.9, 130); c.set_ylim(0, 1.02)
    c.set_xlabel("rank of $W_{q,q}$ among 120 effects", fontsize=8); c.set_ylabel("fraction of cells", fontsize=8)
    c.tick_params(labelsize=7.5)
    fs.panel_title(c, "c", "rank in the family")
    fs.boxed_legend(c, c.get_legend_handles_labels()[0], loc="lower right", **fs.LEG)
    fs.save(f, FIG / "fig5_dose_reference")


# ============================================================================================== fig6
def fig6_controls(blocks):
    f = fig(fs.H1); gs = f.add_gridspec(1, 4)
    keys = [k for k in blocks if t3(blocks[k], "all|connector_median_O")]
    # (a) connector vs primary
    a = f.add_subplot(gs[0])
    for k in keys:
        b = blocks[k]; prim = np.median([v["O_q"] for v in b["s"]["core"]["per_question"].values()])
        conn = t3(b, "all|connector_median_O")[0]
        a.plot(prim, conn, marker=DS_MK[k[1]], color=DS_COL[k[1]], ms=4.5, mec="white", mew=0.5, ls="none", zorder=4)
    lim = (-0.9, 0.9)
    a.axhline(0, color=fs.CHANCE, lw=0.6); a.axvline(0, color=fs.CHANCE, lw=0.6)
    a.set_xlim(*lim); a.set_ylim(-0.12, 0.12)
    a.set_xlabel("primary locus: median $O$", fontsize=8); a.set_ylabel("connector locus: median $O$", fontsize=8); a.tick_params(labelsize=7.5)
    fs.panel_title(a, "a", "connector locus")

    def strip(ax, key, letter, title, ylab, symlog=False):
        rng = np.random.default_rng(2)
        present = [ds for ds in DATASETS if any(k[1] == ds for k in keys)]
        for i, ds in enumerate(present):
            vals = [t3(blocks[k], key)[0] for k in keys if k[1] == ds and t3(blocks[k], key)]
            if not vals:
                continue
            x = i + rng.uniform(-0.18, 0.18, len(vals))
            ax.plot(x, vals, marker=DS_MK[ds], color=DS_COL[ds], ms=4.2, mec="white", mew=0.5, ls="none", zorder=4)
            ax.hlines(np.median(vals), i - 0.3, i + 0.3, color=fs.CHARCOAL, lw=1.2, zorder=5)
        ax.set_xticks(range(len(present))); ax.set_xticklabels([DS_SHORT[d] for d in present], fontsize=7.5); ax.set_xlim(-0.6, len(present) - 0.4)
        if symlog:
            ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylabel(ylab, fontsize=8); ax.tick_params(axis="y", labelsize=7.5)
        fs.panel_title(ax, letter, title)
    strip(f.add_subplot(gs[1]), "all|median_refit_O_sd", "b", "refit stability", "SD of $O$ over 3 refits")
    cx = f.add_subplot(gs[2]); strip(cx, "all|median_label_gap", "c", "label shift", "label-shift gap (symlog)", symlog=True)
    cx.axhline(0, color=fs.CHANCE, lw=0.6)
    # (d) prompt contrasts
    d = f.add_subplot(gs[3]); rng = np.random.default_rng(3)
    for j, (kind, lab) in enumerate((("wording_IY_minus_WY_O", "wording (IY$-$WY)"), ("mapping_IA_minus_IB_O", "mapping (IA$-$IB)"))):
        for ds in DATASETS:
            vals = []
            for k in keys:
                if k[1] != ds:
                    continue
                for kk, v in blocks[k]["s"].get("t3", {}).items():
                    if kk.endswith(kind) and isinstance(v, dict) and v.get("estimate") is not None:
                        vals.append(v["estimate"])
            if vals:
                x = j + {"nih": -0.14, "chexpert": 0, "coco": 0.14}[ds] + rng.uniform(-0.05, 0.05, len(vals))
                d.plot(x, vals, marker=DS_MK[ds], color=DS_COL[ds], ms=4, mec="white", mew=0.5, ls="none", zorder=4)
    d.axhline(0, color=fs.CHANCE, lw=0.8)
    d.set_xticks([0, 1]); d.set_xticklabels(["wording\nIY$-$WY", "mapping\nIA$-$IB"], fontsize=7.5); d.set_xlim(-0.6, 1.6)
    d.set_ylabel("difference in $O$ between templates", fontsize=8); d.tick_params(axis="y", labelsize=7.5)
    fs.panel_title(d, "d", "prompts")
    fs.boxed_legend(d, [Line2D([], [], marker=DS_MK[ds], color=DS_COL[ds], ls="none", ms=4.2, mec="white", label=DS_SHORT[ds]) for ds in ("nih", "coco")],
                    loc="lower left", **fs.LEG)
    fs.save(f, FIG / "fig6_controls")


# ============================================================================================== appendix per-block
def figA10_controls_all(blocks):
    keys = sorted([k for k in blocks if t3(blocks[k], "all|connector_median_O")], key=lambda k: (k[1], ORDER.index(k[0])))
    f = fig(fs.H3); gs = f.add_gridspec(1, 3)
    axes = [f.add_subplot(gs[i]) for i in range(3)]
    y = np.arange(len(keys))
    for ax, (metric, lab, letter) in zip(axes, [("all|connector_median_O", "connector-locus median $O$", "a"), ("all|median_refit_O_sd", "refit SD of $O$ (3 seeds)", "b"),
                                                ("all|median_label_gap", "label-shift gap", "c")]):
        for i, k in enumerate(keys):
            e, lo, hi = t3(blocks[k], metric); col = DS_COL[k[1]]
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [i, i], color=col, lw=1.0, zorder=3)
            ax.plot(e, i, marker=DS_MK[k[1]], color=col, ms=3.6, mec="white", mew=0.4, ls="none", zorder=4)
        ax.axvline(0, color=fs.CHANCE, lw=0.8); ax.set_ylim(-0.7, len(keys) - 0.3); ax.invert_yaxis()
        ax.set_xlabel(lab, fontsize=8); ax.tick_params(axis="x", labelsize=7.2)
        fs.panel_title(ax, letter, lab.split(" (")[0])
        ax.set_yticks(y); ax.set_yticklabels([f"{SHORT[m]} / {DS_SHORT[d]}" for m, d in keys] if ax is axes[0] else [], fontsize=6.2)
        ax.tick_params(axis="y", length=0)
    fs.save(f, FIG / "figA10_controls_all")


def figA11_prompt_dependence(blocks):
    keys = sorted([k for k in blocks if any(kk.endswith("wording_IY_minus_WY_O") for kk in blocks[k]["s"].get("t3", {}))], key=lambda k: (k[1], ORDER.index(k[0])))
    f = fig(fs.H3); ax = f.add_subplot(111)
    for i, k in enumerate(keys):
        for kk, v in blocks[k]["s"]["t3"].items():
            if not isinstance(v, dict) or v.get("estimate") is None:
                continue
            if "|wording" in kk or "|mapping" in kk:
                mk = "o" if "wording" in kk else "s"; col = fs.SAGE if "wording" in kk else fs.LILAC
                first = kk.split("|")[0] in ("Effusion", "person")
                if v.get("ci_low") is not None:
                    ax.plot([v["ci_low"], v["ci_high"]], [i, i], color=col, lw=0.8, alpha=0.7, zorder=3)
                ax.plot(v["estimate"], i, marker=mk, color=col if first else "white", mec=col, ms=4, mew=0.9, ls="none", zorder=4)
    ax.axvline(0, color=fs.CHANCE, lw=0.8); ax.set_yticks(range(len(keys))); ax.set_yticklabels([f"{SHORT[m]} / {DS_SHORT[d]}" for m, d in keys], fontsize=6.4)
    ax.set_ylim(-0.7, len(keys) - 0.3); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("difference in ownership $O$ between templates (two concepts per block, with 95% intervals)", fontsize=8)
    fs.boxed_legend(ax, [Line2D([], [], marker="o", color=fs.SAGE, ls="none", ms=4, mec=fs.SAGE, label="wording contrast IY $-$ WY"),
                         Line2D([], [], marker="s", color=fs.LILAC, ls="none", ms=4, mec=fs.LILAC, label="mapping contrast IA $-$ IB"),
                         Line2D([], [], marker="o", color="white", ls="none", ms=4, mec=fs.CHARCOAL, mew=0.9, label="open: second concept (Mass / bottle)"),
                         Line2D([], [], marker="o", color=fs.CHARCOAL, ls="none", ms=4, mec=fs.CHARCOAL, label="filled: first concept (Effusion / person)")],
                    loc="lower right", ncol=2, **fs.LEG)
    fs.save(f, FIG / "figA11_prompt_dependence")


# ============================================================================================== appendix seed study
def figA1_prompt_score():
    f = plt.figure(figsize=(fs.WIDTH, fs.H1)); ax = f.add_axes((0, 0, 1, 1)); ax.set(xlim=(0, 5.5), ylim=(2.4, 0)); ax.set_axis_off(); ax.grid(False)
    def text(x, y, s, size=7.8, color=fs.INK, weight="normal", ha="left", va="top"):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, linespacing=1.35)
    text(0.12, 0.10, "Fixed clinical question", 8.5, weight="bold")
    ax.add_patch(Rectangle((0.12, 0.32), 5.26, 0.44, facecolor="#F5F1EC", edgecolor="none"))
    ax.plot((0.12, 0.12), (0.32, 0.76), color=fs.HAZE, lw=1.4)
    text(0.24, 0.40, "Is there <finding> in this chest radiograph?  Answer yes or no.", 8.0)
    text(0.12, 0.92, "Six ownership findings", 8.5, weight="bold"); text(5.38, 0.94, "only the finding phrase changes", 7.2, color=fs.MUTED, ha="right")
    findings = (("Effusion", "a pleural effusion"), ("Atelectasis", "atelectasis"), ("Pneumothorax", "a pneumothorax"),
                ("Cardiomegaly", "cardiomegaly"), ("Mass", "a lung mass"), ("Nodule", "a lung nodule"))
    for i, (name, phrase) in enumerate(findings):
        col, row = divmod(i, 3); x, y = 0.12 + col * 2.75, 1.16 + row * 0.22
        text(x, y, name, 7.8); text(x + 1.05, y, phrase, 7.8)
        ax.plot((x, x + 2.5), (y + 0.18, y + 0.18), color="#e6e4e1", lw=0.7)
    text(0.12, 1.86, "First-answer scoring", 8.5, weight="bold")
    centers = (0.62, 1.78, 3.15, 4.68)
    entries = ("image +\nfixed question", "first-answer-position\nlogits (fp32)", "max log probability per answer\nover valid token variants",
               r"$P(\mathrm{yes})=\sigma(\ell_{\mathrm{yes}}-\ell_{\mathrm{no}})$")
    for x, e in zip(centers, entries):
        text(x, 2.2, e, 7.4, ha="center", va="center")
    for s, e in ((1.06, 1.26), (2.32, 2.50), (3.92, 4.12)):
        ax.add_patch(FancyArrowPatch((s, 2.2), (e, 2.2), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=fs.MUTED))
    fs.save(f, FIG / "figA1_prompt_score")


def figA3_ownership_matrix(ownership):
    concepts = ownership["concepts"]
    M = np.array([[ownership["effect_matrix"][d][q] for q in concepts] for d in concepts])   # rows directions, cols questions
    W = M.T                                                                                   # rows questions, cols directions
    f = plt.figure(figsize=(fs.WIDTH, fs.H2), constrained_layout=True)
    gs = f.add_gridspec(1, 2, width_ratios=[1, 0.05])
    ax = f.add_subplot(gs[0]); cax = f.add_subplot(gs[1])
    vlim = float(np.nanmax(np.abs(W)))
    im = draw_matrix(ax, concepts, W, set(), vlim, values=True, title=None)
    ax.set_xticklabels(concepts, rotation=35, ha="right", fontsize=8, rotation_mode="anchor"); ax.set_yticklabels(concepts, fontsize=8)
    ax.set_xlabel("steering direction $d$", fontsize=9); ax.set_ylabel("clinical question $q$", fontsize=9)
    ax.set_title(f"Qwen2.5-VL-7B, {ownership['n_eval_patients']} patients, $\\alpha=+{ownership['alpha']:.2f}$; dot = column-wise strongest direction (all off-diagonal)", fontsize=8)
    cb = f.colorbar(im, cax=cax); cb.set_label("$W_{q,d}$", fontsize=9); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8)
    fs.save(f, FIG / "figA3_ownership_matrix")


def figA4_shared_alias(ownership):
    alias = ownership["shared_alias"]; concepts = ownership["concepts"]
    f = fig(fs.H1); gs = f.add_gridspec(1, 2, width_ratios=[1.3, 1])
    a = f.add_subplot(gs[0]); b = f.add_subplot(gs[1])
    for i, d in enumerate(concepts[::-1]):
        v = alias["by_direction"][d]["off_diagonal_effect"]; col = fs.TERRACOTTA if d == alias["candidate_direction"] else fs.HAZE
        a.plot([0, v], [i, i], color=col, lw=1.6, alpha=0.7, zorder=2); a.plot(v, i, marker="o", ms=5.5, color=col, mec="white", ls="none", zorder=3)
        a.annotate(f"{v:.4f}", (v, i), xytext=(4, 0), textcoords="offset points", fontsize=7, va="center", color=fs.INK)
    a.set_yticks(range(len(concepts))); a.set_yticklabels(concepts[::-1], fontsize=8); a.axvline(0, color=fs.CHANCE, lw=0.8)
    a.set_xlim(-0.03, 0.46); a.set_xlabel("off-diagonal $\\Delta P(\\mathrm{yes})$", fontsize=8); a.tick_params(axis="y", length=0)
    fs.panel_title(a, "a", "clinical directions as shared aliases")
    items = [(alias["candidate_direction"] + "\ncandidate", alias["off_diagonal_effect"], fs.TERRACOTTA, "o"),
             ("random\nmaximum", alias["global_random_effect_max"], fs.GREY, "D"), ("|sham|\nmaximum", alias["global_absolute_sham_effect_max"], fs.SAGE, "s")]
    for i, (lab, v, col, mk) in enumerate(items[::-1]):
        b.plot([0, v], [i, i], color=col, lw=1.6, alpha=0.7, zorder=2); b.plot(v, i, marker=mk, ms=5.5, color=col, mec="white", ls="none", zorder=3)
        b.annotate(f"{v:.4f}", (v, i), xytext=(4, 0), textcoords="offset points", fontsize=7, va="center", color=fs.INK)
    b.set_yticks(range(3)); b.set_yticklabels([it[0] for it in items[::-1]], fontsize=7.5); b.axvline(0, color=fs.CHANCE, lw=0.8)
    b.set_xlim(-0.03, 0.46); b.set_xlabel("off-diagonal $\\Delta P(\\mathrm{yes})$", fontsize=8); b.tick_params(axis="y", length=0)
    fs.panel_title(b, "b", "global controls")
    b.text(0.02, 0.5, f"Effusion simultaneous 95% lower bounds:\neffect {alias['off_diagonal_effect_simultaneous_lower_95']:.4f}; "
           f"relative dominance {alias['relative_dominance_simultaneous_lower_95']:.4f}", transform=b.transAxes, fontsize=6.4, color=fs.INK, ha="left", va="center")
    fs.save(f, FIG / "figA4_shared_alias")


def figA5_input_closure(closure):
    f = fig(fs.H2); gs = f.add_gridspec(2, 2)
    def one_sided(ax, est, lower, lim, xlabel, color, letter, title, note):
        ax.set_xlim(*lim); ax.set_ylim(0, 1); ax.set_yticks([]); ax.axvline(0, color=fs.CHANCE, lw=0.8)
        ax.annotate("", xy=(lim[1] - 0.03 * (lim[1] - lim[0]), 0.5), xytext=(lower, 0.5), arrowprops=dict(arrowstyle="->", color=color, lw=1.1))
        ax.plot(lower, 0.5, marker="|", ms=11, color=color, mew=1.3, ls="none"); ax.plot(est, 0.5, marker="o", ms=6, color=color, mec="white", ls="none")
        ax.set_xlabel(xlabel, fontsize=8); ax.tick_params(labelsize=7.5); ax.grid(False, axis="y")
        fs.panel_title(ax, letter, title); ax.text(0.02, 0.9, note, transform=ax.transAxes, fontsize=6.8, color=fs.INK, va="top")
    a = f.add_subplot(gs[0, 0]); p = closure["probe"]; lo, hi = p["selectivity_ci95"]
    a.errorbar(p["selectivity"], 0.5, xerr=[[p["selectivity"] - lo], [hi - p["selectivity"]]], fmt="o", color=fs.HAZE, ms=6, mec="white", capsize=3, elinewidth=1.2, ecolor="#555555")
    a.set_xlim(-0.005, 0.125); a.set_ylim(0, 1); a.set_yticks([]); a.axvline(0, color=fs.CHANCE, lw=0.8); a.grid(False, axis="y")
    a.set_xlabel("reader selectivity $S$", fontsize=8); a.tick_params(labelsize=7.5)
    fs.panel_title(a, "a", "reader eligibility"); a.text(0.02, 0.9, f"AUROC {p['real_auroc']:.4f}; $S$ = {p['selectivity']:.4f}  [{lo:.4f}, {hi:.4f}]", transform=a.transAxes, fontsize=6.8, va="top")
    dd = closure["input_displacement"]
    one_sided(f.add_subplot(gs[0, 1]), dd["estimate"], dd["one_sided_lower_95"], (-0.06, 0.19), "mean displacement", fs.SAGE, "b", "paired displacement",
              f"{closure['n_pairs']} patient pairs; estimate {dd['estimate']:.4f}; 95% lower bound {dd['one_sided_lower_95']:.4f}")
    c = f.add_subplot(gs[1, 0])
    items = [("Consolidation", closure["concept_closure_gain"], fs.TERRACOTTA, "o"), ("random max.", closure["random_closure_gain_max"], fs.GREY, "D"),
             (f"best clinical\n({closure['maximum_unrelated']['concept']})", closure["maximum_unrelated"]["closure_gain"], fs.HAZE, "o"), ("sham", closure["sham_closure_gain"], fs.SAGE, "s")]
    for i, (lab, v, col, mk) in enumerate(items[::-1]):
        c.plot([0, v * 1000], [i, i], color=col, lw=1.6, alpha=0.7, zorder=2); c.plot(v * 1000, i, marker=mk, ms=5.5, color=col, mec="white", ls="none", zorder=3)
        c.annotate(f"{v * 1000:.2f}", (v * 1000, i), xytext=(4, 0), textcoords="offset points", fontsize=7, va="center")
    c.set_yticks(range(4)); c.set_yticklabels([it[0] for it in items[::-1]], fontsize=7.5); c.axvline(0, color=fs.CHANCE, lw=0.8); c.tick_params(axis="y", length=0)
    c.set_xlim(-1.1, 3.9); c.set_xlabel(r"closure gain ($\times 10^{-3}$)", fontsize=8); c.tick_params(axis="x", labelsize=7.5)
    fs.panel_title(c, "c", "closure gains")
    m = closure["clinical_familywise_margin"]
    one_sided(f.add_subplot(gs[1, 1]), m["estimate"] * 1000, m["one_sided_lower_95"] * 1000, (-2.2, 3.4), r"clinical margin ($\times 10^{-3}$)", fs.TERRACOTTA, "d", "clinical margin",
              f"estimate {m['estimate'] * 1000:.2f}; 95% lower bound {m['one_sided_lower_95'] * 1000:.2f}; exact random-rank $p$={closure['exact_random_rank_p']:.3f}")
    fs.save(f, FIG / "figA5_input_closure")


def figA12_seed_dose(results):
    cells = {c["key"]: c for c in results["cells"]}
    order = [("llava_effusion", "LLaVA-1.5-7B, Effusion"), ("llava_edema", "LLaVA-1.5-7B, Edema"), ("qwen_effusion", "Qwen2.5-VL-7B, Effusion")]
    f = fig(fs.H1); axes = [f.add_subplot(1, 3, k + 1) for k in range(3)]
    for k, (ax, (key, title)) in enumerate(zip(axes, order)):
        cell = cells[key]
        al = np.array([d["alpha"] for d in cell["dose"]]); ch = np.array([d["raw_change"] for d in cell["dose"]])
        ctrl = {float(a): v for a, v in cell["controls"].items()}; ca = np.array(sorted(ctrl))
        ax.fill_between(ca, [ctrl[a]["random_p05"] for a in ca], [ctrl[a]["random_p95"] for a in ca], color=fs.STONE, alpha=0.6, lw=0, zorder=1)
        ax.plot(al, ch, ls="--", marker="o", color=fs.HAZE, ms=4.5, zorder=3)
        ax.plot(ca, [ctrl[a]["sham_effect"] for a in ca], ls="none", marker="x", color=fs.TERRACOTTA, ms=5.5, mew=1.2, mec=fs.TERRACOTTA, zorder=4)
        ax.axhline(0, color=fs.CHANCE, lw=0.8); ax.axvline(0, color=fs.CHANCE, lw=0.6, ls=":")
        ax.set_xlim(-1.08, 1.08); ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_ylim(-0.32, 0.32)
        ax.set_xlabel(r"relative-token dose $\alpha$", fontsize=8); ax.tick_params(labelsize=7.5)
        fs.panel_title(ax, "abc"[k], title)
        if k:
            ax.set_yticklabels([])
    axes[0].set_ylabel(r"change in mean $P(\mathrm{yes})$", fontsize=8)
    fs.boxed_legend(axes[2], [Line2D([], [], ls="--", marker="o", color=fs.HAZE, ms=4.5, mec="white", label="probe normal"),
                              Patch(facecolor="#E9E5DF", edgecolor=fs.GREY, lw=0.5, label="random 5–95%"),
                              Line2D([], [], ls="none", marker="x", color=fs.TERRACOTTA, mec=fs.TERRACOTTA, ms=5.5, mew=1.2, label="sham")], loc="upper left", **fs.LEG)
    fs.save(f, FIG / "figA12_seed_dose")


def figA13_seed_specificity(results):
    spec = results["specificity"]; eff = spec["direction_effects"]
    randoms = np.array([v for k, v in eff.items() if k.startswith("random")])
    alt = [("unrelated_Atelectasis", "Atel."), ("unrelated_Cardiomegaly", "Card."), ("unrelated_Mass", "Mass"), ("unrelated_Nodule", "Nodule"), ("unrelated_Pneumothorax", "Pneumo.")]
    f = fig(fs.H1); gs = f.add_gridspec(1, 2, width_ratios=[2.4, 1]); ax = f.add_subplot(gs[0]); bx = f.add_subplot(gs[1])
    rng = np.random.default_rng(0); jit = rng.uniform(-0.14, 0.14, len(randoms))
    ax.plot(jit, randoms, ls="none", marker="o", ms=3.6, color=fs.GREY, mec="white", mew=0.5, zorder=3)
    ax.hlines(spec["random_effect_p95"], -0.28, 0.28, color=fs.RED, ls="--", lw=1.0, zorder=4)
    ax.annotate("p95", (0.30, spec["random_effect_p95"]), fontsize=7, color=fs.RED, va="center")
    ax.plot(1, eff["concept"], ls="none", marker="o", ms=8, color=fs.HAZE, mec="white", zorder=5)
    ax.plot(2, eff["sham"], ls="none", marker="X", ms=8, color=fs.LILAC, mec="white", zorder=5)
    for i, (key, lab) in enumerate(alt):
        col = fs.TERRACOTTA if key == spec["primary"]["maximum_unrelated_direction"] else fs.SAGE
        ax.plot(3 + i, eff[key], ls="none", marker="D", ms=7, color=col, mec="white", zorder=5)
    ax.axhline(0, color=fs.CHANCE, lw=0.8)
    ax.set_xticks(range(8)); ax.set_xticklabels(["Random", "Effusion", "Sham"] + [l for _, l in alt], rotation=30, ha="right", fontsize=7.5)
    ax.set_xlim(-0.6, 7.6); ax.set_ylabel(r"change in mean $P(\mathrm{yes})$", fontsize=8); ax.tick_params(axis="y", labelsize=7.5)
    fs.panel_title(ax, "a", rf"locked dose $\alpha=+{spec['alpha']:.2f}$, {spec['n_eval_patients']} new patients")
    fs.boxed_legend(ax, [Line2D([], [], ls="none", marker="o", color=fs.GREY, ms=3.6, mec="white", label="20 random directions"),
                         Line2D([], [], color=fs.RED, ls="--", lw=1.0, label="random p95"),
                         Line2D([], [], ls="none", marker="o", color=fs.HAZE, ms=6, mec="white", label="Effusion direction"),
                         Line2D([], [], ls="none", marker="X", color=fs.LILAC, ms=6, mec="white", label="sham"),
                         Line2D([], [], ls="none", marker="D", color=fs.SAGE, ms=5.5, mec="white", label="clinical alternatives"),
                         Line2D([], [], ls="none", marker="D", color=fs.TERRACOTTA, ms=5.5, mec="white", label="strongest alternative")],
                    loc="upper left", ncol=2, fontsize=6.4, handlelength=1.2, borderpad=0.35, labelspacing=0.3, handletextpad=0.4, columnspacing=0.8)
    p = spec["primary"]; lo, hi = p["ci95"]
    bx.errorbar(p["margin"], 0, xerr=[[p["margin"] - lo], [hi - p["margin"]]], fmt="o", color=fs.TERRACOTTA, ms=6, mec="white", ecolor="#555555", elinewidth=0.7, capsize=2.5, zorder=4)
    bx.axvline(0, color=fs.CHANCE, lw=0.8); bx.plot([p["one_sided_lower_95"]], [0], marker="|", ms=9, color=fs.TERRACOTTA, mew=1.2, ls="none", zorder=5)
    bx.set_yticks([0]); bx.set_yticklabels([r"$O_{\mathrm{Effusion}}$"], fontsize=8); bx.set_ylim(-1, 1); bx.set_xlim(-0.095, 0.015); bx.set_xticks([-0.08, -0.04, 0])
    bx.set_xlabel("Effusion ownership contrast", fontsize=8); bx.tick_params(axis="x", labelsize=7.5); bx.grid(False, axis="y")
    fs.panel_title(bx, "b", "95% patient bootstrap")
    bx.annotate(f"{p['margin']:.4f}  [{lo:.4f}, {hi:.4f}]", (p["margin"], 0), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7, color=fs.TERRACOTTA, fontweight="bold")
    fs.save(f, FIG / "figA13_seed_specificity")


# ============================================================================================== main
def main(only=None):
    fs.use_house_style()
    blocks = load_blocks(); results, ownership = seed()
    todo = {
        "fig1": lambda: fig1_framework(blocks), "fig2": lambda: fig2_landscape(blocks), "fig3": lambda: fig3_write_matrices(blocks),
        "fig4": lambda: fig4_readers(blocks), "fig5": lambda: fig5_dose_reference(blocks), "fig6": lambda: fig6_controls(blocks),
        "figA1": figA1_prompt_score, "figA3": lambda: figA3_ownership_matrix(ownership), "figA4": lambda: figA4_shared_alias(ownership),
        "figA5": lambda: figA5_input_closure(results["input_closure"]),
        "figA7": lambda: matrices_all(blocks, "nih", "figA7_write_matrices_nih"), "figA8": lambda: matrices_all(blocks, "chexpert", "figA8_write_matrices_chexpert"),
        "figA9": lambda: matrices_all(blocks, "coco", "figA9_write_matrices_coco"),
        "figA10": lambda: figA10_controls_all(blocks), "figA11": lambda: figA11_prompt_dependence(blocks),
        "figA12": lambda: figA12_seed_dose(results), "figA13": lambda: figA13_seed_specificity(results),
    }
    for name, fn in todo.items():
        if only and name not in only:
            continue
        try:
            fn(); print("ok", name)
        except Exception as e:      # keep going; report at the end
            import traceback; traceback.print_exc(); print("FAILED", name, repr(e))


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
