"""Every figure of the paper, generated from the packaged campaign statistics and the seed-study data.

Size system: each figure is authored at the ICLR text width (5.5 in) with a height chosen so that no panel is
squashed, and is included with width=\textwidth, so nothing is rescaled. Style: figstyle (house typography, Morandi
palette for categorical series). Heat maps follow standard practice: square cells separated by thin white gaps, no
frame, a soft diverging map centred on zero (white at zero) with symmetric limits per figure, one slim shared
colour bar, sparse annotations with automatic text colour, and subtle emphasis (a thin outline on the diagonal, a
small dot on owned cells) rather than heavy borders. Numbers are copied from the data files, never recomputed
beyond means, medians, and quantiles stated in the captions. Every figure is checked for overlapping text after
rendering (check_overlaps), and the check is printed with the figure name.

Main text
  fig1_framework        read / write / answer schematic with the seed inset
  fig2_overview         (a) graded fractions per dataset, (b) model landscape, (c) dose response, (d) rank ECDF
  fig3_example          one chest radiograph and one natural image: P(yes) per question under three inputs
  fig4_write_matrices   6x6 write matrices W_{q,d} for three checkpoints on NIH (top) and COCO (bottom)
  fig5_same_write       same write, different reader: ownership of Gemma 3 4B/12B/27B and MedGemma 4B/27B
  fig6_ladders          size ladders: Qwen2.5-VL, Qwen3-VL (NIH), Lingshu (CheXpert)
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm   # noqa: E402
from matplotlib.lines import Line2D                     # noqa: E402
from matplotlib.text import Text                        # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle   # noqa: E402
from matplotlib.transforms import Bbox, offset_copy     # noqa: E402
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
SHORT_CONCEPT = {"Effusion": "Effus.", "Atelectasis": "Atel.", "Pneumothorax": "Pneu.", "Cardiomegaly": "Card.",
                 "Mass": "Mass", "Nodule": "Nodule", "Consolidation": "Cons.", "Edema": "Edema",
                 "person": "person", "dog": "dog", "car": "car", "chair": "chair", "bottle": "bottle", "bicycle": "bicycle"}

# heat-map conventions shared by every matrix figure
DIV = LinearSegmentedColormap.from_list("div_soft", ["#2c6fac", "#f7f7f7", "#c0392b"], N=256)
GAP = 1.2                 # white gap between cells (points)
OUTLINE = 0.8             # diagonal outline (points), drawn inside the white gap
DOT_MS = 3.2              # owned-cell dot diameter (points)
DOT_OFF = 2.8             # dot inset from the top-right corner (points)
ANNOT_MIN = 0.05          # annotate |value| >= this
ANNOT_SIZE = 6.5
TICK_SIZE = 8
HEAT_BAD = "#e9e7e3"


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


def O_of(blocks, mk, ds):
    b = blocks.get((mk, ds))
    return {c: v["O_q"] for c, v in b["s"]["core"]["per_question"].items()} if b else None


def dose_cache():
    p = RUNS / "figures" / "dose_curves.json"
    return json.loads(p.read_text()) if p.exists() else {}


def panel(ax, letter, text):
    fs.panel_title(ax, letter, text)


# ============================================================================================== layout helpers
def rect(fw, fh, x, y, w, h):
    """Axes rectangle in figure fractions from inches (x, y from the bottom-left corner)."""
    return (x / fw, y / fh, w / fw, h / fh)


def text_bboxes(fig):
    """Every visible non-empty text of the figure with its display bbox, plus every legend frame."""
    r = fig.canvas.get_renderer(); out = []
    for ax in fig.axes:
        items = [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        for t in items:
            if t.get_visible() and t.get_text().strip():      # Text.get_window_extent: the glyph box only, never a leader arrow
                out.append((f"{ax.get_label() or 'ax'}:{t.get_text()[:28]!r}", Text.get_window_extent(t, r)))
        for axis in ("x", "y"):
            for t in (ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()):
                if t.get_visible() and t.get_text().strip():
                    out.append((f"{ax.get_label() or 'ax'}:tick{axis}:{t.get_text()[:20]!r}", t.get_window_extent(r)))
        if ax.get_legend() is not None:
            out.append((f"{ax.get_label() or 'ax'}:legend", ax.get_legend().get_window_extent(r)))
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            out.append((f"fig:{t.get_text()[:28]!r}", t.get_window_extent(r)))
    for lg in fig.legends:
        out.append(("fig:legend", lg.get_window_extent(r)))
    return out


def check_overlaps(fig, name, ignore=()):
    """Print every pair of overlapping texts / legends and every text outside the figure. Tick labels of the same
    axis are skipped (their spacing is the axis's business); `ignore` lists label prefixes to skip."""
    fig.canvas.draw()
    items = text_bboxes(fig); bad = 0
    W, H = fig.get_size_inches() * fig.dpi
    for i in range(len(items)):
        li, bi = items[i]
        if any(li.startswith(p) for p in ignore):
            continue
        if (bi.x0 < -0.5 or bi.y0 < -0.5 or bi.x1 > W + 0.5 or bi.y1 > H + 0.5) and ":tick" not in li:
            print(f"  [{name}] OUTSIDE figure: {li}"); bad += 1
        for j in range(i + 1, len(items)):
            lj, bj = items[j]
            if any(lj.startswith(p) for p in ignore):
                continue
            if ":tick" in li and ":tick" in lj and li.split(":tick")[0] == lj.split(":tick")[0] and li.split(":")[1] == lj.split(":")[1]:
                continue      # tick labels of the same axis: rotated labels have overlapping boxes but not glyphs
            if bi.overlaps(bj) and bi.width > 0 and bj.width > 0:
                ov = Bbox.intersection(bi, bj)
                if ov is not None and ov.width > 0.5 and ov.height > 0.5:
                    print(f"  [{name}] OVERLAP: {li}  x  {lj}"); bad += 1
    print(f"  [{name}] overlap check: {'clean' if bad == 0 else f'{bad} issue(s)'}")
    return bad


def marker_display_points(ax, lines):
    r = ax.figure.canvas.get_renderer()
    pts = []
    for ln in lines:
        x, y = ln.get_data()
        pts.append(ax.transData.transform(np.column_stack([x, y])))
    return np.vstack(pts) if pts else np.zeros((0, 2))


def place_free_text(fig, ax, text, candidates, avoid_pts, margin=3, **kw):
    """Put `text` at the first candidate (axes-fraction (x, y)) whose bbox, grown by `margin` points, contains
    no point of `avoid_pts` (display coords) and stays inside the axes."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); axbb = ax.get_window_extent(r); m = margin * fig.dpi / 72
    for (x, y) in candidates:
        t = ax.text(x, y, text, transform=ax.transAxes, **kw)
        bb = t.get_window_extent(r).expanded(1.0, 1.0)
        bb = Bbox([[bb.x0 - m, bb.y0 - m], [bb.x1 + m, bb.y1 + m]])
        hit = ((avoid_pts[:, 0] > bb.x0) & (avoid_pts[:, 0] < bb.x1) & (avoid_pts[:, 1] > bb.y0) & (avoid_pts[:, 1] < bb.y1)).any()
        inside = bb.x0 >= axbb.x0 and bb.x1 <= axbb.x1 and bb.y0 >= axbb.y0 and bb.y1 <= axbb.y1
        if not hit and inside:
            return t
        t.remove()
    t = ax.text(*candidates[0], text, transform=ax.transAxes, **kw)
    print("  WARNING: no free candidate for", text.replace("\n", " / ")); return t


def stack_labels(fig, texts, step=1.0):
    """Annotations with offset-point positions: raise any label that overlaps an earlier one (left to right)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); placed = []
    for t in sorted(texts, key=lambda t: t.get_window_extent(r).x0):
        for _ in range(40):
            bb = t.get_window_extent(r).expanded(1.08, 1.15)
            if not any(bb.overlaps(p) for p in placed):
                break
            dx, dy = t.xyann; t.xyann = (dx, dy + step); fig.canvas.draw()
        placed.append(t.get_window_extent(r).expanded(1.08, 1.15))


def repel(ys, step, lo, hi, iters=200):
    """Spread label centres (any order) so neighbours are >= step apart, staying within [lo, hi]; order preserved."""
    ys = np.array(ys, float); order = np.argsort(ys); y = ys[order].copy()
    for _ in range(iters):
        moved = False
        for a in range(len(y) - 1):
            gap = y[a + 1] - y[a]
            if gap < step - 1e-9:
                d = (step - gap) / 2; y[a] -= d; y[a + 1] += d; moved = True
        y = np.clip(y, lo, hi)
        if not moved:
            break
    out = np.empty_like(y); out[order] = y
    return out


# ============================================================================================== heat-map primitives
def lum(rgba):
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def heat(ax, M, norm, xedges=None, yedges=None, cmap=DIV, spacer_rows=()):
    """Square-cell heat map with thin white gaps, no frame, row 0 at the top. Returns (xedges, yedges).
    Rows listed in spacer_rows are all-NaN separators drawn as white space (no grey patch)."""
    nr, nc = M.shape
    xe = np.arange(nc + 1, dtype=float) if xedges is None else np.asarray(xedges, float)
    ye = np.arange(nr + 1, dtype=float) if yedges is None else np.asarray(yedges, float)
    for i in range(nr):
        for j in range(nc):
            if not np.isfinite(M[i, j]) and i not in spacer_rows:
                ax.add_patch(Rectangle((xe[j], ye[i]), xe[j + 1] - xe[j], ye[i + 1] - ye[i], facecolor=HEAT_BAD, edgecolor="white", lw=GAP, zorder=1))
    ax.pcolormesh(xe, ye, np.ma.masked_invalid(M), cmap=cmap, norm=norm, edgecolors="white", linewidth=GAP, zorder=2)
    ax.set_xlim(xe[0], xe[-1]); ax.set_ylim(ye[-1], ye[0]); ax.set_aspect("equal")
    ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return xe, ye


def heat_ticks(ax, xe, ye, xlabels, ylabels, size=TICK_SIZE):
    ax.set_xticks((xe[:-1] + xe[1:]) / 2); ax.set_yticks((ye[:-1] + ye[1:]) / 2)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", rotation_mode="anchor", fontsize=size)
    ax.set_yticklabels(ylabels, fontsize=size)


def heat_cell_marks(ax, xe, ye, i, j, outline=False, dot=False):
    """Thin charcoal outline (inside the white gap) and/or a small charcoal dot at the top-right corner."""
    if outline:
        ax.add_patch(Rectangle((xe[j], ye[i]), xe[j + 1] - xe[j], ye[i + 1] - ye[i], fill=False, edgecolor=fs.CHARCOAL, lw=OUTLINE, zorder=4))
    if dot:
        tr = offset_copy(ax.transData, fig=ax.figure, x=-DOT_OFF, y=-DOT_OFF, units="points")
        ax.plot(xe[j + 1], ye[i], marker="o", ms=DOT_MS, color=fs.CHARCOAL, mec="none", ls="none", transform=tr, zorder=5, clip_on=False)


def heat_value(ax, xe, ye, i, j, v, norm, bold=False, cmap=DIV):
    if not np.isfinite(v) or abs(v) < ANNOT_MIN:
        return
    col = "white" if lum(cmap(norm(v))) < 0.5 else fs.INK
    ax.text((xe[j] + xe[j + 1]) / 2, (ye[i] + ye[i + 1]) / 2, f"{v:+.2f}", ha="center", va="center", fontsize=ANNOT_SIZE, color=col,
            fontweight="bold" if bold else "normal", zorder=6)


def heat_colorbar(fig, cax, norm, label, cmap=DIV, ticks=None):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7, length=2, width=0.5)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.set_label(label, fontsize=7.5, labelpad=2)
    return cb


def sym_norm(vlim):
    return TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)


def round_up(v, q=0.1):
    return float(np.ceil(v / q) * q)


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
    for ax, lab in zip(axes.ravel(), "abcd"):
        ax.set_label(lab)
    # (a) graded fractions -------------------------------------------------------------------------------------
    ax = axes[0, 0]; grades = ["readable", "answerable", "owned"]; w = 0.26; counts = []
    for k, ds in enumerate(DATASETS):
        read, ans, own = grade_cells(blocks, ds)
        for gi, flags in enumerate((read, ans, own)):
            m, lo, hi = boot_frac(flags)
            x = gi + (k - 1) * w
            ax.bar(x, m, w - 0.03, color=DS_COL[ds], edgecolor="none", zorder=3)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#555555", elinewidth=0.7, capsize=1.5, zorder=4)
            counts.append(ax.annotate(f"{int(flags.sum())}/{len(flags)}", (x, hi), xytext=(0, 2.5), textcoords="offset points",
                                      ha="center", va="bottom", fontsize=6, color=fs.INK, zorder=6))
    ax.set_xticks(range(3)); ax.set_xticklabels(grades); ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("fraction of concept cells"); ax.grid(False, axis="x")
    panel(ax, "a", "graded outcomes per concept cell")

    # (b) model landscape: markers only, clusters named in empty space ------------------------------------------
    ax = axes[0, 1]; lines = []
    for ds in DATASETS:
        S, O = [], []
        for (mk, d), b in blocks.items():
            if d != ds:
                continue
            cal = b["s"].get("calibration", {}); core = b["s"]["core"]["per_question"]
            S.append(np.nanmean([cal[c]["selectivity"] for c in core if isinstance(cal.get(c), dict) and "selectivity" in cal[c]]))
            O.append(np.nanmean([v["O_q"] for v in core.values() if v.get("O_q") is not None]))
        lines += ax.plot(S, O, marker=DS_MK[ds], color=DS_COL[ds], ls="none", ms=5.5, mec="white", mew=0.7, zorder=3)
    ax.axhline(0, color=fs.CHANCE, ls="--", lw=0.8)
    ax.set_xlabel("mean readability $S$ over concepts"); ax.set_ylabel("mean ownership $O$ over concepts")
    ax.set_ylim(-0.5, 0.8); ax.set_xlim(-0.02, 0.34)
    panel(ax, "b", "model landscape, one marker per block")
    pts = marker_display_points(ax, lines)
    place_free_text(f, ax, "natural objects:\nwritten", [(0.50, 0.90), (0.55, 0.85), (0.62, 0.90), (0.40, 0.92)], pts,
                    fontsize=7.4, color=fs.INK, ha="left", va="top", linespacing=1.15)
    place_free_text(f, ax, "chest findings:\nread, not written", [(0.97, 0.19), (0.97, 0.22), (0.97, 0.26), (0.97, 0.30), (0.30, 0.15)], pts,
                    fontsize=7.4, color=fs.INK, ha="right", va="top", linespacing=1.15)

    # (c) dose response -----------------------------------------------------------------------------------------
    ax = axes[1, 0]; curves = dose_cache(); ends = {}
    for ds in ("nih", "coco"):
        rows = [v for k, v in curves.items() if k.endswith("/" + ds) and v]
        alphas = sorted({float(a) for r in rows for a in r})
        M = np.array([[r.get(str(a), np.nan) for a in alphas] for r in rows])
        med, lo, hi = np.nanmedian(M, axis=0), np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0)
        ax.fill_between(alphas, lo, hi, color=DS_COL[ds], alpha=0.18, lw=0)
        ax.plot(alphas, med, marker=DS_MK[ds], color=DS_COL[ds], ls="-", lw=1.5, ms=5, mec="white")
        ends[ds] = (alphas[-1], med[-1], len(rows))
    ax.axhline(0, color=fs.CHANCE, ls="--", lw=0.8); ax.axvline(0, color=fs.CHANCE, lw=0.6)
    ax.set_xlabel(r"relative dose $\alpha$"); ax.set_ylabel("median ownership $O$ over questions")
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5]); ax.set_xlim(-0.58, 0.58); ax.set_ylim(-0.32, 0.66)
    ax.annotate(f"COCO, {ends['coco'][2]} blocks", (ends["coco"][0], ends["coco"][1]), xytext=(-2, 9), textcoords="offset points",
                ha="right", va="bottom", fontsize=7, color=DS_COL["coco"])
    ax.annotate(f"NIH, {ends['nih'][2]} blocks", (ends["nih"][0], ends["nih"][1]), xytext=(-2, -9), textcoords="offset points",
                ha="right", va="top", fontsize=7, color=DS_COL["nih"])
    panel(ax, "c", "dose response, median and IQR over blocks")

    # (d) rank ECDF ---------------------------------------------------------------------------------------------
    ax = axes[1, 1]; xs = np.arange(1, 121)
    # direct labels in the empty regions: above the COCO curve, above-left of the CheXpert curve, below-right of NIH
    lab_pos = {"coco": (1.5, 0.885, "left", "bottom"), "chexpert": (9.0, 0.66, "right", "top"), "nih": (50.0, 0.30, "left", "top")}
    for ds in DATASETS:
        ranks = np.array([v["rank_in_random_family"] for (mk, d), b in blocks.items() if d == ds
                          for v in b["s"]["core"]["per_question"].values() if v.get("rank_in_random_family")])
        ecdf = np.array([(ranks <= x).mean() for x in xs])
        ax.step(xs, ecdf, where="post", color=DS_COL[ds], lw=1.5)
        r1 = (ranks == 1).mean(); x0, y0, ha, va = lab_pos[ds]
        ax.text(x0, y0, f"{DS_SHORT[ds]}: {len(ranks)} cells\nrank 1 in {100 * r1:.0f}%", ha=ha, va=va, fontsize=7, color=DS_COL[ds], linespacing=1.15)
    ax.set_xscale("log"); ax.set_xlim(0.9, 125); ax.set_ylim(0, 1.06)
    ax.set_xlabel("rank of the concept write among 120 effects"); ax.set_ylabel("fraction of concept cells")
    panel(ax, "d", "is the concept write the strongest effect?")

    handles = [Line2D([], [], marker=DS_MK[d], ms=6, color=DS_COL[d], ls="none", mec="white", label=DS_LAB[d]) for d in DATASETS]
    f.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7.5, frameon=True, edgecolor="#8a8a8a", columnspacing=1.6, handletextpad=0.4)
    stack_labels(f, counts)
    check_overlaps(f, "fig2_overview")
    fs.save(f, FIG / "fig2_overview")


# ============================================================================================== fig3: qualitative example
def row_outcomes(mk, ds, row_id, concepts):
    import pyarrow.parquet as pq
    p = RUNS / mk / ds / "outcomes" / "CORE.parquet"
    df = pq.read_table(p, columns=["row_id", "concept", "template_id", "fit_seed", "direction_id", "direction_kind", "p_present", "sample_status"]).to_pandas()
    df = df[(df.row_id == row_id) & (df.sample_status == "OK") & (df.template_id == "IY") & (df.fit_seed == 0)]
    base = df[df.direction_kind == "baseline"].set_index("concept")["p_present"]
    writes = {d: df[df.direction_id == f"concept:{d}"].set_index("concept")["p_present"] for d in concepts}
    return base, writes


EXAMPLES = [("q25-7", "nih", "00005446_000", "Effusion", "Nodule", "/rodata/azradonc_dev/m253405/cache/nih/images/00005446_000.png",
             "NIH ChestX-ray14\nno finding reported", "(a)  chest radiograph: the Effusion write raises the Effusion answer less than the Nodule write does"),
            ("q25-7", "coco", "000000028449", "dog", None, "/rodata/azradonc_dev/m253405/cf-transfer/data/coco/val2017/000000028449.jpg",
             "COCO val2017\nno dog in the image", "(b)  natural image: the dog write raises only the dog answer, and its strongest competitor does not move it")]
EX_COL = {"clean": "#9d9a95", "concept": fs.HAZE, "competitor": fs.TERRACOTTA}


def colored_annotation(fig, ax, xy, pieces, dx=5, fontsize=7.5):
    """Consecutive text pieces in different colours, starting `dx` points right of data point `xy`."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); x = dx
    for txt, col in pieces:
        t = ax.annotate(txt, xy, xytext=(x, 0), textcoords="offset points", ha="left", va="center", fontsize=fontsize, color=col,
                        fontweight="bold" if txt.strip() and txt.strip()[0].isdigit() else "normal", annotation_clip=False)
        x += t.get_window_extent(r).width * 72 / fig.dpi


def fig3_example(blocks):
    from PIL import Image
    FW, FH = fs.WIDTH, 4.2
    f = plt.figure(figsize=(FW, FH))
    img_w, img_x = 1.15, 0.12; bar_x, bar_w = 2.08, 2.47; ax_h = 1.30
    row_y = [2.56, 0.86]; title_y = [3.92, 2.20]
    for r, (mk, ds, rid, target, comp, path, desc, claim) in enumerate(EXAMPLES):
        concepts = list(blocks[(mk, ds)]["s"]["core"]["per_question"])
        base, writes = row_outcomes(mk, ds, rid, concepts)
        if comp is None:
            comp = max((d for d in concepts if d != target), key=lambda d: writes[d].get(target, -1))
        im = Image.open(path).convert("L" if ds == "nih" else "RGB"); iw, ih = im.size
        h_img = min(ax_h, img_w * ih / iw); w_img = h_img * iw / ih
        axi = f.add_axes(rect(FW, FH, img_x + (img_w - w_img) / 2, row_y[r] + ax_h - h_img, w_img, h_img), label=f"img{r}")
        axi.imshow(im, cmap="gray" if ds == "nih" else None, interpolation="lanczos"); axi.set_xticks([]); axi.set_yticks([]); axi.grid(False)
        for sp in axi.spines.values():
            sp.set_edgecolor("#8a8a8a"); sp.set_linewidth(0.6)
        f.text((img_x + img_w / 2) / FW, (row_y[r] + ax_h - h_img - 0.06) / FH, desc, ha="center", va="top", fontsize=7, color=fs.MUTED, linespacing=1.15)
        ax = f.add_axes(rect(FW, FH, bar_x, row_y[r], bar_w, ax_h), label=f"bars{r}")
        y = np.arange(len(concepts)); h = 0.27
        vals = [[base.get(c, np.nan) for c in concepts], [writes[target].get(c, np.nan) for c in concepts], [writes[comp].get(c, np.nan) for c in concepts]]
        for k, (v, key) in enumerate(zip(vals, ("clean", "concept", "competitor"))):
            ax.barh(y + (k - 1) * h, v, h, color=EX_COL[key], edgecolor="none", zorder=3)
        jt = concepts.index(target)
        ax.axhspan(jt - 0.47, jt + 0.47, color="#f0eeea", zorder=0)
        ax.set_yticks(y); ax.set_yticklabels(concepts, fontsize=7.5); ax.set_ylim(len(concepts) - 0.5, -0.5)
        ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(axis="x", labelsize=7); ax.grid(False, axis="y")
        if r == 1:
            ax.set_xlabel(r"$P(\mathrm{yes})$ to each question at $\alpha=+0.25$, Qwen2.5-VL-7B", fontsize=7.5)
        b0, bt, bc = vals[0][jt], vals[1][jt], vals[2][jt]
        pieces = [(f"{b0:.2f}", EX_COL["clean"]), (" → ", fs.INK), (f"{bt:.2f}", EX_COL["concept"]), (" / ", fs.INK), (f"{bc:.2f}", EX_COL["competitor"])]
        colored_annotation(f, ax, (1.0, jt), pieces, dx=5, fontsize=7.5)
        f.text(img_x / FW, title_y[r] / FH, claim, ha="left", va="bottom", fontsize=8.5, color=fs.INK)
    handles = [Rectangle((0, 0), 1, 1, color=EX_COL["clean"], label="clean (no write)"),
               Rectangle((0, 0), 1, 1, color=EX_COL["concept"], label="concept write (Effusion in a, dog in b)"),
               Rectangle((0, 0), 1, 1, color=EX_COL["competitor"], label="strongest competing write (Nodule in a, bottle in b)")]
    f.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=3, fontsize=6.8, frameon=True, edgecolor="#8a8a8a",
             handlelength=1.4, handleheight=0.8, columnspacing=1.2, handletextpad=0.5, borderpad=0.45)
    check_overlaps(f, "fig3_example")
    fs.save(f, FIG / "fig3_example")


# ============================================================================================== fig4 / appendix matrices
MATRIX_PICKS = [("q25-7", "nih"), ("q25-72", "nih"), ("lingshu-32", "nih"), ("q25-7", "coco"), ("q3-32", "coco"), ("lingshu-32", "coco")]
CB_LABEL_W = ("$W_{q,d}$: mean change in $P(\\mathrm{yes}\\mid q)$ of question $q$ (row) under the write of direction $d$ (column)\n"
              "$\\bullet$ owned concept;  outlined: diagonal $d=q$")


def draw_matrix(ax, concepts, W, own, norm, values=True, ylabels=True, xlabels=True):
    xe, ye = heat(ax, W, norm); n = len(concepts)
    lab = [SHORT_CONCEPT.get(c, c) for c in concepts]
    heat_ticks(ax, xe, ye, lab if xlabels else [""] * n, lab if ylabels else [""] * n)
    for i in range(n):
        heat_cell_marks(ax, xe, ye, i, i, outline=True, dot=concepts[i] in own)
        if values:
            for j in range(n):
                heat_value(ax, xe, ye, i, j, W[i, j], norm, bold=(i == j and concepts[i] in own))


def fig4_write_matrices(blocks):
    mats = [(mk, ds, *wmatrix(blocks[(mk, ds)]["s"]), owned_set(blocks[(mk, ds)]["s"])) for mk, ds in MATRIX_PICKS]
    vlim = round_up(np.nanmax([np.nanmax(np.abs(W)) for *_, W, _ in mats])); norm = sym_norm(vlim)
    FW, FH = fs.WIDTH, 4.6; P = 1.35; gap = 0.26; x0 = 0.64
    rows_y = [FH - 0.31 - P, 0.97]
    f = plt.figure(figsize=(FW, FH))
    for k, (mk, ds, concepts, W, own) in enumerate(mats):
        r, c = divmod(k, 3)
        ax = f.add_axes(rect(FW, FH, x0 + c * (P + gap), rows_y[r], P, P), label=f"m{k}")
        draw_matrix(ax, concepts, W, own, norm, ylabels=(c == 0))
        ax.set_title(f"({'abcdef'[k]})  {NAMES[mk]}, {DS_SHORT[ds]}", fontsize=8.5, pad=4)
        if c == 0:
            ax.set_ylabel("question $q$", fontsize=8, labelpad=3)
    cax = f.add_axes(rect(FW, FH, 1.55, 0.50, 2.4, 0.08), label="cbar")
    heat_colorbar(f, cax, norm, CB_LABEL_W, ticks=[-vlim, -vlim / 2, 0, vlim / 2, vlim])
    check_overlaps(f, "fig4_write_matrices")
    fs.save(f, FIG / "fig4_write_matrices")


def matrices_all(blocks, ds, name):
    keys = [(mk, d) for (mk, d) in blocks if d == ds]
    mats = [(mk, *wmatrix(blocks[(mk, ds)]["s"]), owned_set(blocks[(mk, ds)]["s"])) for mk, _ in keys]
    vlim = round_up(np.nanmax([np.nanmax(np.abs(W)) for _, _, W, _ in mats])); norm = sym_norm(vlim)
    ncol = 4; nrow = int(np.ceil(len(mats) / ncol)); P = 1.06; gap = 0.16; x0 = 0.52; pitch = 1.32
    FW = fs.WIDTH; FH = 0.24 + nrow * pitch + 0.30 + 0.60
    f = plt.figure(figsize=(FW, FH))
    for k, (mk, concepts, W, own) in enumerate(mats):
        r, c = divmod(k, ncol); below = k + ncol < len(mats)
        ax = f.add_axes(rect(FW, FH, x0 + c * (P + gap), FH - 0.24 - P - r * pitch, P, P), label=f"m{k}")
        draw_matrix(ax, concepts, W, own, norm, values=False, ylabels=(c == 0), xlabels=not below)
        ax.set_title(NAMES[mk], fontsize=8, pad=3)
    cax = f.add_axes(rect(FW, FH, 1.55, 0.50, 2.4, 0.08), label="cbar")
    heat_colorbar(f, cax, norm, "$W_{q,d}$: change in $P(\\mathrm{yes}\\mid q)$ of question $q$ (row) under the write of direction $d$ (column)\n"
                  "$\\bullet$ owned concept;  outlined: diagonal $d=q$", ticks=[-vlim, -vlim / 2, 0, vlim / 2, vlim])
    check_overlaps(f, name)
    fs.save(f, FIG / name)


# ============================================================================================== fig5: same write, different reader
FAM = [("gemma3-4", "Gemma 3 4B"), ("gemma3-12", "Gemma 3 12B"), ("gemma3-27", "Gemma 3 27B"), ("medgemma-4", "MedGemma 4B"), ("medgemma-27", "MedGemma 27B")]
FAM_GAP = 0.32   # extra white gap (cell units) between the Gemma 3 and MedGemma rows


def fig5_same_write(blocks):
    Ms = {}
    for ds in DATASETS:
        concepts = list(next(b for (m, d), b in blocks.items() if d == ds)["s"]["core"]["per_question"])
        O = [O_of(blocks, mk, ds) for mk, _ in FAM]
        Ms[ds] = (concepts, np.array([[o.get(c, np.nan) if o else np.nan for c in concepts] for o in O]),
                  [owned_set(blocks[(mk, ds)]["s"]) if (mk, ds) in blocks else set() for mk, _ in FAM])
    vlim = round_up(np.nanmax([np.nanmax(np.abs(M)) for _, M, _ in Ms.values()])); norm = sym_norm(vlim)
    ye = np.array([0, 1, 2, 3, 3 + FAM_GAP, 4 + FAM_GAP, 5 + FAM_GAP]); real = [0, 1, 2, 4, 5]   # row 3 is the family spacer
    FW = fs.WIDTH; P = 1.36; gap = 0.20; x0 = 0.98; Ph = P * ye[-1] / 6
    FH = 0.10 + 0.26 + Ph + 0.34 + 0.34 + 0.26
    f = plt.figure(figsize=(FW, FH))
    for k, ds in enumerate(DATASETS):
        concepts, M, own = Ms[ds]
        ax = f.add_axes(rect(FW, FH, x0 + k * (P + gap), FH - 0.36 - Ph, P, Ph), label=f"h{k}")
        M6 = np.full((6, M.shape[1]), np.nan); M6[real] = M
        xe, _ = heat(ax, M6, norm, yedges=ye, spacer_rows=(3,))
        ax.set_xticks((xe[:-1] + xe[1:]) / 2); ax.set_yticks([(ye[i] + ye[i + 1]) / 2 for i in real])
        ax.set_xticklabels([SHORT_CONCEPT.get(c, c) for c in concepts], rotation=45, ha="right", rotation_mode="anchor", fontsize=TICK_SIZE)
        ax.set_yticklabels([lab for _, lab in FAM] if k == 0 else [""] * len(FAM), fontsize=TICK_SIZE)
        for i, ri in enumerate(real):
            for j, c in enumerate(concepts):
                o = c in own[i]
                heat_value(ax, xe, ye, ri, j, M[i, j], norm, bold=o)
                if o:
                    heat_cell_marks(ax, xe, ye, ri, j, dot=True)
        ax.set_title(f"({'abc'[k]})  {DS_LAB[ds]}", fontsize=8.5, pad=4)
    cax = f.add_axes(rect(FW, FH, 1.55, 0.40, 2.4, 0.08), label="cbar")
    heat_colorbar(f, cax, norm, r"ownership $O_q$ of the shared write vector, by reader      ($\bullet$ owned)",
                  ticks=[-vlim, -vlim / 2, 0, vlim / 2, vlim])
    check_overlaps(f, "fig5_same_write")
    fs.save(f, FIG / "fig5_same_write")


# ============================================================================================== fig6: size ladders
LADDERS = [("(a)  Qwen2.5-VL on NIH\nownership appears at 32B and 72B", [("q25-3", "3B"), ("q25-7", "7B"), ("q25-32", "32B"), ("q25-72", "72B")], "nih"),
           ("(b)  Qwen3-VL on NIH\nonly Nodule, at 32B", [("q3-4", "4B"), ("q3-8", "8B"), ("q3-32", "32B")], "nih"),
           ("(c)  Lingshu on CheXpert\nfive of six owned at 32B", [("lingshu-7", "7B"), ("lingshu-32", "32B")], "chexpert")]


def fig6_ladders(blocks):
    f, axes = plt.subplots(1, 3, figsize=(fs.WIDTH, 2.9), constrained_layout=True, sharey=True, gridspec_kw={"width_ratios": [4, 3, 2.5]})
    pad = {4: 0.85, 3: 0.85, 2: 0.80}
    for ax, (title, sizes, ds), letter in zip(axes, LADDERS, "abc"):
        ax.set_label(letter)
        sizes = [(m, l) for m, l in sizes if (m, ds) in blocks]
        concepts = list(blocks[(sizes[0][0], ds)]["s"]["core"]["per_question"]); xs = np.arange(len(sizes))
        ends = []
        for c in concepts:
            col = CONCEPT_COLOR[c]
            ys = [O_of(blocks, mk, ds)[c] for mk, _ in sizes]; ow = [c in owned_set(blocks[(mk, ds)]["s"]) for mk, _ in sizes]
            ax.plot(xs, ys, color=col, lw=1.3, zorder=2)
            for x, y, o in zip(xs, ys, ow):
                ax.plot(x, y, marker="o", ms=4.8, mfc=col if o else "white", mec=col, mew=1.2, ls="none", zorder=3)
            ends.append((ys[-1], SHORT_CONCEPT.get(c, c), col))
        ax.axhline(0, color=fs.CHANCE, lw=0.8, ls="--", zorder=1)
        ax.set_xticks(xs); ax.set_xticklabels([l for _, l in sizes], fontsize=7.5); ax.set_xlim(-0.3, len(sizes) - 1 + pad[len(sizes)])
        ax.set_title(title, fontsize=8, loc="left", pad=5, linespacing=1.2); ax.grid(False, axis="x"); ax.set_xlabel("model size", fontsize=7.5)
        ax._ends = (ends, xs[-1])
    axes[0].set_ylim(-0.6, 0.66)
    lo, hi = axes[0].get_ylim(); step = 0.098 * (hi - lo)
    for ax in axes:
        ends, xl = ax._ends
        ly = repel([e[0] for e in ends], step, lo + step * 0.6, hi - step * 0.6)
        for (y0, name, col), y1 in zip(ends, ly):
            ax.annotate(name, (xl, y0), xytext=(xl + 0.14, y1), textcoords="data", fontsize=8, color=col, va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.45, alpha=0.7, shrinkA=0, shrinkB=3) if abs(y1 - y0) > 0.004 else None)
    axes[0].set_ylabel("ownership $O_q$  (filled: owned)", fontsize=8)
    check_overlaps(f, "fig6_ladders")
    fs.save(f, FIG / "fig6_ladders")


# ============================================================================================== main
KEEP = {"fig1_framework", "fig2_overview", "fig3_example", "fig4_write_matrices", "fig5_same_write", "fig6_ladders",
        "figA1_write_matrices_nih", "figA2_write_matrices_chexpert", "figA3_write_matrices_coco"}


def main(only=None):
    fs.use_house_style()
    blocks = load_blocks()
    jobs = {"fig1_framework": lambda: fig1_framework(blocks), "fig2_overview": lambda: fig2_overview(blocks),
            "fig3_example": lambda: fig3_example(blocks), "fig4_write_matrices": lambda: fig4_write_matrices(blocks),
            "fig5_same_write": lambda: fig5_same_write(blocks), "fig6_ladders": lambda: fig6_ladders(blocks),
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
