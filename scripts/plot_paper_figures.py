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
  fig1_method           (a) read and write at the consumed block, (b) the three comparisons of the grade, (c) the seed cell from macros
  fig2_overview         (a) graded fractions per dataset, (b) model landscape, (c) dose response, (d) rank ECDF
  fig3_example          one chest radiograph and one natural image: P(yes) per question under three inputs
  fig4_write_matrices   6x6 write matrices W_{q,d} for three checkpoints on NIH (top) and COCO (bottom)
  fig5_same_write       same write, different reader: ownership of Gemma 3 4B/12B/27B and MedGemma 4B/27B
  fig6_ladders          size ladders: Qwen2.5-VL, Qwen3-VL (NIH), Lingshu (CheXpert)
Appendix
  figA1_write_structure  median write matrix per dataset (top) and own write vs strongest competitor per cell (bottom)
  figA2_examples         ten representative per-image examples (image, question, label, probe, clean / own / competitor answers)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fs                                   # noqa: E402
import cf_inclusion as ci                               # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm   # noqa: E402
from matplotlib.lines import Line2D                     # noqa: E402
from matplotlib.text import Text                        # noqa: E402
from matplotlib.ticker import MaxNLocator               # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle   # noqa: E402
from matplotlib.path import Path as MPath               # noqa: E402
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
                 "Nodule": fs.SLATE, "Consolidation": fs.MUSTARD, "Edema": fs.TERRACOTTA,
                 "person": fs.ROSE, "dog": fs.SAGE, "car": fs.HAZE, "chair": fs.OAT, "bottle": fs.LILAC, "bicycle": fs.MUSTARD}
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
ANNOT_SIZE = 6.3          # 6.5 pt leaves 0.4 pt between adjacent cell values at this cell size; 6.3 leaves 2.5 pt
TICK_SIZE = 8
HEAT_BAD = "#e9e7e3"


# ============================================================================================== data
def load_blocks():
    """The blocks with a scored write matrix under the paper's one inclusion rule (cf_inclusion.block_included:
    CORE and CALIBRATION in run.json completed_modules), in catalogue order. Blocks whose CORE is ineligible or
    incomplete are absent, so no figure can read their partial core statistics."""
    return {k: {"s": b["s"], "run": b["run"]} for k, b in ci.write_blocks(ci.load_runs(RUNS, order=ORDER)).items()}


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


def dose_blocks_expected():
    """The one inclusion rule for the dose panel and the coverage table: a block included by the manifest
    (CORE and CALIBRATION completed) whose completed_modules also contain DOSE."""
    done = {}
    for r in ci.read_manifest():
        done.setdefault((r["model_key"], r["dataset"]), set(m for m in r["completed_modules"].split("|") if m))
    return {f"{mk}/{ds}" for (mk, ds), b in ci.load_runs().items() if b["included"] and "DOSE" in done[(mk, ds)]}


def dose_cache():
    """runs/figures/dose_curves.json, gated on the same rule as Table cf-coverage: the cache must carry exactly the
    included blocks with DOSE, so Figure 2(c) and the coverage table can never disagree."""
    p = RUNS / "figures" / "dose_curves.json"
    curves = json.loads(p.read_text()) if p.exists() else {}
    curves = {k: v for k, v in curves.items() if v}
    expected = dose_blocks_expected()
    if set(curves) != expected:
        raise RuntimeError(f"{p}: dose curves differ from the included blocks with DOSE by {sorted(set(curves) ^ expected)}; "
                           f"delete the cache and rerun cftransfer.figures.dose_curves in the code repository")
    return curves


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
        for axis, xa in (("x", ax.xaxis), ("y", ax.yaxis)):
            if not ax.axison:
                break
            lo, hi = sorted(xa.get_view_interval())
            for tk in xa.get_major_ticks():
                t = tk.label1
                if t.get_visible() and t.get_text().strip() and lo - 1e-9 <= tk.get_loc() <= hi + 1e-9:
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
def macros():
    """The paper's number macros (tables/cf_numbers.json, written by build_numbers.py with tables/cf_numbers.tex). Figures
    that print a campaign number take it from here, as the macro's exact string, so figure and prose cannot disagree."""
    return json.loads((ROOT / "tables" / "cf_numbers.json").read_text())["macros"]


def element_boxes(fig, ax):
    """Display bboxes of the schematic's framed elements (boxes, tokens, circles) and every inset axes, plus sampled
    display points along every arrow and connector line of `ax`. Used by check_elements."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    boxes = [(p.get_label() or type(p).__name__, p.get_window_extent(r)) for p in ax.patches
             if not isinstance(p, FancyArrowPatch) and p.get_label() != "_bracket"]
    boxes += [(f"axes:{a.get_label()}", a.get_window_extent(r)) for a in fig.axes if a is not ax]
    lines = []
    for p in ax.patches:
        if isinstance(p, FancyArrowPatch):
            path = p.get_path()                                         # display coordinates, arrow head included
            pts = path.interpolated(40).vertices if len(path.vertices) > 1 else path.vertices
            lines.append((p.get_label() or "arrow", pts))
    for ln in ax.lines:
        xy = ax.transData.transform(np.column_stack(ln.get_data()))
        dense = np.vstack([np.linspace(xy[i], xy[i + 1], 40) for i in range(len(xy) - 1)]) if len(xy) > 1 else xy
        lines.append((ln.get_label() or "line", dense))
    return boxes, lines


def check_elements(fig, ax, name, pad_pt=0.8):
    """Schematic overlap check on top of check_overlaps: (1) no text crosses an arrow or connector line; (2) every text
    lies either wholly inside a framed element or wholly outside it (no text straddles a frame); (3) framed elements do
    not partially overlap each other (nesting is allowed). Prints each violation and a one-line verdict."""
    boxes, lines = element_boxes(fig, ax); texts = text_bboxes(fig); bad = 0
    m = pad_pt * fig.dpi / 72
    for lt, tb in texts:
        if ":tick" in lt or lt.endswith(":legend"):
            continue
        inner = Bbox([[tb.x0 + m, tb.y0 + m], [tb.x1 - m, tb.y1 - m]])
        for ll, pts in lines:
            if ((pts[:, 0] > inner.x0) & (pts[:, 0] < inner.x1) & (pts[:, 1] > inner.y0) & (pts[:, 1] < inner.y1)).any():
                print(f"  [{name}] TEXT x LINE: {lt}  x  {ll}"); bad += 1
        for lb, bb in boxes:
            if lt.startswith(lb.replace("axes:", "") + ":"):
                continue                                                # a text of the inset axes itself
            inside = tb.x0 >= bb.x0 - 0.5 and tb.x1 <= bb.x1 + 0.5 and tb.y0 >= bb.y0 - 0.5 and tb.y1 <= bb.y1 + 0.5
            ov = Bbox.intersection(inner, bb)
            if ov is not None and ov.width > 0 and ov.height > 0 and not inside:
                print(f"  [{name}] TEXT STRADDLES FRAME: {lt}  x  {lb}"); bad += 1
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (li, bi), (lj, bj) = boxes[i], boxes[j]
            ov = Bbox.intersection(bi, bj)
            if ov is None or ov.width <= 1 or ov.height <= 1:
                continue
            nested = (bi.x0 <= bj.x0 and bi.x1 >= bj.x1 and bi.y0 <= bj.y0 and bi.y1 >= bj.y1) or \
                     (bj.x0 <= bi.x0 and bj.x1 >= bi.x1 and bj.y0 <= bi.y0 and bj.y1 >= bi.y1)
            if not nested:
                print(f"  [{name}] FRAMES OVERLAP: {li}  x  {lj}"); bad += 1
    print(f"  [{name}] element check: {'clean' if bad == 0 else f'{bad} issue(s)'}")
    return bad


FIG1_IMAGE = "/rodata/azradonc_dev/m253405/cache/nih/images/00005446_000.png"     # the NIH row of Figure 3(a)


def fig1_method(blocks):
    """Figure 1: the method and the result in one view. (a) one image through the vision tower to the final visual block the
    language model consumes; the probe reads the concept from the tokens pooled over the consumed positions; the lifted
    direction is written back onto the same positions at a relative dose; the connector and language model answer one
    question. (b) the grade: the same write against its three comparisons. (c) the seed cell, every number a macro."""
    from PIL import Image
    Mx = macros()
    FW, FH = fs.WIDTH, 3.40
    f = plt.figure(figsize=(FW, FH))
    ax = f.add_axes((0, 0, 1, 1), label="schematic"); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.set_axis_off(); ax.grid(False)
    OWN, COMP, REF, OTHER = fs.HAZE, fs.TERRACOTTA, fs.SLATE, fs.STONE
    INKC = fs.INK

    def box(x, y, w, h, text, edge, label, fontsize=7.5, fill="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", facecolor=fill, edgecolor=edge,
                                    linewidth=0.9, zorder=2, label=label))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INKC, linespacing=1.2, zorder=4)

    def arrow(p, q, label, color=fs.CHARCOAL, lw=0.9, style="-|>", ms=7, connection=None):
        kw = dict(connectionstyle=connection) if connection else {}
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                                     zorder=3, label=label, **kw))

    # ------------------------------------------------------------------------------------------------ (a) pipeline
    ax.text(0.06, FH - 0.04, "(a)  read and write at the final visual block the language model consumes", ha="left", va="top",
            fontsize=8.5, color=INKC)
    yc = 2.30                                                           # centre line of the pipeline row
    # image
    im = Image.open(FIG1_IMAGE).convert("L"); iw, ih = im.size; S = 0.66
    axi = f.add_axes(rect(FW, FH, 0.08, yc - S / 2, S, S), label="image")
    axi.imshow(im, cmap="gray", interpolation="lanczos"); axi.set_xticks([]); axi.set_yticks([]); axi.grid(False)
    for sp in axi.spines.values():
        sp.set_edgecolor("#8a8a8a"); sp.set_linewidth(0.6)
    ax.text(0.08 + S / 2, yc - S / 2 - 0.05, "image $i$", ha="center", va="top", fontsize=7.5, color=fs.MUTED)
    # vision tower
    tx0, tw, th = 0.86, 0.50, 0.50
    arrow((0.08 + S + 0.02, yc), (tx0 - 0.02, yc), "a:img-tower")
    box(tx0, yc - th / 2, tw, th, "vision\ntower", fs.CHARCOAL, "tower")
    # token strips: block output h_it and written output h'_it, the same positions; position 0 is not consumed
    NT, TS, TG = 5, 0.13, 0.035; pitch = TS + TG; strip_w = NT * pitch - TG; T_MARK = 2
    s1 = tx0 + tw + 0.14
    xp = s1 + strip_w + 0.20                                            # the add node between the two strips
    s2 = xp + 0.20

    def strip(x0, written, tag):
        for k in range(NT):
            x = x0 + k * pitch
            consumed = k > 0
            face = "white" if not consumed else (tint(OWN, 0.45) if written else tint(REF, 0.35))
            ax.add_patch(Rectangle((x, yc - TS / 2), TS, TS, facecolor=face, edgecolor=fs.CHARCOAL if k == T_MARK else "#8a8a8a",
                                   linewidth=1.0 if k == T_MARK else 0.6, hatch=None if consumed else "//////", zorder=2,
                                   label=f"{tag}-tok{k}"))
        ax.text(x0 + T_MARK * pitch + TS / 2, yc - TS / 2 - 0.04, "$t$", ha="center", va="top", fontsize=7.5, color=INKC)

    arrow((tx0 + tw + 0.02, yc), (s1 - 0.02, yc), "a:tower-strip")
    strip(s1, False, "h")
    strip(s2, True, "hw")
    ax.text(s1 + strip_w / 2, yc - TS / 2 - 0.15, "$h_{it}$", ha="center", va="top", fontsize=8, color=INKC)
    ax.text(s2 + strip_w / 2, yc - TS / 2 - 0.15, "$h^{\\prime}_{it}$", ha="center", va="top", fontsize=8, color=INKC)
    ax.text((s1 + s2 + strip_w) / 2, yc - TS / 2 - 0.36, "same consumed positions $t$ (hatched: not consumed)", ha="center",
            va="top", fontsize=7.3, color=INKC)
    # add node
    R_ADD = 0.065
    arrow((s1 + strip_w + 0.02, yc), (xp - R_ADD - 0.01, yc), "a:strip-add")
    ax.add_patch(Circle((xp, yc), R_ADD, facecolor="white", edgecolor=fs.CHARCOAL, linewidth=0.9, zorder=2, label="add"))
    ax.plot([xp - 0.035, xp + 0.035], [yc, yc], color=fs.CHARCOAL, lw=0.9, zorder=3, label="_plus")
    ax.plot([xp, xp], [yc - 0.035, yc + 0.035], color=fs.CHARCOAL, lw=0.9, zorder=3, label="_plus")
    arrow((xp + R_ADD + 0.01, yc), (s2 - 0.02, yc), "a:add-strip")
    # read: bracket over the consumed tokens of h, up to the probe
    bx0, bx1, by = s1 + pitch - 0.01, s1 + strip_w + 0.01, yc + TS / 2 + 0.05
    ax.plot([bx0, bx0, bx1, bx1], [by, by + 0.04, by + 0.04, by], color=fs.CHARCOAL, lw=0.7, zorder=3, label="_bracket_line")
    bmid = (bx0 + bx1) / 2
    py0, ph, pw = 2.86, 0.28, 0.86
    arrow((bmid, by + 0.04), (bmid, py0 - 0.02), "a:pool-probe")
    ax.text(bmid - 0.05, (by + 0.04 + py0) / 2, "mean over\nconsumed $t$", ha="right", va="center", fontsize=7.3, color=INKC, linespacing=1.1)
    box(bmid - pw / 2, py0, pw, ph, "probe reads\nEffusion", OWN, "probe", fontsize=7.6)
    # lift and write: from the probe to the add node
    arrow((bmid + pw / 2 + 0.02, py0 + ph / 2), (xp, yc + R_ADD + 0.01), "a:lift-write", color=OWN, lw=1.1,
          connection="angle,angleA=0,angleB=90,rad=0")
    ax.text(bmid + pw / 2 + 0.10, py0 + ph / 2 + 0.035, "lift $\\hat w_c$", ha="left", va="bottom", fontsize=7.5, color=INKC)
    ax.text(xp + 0.07, yc + 0.40, "add $\\alpha\\,\\Vert h_{it}\\Vert\\,\\hat w_d$ to every consumed $t$, $\\alpha=+0.25$\n$d$: $c$, five clinical, 119 random, sham",
            ha="left", va="center", fontsize=7.3, color=INKC, linespacing=1.25)
    # connector, language model, answer
    cx0, cw = s2 + strip_w + 0.14, 0.56
    arrow((s2 + strip_w + 0.02, yc), (cx0 - 0.02, yc), "a:strip-conn")
    box(cx0, yc - th / 2, cw, th, "connector", fs.CHARCOAL, "connector", fontsize=7.6)
    lx0, lw_ = cx0 + cw + 0.12, 0.62
    arrow((cx0 + cw + 0.02, yc), (lx0 - 0.02, yc), "a:conn-lm")
    box(lx0, yc - th / 2, lw_, th, "language\nmodel", fs.CHARCOAL, "lm", fontsize=7.6)
    ax.text(lx0 + lw_ / 2, py0 + ph, "question $q$: Is there a pleural\neffusion in this chest radiograph?", ha="center", va="top",
            fontsize=7.3, color=INKC, linespacing=1.15)
    arrow((lx0 + lw_ / 2, py0 - 0.02), (lx0 + lw_ / 2, yc + th / 2 + 0.02), "a:q-lm")
    ox = lx0 + lw_ + 0.12
    arrow((lx0 + lw_ + 0.02, yc), (ox - 0.02, yc), "a:lm-out")
    ax.text(ox, yc, "$P(\\mathrm{yes}\\mid q)$", ha="left", va="center", fontsize=8, color=INKC)

    # ------------------------------------------------------------------------------------------------ (b) the grade
    W = {"Effusion": Mx["cfSeedWEff"], "Atelectasis": Mx["cfSeedWAtel"], "Pneumothorax": Mx["cfSeedWPneu"],
         "Cardiomegaly": Mx["cfSeedWCard"], "Mass": Mx["cfSeedWMass"], "Nodule": Mx["cfSeedWNodule"]}
    p95, sham = Mx["cfSeedEffRandP"], Mx["cfSeedEffSham"]
    O, Olo, Ohi = Mx["cfSeedEffO"], Mx["cfSeedEffOLo"], Mx["cfSeedEffOHi"]
    wq = float(W["Effusion"])
    c1 = wq > 0
    c2 = wq > float(p95) and wq > float(sham)
    c3 = float(Olo) > 0                                                 # every simultaneous lower bound positive
    top_b = 1.74
    ax.text(0.06, top_b, "(b)  the grade: one write, three comparisons", ha="left", va="top", fontsize=8.5, color=INKC)
    ax.text(0.06, top_b - 0.24, "$W_{q,d}$: mean change in $P(\\mathrm{yes}\\mid q)$ under the write of $d$", ha="left", va="top",
            fontsize=7.5, color=INKC)
    rows = [(OWN, "own effect", "$W_{q,q}>0$", c1),
            (REF, "random and sham", "$W_{q,q}>$ random p95 and $>|\\mathrm{sham}|$", c2),
            (COMP, "five clinical directions", "$W_{q,q}-W_{q,d}>0$ for every $d\\neq q$", c3)]
    y = top_b - 0.58
    for col, name, rule, ok in rows:
        ax.add_patch(Rectangle((0.10, y - 0.045), 0.09, 0.09, facecolor=col, edgecolor="none", zorder=2, label=f"key-{name}"))
        ax.text(0.26, y + 0.01, name, ha="left", va="bottom", fontsize=8, color=INKC)
        ax.text(0.26, y - 0.01, rule, ha="left", va="top", fontsize=7.8, color=INKC)
        f.text(2.42 / FW, y / FH, "✓" if ok else "✗", fontsize=6.5, color="white", ha="center", va="center", family="DejaVu Sans",
               fontweight="bold", bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.25", fc=fs.SAGE if ok else fs.TERRACOTTA, ec="none"), zorder=6)
        y -= 0.31
    verdict = "owned" if (c1 and c2 and c3) else ("stronger competitor" if float(Ohi) < 0 else "unresolved")
    ax.text(0.06, y + 0.10, f"owned: all three hold under simultaneous max-$T$ bounds\nseed cell: {verdict}", ha="left", va="top", fontsize=7.8,
            color=INKC, linespacing=1.2)

    # ------------------------------------------------------------------------------------------------ (c) the seed cell
    names = list(W); vals = np.array([float(W[c]) for c in names])
    AX0, AY0, AW, AH = 3.12, 0.36, 2.30, 1.04
    axc = f.add_axes(rect(FW, FH, AX0, AY0, AW, AH), label="seed")
    cols = [OWN if c == "Effusion" else COMP if c == "Nodule" else tint(COMP, 0.35) for c in names]
    xs = np.arange(len(names))
    axc.bar(xs, vals, width=0.62, color=cols, edgecolor="none", zorder=3)
    axc.set_xlim(-0.55, len(names) - 0.45); axc.set_ylim(0, 0.33)
    axc.set_yticks([0, 0.1, 0.2, 0.3]); axc.tick_params(labelsize=7.5); axc.grid(True, axis="y"); axc.grid(False, axis="x")
    axc.set_xticks(xs); axc.set_xticklabels([SHORT_CONCEPT[c] for c in names], fontsize=7.5)
    axc.set_ylabel("$W_{q,d}$, $q$ = Effusion", fontsize=7.8, labelpad=2)
    axc.set_xlabel("written direction $d$", fontsize=7.8, labelpad=2)
    axc.axhline(float(p95), color=REF, ls="--", lw=0.9, zorder=2)
    axc.axhline(float(sham), color=REF, ls=":", lw=1.0, zorder=2)
    axc.annotate(f"random p95 {p95}", (2.5, float(p95)), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                 fontsize=7.2, color=REF)
    axc.annotate(f"$|$sham$|$ {sham}", (2.5, float(sham)), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                 fontsize=7.2, color=REF)
    for c, col in (("Effusion", OWN), ("Nodule", COMP)):
        j = names.index(c)
        axc.annotate(W[c], (j, float(W[c])), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
                     fontweight="bold", color=INKC)
    axc.plot([0.31, len(names) - 1 + 0.42], [wq, wq], color=OWN, lw=0.8, ls=(0, (2, 1.5)), zorder=2)
    axc.annotate("", xy=(len(names) - 1 + 0.42, float(W["Nodule"])), xytext=(len(names) - 1 + 0.42, wq),
                 arrowprops=dict(arrowstyle="<->", lw=0.8, color=fs.CHARCOAL, shrinkA=0, shrinkB=0, mutation_scale=5))
    axc.text(len(names) - 1 - 0.45, float(W["Nodule"]), f"$O_q={O}$  $[{Olo},\\ {{{Ohi}}}]$", ha="right", va="center", fontsize=7.5, color=INKC,
             bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax.text(2.72, top_b, "(c)  seed cell: Effusion, Qwen2.5-VL-7B, NIH", ha="left", va="top", fontsize=8.5,
            color=INKC)
    check_overlaps(f, "fig1_method")
    check_elements(f, ax, "fig1_method")
    fs.save(f, FIG / "fig1_method")


# ============================================================================================== fig2
def grade_cells(blocks, ds):
    """Per concept cell of a dataset: (readable, answerable, owned) flags. Readable runs over every probe-graded cell
    (cf_inclusion.probe_graded: CALIBRATION completed, whether or not the write matrix was scored); answerable and
    owned run over the write-matrix cells of the included blocks only, the same cells as Table 1's Ans/Own counts."""
    read, ans, own = [], [], []
    for (mk, d), b in ci.probe_blocks(ci.load_runs(RUNS, order=ORDER)).items():
        if d == ds:
            read += [bool(v.get("readable")) for v in ci.calibration(b).values()]
    for (mk, d), b in blocks.items():
        if d != ds:
            continue
        cal = b["s"].get("calibration", {}); core = b["s"]["core"]["per_question"]
        for c, v in core.items():
            cc = cal.get(c, {}) if isinstance(cal.get(c, {}), dict) else {}
            ans.append(bool(cc.get("answer_capable"))); own.append(owned(v))
    return np.array(read), np.array(ans), np.array(own)


def boot_frac(flags, n=2000, seed=0):
    rng = np.random.default_rng(seed); flags = np.asarray(flags, dtype=float)
    draws = rng.choice(flags, size=(n, len(flags)), replace=True).mean(axis=1)
    return flags.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def fig2_overview(blocks):
    f, axes = plt.subplots(2, 2, figsize=(fs.WIDTH, 4.1), constrained_layout=True)
    for ax, lab in zip(axes.ravel(), "abcd"):
        ax.set_label(lab)
    # (a) graded fractions -------------------------------------------------------------------------------------
    ax = axes[0, 0]; grades = ["readable", "answerable", "owned"]; w = 0.26; counts = []; grade_counts = {}
    for k, ds in enumerate(DATASETS):
        read, ans, own = grade_cells(blocks, ds)
        for gi, flags in enumerate((read, ans, own)):
            grade_counts.setdefault(ds, {})[grades[gi]] = [int(flags.sum()), int(len(flags))]
            m, lo, hi = boot_frac(flags)
            x = gi + (k - 1) * w
            ax.bar(x, m, w - 0.03, color=DS_COL[ds], edgecolor="none", zorder=3)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="#555555", elinewidth=0.7, capsize=1.5, zorder=4)
            counts.append(ax.annotate(f"{int(flags.sum())}/{len(flags)}", (x, hi), xytext=(0, 2.5), textcoords="offset points",
                                      ha="center", va="bottom", fontsize=6, color=fs.INK, zorder=6))
    ax.set_xticks(range(3)); ax.set_xticklabels(grades); ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("fraction of concept cells"); ax.grid(False, axis="x")
    panel(ax, "a", "graded outcomes per concept cell")
    (FIG / "fig2_counts.json").write_text(json.dumps(grade_counts, indent=1) + "\n")
    print("  fig2(a) counts (readable over probe-graded cells; answerable / owned over write-matrix cells):")
    for ds, g in grade_counts.items():
        print(f"    {ds:8s} " + "  ".join(f"{k} {a}/{n}" for k, (a, n) in g.items()))

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
    (FIG / "fig2_dose_blocks.json").write_text(json.dumps({ds: ends[ds][2] for ds in ends}, indent=1) + "\n")
    print("  fig2(c) dose blocks: " + "  ".join(f"{ds} {ends[ds][2]}" for ds in ends))

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
    FW, FH = fs.WIDTH, 3.53          # 3.7 in until the round-2 prose; 0.12 in of top and 0.05 in of bottom padding removed
    f = plt.figure(figsize=(FW, FH))
    img_w, img_x = 1.02, 0.12; bar_x, bar_w = 1.96, 2.58; ax_h = 1.15
    row_y = [2.15, 0.63]; title_y = [3.36, 1.83]
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
    f.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.8, frameon=True, edgecolor="#8a8a8a",
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
    FW, FH = fs.WIDTH, 4.01; P = 1.15; gap = 0.26; x0 = 0.64      # 4.2 in until the round-2 prose; top and row-gap padding trimmed
    rows_y = [FH - 0.18 - P, 0.97]
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
    ceiling = set()
    pj = RUNS / "robustness" / "pairs.json"
    if pj.exists():
        ceiling = {(c["model"], c["dataset"], c["concept"]) for c in json.loads(pj.read_text())["task2_ceiling"]["cells"] if c.get("ceiling")}
    ye = np.array([0, 1, 2, 3, 3 + FAM_GAP, 4 + FAM_GAP, 5 + FAM_GAP]); real = [0, 1, 2, 4, 5]   # row 3 is the family spacer
    FW = fs.WIDTH; P = 1.36; gap = 0.20; x0 = 0.98; Ph = P * ye[-1] / 6
    FH = 0.18 + Ph + 0.34 + 0.34 + 0.26      # top padding 0.36 in until the round-2 prose
    f = plt.figure(figsize=(FW, FH))
    for k, ds in enumerate(DATASETS):
        concepts, M, own = Ms[ds]
        ax = f.add_axes(rect(FW, FH, x0 + k * (P + gap), FH - 0.18 - Ph, P, Ph), label=f"h{k}")
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
                if (FAM[i][0], ds, c) in ceiling:
                    ax.add_patch(Rectangle((xe[j], ye[ri]), xe[j + 1] - xe[j], ye[ri + 1] - ye[ri], fill=False, hatch="////",
                                           edgecolor="#8a8a8a", linewidth=0, zorder=4))
        ax.set_title(f"({'abc'[k]})  {DS_LAB[ds]}", fontsize=8.5, pad=4)
    cax = f.add_axes(rect(FW, FH, 1.55, 0.40, 2.4, 0.08), label="cbar")
    heat_colorbar(f, cax, norm, r"ownership $O_q$ of the shared write vector, by reader      ($\bullet$ owned;  hatched: clean answer at a ceiling)",
                  ticks=[-vlim, -vlim / 2, 0, vlim / 2, vlim])
    check_overlaps(f, "fig5_same_write")
    fs.save(f, FIG / "fig5_same_write")


# ============================================================================================== fig6: size ladders
LADDERS = [("Qwen2.5-VL, NIH", [("q25-3", 3, "3B"), ("q25-7", 7, "7B"), ("q25-32", 32, "32B"), ("q25-72", 72, "72B")], "nih"),
           ("Qwen3-VL, NIH", [("q3-4", 4, "4B"), ("q3-8", 8, "8B"), ("q3-32", 32, "32B")], "nih"),
           ("Lingshu, CheXpert", [("lingshu-7", 7, "7B"), ("lingshu-32", 32, "32B")], "chexpert")]
LADDER_MARKERS = ["o", "s", "^", "D", "v", "P"]
LEGEND_FRAME = dict(frameon=True, facecolor="white", edgecolor="#8a8a8a", fontsize=6.5, handlelength=2.0, borderpad=0.4, labelspacing=0.25, handletextpad=0.5)


def sampled_display_points(ax, lines, n=25):
    """Display coordinates of every marker and of n samples along every segment of the given lines."""
    pts = []
    for ln in lines:
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if ax.get_xscale() == "log":
            xs = np.concatenate([np.geomspace(x[i], x[i + 1], n) for i in range(len(x) - 1)]) if len(x) > 1 else x
            ys = np.concatenate([np.interp(np.log(np.geomspace(x[i], x[i + 1], n)), np.log(x[i:i + 2]), y[i:i + 2]) for i in range(len(x) - 1)]) if len(x) > 1 else y
        else:
            xs = np.concatenate([np.linspace(x[i], x[i + 1], n) for i in range(len(x) - 1)]) if len(x) > 1 else x
            ys = np.concatenate([np.linspace(y[i], y[i + 1], n) for i in range(len(x) - 1)]) if len(x) > 1 else y
        pts.append(ax.transData.transform(np.column_stack([xs, ys])))
    return np.vstack(pts) if pts else np.zeros((0, 2))


def bbox_hits(bb, pts, margin_pt, fig):
    m = margin_pt * fig.dpi / 72
    return ((pts[:, 0] > bb.x0 - m) & (pts[:, 0] < bb.x1 + m) & (pts[:, 1] > bb.y0 - m) & (pts[:, 1] < bb.y1 + m)).any()


def free_legend(fig, ax, handles, pts, locs=("lower right", "upper left", "lower left", "upper right"), **kw):
    """Framed legend at the first location whose frame (plus 3 pt) contains no data point; None if no corner is free."""
    for loc in locs:
        lg = ax.legend(handles=handles, loc=loc, **kw)
        fig.canvas.draw(); bb = lg.get_window_extent(fig.canvas.get_renderer())
        if not bbox_hits(bb, pts, 3, fig):
            return lg
        lg.remove()
    return None


def tint(color, strength=0.8):
    """Mix a colour with white: strength 1 = the colour itself, 0 = white (no alpha, so overlaps stay opaque)."""
    from matplotlib.colors import to_rgb, to_hex
    c = np.array(to_rgb(color)); return to_hex(strength * c + (1 - strength))


def fig6_ladders(blocks):
    f, axes = plt.subplots(1, 3, figsize=(fs.WIDTH, 2.55), constrained_layout=True, sharey=True)
    panels, emph_names = [], []
    for ax, (title, sizes, ds), letter in zip(axes, LADDERS, "abc"):
        ax.set_label(letter)
        sizes = [(m, b, l) for m, b, l in sizes if (m, ds) in blocks]
        concepts = list(blocks[(sizes[0][0], ds)]["s"]["core"]["per_question"]); xs = np.array([b for _, b, _ in sizes], float)
        O_last = O_of(blocks, sizes[-1][0], ds); emph = max(concepts, key=lambda c: O_last[c]); emph_names.append(emph)
        handles, data_lines, mk_iter = [], [], iter(["o", "s", "^", "v", "P"])
        order = [emph] + [c for c in concepts if c != emph]          # emphasised series drawn last (on top), listed first
        for c in reversed(order):
            ys = np.array([O_of(blocks, m, ds)[c] for m, _, _ in sizes]); ow = [c in owned_set(blocks[(m, ds)]["s"]) for m, _, _ in sizes]
            if c == emph:
                col, mk, lw, ms = CONCEPT_COLOR[c], "D", 1.7, 4.2
            else:
                col, mk, lw, ms = tint(CONCEPT_COLOR[c], 0.8), next(mk_iter), 0.9, 2.6
            data_lines += ax.plot(xs, ys, color=col, ls=(0, (4, 2)), lw=lw, zorder=3 if c != emph else 5)
            for x, y, o in zip(xs, ys, ow):
                data_lines += ax.plot(x, y, marker=mk, ms=ms, mfc=col, mec=fs.CHARCOAL if o else col, mew=0.7 if o else 0.5, ls="none", zorder=4 if c != emph else 6)
            handles.append(Line2D([], [], color=col, ls=(0, (4, 2)), lw=lw, marker=mk, ms=ms, mfc=col, mec=col, label=c))
        handles = handles[::-1]
        ax.set_xscale("log", base=2); ax.set_xlim(xs[0] / 1.45, xs[-1] * 1.8)
        ax.set_xticks(xs); ax.set_xticklabels([l for _, _, l in sizes]); ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(labelsize=8)
        ax.grid(True, which="major", axis="both", ls=":", lw=0.5, color="#8a8a8a", alpha=0.6)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor("#8a8a8a"); sp.set_linewidth(0.7)
        ax.set_title(f"({letter})  {title}", fontsize=9.5, pad=5); ax.set_xlabel("model size", fontsize=8.5)
        panels.append((ax, handles, data_lines))
    axes[0].set_ylim(-0.6, 0.65); axes[0].set_yticks([-0.6, -0.3, 0, 0.3, 0.6]); axes[0].set_ylabel("ownership $O_q$  (ringed: owned)", fontsize=8.5)
    for ax, _, _ in panels:
        ax.axhline(0, color="#9a9a9a", ls="--", lw=0.9, zorder=2)
    f.canvas.draw()
    locs = ("upper left", "upper right", "lower right", "lower left")
    chosen = None
    for variant in (dict(fontsize=6.5), dict(fontsize=6.0), dict(fontsize=6.0, ncol=2)):    # one variant for all panels, for a uniform look
        lgs = []
        for ax, handles, data_lines in panels:
            pts = sampled_display_points(ax, data_lines)
            lgs.append(free_legend(f, ax, handles, pts, locs=locs, **{**LEGEND_FRAME, **variant}))
        if all(lg is not None for lg in lgs):
            chosen = variant; break
        for lg in lgs:
            if lg is not None:
                lg.remove()
    if chosen is None:
        allh = panels[0][1] + [h for h in panels[2][1] if h.get_label() not in {h0.get_label() for h0 in panels[0][1]}]
        f.set_size_inches(fs.WIDTH, 2.85)
        f.legend(handles=allh, loc="outside lower center", ncol=4, **{**LEGEND_FRAME, "columnspacing": 1.2})
        print("  fig6: no free corner in some panel -> one shared legend below")
    else:
        for lg in lgs:
            lg.get_frame().set_linewidth(0.6)
        print("  fig6: legends inside every panel:", chosen, [(ax.get_label(), lg._loc) for (ax, _, _), lg in zip(panels, lgs)], "emphasised:", emph_names)
    check_overlaps(f, "fig6_ladders")
    fs.save(f, FIG / "fig6_ladders")


# ============================================================================================== figA1: write structure
FAMILY = [("q25", "Qwen2.5-VL", fs.ROSE), ("q3", "Qwen3-VL", fs.SAGE), ("iv35", "InternVL3.5", fs.HAZE), ("gemma3", "Gemma 3", fs.OAT),
          ("medgemma", "MedGemma", fs.LILAC), ("lingshu", "Lingshu", fs.MUSTARD), ("llava15", "LLaVA-1.5", fs.TERRACOTTA),
          ("llavamed", "LLaVA-Med", fs.SLATE), ("llama32", "Llama 3.2", "#7F9C9A")]


def family_of(mk):
    for pre, name, col in FAMILY:
        if mk.split("-")[0] == pre:
            return name, col
    return mk, fs.GREY


def place_note(fig, ax, text, candidates, pts, **kw):
    """Corner note at the candidate (axes fraction) with the fewest data points within its box (+4 pt)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); best = None
    for (x, y, ha, va) in candidates:
        t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, **kw); bb = t.get_window_extent(r)
        m = 4 * fig.dpi / 72
        n = int(((pts[:, 0] > bb.x0 - m) & (pts[:, 0] < bb.x1 + m) & (pts[:, 1] > bb.y0 - m) & (pts[:, 1] < bb.y1 + m)).sum())
        t.remove()
        if best is None or n < best[0]:
            best = (n, x, y, ha, va)
    n, x, y, ha, va = best
    if n:
        print(f"  WARNING: note {text!r} overlaps {n} point(s) at its best position")
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, **kw)


def grid_note(fig, ax, text, pts, above, n=6, **kw):
    """Region note placed by search over an n x n grid of axes-fraction positions restricted to one half-plane of the
    identity line (equal limits, so fy > fx is 'above'); score = min distance (points) from the text box to any data
    point and to the axes edges; the best-scoring position wins."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); axbb = ax.get_window_extent(r)
    probe = ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", **kw); tb = probe.get_window_extent(r); probe.remove()
    w, h = tb.width, tb.height; best = None
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            fx, fy = i / (n + 1), j / (n + 1)
            if (fy > fx) != above:
                continue
            cx, cy = axbb.x0 + fx * axbb.width, axbb.y0 + fy * axbb.height
            x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
            edge = min(x0 - axbb.x0, axbb.x1 - x1, y0 - axbb.y0, axbb.y1 - y1)
            # identity line in display space runs corner to corner; keep the box on its side: distance of the nearest corner
            corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
            fc = (corners - [axbb.x0, axbb.y0]) / [axbb.width, axbb.height]
            side = (fc[:, 1] - fc[:, 0]) * (1 if above else -1)
            line_d = side.min() * min(axbb.width, axbb.height) / np.sqrt(2)
            if len(pts):
                dx = np.maximum(0, np.maximum(x0 - pts[:, 0], pts[:, 0] - x1)); dy = np.maximum(0, np.maximum(y0 - pts[:, 1], pts[:, 1] - y1))
                pd = float(np.sqrt(dx ** 2 + dy ** 2).min())
            else:
                pd = 1e9
            score = min(pd, edge, line_d) * 72 / fig.dpi
            if best is None or score > best[0]:
                best = (score, fx, fy)
    score, fx, fy = best
    if score < 2:
        print(f"  WARNING: note {text!r} placed with only {score:.1f} pt clearance")
    return ax.text(fx, fy, text, transform=ax.transAxes, ha="center", va="center", **kw)


def table1_order(blocks):
    """Checkpoints in the order of Table 1: clinical owned share desc., then COCO owned share desc., then catalogue order."""
    def rate(mk, dss):
        cells = [(c, v) for ds in dss if (mk, ds) in blocks for c, v in blocks[(mk, ds)]["s"]["core"]["per_question"].items()]
        return sum(owned(v) for _, v in cells) / len(cells) if cells else -1
    mks = [m for m in ORDER if any((m, d) in blocks for d in DATASETS)]
    return sorted(mks, key=lambda m: (-rate(m, ("nih", "chexpert")), -rate(m, ("coco",)), ORDER.index(m)))


def figA1_write_structure(blocks):
    FW, FH = fs.WIDTH, 5.75; P = 1.23; gap = 0.5; x0 = 0.55
    f = plt.figure(figsize=(FW, FH))
    # ---- top row: median write matrix per dataset
    meds = {}
    for ds in DATASETS:
        keys = [(mk, d) for (mk, d) in blocks if d == ds]
        concepts = list(blocks[keys[0]]["s"]["core"]["per_question"])
        Ws = np.stack([wmatrix(blocks[k]["s"])[1] for k in keys])
        meds[ds] = (concepts, np.nanmedian(Ws, axis=0), len(keys))
    vlim = round_up(np.nanmax([np.nanmax(np.abs(M)) for _, M, _ in meds.values()])); norm = sym_norm(vlim)
    y_top = FH - 0.42 - P
    for k, ds in enumerate(DATASETS):
        concepts, M, n = meds[ds]
        ax = f.add_axes(rect(FW, FH, x0 + k * (P + gap), y_top, P, P), label=f"med{k}")
        xe, ye = heat(ax, M, norm); lab = [SHORT_CONCEPT.get(c, c) for c in concepts]
        heat_ticks(ax, xe, ye, lab, lab)
        for i in range(len(concepts)):
            heat_cell_marks(ax, xe, ye, i, i, outline=True)
            for j in range(len(concepts)):
                v = M[i, j]
                col = "white" if lum(DIV(norm(v))) < 0.5 else fs.INK
                txt = "0.00" if abs(v) < 0.005 else f"{v:+.2f}"          # no signed zero
                ax.text((xe[j] + xe[j + 1]) / 2, (ye[i] + ye[i + 1]) / 2, txt, ha="center", va="center", fontsize=ANNOT_SIZE, color=col, zorder=6)
        ax.set_title(f"({'abc'[k]})  {DS_LAB[ds].replace(' (control)', '')}\nmedian $W_{{q,d}}$ over {n} checkpoints", fontsize=8, pad=4, linespacing=1.2)
        if k == 0:
            ax.set_ylabel("question $q$", fontsize=8, labelpad=3)
    cax = f.add_axes(rect(FW, FH, 1.55, y_top - 0.50, 2.4, 0.07), label="cbar")
    heat_colorbar(f, cax, norm, "median $W_{q,d}$: change in $P(\\mathrm{yes}\\mid q)$ under the write of $d$ (column);  outlined: diagonal $d=q$",
                  ticks=[-vlim, -vlim / 2, 0, vlim / 2, vlim])
    # ---- bottom: write-flow band charts ("where does the write go"), one per dataset
    BAR_W_IN, GAP_FRAC, PH, PW = 0.05, 0.04, 2.3, 1.72
    xl = 0.10; pgap = (FW - 2 * xl - 3 * PW) / 2; y0 = 0.36
    summary = {}
    for k, ds in enumerate(DATASETS):
        keys = [(mk, d) for (mk, d) in blocks if d == ds]
        concepts = list(blocks[keys[0]]["s"]["core"]["per_question"]); n = len(concepts)
        Pm = np.nanmean(np.stack([np.maximum(wmatrix(blocks[kk]["s"])[1], 0.0) for kk in keys]), axis=0)   # P[q, d] = mean over checkpoints of max(W_qd, 0)
        total = float(Pm.sum()); own_share = float(np.trace(Pm) / total)
        summary[ds] = dict(total=total, diag=float(np.trace(Pm)), own_share=own_share, n=len(keys), outflow={c: float(Pm[:, j_].sum()) for j_, c in enumerate(concepts)},
                           inflow={c: float(Pm[i_, :].sum()) for i_, c in enumerate(concepts)})
        ax = f.add_axes(rect(FW, FH, xl + k * (PW + pgap), y0, PW, PH), label=f"flow{k}")
        ax.set_xlim(-0.45, 1.45); ax.set_ylim(-0.11, 1.09); ax.set_axis_off(); ax.grid(False)
        bar_w = BAR_W_IN / (PW / 1.9)                        # 0.05 in expressed in data units
        usable = 1.0 - GAP_FRAC * (n - 1)
        out_h = {c: usable * Pm[:, j_].sum() / total for j_, c in enumerate(concepts)}
        in_h = {c: usable * Pm[i_, :].sum() / total for i_, c in enumerate(concepts)}
        # node spans, top to bottom in concept order
        left, right, yt = {}, {}, 1.0
        for c in concepts:
            left[c] = (yt - out_h[c], yt); yt = yt - out_h[c] - GAP_FRAC
        yt = 1.0
        for c in concepts:
            right[c] = (yt - in_h[c], yt); yt = yt - in_h[c] - GAP_FRAC
        # band sub-spans: on the writer d in question order, on the question q in writer order
        lcur = {c: left[c][1] for c in concepts}; rcur = {c: right[c][1] for c in concepts}
        bands = []
        for j_, d in enumerate(concepts):
            for i_, q in enumerate(concepts):
                w = usable * Pm[i_, j_] / total
                if w <= 0:
                    continue
                yl1, yl0 = lcur[d], lcur[d] - w; lcur[d] = yl0
                bands.append((d, q, yl0, yl1))
        rb = {}
        for j_, d in enumerate(concepts):
            for i_, q in enumerate(concepts):
                w = usable * Pm[i_, j_] / total
                if w <= 0:
                    continue
                yr1, yr0 = rcur[q], rcur[q] - w; rcur[q] = yr0
                rb[(d, q)] = (yr0, yr1)
        for d, q, yl0, yl1 in bands:
            yr0, yr1 = rb[(d, q)]
            verts = [(0, yl1), (0.5, yl1), (0.5, yr1), (1, yr1), (1, yr0), (0.5, yr0), (0.5, yl0), (0, yl0), (0, yl1)]
            codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
            own = d == q
            ax.add_patch(PathPatch(MPath(verts, codes), facecolor=CONCEPT_COLOR.get(d, fs.SLATE) if own else "#c9c6c0", edgecolor="none",
                                   alpha=0.9 if own else 0.75, zorder=3 if own else 2))
        for c in concepts:
            ax.add_patch(Rectangle((-bar_w, left[c][0]), bar_w, left[c][1] - left[c][0], facecolor=CONCEPT_COLOR.get(c, fs.SLATE), edgecolor="none", zorder=4))
            ax.add_patch(Rectangle((1, right[c][0]), bar_w, right[c][1] - right[c][0], facecolor=fs.CHARCOAL, edgecolor="none", zorder=4))
        # node labels, repelled vertically where nodes are thin
        step = 9.0 / (PH * 72 / 1.11)          # 9 pt in data units (axes spans 1.11 data units over PH inches)
        for side, spans, x, ha in (("L", left, -bar_w - 0.05, "right"), ("R", right, 1 + bar_w + 0.05, "left")):
            cs = list(concepts); ys = [(spans[c][0] + spans[c][1]) / 2 for c in cs]
            ly = repel(ys, step, 0.0, 1.0)
            for c, y_, y_lab in zip(cs, ys, ly):
                ax.text(x, y_lab, SHORT_CONCEPT.get(c, c), ha=ha, va="center", fontsize=7, color=fs.INK)
        ax.text(0.5, -0.075, f"own-question share of the write: {100 * own_share:.0f}%", ha="center", va="center", fontsize=7.5, color=fs.INK)
        ax.text(-bar_w / 2, 1.025, "write $d$", ha="center", va="bottom", fontsize=7, color=fs.MUTED)
        ax.text(1 + bar_w / 2, 1.025, "question $q$", ha="center", va="bottom", fontsize=7, color=fs.MUTED)
        ax.set_title(f"({'def'[k]})  {DS_LAB[ds].replace(' (control)', '')}\nwhere the write goes", fontsize=8, pad=3, linespacing=1.2)
    f.text(0.5, 0.16 / FH, "coloured: write lands on its own question;   grey: write lands on another question", ha="center", va="center", fontsize=7, color=fs.INK)
    (FIG / "figA1_own_share.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("  figA1 write flow (mean over checkpoints of max(W_qd, 0)):")
    for ds, v in summary.items():
        print(f"    {ds:8s} n={v['n']} total={v['total']:.3f} diagonal={v['diag']:.3f} own_share={100 * v['own_share']:.1f}%")
        print("      outflow by written direction:", {c: round(x, 3) for c, x in v["outflow"].items()})
        print("      inflow by question:          ", {c: round(x, 3) for c, x in v["inflow"].items()})
    check_overlaps(f, "figA1_write_structure")
    fs.save(f, FIG / "figA1_write_structure")


# ============================================================================================== figA2: per-image examples
EXAMPLE_SPEC = [("q25-7", "nih", "Effusion", "deficit"), ("q25-7", "nih", "Atelectasis", "deficit"), ("q25-7", "chexpert", "Effusion", "deficit"),
                ("q25-7", "chexpert", "Consolidation", "deficit"), ("lingshu-32", "chexpert", "Edema", "owned"), ("q25-72", "nih", "Atelectasis", "owned"),
                ("q25-7", "coco", "dog", "coco"), ("q25-7", "coco", "chair", "coco"), ("lingshu-32", "coco", "person", "coco"), ("medgemma-4", "coco", "car", "coco")]
DATA_ROOT = {"nih": Path("/rodata/azradonc_dev/m253405/cache/nih"), "chexpert": Path("/rodata/azradonc_dev/m253405/cf-transfer/data/chexpert/images"),
             "coco": Path("/rodata/azradonc_dev/m253405/cf-transfer/data/coco")}


def norm_row_id(ds, rid):
    return str(int(rid)).zfill(12) if ds == "coco" else str(rid)


def block_rows(mk, ds, concept):
    """Per test row of a block and question concept: label, probe probability, clean p, p under every concept write."""
    import pandas as pd
    import pyarrow.parquet as pq
    d = RUNS / mk / ds
    lab = pd.read_csv(d / "manifests" / "labels.csv"); lab["row_id"] = [norm_row_id(ds, r) for r in lab.row_id]
    lab = lab[lab.concept == concept].set_index("row_id")["label"]
    coh = pd.read_csv(d / "manifests" / "cohort.csv"); coh["row_id"] = [norm_row_id(ds, r) for r in coh.row_id]
    test = coh[coh.role == "test"].set_index("row_id")["relative_image_path"]
    ps = pq.read_table(d / "probe_scores.vis.last.parquet", filters=[("probe_kind", "=", "real"), ("fit_seed", "=", 0), ("concept", "=", concept), ("role", "=", "test")]).to_pandas()
    ps = ps.set_index("row_id")["probability"]
    core = pq.read_table(d / "outcomes" / "CORE.parquet", columns=["row_id", "direction_id", "direction_kind", "p_present"],
                         filters=[("template_id", "=", "IY"), ("fit_seed", "=", 0), ("concept", "=", concept), ("sample_status", "=", "OK")]).to_pandas()
    core = core[core.direction_kind.isin(["baseline", "concept"])]
    concepts = list(json.loads((d / "summary.json").read_text())["core"]["per_question"])
    piv = core.pivot_table(index="row_id", columns="direction_id", values="p_present", aggfunc="first")
    rows = []
    for r in test.index:
        if r not in piv.index or r not in ps.index or r not in lab.index or not np.isfinite(lab[r]):
            continue
        pr = piv.loc[r]
        if "baseline" not in pr or any(f"concept:{c}" not in pr or not np.isfinite(pr[f"concept:{c}"]) for c in concepts):
            continue
        p0 = float(pr["baseline"]); pw = {c: float(pr[f"concept:{c}"]) for c in concepts}
        comp = max((c for c in concepts if c != concept), key=lambda c: pw[c] - p0)
        rows.append(dict(row_id=r, label=int(lab[r]), probe=float(ps[r]), p_clean=p0, p_own=pw[concept], comp=comp, p_comp=pw[comp], image=DATA_ROOT[ds] / test[r]))
    question = json.loads((d / "prompts.json").read_text())[f"{ds}|{concept}|IY"]["question"]
    return rows, concepts, question


def pick_example(mk, ds, concept, kind, used_rows=(), used_concepts=()):
    """Representative example for one slot (see EXAMPLE_SPEC), delta rule.
    Pool M: chest -> label present, probe > 0.5, clean p < 0.5; COCO -> label present with probe > 0.5 or absent with probe < 0.5, clean p < 0.5.
    d_own = own-write p - clean p; d_comp = strongest-competitor p - clean p (competitor = concept whose write raises the target most on that row).
    deficit slots: rows with d_comp > d_own; owned / coco slots: rows with d_own > d_comp; the shown row needs max(d_own, d_comp) >= 0.2
    (floor relaxed to 0.1, then 0 if the slot is empty, reported); among candidates the row whose |d_comp - d_own| is nearest the median.
    'count' = rows of M with the slot's ordering (no floor); shown as the representativeness line."""
    rows, concepts, question = block_rows(mk, ds, concept)
    O = O_of(blocks_global, mk, ds)
    order = [concept] + sorted((c for c in concepts if c != concept and c not in used_concepts), key=lambda c: abs(O[c] - O[concept]))
    levels = []
    for c_used in order:
        if c_used != concept:
            rows, concepts, question = block_rows(mk, ds, c_used)
        if kind == "coco":
            pool = [x for x in rows if x["p_clean"] < 0.5 and ((x["label"] == 1 and x["probe"] > 0.5) or (x["label"] == 0 and x["probe"] < 0.5))]
        else:
            pool = [x for x in rows if x["label"] == 1 and x["probe"] > 0.5 and x["p_clean"] < 0.5]
        for x in pool:
            x["d_own"] = x["p_own"] - x["p_clean"]; x["d_comp"] = x["p_comp"] - x["p_clean"]
        direction = (lambda x: x["d_comp"] > x["d_own"]) if kind == "deficit" else (lambda x: x["d_own"] > x["d_comp"])
        ordered = [x for x in pool if direction(x)]
        for floor in (0.2, 0.1, 0.0):
            cands = [x for x in ordered if max(x["d_own"], x["d_comp"]) >= floor and x["row_id"] not in used_rows]
            if kind != "deficit":                      # owned / COCO slots: show a row whose competitor answer stays No, when any exists
                below = [x for x in cands if x["p_comp"] < 0.5]
                cands = below or cands
            levels.append(f"{c_used} floor={floor}: M={len(pool)} ordered={len(ordered)} candidates={len(cands)}")
            if not cands:
                continue
            gaps = [abs(x["d_comp"] - x["d_own"]) for x in cands]; med = float(np.median(gaps))
            best = dict(min(cands, key=lambda x: (abs(abs(x["d_comp"] - x["d_own"]) - med), x["row_id"])))
            best.update(mk=mk, ds=ds, concept=c_used, kind=kind, question=question, label="present" if best["label"] == 1 else "absent",
                        M=len(pool), count=len(ordered), median_gap=med, floor=floor, relaxed=(c_used != concept or floor != 0.2), requested=concept, levels=levels)
            return best
    raise RuntimeError(f"no example for {mk}/{ds}/{concept}: " + "; ".join(levels))


def wrap_to_width(fig, text, x_in, max_w_in, fontsize, **kw):
    """Wrap `text` so that every rendered line is narrower than max_w_in; returns the lines."""
    import textwrap
    r = fig.canvas.get_renderer()
    for width in range(48, 18, -2):
        lines = textwrap.wrap(text, width)
        ok = True
        for ln in lines:
            t = fig.text(0, 0, ln, fontsize=fontsize, **kw); w = t.get_window_extent(r).width / fig.dpi; t.remove()
            if w > max_w_in:
                ok = False; break
        if ok:
            return lines
    return textwrap.wrap(text, 18)


def mark(fig, x_in, y_in, ok, FW, FH):
    """Small rounded square with a white check or cross glyph (DejaVu Sans carries both glyphs)."""
    fig.text(x_in / FW, y_in / FH, "✓" if ok else "✗", fontsize=5.6, color="white", ha="center", va="center", family="DejaVu Sans",
             fontweight="bold", bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.25", fc=fs.SAGE if ok else fs.TERRACOTTA, ec="none"), zorder=6)


def figA2_examples(blocks):
    from PIL import Image
    FW, FH = fs.WIDTH, 7.35; IMG = 1.05; col_x = [0.15, 2.90]; text_dx = 0.13; text_w = 2.75 - 0.15 - IMG - text_dx - 0.06
    row_pitch = 1.40; top = FH - 0.22; line_h = 0.118      # 7 pt at 1.2 line spacing
    f = plt.figure(figsize=(FW, FH)); f.canvas.draw()
    report = []
    global blocks_global; blocks_global = blocks
    used_rows, used_concepts = set(), {}
    for k, (mk, ds, concept, kind) in enumerate(EXAMPLE_SPEC):
        if (mk, ds) not in blocks:
            print(f"  WARNING: block {mk}/{ds} missing; example skipped"); continue
        ex = pick_example(mk, ds, concept, kind, used_rows, used_concepts.get((mk, ds), set())); report.append(ex); concept = ex["concept"]
        used_rows.add(ex["row_id"]); used_concepts.setdefault((mk, ds), set()).add(concept)
        r, c = divmod(k, 2); x0 = col_x[c]; y_top = top - r * row_pitch
        im = Image.open(ex["image"]); im = im.convert("L") if ds != "coco" else im.convert("RGB"); iw, ih = im.size
        axi = f.add_axes(rect(FW, FH, x0, y_top - IMG, IMG, IMG), label=f"img{k}")     # fixed square box: longer side fits, shorter side centred
        sc = 1.0 / max(iw, ih); w_, h_ = iw * sc, ih * sc
        axi.imshow(im, cmap="gray" if ds != "coco" else None, interpolation="lanczos", extent=((1 - w_) / 2, (1 + w_) / 2, (1 + h_) / 2, (1 - h_) / 2))
        axi.set_xlim(0, 1); axi.set_ylim(1, 0); axi.set_aspect("equal"); axi.set_xticks([]); axi.set_yticks([]); axi.grid(False); axi.set_facecolor("white")
        for sp in axi.spines.values():
            sp.set_edgecolor("#8a8a8a"); sp.set_linewidth(0.6)
        f.text((x0 + IMG / 2) / FW, (y_top - IMG - 0.05) / FH, f"{DS_LAB[ds].replace(' (control)', '')}\n{NAMES[mk]}", ha="center", va="top", fontsize=6.5, color=fs.MUTED, linespacing=1.15)
        # text block
        tx = x0 + IMG + text_dx; y = y_top - 0.04
        qlines = wrap_to_width(f, ex["question"], tx, text_w, 7, style="italic")
        if len(qlines) > 3:
            print(f"  WARNING: question of example {k + 1} needs {len(qlines)} lines")
        lines = [(ln, dict(style="italic"), None) for ln in qlines]
        probe_ok = (ex["probe"] > 0.5) == (ex["label"] == "present")
        own_ok = ex["d_own"] > ex["d_comp"]
        yn = lambda p: "Yes" if p > 0.5 else "No"
        who = "competing write" if ex["kind"] == "deficit" else "own write"
        lines += [(f"label: {concept} present" if ex["label"] == "present" else f"label: no {concept}", {}, None), (f"probe reads {ex['probe']:.2f}", {}, probe_ok),
                  (f"clean: {yn(ex['p_clean'])} ({ex['p_clean']:.2f})", {}, None),
                  (f"{concept} write: {yn(ex['p_own'])} ({ex['p_own']:.2f})", {}, own_ok),
                  (f"{ex['comp']} write: {yn(ex['p_comp'])} ({ex['p_comp']:.2f})", {}, None),
                 ]
        lines += [(ln, dict(fontsize=6.5, color=fs.MUTED), None) for ln in wrap_to_width(f, f"{who} moves it more on {ex['count']} of {ex['M']} such rows", tx, text_w, 6.5)]
        for txt, kw, ok in lines:
            t = f.text(tx / FW, y / FH, txt, ha="left", va="top", **{"fontsize": 7, "color": fs.INK, **kw})
            if ok is not None:
                bb = t.get_window_extent(f.canvas.get_renderer())
                mark(f, bb.x1 / f.dpi + 0.09, y - 0.045, ok, FW, FH)
            y -= line_h
        ex["own_ok"] = own_ok; ex["probe_ok"] = probe_ok
    check_overlaps(f, "figA2_examples")
    fs.save(f, FIG / "figA2_examples")
    print("  figA2 examples (delta rule):")
    for ex in report:
        print(f"    {ex['mk']:>10s} {ex['ds']:8s} {ex['row_id']:36s} {ex['concept']:13s} [{ex['kind']}] label={ex['label']} probe={ex['probe']:.2f}{'✓' if ex['probe_ok'] else '✗'} "
              f"clean={ex['p_clean']:.2f} own={ex['p_own']:.2f}{'✓' if ex['own_ok'] else '✗'} comp={ex['comp']} {ex['p_comp']:.2f} "
              f"d_own={ex['d_own']:+.3f} d_comp={ex['d_comp']:+.3f} count/M={ex['count']}/{ex['M']} median_gap={ex['median_gap']:.3f} floor={ex['floor']}"
              f"{' RELAXED from ' + ex['requested'] if ex['relaxed'] else ''}")
        if ex["relaxed"]:
            for lv in ex["levels"]:
                print("        tried", lv)
    return report


# ============================================================================================== main
KEEP = {"fig1_method", "fig2_overview", "fig3_example", "fig4_write_matrices", "fig5_same_write", "fig6_ladders",
        "figA1_write_structure", "figA2_examples"}


def main(only=None):
    fs.use_house_style()
    blocks = load_blocks()
    jobs = {"fig1_method": lambda: fig1_method(blocks), "fig2_overview": lambda: fig2_overview(blocks),
            "fig3_example": lambda: fig3_example(blocks), "fig4_write_matrices": lambda: fig4_write_matrices(blocks),
            "fig5_same_write": lambda: fig5_same_write(blocks), "fig6_ladders": lambda: fig6_ladders(blocks),
            "figA1_write_structure": lambda: figA1_write_structure(blocks), "figA2_examples": lambda: figA2_examples(blocks)}
    # matrices_all (one matrix per checkpoint) is kept for reference but not part of the paper's figure set
    for name, fn in jobs.items():
        if only and name not in only:
            continue
        fn(); print("ok", name)
    for p in FIG.iterdir():
        if p.suffix in (".pdf", ".png", ".svg") and p.stem not in KEEP:
            p.unlink(); print("removed", p.name)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
