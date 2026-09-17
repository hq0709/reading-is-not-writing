"""Build the cf-transfer-v1 LaTeX tables for the paper from the packaged campaign results.

Inputs (read-only): the campaign run root (leaderboard.csv, runs/<model>/<dataset>/{summary.json, run.json,
preflight.vis.last.json, template_eligibility.json}). Numbers are copied, never recomputed; the only derived
quantities are means/medians over cells and the pass fractions that the captions define.

Inclusion (one rule, shared with every other paper script through cf_inclusion.py and with runs/manifest.csv):
a block enters the write-matrix counts (Ans, Own, ownership tables, controls) only if its run.json lists CORE and
CALIBRATION in completed_modules; a block whose CALIBRATION completed but whose CORE is ineligible or incomplete is
probe-graded only and contributes to the Read column alone.

No table is scaled with \\resizebox: a table that does not fit the text width at \\scriptsize loses columns or is split.

Outputs (tables/):
  cf_colors.tex            Morandi tints used by \\cellcolor (input by the tables)
  table_cf_main.tex        Table: checkpoint x dataset benchmark with heat-mapped cells (sorted by chest ownership)
  table_cf_controls.tex    Table: reference family and controls per dataset (median, IQR over blocks)
  table_cf_seed.tex        Table: Qwen2.5-VL-7B on NIH per concept (seed replication)
  table_seed_mass.tex      Appendix: the seed study's Mass competition, discovery and confirmation (from the registered tables)
  table_cf_own_{nih,chexpert,coco}.tex   Appendix: compact ownership tables, checkpoint x concept, O_q per cell
  table_cf_models.tex      Appendix: checkpoints and families (checkpoint sheet under family rows with the family-level grades)
  table_cf_gates.tex       Appendix: the seven preflight gates and their observed ranges
  table_cf_coverage.tex    Appendix: modules completed per block (from runs/manifest.csv), primary template, ineligible modules
  table_cf_leaderboard.tex thin wrapper that inputs table_cf_main.tex (kept for the existing \\input)
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cf_inclusion as ci   # noqa: E402

RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ci.RUNS
OUT = HERE.parent / "tables"
DATASETS = [("nih", "NIH ChestX-ray14"), ("chexpert", "CheXpert Plus"), ("coco", "COCO")]
CHEST = ("nih", "chexpert")
TEMPLATES = ["IY", "WY", "IA", "IB", "WA", "WB"]
CHECK = "$\\checkmark$"

# checkpoint metadata: display name, family, params (B), vision tower, consumed tokens at the primary locus,
# image size, medical fine-tune
META = {
    "q25-3":       ("Qwen2.5-VL-3B",        "Qwen2.5-VL",      3,  "native ViT 675M",    "576$\\to$144", "$336^2$",  False),
    "q25-7":       ("Qwen2.5-VL-7B",        "Qwen2.5-VL",      7,  "native ViT 675M",    "576$\\to$144", "$336^2$",  False),
    "q25-32":      ("Qwen2.5-VL-32B",       "Qwen2.5-VL",      32, "native ViT 675M",    "576$\\to$144", "$336^2$",  False),
    "q25-72":      ("Qwen2.5-VL-72B",       "Qwen2.5-VL",      72, "native ViT 675M",    "576$\\to$144", "$336^2$",  False),
    "q3-4":        ("Qwen3-VL-4B",          "Qwen3-VL",        4,  "SigLIP2-based",      "$\\leq$1024",  "$\\leq 512^2$", False),
    "q3-8":        ("Qwen3-VL-8B",          "Qwen3-VL",        8,  "SigLIP2-based",      "$\\leq$1024",  "$\\leq 512^2$", False),
    "q3-32":       ("Qwen3-VL-32B",         "Qwen3-VL",        32, "SigLIP2-based",      "$\\leq$1024",  "$\\leq 512^2$", False),
    "iv35-8":      ("InternVL3.5-8B",       "InternVL3.5",     8,  "InternViT-300M",     "1024",         "$448^2$",  False),
    "iv35-14":     ("InternVL3.5-14B",      "InternVL3.5",     14, "InternViT-300M",     "1024",         "$448^2$",  False),
    "iv35-38":     ("InternVL3.5-38B",      "InternVL3.5",     38, "InternViT-6B",       "1024",         "$448^2$",  False),
    "gemma3-4":    ("Gemma 3 4B",           "Gemma 3",         4,  "SigLIP-So400m",      "4096",         "$896^2$",  False),
    "gemma3-12":   ("Gemma 3 12B",          "Gemma 3",         12, "SigLIP-So400m",      "4096",         "$896^2$",  False),
    "gemma3-27":   ("Gemma 3 27B",          "Gemma 3",         27, "SigLIP-So400m",      "4096",         "$896^2$",  False),
    "medgemma-4":  ("MedGemma 4B",          "MedGemma",        4,  "MedSigLIP",          "4096",         "$896^2$",  True),
    "medgemma-27": ("MedGemma 27B",         "MedGemma",        27, "MedSigLIP",          "4096",         "$896^2$",  True),
    "lingshu-7":   ("Lingshu 7B",           "Lingshu",         7,  "Qwen2.5-VL ViT",     "576$\\to$144", "$336^2$",  True),
    "lingshu-32":  ("Lingshu 32B",          "Lingshu",         32, "Qwen2.5-VL ViT",     "576$\\to$144", "$336^2$",  True),
    "llava15-7":   ("LLaVA-1.5-7B",         "LLaVA-1.5",       7,  "CLIP ViT-L/14-336",  "576",          "$336^2$",  False),
    "llava15-13":  ("LLaVA-1.5-13B",        "LLaVA-1.5",       13, "CLIP ViT-L/14-336",  "576",          "$336^2$",  False),
    "llavamed-7":  ("LLaVA-Med 1.5 7B",     "LLaVA-Med",       7,  "CLIP ViT-L/14-336",  "576",          "$336^2$",  True),
    "llama32-11":  ("Llama 3.2 Vision 11B", "Llama 3.2 Vision", 11, "ViT-H/14",          "1601",         "$560^2$",  False),
}
ORDER = list(META)
REV = {'q25-3': '6628554', 'q25-7': 'cc59489', 'q25-32': '7cfb30d', 'q25-72': '89c8620', 'q3-4': 'ebb281e', 'q3-8': '0c351dd', 'q3-32': '0cfaf48', 'iv35-8': '741a7d0', 'iv35-14': '226b96d', 'iv35-38': '7c830fc', 'gemma3-4': '093f9f3', 'gemma3-12': '96b6f1e', 'gemma3-27': '005ad34', 'medgemma-4': '290cda5', 'medgemma-27': '2d3e00e', 'llama32-11': '9eb2daa', 'llava15-7': 'b234b80', 'llava15-13': '5dda288', 'lingshu-7': 'b98aecd', 'lingshu-32': '36b9827', 'llavamed-7': '91bb16c'}   # Hugging Face revision (7-char) of every checkpoint
FAMILY_ORDER = ["Qwen2.5-VL", "Qwen3-VL", "InternVL3.5", "Gemma 3", "MedGemma", "Lingshu", "LLaVA-1.5", "LLaVA-Med", "Llama 3.2 Vision"]
NAMES = {k: v[0] for k, v in META.items()}

# --------------------------------------------------------------------------------------------- colours
COLORS = r"""% Morandi tints for the cf-transfer tables (muted, low saturation); darker = larger value
\definecolor{cfSage1}{HTML}{EEF1EA}
\definecolor{cfSage2}{HTML}{D9E1D2}
\definecolor{cfSage3}{HTML}{C1CDB8}
\definecolor{cfSage4}{HTML}{A6B79C}
\definecolor{cfSlate1}{HTML}{ECEFF2}
\definecolor{cfSlate2}{HTML}{D3DAE2}
\definecolor{cfSlate3}{HTML}{B7C3D0}
\definecolor{cfSlate4}{HTML}{98A8B9}
\definecolor{cfTerra1}{HTML}{F5ECE9}
\definecolor{cfTerra2}{HTML}{E9D3CC}
\definecolor{cfTerra3}{HTML}{DAB5AA}
\definecolor{cfTerra4}{HTML}{C9968A}
\definecolor{cfOat1}{HTML}{F4EFE6}
\definecolor{cfOat2}{HTML}{E8DDC8}
\definecolor{cfGrey}{HTML}{E4E1DC}
"""


def shade_pos(x: float, edges: tuple[float, ...], base: str) -> str:
    """Tint level 1..4 by thresholds (values >= edge[i] get level i+2)."""
    level = 1 + sum(x >= e for e in edges)
    return f"\\cellcolor{{{base}{min(level, 4)}}}"


def shade_own(x: float | None) -> str:
    if x is None:
        return "\\cellcolor{cfGrey}"
    if x < 0:
        return shade_pos(-x, (0.02, 0.10, 0.25), "cfTerra")
    return shade_pos(x, (0.02, 0.10, 0.25), "cfSlate")


def fmt(x, nd=2, signed=False):
    if x is None or (isinstance(x, float) and x != x):
        return "--"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


# --------------------------------------------------------------------------------------------- data
def load_blocks():
    """Every block with a run.json, in catalogue order, under the one inclusion rule (cf_inclusion): `included` blocks
    carry the write matrix (core, t3), `probe` blocks carry the calibration grades; a block that is neither keeps a
    row in Table 1 with no counts. `why` names the grey cell: inel. (CORE ineligible) or inc. (CORE incomplete)."""
    blocks = {}
    for (mk, ds), b in ci.load_runs(RUNS, order=ORDER).items():
        blocks[(mk, ds)] = {"cal": ci.calibration(b) if b["probe"] else {}, "core": ci.core(b),
                            "t3": (b["s"].get("t3") or {}) if b["included"] else {}, "run": b["run"], "elig": b["elig"], "pf": b["pf"],
                            "included": b["included"], "probe": b["probe"],
                            "why": "inel." if "CORE" in (b["run"].get("ineligible_modules") or []) else "inc."}
    return blocks


load = load_blocks


def owned(v):
    return bool(v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage")


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else None


def med_iqr(xs):
    xs = sorted(x for x in xs if x is not None and x == x)
    if not xs:
        return None, None, None
    q = statistics.quantiles(xs, n=4) if len(xs) >= 2 else [xs[0], xs[0], xs[0]]
    return statistics.median(xs), q[0], q[2]


def block_stats(b):
    """Read (S, n_read, n_probe) for probe-graded blocks; Ans (A, n_ans) and Own (O, n_own) over the n write-matrix
    cells of included blocks only; None where the rule excludes the block."""
    cal, core = b["cal"], b["core"]
    S = mean(v.get("selectivity") for v in cal.values()) if b["probe"] else None
    n_read = sum(bool(v.get("readable")) for v in cal.values()) if b["probe"] else None
    if b["included"]:
        A = mean(v.get("answer_auroc") for v in cal.values() if "answer_auroc" in v)
        n_ans = sum(bool(v.get("answer_capable")) for v in cal.values())
        O = mean(v.get("O_q") for v in core.values()); n_own = sum(owned(v) for v in core.values()); n_cells = len(core)
    else:
        A = n_ans = O = n_own = None; n_cells = 0
    return {"S": S, "n_read": n_read, "n_probe": len(cal) if b["probe"] else 0, "n": n_cells, "A": A, "n_ans": n_ans,
            "O": O, "n_own": n_own, "answer_defined": A is not None, "why": b["why"]}


# --------------------------------------------------------------------------------------------- Table 1
def table_main(blocks):
    models = [m for m in ORDER if any((m, d) in blocks for d, _ in DATASETS)]
    stats = {k: block_stats(v) for k, v in blocks.items()}
    def chest_rate(m):
        own = tot = 0
        for d in CHEST:
            st = stats.get((m, d))
            if st and st["n_own"] is not None:
                own += st["n_own"]; tot += st["n"]
        return (own / tot) if tot else -1, own, tot
    def coco_rate(m):
        st = stats.get((m, "coco"))
        return (st["n_own"] / st["n"], st["n_own"], st["n"]) if st and st["n_own"] is not None else (None, 0, 0)
    models.sort(key=lambda m: (-chest_rate(m)[0], -(coco_rate(m)[0] or -1), ORDER.index(m)))
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}", r"\renewcommand{\arraystretch}{1.08}",
         r"\input{tables/cf_colors}",
         r"\caption{\textbf{Reading, answering, and writing per checkpoint.} For each dataset: \textsc{Read} = mean probe selectivity "
         r"$S$ over the six concepts, with the number of readable concepts (of six; selectivity lower bound $>0$) as superscript; "
         r"\textsc{Ans} = mean clean-answer AUROC, with the number of answer-capable concepts (lower bound $>0.5$); "
         r"\textsc{Own} = mean ownership $O_q$, with the number of owned concepts (steering reference met and all simultaneous "
         r"lower bounds $>0$). The three right columns give the owned share of clinical cells (NIH + CheXpert), of COCO cells, "
         r"and their ratio. A block enters the \textsc{Ans} and \textsc{Own} counts only if both its write matrix and its "
         r"calibration module completed; the \textsc{Read} count in the last row is over every probe-graded cell (calibration "
         r"completed, including blocks whose write matrix is ineligible or incomplete), and the \textsc{Ans}/\textsc{Own} counts "
         r"over the cells with a scored write matrix. Cells are shaded by value: sage for \textsc{Read}/\textsc{Ans} (darker = higher), slate for positive and "
         r"terracotta for negative mean ownership (darker = larger magnitude); grey ``inel.'' = ineligible yes/no template, "
         r"grey ``inc.'' = write matrix incomplete at packaging; ``--'' = block not run. Checkpoints are sorted by clinical owned share, descending.}",
         r"\label{tab:cf-main}",
         r"\begin{tabular}{l@{\hspace{3pt}}ccc@{\hspace{4pt}}ccc@{\hspace{4pt}}ccc@{\hspace{4pt}}ccc}", r"\toprule",
         r"& \multicolumn{3}{c}{NIH ChestX-ray14} & \multicolumn{3}{c}{CheXpert Plus} & \multicolumn{3}{c}{COCO (control)} & \multicolumn{3}{c}{Owned share} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
         r"Checkpoint & \textsc{Read} & \textsc{Ans} & \textsc{Own} & \textsc{Read} & \textsc{Ans} & \textsc{Own} & \textsc{Read} & \textsc{Ans} & \textsc{Own} & clin. & COCO & ratio \\",
         r"\midrule"]
    agg = {d: {"S": [], "A": [], "O": [], "read": 0, "ans": 0, "own": 0, "n": 0, "n_ans": 0, "n_own": 0} for d, _ in DATASETS}
    for m in models:
        cells = []
        for d, _ in DATASETS:
            st = stats.get((m, d))
            if not st:
                cells += ["--", "--", "--"]; continue
            if st["n_read"] is not None:
                cells.append(f"{shade_pos(st['S'], (0.05, 0.12, 0.20), 'cfSage')}{fmt(st['S'])}$^{{{st['n_read']}}}$")
                agg[d]["S"].append(st["S"]); agg[d]["read"] += st["n_read"]; agg[d]["n"] += st["n_probe"]
            else:
                cells.append(f"\\cellcolor{{cfGrey}}{st['why']}")
            if st["answer_defined"]:
                cells.append(f"{shade_pos(st['A'], (0.6, 0.75, 0.9), 'cfSage')}{fmt(st['A'])}$^{{{st['n_ans']}}}$")
                agg[d]["A"].append(st["A"]); agg[d]["ans"] += st["n_ans"]; agg[d]["n_ans"] += st["n"]
            else:
                cells.append(f"\\cellcolor{{cfGrey}}{st['why']}")
            if st["n_own"] is None:
                cells.append(f"\\cellcolor{{cfGrey}}{st['why']}")
            else:
                cells.append(f"{shade_own(st['O'])}{fmt(st['O'], 2, True)}$^{{{st['n_own']}}}$")
                agg[d]["O"].append(st["O"]); agg[d]["own"] += st["n_own"]; agg[d]["n_own"] += st["n"]
        cr, co, ct = chest_rate(m); qr, qo, qt = coco_rate(m)
        if ct:
            cells.append(f"{shade_pos(cr, (0.01, 0.15, 0.35), 'cfSlate')}{100 * cr:.0f}\\%")
        else:
            cells.append(r"\cellcolor{cfGrey}inel.")
        cells.append(f"{shade_pos(qr, (0.01, 0.5, 0.9), 'cfSlate')}{100 * qr:.0f}\\%" if qr is not None else "--")
        cells.append(f"{100 * cr / qr:.0f}\\%" if (ct and qr) else "--")
        L.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    # the totals: means (and the owned shares) on one row, the graded counts over their denominators on the row below, so
    # the nine count/denominator pairs do not widen the value columns (no \resizebox on any table)
    tot, cnt = [], []
    for d, _ in DATASETS:
        a = agg[d]
        tot += [fmt(mean(a['S'])), fmt(mean(a['A'])), fmt(mean(a['O']), 2, True)]
        cnt += [f"{a['read']}/{a['n']}", f"{a['ans']}/{a['n_ans']}", f"{a['own']}/{a['n_own']}"]
    chest_own = agg["nih"]["own"] + agg["chexpert"]["own"]; chest_n = agg["nih"]["n_own"] + agg["chexpert"]["n_own"]
    coco_share = agg["coco"]["own"] / agg["coco"]["n_own"] if agg["coco"]["n_own"] else 0
    chest_share = chest_own / chest_n if chest_n else 0
    tot += [f"{100 * chest_share:.0f}\\%", f"{100 * coco_share:.0f}\\%", f"{100 * chest_share / coco_share:.0f}\\%" if coco_share else "--"]
    L.append(r"\textbf{All cells} & " + " & ".join(tot) + r" \\")
    L.append(r"\quad count / cells & " + " & ".join(cnt) + r" & & & \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_main.tex").write_text("\n".join(L) + "\n")
    (OUT / "table_cf_leaderboard.tex").write_text("\\input{tables/table_cf_main}\n")
    return stats, models


# --------------------------------------------------------------------------------------------- family-level grades (merged checkpoints table)
def family_stats(blocks, stats, mks):
    """Mean probe selectivity and clean-answer AUROC over the family's clinical cells, and owned clinical / COCO cells."""
    S, A = [], []
    own_c = tot_c = own_q = tot_q = 0
    for m in mks:
        for d, _ in DATASETS:
            b = blocks.get((m, d)); st = stats.get((m, d))
            if not b:
                continue
            if d in CHEST:
                if b["probe"]:
                    S += [v.get("selectivity") for v in b["cal"].values()]
                if b["included"]:
                    A += [v.get("answer_auroc") for v in b["cal"].values() if "answer_auroc" in v]
                if st["n_own"] is not None:
                    own_c += st["n_own"]; tot_c += st["n"]
            elif st["n_own"] is not None:
                own_q += st["n_own"]; tot_q += st["n"]
    return mean(S), mean(A), own_c, tot_c, own_q, tot_q


# --------------------------------------------------------------------------------------------- Table 3
def table_controls(blocks):
    rows = [("Random-family p95 of $W$", lambda b: [v.get("random_p95") for v in b["core"].values()]),
            ("$|$Sham write$|$", lambda b: [v.get("abs_sham") for v in b["core"].values()]),
            ("Concept write $W_{qq}$", lambda b: [v.get("W_qq") for v in b["core"].values()]),
            ("Ownership $O_q$", lambda b: [v.get("O_q") for v in b["core"].values()]),
            ("Signed dose range of $O_q$", lambda b: [b["t3"].get("all|median_signed_dose_O_range", {}).get("estimate")]),
            ("Refit SD of $O_q$ (3 seeds)", lambda b: [b["t3"].get("all|median_refit_O_sd", {}).get("estimate")]),
            ("Connector-locus median $O_q$", lambda b: [b["t3"].get("all|connector_median_O", {}).get("estimate")]),
            ("Label gap (logits)", lambda b: [b["t3"].get("all|median_label_gap", {}).get("estimate")]),
            ("$|$Wording contrast$|$ (IY$-$WY)", lambda b: [abs(v["estimate"]) for k, v in b["t3"].items() if k.endswith("wording_IY_minus_WY_O") and isinstance(v, dict) and v.get("estimate") is not None]),
            ("$|$Mapping contrast$|$ (IA$-$IB)", lambda b: [abs(v["estimate"]) for k, v in b["t3"].items() if k.endswith("mapping_IA_minus_IB_O") and isinstance(v, dict) and v.get("estimate") is not None])]
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Reference family and controls.} Median and interquartile range over all cells (first four rows) or over all "
         r"blocks with the corresponding module (next six rows), per dataset. The last three rows count the concept cells whose write "
         r"meets the steering reference (above the random p95 and the sham), the owned cells, and the reference-meeting cells that miss "
         r"the complete owned grade: their simultaneous advantage over the five clinical competitors is not positive. "
         r"Table~\ref{tab:cf-contingency} splits these cells into stronger-competitor and unresolved cells.}",
         r"\label{tab:cf-controls}",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"& \multicolumn{2}{c}{NIH ChestX-ray14} & \multicolumn{2}{c}{CheXpert Plus} & \multicolumn{2}{c}{COCO (control)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Quantity & median & IQR & median & IQR & median & IQR \\", r"\midrule"]
    for label, f in rows:
        cells = []
        for d, _ in DATASETS:
            xs = []
            for (m, dd), b in blocks.items():
                if dd == d and b["included"]:
                    xs += f(b)
            med, lo, hi = med_iqr(xs)
            nd = 1 if "Label" in label else 3
            cells += [fmt(med, nd, True) if med is not None else "--", f"[{fmt(lo, nd, True)}, {fmt(hi, nd, True)}]" if med is not None else "--"]
        L.append(f"{label} & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    ref, own, notown = [], [], []
    for d, _ in DATASETS:
        n_ref = n_own = n_cells = 0
        for (m, dd), b in blocks.items():
            if dd == d and b["included"]:
                for v in b["core"].values():
                    n_cells += 1; n_ref += bool(v.get("steering_reference")); n_own += owned(v)
        ref.append(f"\\multicolumn{{2}}{{c}}{{{n_ref} of {n_cells}}}")
        own.append(f"\\multicolumn{{2}}{{c}}{{{n_own} of {n_cells}}}")
        notown.append(f"\\multicolumn{{2}}{{c}}{{{n_ref - n_own} of {n_cells}}}")
    L.append("Reference met & " + " & ".join(ref) + r" \\")
    L.append("Owned & " + " & ".join(own) + r" \\")
    L.append("Reference met but not owned & " + " & ".join(notown) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_controls.tex").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------------------------- Table 4
def table_seed(blocks):
    b = blocks[("q25-7", "nih")]
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}", r"\input{tables/cf_colors}",
         r"\caption{\textbf{Seed replication: Qwen2.5-VL-7B on NIH ChestX-ray14, 600 new patients.} Per concept: probe selectivity $S$, "
         r"clean-answer AUROC, concept write $W_{qq}$, the random-family 95th percentile and the absolute sham at the same dose, the "
         r"rank of $W_{qq}$ among the 120 compared effects, the ownership contrast $O_q$ with its 95\% percentile interval, the "
         r"strongest competing direction, and the max-$T$ verdict. Negative ownership is shaded terracotta.}",
         r"\label{tab:cf-seed}",
         r"\begin{tabular}{lrrrrrrrll}", r"\toprule",
         r"Concept & $S$ & AUROC & $W_{qq}$ & p95 & $|$sham$|$ & rank & $O_q$ [95\% CI] & Competitor & Verdict \\", r"\midrule"]
    for c, v in b["core"].items():
        cc = b["cal"].get(c, {}); ci = v.get("O_q_ci95_percentile") or [None, None]
        verdict = "owned" if owned(v) else {"stronger_competitor": "competitor", "unresolved": "unresolved"}.get(v.get("verdict"), v.get("verdict"))
        L.append(f"{c} & {fmt(cc.get('selectivity'), 3)} & {fmt(cc.get('answer_auroc'), 3)} & {fmt(v['W_qq'], 3, True)} & {fmt(v['random_p95'], 3)} & "
                 f"{fmt(v['abs_sham'], 3)} & {v['rank_in_random_family']}/120 & {shade_own(v['O_q'])}{fmt(v['O_q'], 3, True)} [{fmt(ci[0], 3, True)}, {fmt(ci[1], 3, True)}] & {v.get('argmax_other')} & {verdict} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_seed.tex").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------------------------- seed study: Mass across prompts
SEED_TABLES = HERE.parent / "tables"   # the registered seed-study tables (hash-pinned in MANIFEST.sha256; not input by the paper)


def _registered_rows(name: str, ncols: int) -> list[list[str]]:
    """Body rows of a registered seed-study table, cells verbatim."""
    tex = (SEED_TABLES / name).read_text()
    body = tex.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    rows = [[c.strip() for c in ln.strip().rstrip("\\").strip().split(" & ")] for ln in body.strip().splitlines() if ln.strip()]
    for r in rows:
        if len(r) != ncols:
            raise RuntimeError(f"{name}: expected {ncols} cells per row, got {r}")
    return rows


def table_seed_mass():
    """One table for the seed study's Mass competition: the discovery crossover (table_encoding_mass.tex) and the registered
    confirmation on new patients (table_mass_confirmation.tex), cells copied verbatim from the two registered tables."""
    disc = _registered_rows("table_encoding_mass.tex", 6)
    conf = _registered_rows("table_mass_confirmation.tex", 5)
    L = [r"\begin{table}[H]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{5pt}",
         r"\caption{\textbf{Mass clinical competition across prompts.} Clinical margin: the Mass effect minus the largest of the five other "
         r"clinical effects, in finding-present logit units at $\alpha=+0.25$. Top: the discovery crossover (pointwise 95\% interval; "
         r"largest of 20 random effects). Bottom: the registered confirmation on new patients (smallest simultaneous lower bound of the "
         r"cell's five contrasts; largest of 119 random effects).}",
         r"\label{tab:seed-mass}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Prompt & Clinical margin & 95\% CI & Mass effect & Random max & $|$Sham$|$ \\", r"\midrule"]
    L += [" & ".join(r) + r" \\" for r in disc]
    L += [r"\midrule", r"Condition & Clinical margin & Minimum LCB & Mass effect & Random max & \\", r"\midrule"]
    L += [" & ".join(r) + r" & \\" for r in conf]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_seed_mass.tex").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------------------------- appendix ownership tables
def table_contingency():
    """Steering reference x verdict over the primary write-matrix cells of included blocks: chest (all cells and the
    readable-and-answerable subset) and COCO."""
    import csv, collections
    rows = [r for r in csv.DictReader(open(ci.RUNS / "manifest.csv"))
            if r["module"] == "CORE" and r["verdict"] and r["block_included"].lower() in ("true", "1")
            and r["is_primary_template"].lower() in ("true", "1")]
    t = lambda x: x.lower() in ("true", "1")
    V = [("fixed_family_advantage", "advantage"), ("stronger_competitor", "stronger competitor"), ("unresolved", "unresolved")]
    def counts(sel):
        c = collections.Counter((t(r["steering_reference"]), r["verdict"]) for r in sel)
        return c
    groups = [("Chest, all cells", [r for r in rows if r["dataset"] in ("nih", "chexpert")]),
              ("Chest, readable and answerable", [r for r in rows if r["dataset"] in ("nih", "chexpert") and t(r["readable"]) and t(r["answer_capable"])]),
              ("COCO, all cells", [r for r in rows if r["dataset"] == "coco"])]
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Steering reference against verdict.} Primary write-matrix cells of the included blocks, split by whether the "
         r"concept write meets the steering reference ($W_{q,q}>0$, above the 95th percentile of the 119 random writes, and above the "
         r"absolute sham) and by the simultaneous-bound verdict of the comparison with the five other clinical directions: advantage "
         r"(every lower bound of $W_{q,q}-W_{q,d}$ positive), stronger competitor (some upper bound below zero), or unresolved. Owned "
         r"cells are the reference-meeting cells with the advantage verdict. The middle rows restrict the chest cells to those that "
         r"are both readable and answerable.}", r"\label{tab:cf-contingency}",
         r"\begin{tabular}{llrrrr}", r"\toprule", r"Cells & Steering reference & advantage & stronger competitor & unresolved & total \\", r"\midrule"]
    for name, sel in groups:
        c = counts(sel)
        for met, lab in ((True, "met"), (False, "not met")):
            n = [c[(met, v)] for v, _ in V]
            L.append(f"{name if met else ''} & {lab} & " + " & ".join(str(x) for x in n) + f" & {sum(n)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_contingency.tex").write_text("\n".join(L) + "\n")
    print("table_cf_contingency.tex", len(rows), "cells")


def table_ownership(blocks, ds, label, order):
    """Compact checkpoint x concept table of O_q for one dataset (same checkpoint order as Table 1)."""
    models = [m for m in order if (m, ds) in blocks]
    concepts = None
    for m in models:
        b = blocks[(m, ds)]
        if b["included"]:
            concepts = list(b["core"].keys()); break
    if concepts is None:
        concepts = list(blocks[(models[0], ds)]["cal"].keys())
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.08}",
         r"\input{tables/cf_colors}",
         r"\caption{\textbf{Ownership on " + label + r".} Each cell is the ownership contrast $O_q$ of the concept's own write against its strongest "
         r"clinical competitor at $\alpha=+0.25$. Cells are shaded slate for positive and terracotta for negative values (darker = larger magnitude, "
         r"thresholds $0.02$, $0.10$, $0.25$). Bold marks owned cells (steering reference met and all simultaneous lower bounds $>0$). "
         r"A dagger marks cells whose write meets the steering reference but has a stronger competitor; a double dagger marks reference-meeting cells whose comparison is unresolved. Grey ``inel.'' marks checkpoints whose "
         r"yes/no template is ineligible and grey ``inc.'' a write matrix incomplete at packaging (neither enters any count). Checkpoints are in the order of Table~\ref{tab:cf-main}.}",
         r"\label{tab:cf-own-" + ds + r"}",
         r"\begin{tabular}{l" + "r" * len(concepts) + r"}", r"\toprule",
         "Checkpoint & " + " & ".join(concepts) + r" \\", r"\midrule"]
    n_own = {c: 0 for c in concepts}
    for m in models:
        b = blocks[(m, ds)]
        if not b["included"]:
            L.append(f"{NAMES[m]} & " + " & ".join(f"\\cellcolor{{cfGrey}}{b['why']}" for _ in concepts) + r" \\")
            continue
        cells = []
        for c in concepts:
            v = b["core"].get(c)
            if v is None or v.get("O_q") is None:
                cells.append("--"); continue
            txt = fmt(v["O_q"], 2, True)
            if owned(v):
                txt = r"\textbf{" + txt + "}"; n_own[c] += 1
            elif v.get("steering_reference"):
                txt += r"$^{\dagger}$" if v.get("verdict") == "stronger_competitor" else r"$^{\ddagger}$"
            cells.append(shade_own(v["O_q"]) + txt)
        L.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
    L += [r"\midrule", r"\textbf{Owned} & " + " & ".join(str(n_own[c]) for c in concepts) + r" \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / f"table_cf_own_{ds}.tex").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------------------------- appendix sheets
def table_models(blocks, stats):
    """Checkpoints and families in one table: a bold family row carries the family-level grades (mean selectivity and
    clean-answer AUROC over clinical cells, owned clinical and COCO cells), and the checkpoint rows under it carry the
    checkpoint sheet (vision tower, consumed tokens, image size, eligible templates, pinned revision)."""
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\input{tables/cf_colors}",
         r"\caption{\textbf{Checkpoints and families.} Per checkpoint: vision tower, consumed visual tokens per image at the primary "
         r"locus, input image size, templates eligible after the image-free mapping gate (I/W: \emph{is}/\emph{show} wording; Y: yes/no; "
         r"A/B: letter mappings), and the pinned Hugging Face revision. Per family (bold rows): mean probe selectivity $\bar S$ and "
         r"clean-answer AUROC over clinical cells, and owned clinical and COCO cells, shaded slate (darker = larger share).}",
         r"\label{tab:cf-models}",
         r"\begin{tabular}{llllll@{\hspace{8pt}}rrrr}", r"\toprule",
         r" & & & & & & & & \multicolumn{2}{c}{owned cells} \\", r"\cmidrule(lr){9-10}",
         r"Checkpoint & Vision tower & Tokens & Image & Templates & Revision & $\bar S$ & $\overline{\mathrm{AUROC}}$ & clinical & COCO \\"]
    for fam in FAMILY_ORDER:
        mks = [m for m in ORDER if META[m][1] == fam and any((m, d) in blocks for d, _ in DATASETS)]
        if not mks:
            continue
        S, A, own_c, tot_c, own_q, tot_q = family_stats(blocks, stats, mks)
        oc = f"{shade_pos(own_c / tot_c if tot_c else 0, (0.01, 0.15, 0.35), 'cfSlate')}{own_c}/{tot_c}" if tot_c else r"\cellcolor{cfGrey}inel."
        oq = f"{shade_pos(own_q / tot_q if tot_q else 0, (0.01, 0.5, 0.9), 'cfSlate')}{own_q}/{tot_q}" if tot_q else "--"
        label = r"\textbf{" + fam + "}" + (r" (medical)" if META[mks[0]][6] else "")
        L += [r"\midrule", f"\\multicolumn{{6}}{{l}}{{{label}}} & {fmt(S, 3)} & {fmt(A, 3)} & {oc} & {oq} \\\\"]
        for m in mks:
            present = [(d, blocks[(m, d)]) for d, _ in DATASETS if (m, d) in blocks]
            name, _, _, tower, tokens, img, _ = META[m]
            elig = present[0][1]["elig"]
            temps = "".join(t for t in TEMPLATES if elig.get(t, {}).get("eligible", True)) if elig else "all"
            temps = "all" if temps == "".join(TEMPLATES) else temps
            L.append(f"\\quad {name} & {tower} & {tokens} & {img} & {temps} & \\texttt{{{REV.get(m, '')}}} & & & & \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_models.tex").write_text("\n".join(L) + "\n")
    (OUT / "table_cf_families.tex").unlink(missing_ok=True)   # merged into this table


def table_gates(blocks):
    A = B = G = 0.0; cons_lo = cons_hi = None; lg_lo = lg_hi = None; outside = 0.0; D = {}; fails = {}
    for (m, d), b in blocks.items():
        c = (b["pf"] or {}).get("checks")
        if not c:
            continue
        A = max(A, c["A_determinism_max_abs_diff"]); B = max(B, c["B_alpha0_max_abs_diff"]); G = max(G, c["G_fp32_vs_model_logits"]["max_abs_diff"])
        r = c["C_reach"]; cons_lo = r["consumer_max_abs_change"] if cons_lo is None else min(cons_lo, r["consumer_max_abs_change"]); cons_hi = r["consumer_max_abs_change"] if cons_hi is None else max(cons_hi, r["consumer_max_abs_change"])
        lg_lo = r["answer_logit_max_abs_change"] if lg_lo is None else min(lg_lo, r["answer_logit_max_abs_change"]); lg_hi = r["answer_logit_max_abs_change"] if lg_hi is None else max(lg_hi, r["answer_logit_max_abs_change"])
        outside = max(outside, r["locus_change_outside_consumed_tokens"])
        D.setdefault(META[m][1], []).append(c["D_batch_vs_single"]["max_abs_candidate_logit_diff"])
        el = b["elig"]
        bad = "".join(t for t in TEMPLATES if el and not el.get(t, {}).get("eligible", True))
        if bad:
            fails.setdefault(NAMES[m], set()).add(bad)
    d_all = [x for xs in D.values() for x in xs]
    d_over = ", ".join(f"{f} ($\\leq${max(xs):.2f})" for f, xs in D.items() if max(xs) > 0.25)
    e_txt = "; ".join(f"{k}: {'/'.join(sorted(v))}" for k, v in sorted(fails.items()))
    rows = [("A", "Determinism", "identical answer logits on a repeated forward", f"max difference {A:g}"),
            ("B", "Identity at $\\alpha=0$", "the armed hook with zero dose changes nothing", f"max difference {B:g}"),
            ("C", "Reach", "the write changes the consumer input and the answer logits; no change outside consumed tokens",
             f"consumer change {cons_lo:.2f}--{cons_hi:.2f}; answer logits {lg_lo:.2f}--{lg_hi:.2f}; outside consumed tokens {outside:g}"),
            ("D", "Batch consistency", "single- vs batched-forward answer logits within 0.25", f"{min(d_all):.2f}--{max(d_all):.2f}; above tolerance (declared, neutralised by fixed batch composition): {d_over}"),
            ("E", "Semantic mapping", "image-free ``present/absent'' statements map to the right answer token under each template", f"failing templates: {e_txt}"),
            ("F", "Throughput", "rows per second recorded for scheduling", "3.8--35 rows/s"),
            ("G", "fp32 answer logits", "fp32 logits recomputed from the final hidden state agree with the model's bf16 logits", f"max difference {G:.2f}")]
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Preflight gates.} Each block is scored only after the seven checks below run on 16 held-out rows at both loci; "
         r"the right column gives the range observed over all completed blocks.}",
         r"\label{tab:cf-gates}",
         r"\begin{tabular}{clp{4.3cm}p{6.0cm}}", r"\toprule", r"Gate & Name & Criterion & Observed \\", r"\midrule"]
    for g, name, crit, obs in rows:
        L.append(f"{g} & {name} & {crit} & {obs} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_gates.tex").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------------------------- coverage (manifest)
# the module's registered key, the letter the table prints, and the name the caption's legend gives it: the paper
# names a module by what it scores, never by the key the run records carry
MODULE_ABBR = [("CORE", "C"), ("CALIBRATION", "Cal"), ("PROMPT", "P"), ("DOSE", "D"), ("REFIT", "R"), ("LOCUS", "L"),
               ("LOCUS_CALIBRATION", "Lc"), ("ALTDIR", "A"), ("ALTDIRD", "Ad"), ("ANSDIR", "An"), ("EXTCOMP", "E"), ("TOKENW", "T"), ("PRECISION", "Pr")]
MODULE_NAME = {"CORE": "write matrix", "CALIBRATION": "calibration pass", "PROMPT": "five additional templates",
               "DOSE": "dose sweep", "REFIT": "two probe refits", "LOCUS": "connector locus",
               "LOCUS_CALIBRATION": "connector-locus calibration", "ALTDIR": "alternative direction constructions",
               "ALTDIRD": "displacement lifts", "ANSDIR": "answer direction", "EXTCOMP": "extended competitor family",
               "TOKENW": "token-weighted writes", "PRECISION": "fp32 and batch-size-one rescoring"}
PLANNED = ("CORE", "CALIBRATION", "PROMPT", "DOSE", "REFIT", "LOCUS", "LOCUS_CALIBRATION")
CHEX_EXTENSION = ("PROMPT", "DOSE", "REFIT", "LOCUS", "LOCUS_CALIBRATION")


def coverage_blocks(rows):
    """(model_key, dataset) -> dict(completed, ineligible, primary) from the manifest's block-level columns."""
    out = {}
    for r in rows:
        k = (r["model_key"], r["dataset"])
        if k not in out:
            out[k] = {"completed": [m for m in r["completed_modules"].split("|") if m], "ineligible": [m for m in r["ineligible_modules"].split("|") if m],
                      "primary": r["primary_template"], "status": r["run_status"]}
    return out


def table_coverage():
    """Modules completed per block, from runs/manifest.csv (the same file the headline numbers come from)."""
    rows = ci.read_manifest(); blocks = coverage_blocks(rows); abbr = dict(MODULE_ABBR)
    def cell(b):
        if b is None:
            return "--"
        done = [abbr.get(m, m) for m, _ in MODULE_ABBR if m in b["completed"]]
        inel = [abbr.get(m, m) for m, _ in MODULE_ABBR if m in b["ineligible"]]
        txt = " ".join(done)
        if inel:
            txt = (txt + "; " if txt else "") + "inel.\\ " + " ".join(inel)
        if b["primary"] and b["primary"] != "IY":
            txt += f" [{b['primary']}]"
        return txt
    n_seven = sum(all(m in b["completed"] for m in PLANNED) for b in blocks.values())
    n_chex = sum(ds == "chexpert" and all(m in b["completed"] for m in CHEX_EXTENSION) for (mk, ds), b in blocks.items())
    legend = ", ".join(f"{a} = {MODULE_NAME[m]}" for m, a in MODULE_ABBR if any(m in b["completed"] or m in b["ineligible"] for b in blocks.values()))
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Coverage.} Modules completed per block at packaging (from the campaign manifest), with the modules whose "
         r"template is ineligible (inel.) and the primary template when it is not the \emph{is}/yes-no template; ``--'' = block not run; "
         r"modules still in progress are not listed. The planned campaign is \cfCoveragePlannedWord{} modules on every block, and "
         r"\cfCoverageBlocksAllSeven{} of the \cfBlocks{} blocks carry all \cfCoveragePlannedWord{}: \cfCoverageNihAllSeven{} of "
         r"\cfCoverageNihBlocks{} on NIH, \cfCoverageChexAllSeven{} of \cfCoverageChexBlocks{} on CheXpert and "
         r"\cfCoverageCocoAllSeven{} of \cfCoverageCocoBlocks{} on COCO. The CheXpert blocks carried the write matrix and the "
         r"calibration pass in the first wave; the \cfCoverageChexAddedWord{} modules amendment A2 added to them are complete on "
         r"\cfCoverageChexFull{} of the \cfCoverageChexBlocks{} CheXpert blocks.}",
         r"\label{tab:cf-coverage}",
         r"\begin{tabular}{llll}", r"\toprule", r"Checkpoint & NIH ChestX-ray14 & CheXpert Plus & COCO \\", r"\midrule"]
    for m in ORDER:
        if not any((m, d) in blocks for d, _ in DATASETS):
            continue
        L.append(f"{NAMES[m]} & " + " & ".join(cell(blocks.get((m, d))) for d, _ in DATASETS) + r" \\")
    L += [r"\midrule", r"\multicolumn{4}{p{0.95\textwidth}}{\textit{Legend:} " + legend + r".} \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_coverage.tex").write_text("\n".join(L) + "\n")
    print(f"table_cf_coverage.tex: {len(blocks)} blocks, {n_seven} with all seven planned modules, {n_chex} CheXpert with the extension modules")
    return n_seven, n_chex


if __name__ == "__main__":
    (OUT / "cf_colors.tex").write_text(COLORS)
    blocks = load_blocks()
    n_inc = sum(b["included"] for b in blocks.values()); n_probe = sum(b["probe"] for b in blocks.values())
    print(f"{len(blocks)} blocks: {n_inc} with a scored write matrix, {n_probe} probe-graded (rule: cf_inclusion.block_included)")
    stats, order = table_main(blocks)
    table_controls(blocks)
    table_seed(blocks)
    table_seed_mass()
    for ds, label in DATASETS:
        table_ownership(blocks, ds, label, order)
    table_contingency()
    for ds, _ in DATASETS:
        (OUT / f"table_cf_{ds}.tex").unlink(missing_ok=True)
    table_models(blocks, stats)
    table_gates(blocks)
    table_coverage()
    print(f"{len(blocks)} blocks -> tables written to {OUT}")
