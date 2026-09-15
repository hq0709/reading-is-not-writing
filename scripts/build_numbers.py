"""Headline numbers of the paper as LaTeX macros, computed from runs/manifest.csv under the one inclusion rule.

    python scripts/build_numbers.py            -> tables/cf_numbers.tex (+ tables/cf_numbers.json for checks)

Every macro is defined \\providecommand-safe (\\providecommand{\\x}{}\\renewcommand{\\x}{value}) with a plain number
as value, so the prose writes e.g. "\\cfChestReadable{} of \\cfChestReadableN{} cells (\\cfChestReadablePct\\%)".

Sources and rules
  runs/manifest.csv      written by `python -m cftransfer.manifest` (code repository); block_included = CORE and
                         CALIBRATION in run.json completed_modules; probe_graded = CALIBRATION completed. The
                         included / probe-graded block sets are cross-checked against the summaries (cf_inclusion).
  counts                 concept cells are the CORE rows with fit_seed 0. Readable counts run over probe-graded
                         cells; every other count over write-matrix cells (included blocks only).
  percentages            round half up of 100 * a / b.
  own-question share     mean over included checkpoints of max(W_qd, 0), trace over total (the write-flow panels
                         of Figure A1, scripts/plot_paper_figures.py figA1_write_structure); read from the summaries.
  ALTDIR                 per family, cells owned (steering reference and fixed-family advantage within the family)
                         over the ALTDIR rows of the manifest; "any" counts cells owned under the logistic normal
                         or any alternative family.
  median W_qq            median over write-matrix cells per dataset, three decimals.
  selectivity            mean probe selectivity of Effusion and Cardiomegaly over probe-graded chest blocks.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cf_inclusion as ci   # noqa: E402

OUT = HERE.parent / "tables"
DS_MACRO = {"nih": "Nih", "chexpert": "Chex", "coco": "Coco"}
FAMILIES = ("dom", "pattern", "orth", "resid")
FAM_MACRO = {"logistic": "Logistic", "dom": "Dom", "pattern": "Pattern", "orth": "Orth", "resid": "Resid", "any": "Any"}
EFFUSION_SMALL = 0.03      # "In N of these, O_Effusion <= 0.03"


def pct(a, b) -> int:
    return int(math.floor(100.0 * a / b + 0.5)) if b else 0


def t(v) -> bool:
    return v == "true"


def f(v):
    return None if v in (None, "") else float(v)


def counts(rows: list[dict]) -> dict:
    cells = [r for r in rows if t(r["cell"])]
    out = {}
    groups = {ds: (ds,) for ds in ci.DATASETS}; groups["chest"] = ci.CHEST; groups["all"] = ci.DATASETS
    for name, dss in groups.items():
        c = [r for r in cells if r["dataset"] in dss]
        probe = [r for r in c if t(r["probe_graded"]) and r["readable"] != ""]
        wm = [r for r in c if t(r["block_included"])]
        own = [r for r in wm if t(r["owned"])]
        ref = [r for r in wm if t(r["steering_reference"])]
        ra = [r for r in wm if t(r["readable"]) and t(r["answer_capable"])]
        out[name] = dict(
            blocks=len({(r["model_key"], r["dataset"]) for r in rows if r["dataset"] in dss}),
            probe_blocks=len({(r["model_key"], r["dataset"]) for r in probe}),
            write_blocks=len({(r["model_key"], r["dataset"]) for r in wm}),
            probe_cells=len(probe), readable=sum(t(r["readable"]) for r in probe),
            write_cells=len(wm), answerable=sum(t(r["answer_capable"]) for r in wm),
            owned=len(own), stronger_competitor=sum(r["verdict"] == "stronger_competitor" for r in wm),
            reference_met=len(ref), reference_met_not_owned=len(ref) - len(own),
            readable_and_answerable=len(ra), owned_among_readable_and_answerable=sum(t(r["owned"]) for r in ra),
            rest=len(wm) - len(ra), owned_among_rest=len(own) - sum(t(r["owned"]) for r in ra),
            effect_floored_owned=sum(f(r["W_qq"]) >= ci.EFFECT_FLOOR and f(r["O_q"]) >= ci.EFFECT_FLOOR for r in own),
            median_W_qq=statistics.median(f(r["W_qq"]) for r in wm) if wm else None)
    return out


def own_share(blocks: dict) -> dict:
    """Same computation as figA1_write_structure: P = mean over included checkpoints of max(W_qd, 0); share = trace / sum."""
    out = {}
    for ds in ci.DATASETS:
        keys = [k for k in blocks if k[1] == ds]
        Ps = []
        for k in keys:
            s = blocks[k]["s"]; concepts = list(s["core"]["per_question"])
            W = np.array([[s["core"]["W"][q].get(f"concept:{d}", np.nan) for d in concepts] for q in concepts])
            Ps.append(np.maximum(W, 0.0))
        Pm = np.nanmean(np.stack(Ps), axis=0)
        out[ds] = dict(n=len(keys), share=float(np.trace(Pm) / Pm.sum()))
    return out


def altdir(rows: list[dict]) -> dict:
    core = {(r["model_key"], r["dataset"], r["concept"]): r for r in rows if t(r["cell"])}
    alt = [r for r in rows if r["module"] == "ALTDIR" and r["fit_seed"] == "0" and r["altdir_owned_dom"] != "" and t(r["block_included"])]
    out = {}
    for ds in ci.DATASETS:
        a = [r for r in alt if r["dataset"] == ds]
        d = {"blocks": len({r["model_key"] for r in a}), "n": len(a),
             "logistic": sum(t(core[(r["model_key"], r["dataset"], r["concept"])]["owned"]) for r in a)}
        for fam in FAMILIES:
            d[fam] = sum(t(r[f"altdir_owned_{fam}"]) for r in a)
        d["any"] = sum(t(core[(r["model_key"], r["dataset"], r["concept"])]["owned"]) or any(t(r[f"altdir_owned_{fam}"]) for fam in FAMILIES) for r in a)
        out[ds] = d
    return out


def main() -> Path:
    rows = ci.read_manifest()
    all_blocks = ci.load_runs()
    ci.check_manifest_agrees(all_blocks, rows)
    wb = ci.write_blocks(all_blocks)
    C = counts(rows); S = own_share(wb); A = altdir(rows)
    ch, co = C["chest"], C["coco"]
    cells = [r for r in rows if t(r["cell"])]
    M = {}
    # campaign
    M["cfCheckpoints"] = len({r["model_key"] for r in rows})
    M["cfBlocks"] = C["all"]["blocks"]; M["cfWriteBlocks"] = C["all"]["write_blocks"]; M["cfProbeBlocks"] = C["all"]["probe_blocks"]
    M["cfCells"] = C["all"]["write_cells"]; M["cfProbeCells"] = C["all"]["probe_cells"]
    M["cfChestBlocks"] = ch["write_blocks"]; M["cfChestProbeBlocks"] = ch["probe_blocks"]
    M["cfNihBlocks"] = C["nih"]["write_blocks"]; M["cfChexBlocks"] = C["chexpert"]["write_blocks"]; M["cfCocoBlocks"] = C["coco"]["write_blocks"]
    mks = {k[0] for k in wb}
    M["cfCheckpointsComplete"] = sum(all((m, d) in wb for d in ci.DATASETS) for m in mks)
    # chest
    M["cfChestReadable"] = ch["readable"]; M["cfChestReadableN"] = ch["probe_cells"]; M["cfChestReadablePct"] = pct(ch["readable"], ch["probe_cells"])
    M["cfChestAnswerable"] = ch["answerable"]; M["cfChestAnswerableN"] = ch["write_cells"]; M["cfChestAnswerablePct"] = pct(ch["answerable"], ch["write_cells"])
    M["cfChestOwned"] = ch["owned"]; M["cfChestOwnedPct"] = pct(ch["owned"], ch["write_cells"])
    M["cfChestCompetitor"] = ch["stronger_competitor"]; M["cfChestCompetitorPct"] = pct(ch["stronger_competitor"], ch["write_cells"])
    M["cfChestRefMet"] = ch["reference_met"]; M["cfChestRefBelow"] = ch["write_cells"] - ch["reference_met"]; M["cfChestRefNotOwned"] = ch["reference_met_not_owned"]
    M["cfReadAns"] = ch["readable_and_answerable"]; M["cfReadAnsOwned"] = ch["owned_among_readable_and_answerable"]
    M["cfReadAnsOwnedPct"] = pct(ch["owned_among_readable_and_answerable"], ch["readable_and_answerable"])
    M["cfRestN"] = ch["rest"]; M["cfRestOwned"] = ch["owned_among_rest"]; M["cfRestOwnedPct"] = pct(ch["owned_among_rest"], ch["rest"])
    M["cfChestFloored"] = ch["effect_floored_owned"]; M["cfChestFlooredPct"] = pct(ch["effect_floored_owned"], ch["write_cells"])
    eff = [r for r in cells if r["dataset"] in ci.CHEST and r["concept"] == "Effusion" and t(r["owned"])]
    M["cfEffusionOwnedBlocks"] = len(eff); M["cfEffusionOwnedSmall"] = sum(f(r["O_q"]) <= EFFUSION_SMALL for r in eff)
    for c, name in (("Effusion", "cfSelEffusion"), ("Cardiomegaly", "cfSelCardiomegaly")):
        xs = [f(r["selectivity"]) for r in cells if r["dataset"] in ci.CHEST and r["concept"] == c and t(r["probe_graded"]) and r["selectivity"] != ""]
        M[name] = f"{sum(xs) / len(xs):.2f}"
    # per dataset
    for ds, D in DS_MACRO.items():
        d = C[ds]
        M[f"cf{D}Readable"] = d["readable"]; M[f"cf{D}ReadableN"] = d["probe_cells"]; M[f"cf{D}Cells"] = d["write_cells"]
        M[f"cf{D}Answerable"] = d["answerable"]; M[f"cf{D}Owned"] = d["owned"]; M[f"cf{D}Competitor"] = d["stronger_competitor"]
        M[f"cf{D}RefMet"] = d["reference_met"]; M[f"cfMedianWqq{D}"] = f"{d['median_W_qq']:.3f}"
        M[f"cfOwnShare{D}"] = pct(S[ds]["share"], 1)
    M["cfCocoOwnedN"] = co["write_cells"]; M["cfCocoOwnedPct"] = pct(co["owned"], co["write_cells"]); M["cfCocoAnswerablePct"] = pct(co["answerable"], co["write_cells"])
    M["cfCocoFloored"] = co["effect_floored_owned"]; M["cfCocoFlooredPct"] = pct(co["effect_floored_owned"], co["write_cells"])
    per_block = {}
    for r in cells:
        if r["dataset"] == "coco" and t(r["block_included"]):
            per_block[r["model_key"]] = per_block.get(r["model_key"], 0) + t(r["owned"])
    M["cfCocoBlocksFourPlus"] = sum(v >= 4 for v in per_block.values())
    # altdir
    M["cfAltdirBlocks"] = sum(A[ds]["blocks"] for ds in ci.DATASETS)
    per = [A[ds]["blocks"] for ds in ci.DATASETS]
    M["cfAltdirBlocksPerDs"] = str(per[0]) if len(set(per)) == 1 else "/".join(map(str, per))
    if len(set(per)) != 1:
        print(f"  NOTE: ALTDIR block counts differ per dataset {dict(zip(ci.DATASETS, per))}; \\cfAltdirBlocksPerDs prints them as nih/chexpert/coco")
    for ds, D in DS_MACRO.items():
        M[f"cfAltdirBlocks{D}"] = A[ds]["blocks"]; M[f"cfAltdir{D}N"] = A[ds]["n"]
        for fam, F in FAM_MACRO.items():
            M[f"cfAltdir{D}{F}"] = A[ds][fam]
    chest_p = [pct(A[ds][fam], A[ds]["n"]) for ds in ci.CHEST for fam in ("logistic",) + FAMILIES]
    coco_p = [pct(A["coco"][fam], A["coco"]["n"]) for fam in ("logistic",) + FAMILIES]
    M["cfAltdirChestPctMin"], M["cfAltdirChestPctMax"] = min(chest_p), max(chest_p)
    M["cfAltdirCocoPctMin"], M["cfAltdirCocoPctMax"] = min(coco_p), max(coco_p)
    # write
    L = ["% Generated by scripts/build_numbers.py from runs/manifest.csv; do not edit.",
         "% Rule: a block enters the write-matrix counts iff run.json completed_modules contains CORE and CALIBRATION;",
         "% readability counts run over probe-graded cells (CALIBRATION completed). See scripts/cf_inclusion.py.",
         f"% {M['cfBlocks']} blocks, {M['cfWriteBlocks']} included, {M['cfCells']} write-matrix cells, {M['cfProbeCells']} probe-graded cells."]
    for k, v in M.items():
        L.append(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}")
    OUT.mkdir(exist_ok=True)
    (OUT / "cf_numbers.tex").write_text("\n".join(L) + "\n")
    (OUT / "cf_numbers.json").write_text(json.dumps({"macros": M, "counts": C, "own_share": S, "altdir": A}, indent=1) + "\n")
    print(f"tables/cf_numbers.tex: {len(M)} macros")
    for k, v in M.items():
        print(f"  \\{k} = {v}")
    return OUT / "cf_numbers.tex"


if __name__ == "__main__":
    main()
