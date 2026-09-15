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
  robustness             runs/robustness/{geometry,scale,pairs,validation}.json (scripts/mayo/robustness_*.py in the code
                         repository, which discover blocks with the same rule); every file's block set must equal the
                         manifest's included set or this script stops. PRECISION, ANSDIR and label-gap numbers come from
                         the included blocks' summary.json. Formats follow the prose (2-3 decimals; signed where the prose
                         prints a sign). Ranges get Min/Max macros. Warnings flag any wording the numbers no longer support.
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
ROB = ci.RUNS / "robustness"
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


def fx(v, nd, signed=False) -> str:
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def fp(p):
    """p-value for math mode: two decimals, or a power of ten below 0.001 (e.g. 8\\times10^{-11})."""
    if p is None or p != p:
        return "--"
    if p >= 0.001:
        return f"{p:.2f}"
    m, e = f"{p:.0e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def robustness_macros(M: dict, rows: list[dict], wb: dict, warn: list) -> None:
    """Macros for every robustness-derived number in the prose (Sections 5.4-5.6 and Appendix C.6)."""
    G, S, P, V = (json.loads((ROB / f"{n}.json").read_text()) for n in ("geometry", "scale", "pairs", "validation"))
    cells = {(r["model_key"], r["dataset"], r["concept"]): r for r in rows if t(r["cell"])}
    inc = {f"{k[0]}/{k[1]}" for k in wb}
    for name, blocks in (("geometry", set(G["meta"]["blocks"])),
                         ("scale", {f"{b['model']}/{b['dataset']}" for b in S["blocks"] if b.get("status") == "OK"}),
                         ("pairs", {f"{b['model']}/{b['dataset']}" for b in P["blocks"] if b.get("write_matrix")}),
                         ("validation", set(V["meta"]["blocks"]))):
        if blocks != inc:
            raise RuntimeError(f"runs/robustness/{name}.json block set differs from the manifest's included blocks by {sorted(blocks ^ inc)}; "
                               f"rerun scripts/mayo/robustness_{name}.py")
    # ---- geometry (Table cf-geometry)
    dg, lg, lo, asc = G["direction_geometry"], G["label_geometry"], G["leave_one_out"], G["association"]
    M["cfGeoCosNih"] = fx(dg["nih"]["offdiag_cos_model_mean"], 3); M["cfGeoCosRandom"] = fx(dg["nih"]["random_pair_abs_cos_mean"], 3)
    M["cfGeoCosChex"] = fx(dg["chexpert"]["offdiag_cos_model_mean"], 2)
    cx = G["labels"]["chexpert"]["concepts"]; i, j = cx.index("Effusion"), cx.index("Consolidation")
    M["cfGeoCosEffCons"] = fx(dg["chexpert"]["cos_model_mean_matrix"][i][j], 2); M["cfGeoPhiEffCons"] = fx(G["labels"]["chexpert"]["phi"][i][j], 2)
    M["cfGeoPhiNih"] = fx(lg["nih"]["offdiag_phi_mean"], 3)
    M["cfGeoOwnWinsNihPct"] = pct(lo["nih"]["O_q_positive"], lo["nih"]["n_cells"]); M["cfGeoOwnWinsChexPct"] = pct(lo["chexpert"]["O_q_positive"], lo["chexpert"]["n_cells"])
    cos_ch = [pct(dg[ds]["coincide_cos_model"]["n_true"], dg[ds]["coincide_cos_model"]["n"]) for ds in ci.CHEST]
    M["cfGeoChanceCosMin"], M["cfGeoChanceCosMax"] = min(cos_ch), max(cos_ch)
    lab = [pct(lg[ds][k]["n_true"], lg[ds][k]["n"]) for ds in ci.CHEST for k in ("coincide_phi", "coincide_cooccurrence")]
    M["cfGeoChanceLabelMin"], M["cfGeoChanceLabelMax"] = min(lab), max(lab)
    M["cfGeoOffdiagCells"] = f"{asc['chest']['n_cells']:,}".replace(",", "{,}")
    M["cfGeoRhoAdvCos"] = fx(asc["chest"]["rho_adv_vs_cos_model"]["rho"], 2); M["cfGeoRhoAdvCosP"] = fx(asc["chest"]["rho_adv_vs_cos_model"]["p"], 2)
    M["cfGeoRhoAdvPhi"] = fx(asc["chest"]["rho_adv_vs_phi"]["rho"], 3)
    M["cfGeoTwoAbove"] = lo["chest"]["n_above_ge2"]; M["cfGeoTwoAboveN"] = lo["chest"]["n_cells"]; M["cfGeoTwoAbovePct"] = pct(lo["chest"]["n_above_ge2"], lo["chest"]["n_cells"])
    fam = [pct(v["O_family_positive"], v["n_cells"]) for v in G["coco_families"].values()]
    M["cfGeoCocoFamMinPct"], M["cfGeoCocoFamMaxPct"] = min(fam), max(fam)
    M["cfGeoFamPhiPersonBicycle"] = fx(G["coco_families"]["person+bicycle"]["phi_within_family_mean"], 2)
    sc = G["seed_cell"]; comp = {c["d"]: c for c in sc["competitors"]}
    M["cfSeedCosNodule"] = fx(comp["Nodule"]["cos_model"], 2); M["cfSeedWNodule"] = fx(comp["Nodule"]["W_qd"], 3); M["cfSeedWEff"] = fx(sc["W_qq"], 3)
    M["cfSeedCosAtel"] = fx(comp["Atelectasis"]["cos_model"], 2); M["cfSeedWAtel"] = fx(comp["Atelectasis"]["W_qd"], 3)
    if sc["argmax_other"] != "Nodule" or min(comp, key=lambda d: comp[d]["cos_model"]) != "Nodule" or max(comp, key=lambda d: comp[d]["cos_model"]) != "Atelectasis":
        warn.append("seed cell: Nodule is no longer the strongest / least similar competitor or Atelectasis the most similar (prose names them)")
    # ---- scale (Table cf-scale)
    i1 = S["item1_margin_scale"]
    M["cfScaleAgree"] = sum(i1[ds]["agreement_owned_p_vs_owned_m"]["a_yes_b_yes"] + i1[ds]["agreement_owned_p_vs_owned_m"]["a_no_b_no"] for ds in ci.DATASETS)
    M["cfScaleAgreeN"] = sum(i1[ds]["agreement_owned_p_vs_owned_m"]["n"] for ds in ci.DATASETS)
    M["cfScaleOwned"] = sum(i1[ds]["owned_p"] for ds in ci.DATASETS); M["cfScaleOwnedKeepFull"] = sum(i1[ds]["owned_p_and_Om_ci_pos_and_beats_random_m"] for ds in ci.DATASETS)
    if sum(i1[ds]["owned_p_and_Om_pos"] for ds in ci.DATASETS) != M["cfScaleOwned"]:
        warn.append("scale: not every owned cell keeps O^m_q > 0 (prose says 'every owned cell keeps')")
    sat = S["item2_ceiling"]["saturation"]["per_dataset_median_block_sat"]
    for ds, D in DS_MACRO.items():
        M[f"cfScaleSat{D}"] = fx(sat[ds], 2)
    uns = [S["item2_ceiling"]["strata"][ds]["unsaturated"] for ds in ci.CHEST]
    M["cfScaleUnsatOwned"] = sum(u["owned_p_evaluable"] for u in uns); M["cfScaleUnsatOwnedKept"] = sum(u["owned_p_O_pos"] for u in uns)
    M["cfScaleUnsatOtherPos"] = sum(u["not_owned_O_pos"] for u in uns); M["cfScaleUnsatOther"] = sum(u["not_owned_evaluable"] for u in uns)
    if M["cfScaleUnsatOwnedKept"] != M["cfScaleUnsatOwned"]:
        warn.append(f"scale: unsaturated rows keep O_q>0 in {M['cfScaleUnsatOwnedKept']} of {M['cfScaleUnsatOwned']} owned chest cells (prose says 'all')")
    d = [abs(c["refit"][f"O_seed{k}"] - c["p_scale"]["O"]) for c in S["cells"] if c.get("refit", {}).get("n_seeds") for k in (1, 2) if c["refit"].get(f"O_seed{k}") is not None]
    M["cfScaleRefitMedian"] = fx(statistics.median(d), 2)
    i3 = S["item3_refit"]["per_dataset"]
    for ds, D in (("nih", "Nih"), ("coco", "Coco")):
        M[f"cfRefitStable{D}"] = i3[ds]["owned_p_O_pos_both_refits"]; M[f"cfRefitOwned{D}"] = i3[ds]["owned_p"]
        if i3[ds]["owned_p"] != M[f"cf{D}Owned"]:
            warn.append(f"scale: refit owned count on {ds} ({i3[ds]['owned_p']}) differs from the manifest ({M[f'cf{D}Owned']})")
    b5 = S["item5_batch"]; dist = b5["distribution"]["vis.last"]
    M["cfScaleBatchMin"] = fx(dist["max_abs_candidate_logit_diff"]["min"], 2); M["cfScaleBatchMax"] = fx(dist["max_abs_candidate_logit_diff"]["max"], 2)
    M["cfScaleBatchAgree"] = dist["margin_sign_agreement_true"]; M["cfScaleBatchN"] = dist["n_blocks"]; M["cfScaleBatchDisagree"] = dist["margin_sign_agreement_false"]
    if b5["owned_cells_in_sign_disagreement_blocks"]:
        warn.append(f"scale: {b5['owned_cells_in_sign_disagreement_blocks']} owned cell(s) in sign-disagreeing blocks (prose says none)")
    cb = b5["clean_baseline_core_vs_locus"]; rest = sorted({round(x["frac_identical"], 2) for x in cb if x["frac_identical"] < 1.0})
    M["cfScaleBaselineN"] = len(cb); M["cfScaleBaselineIdentical"] = sum(x["frac_identical"] >= 1.0 for x in cb)
    M["cfScaleBaselineRestPct"] = pct(min(rest), 1) if rest else 100; M["cfScaleBaselineDpPct"] = f"{100 * max(x['frac_abs_dp_gt_0.1'] for x in cb):.1f}"
    if len(rest) > 1:
        warn.append(f"scale: non-identical baseline blocks have several agreement fractions {rest} (prose quotes one)")
    i4 = S["item4_connector"]["per_dataset"]
    for ds, D in (("nih", "Nih"), ("coco", "Coco")):
        M[f"cfConnWqq{D}"] = fx(i4[ds]["p_scale"]["median_abs_W_qq_connector"], 3); M[f"cfPrimWqq{D}"] = fx(i4[ds]["p_scale"]["median_abs_W_qq_primary"], 3)
    # ---- pairs (Table cf-pairs) and the per-concept O values quoted in Section 5.4 (from the manifest)
    t1 = P["task1_paired_deltas"]
    def pair(small, large, ds):
        return next(x for x in t1 if x["small"] == small and x["large"] == large and x["dataset"] == ds)
    for name, (small, large) in (("FourTwelve", ("gemma3-4", "gemma3-12")), ("TwelveTwentySeven", ("gemma3-12", "gemma3-27"))):
        e = pair(small, large, "nih")["per_concept"]["Effusion"]["delta_O"]
        M[f"cfPairDeltaEff{name}"] = fx(e["estimate"], 2, True); M[f"cfPairDeltaEff{name}Lo"] = fx(e["ci95"][0], 2, True); M[f"cfPairDeltaEff{name}Hi"] = fx(e["ci95"][1], 2, True)
    M["cfPairsFiveOrSix"] = sum(x["n_concepts_delta_O_ci_excludes_zero"] >= 5 for x in t1); M["cfPairsN"] = len(t1)
    ratios = [x["median_abs_delta_O_over_refit_sd"] for x in t1 if x.get("median_abs_delta_O_over_refit_sd") is not None]
    M["cfPairsRatioMin"] = fx(min(ratios), 1); M["cfPairsRatioMax"] = fx(max(ratios), 1)
    M["cfPairsRatioSingleMax"] = fx(max(x["max_abs_delta_O_over_refit_sd"] for x in t1 if x.get("max_abs_delta_O_over_refit_sd") is not None), 1)
    def O(mk, ds, c):
        return fx(float(cells[(mk, ds, c)]["O_q"]), 2, True)
    M["cfOGemmaTwelveEff"] = O("gemma3-12", "nih", "Effusion")
    if O("gemma3-12", "chexpert", "Effusion") != M["cfOGemmaTwelveEff"]:
        warn.append(f"Gemma 3 12B Effusion O differs between NIH {M['cfOGemmaTwelveEff']} and CheXpert {O('gemma3-12', 'chexpert', 'Effusion')} (prose: 'on both chest sets')")
    M["cfOGemmaTwelveMass"] = O("gemma3-12", "nih", "Mass"); M["cfOGemmaTwelvePneu"] = O("gemma3-12", "chexpert", "Pneumothorax")
    M["cfOGemmaTwentySevenEff"] = O("gemma3-27", "nih", "Effusion"); M["cfOGemmaTwentySevenMass"] = O("gemma3-27", "nih", "Mass"); M["cfOGemmaTwentySevenPneu"] = O("gemma3-27", "chexpert", "Pneumothorax")
    M["cfOMedFourPneu"] = O("medgemma-4", "chexpert", "Pneumothorax"); M["cfOMedTwentySevenPneu"] = O("medgemma-27", "chexpert", "Pneumothorax")
    M["cfOMedFourEdema"] = O("medgemma-4", "chexpert", "Edema"); M["cfOMedTwentySevenEdema"] = O("medgemma-27", "chexpert", "Edema")
    ex = P["task2_ceiling"]["examples"]; om = [v for k in ("gemma3-4/nih", "gemma3-4/chexpert") for v in ex[k]["per_concept"].values()]
    M["cfPairsGemmaFourOmNear"] = fx(max(v["O_m"] for v in om), 1); M["cfPairsGemmaFourOmFar"] = fx(min(v["O_m"] for v in om), 1)
    if any(v["O_m_ci95"][1] >= 0 for v in om):
        warn.append("pairs: a Gemma 3 4B chest O^m interval reaches zero (prose says all intervals below zero)")
    for mk, name in (("llava15-7", "cfLlavaSevenCoco"), ("llava15-13", "cfLlavaThirteenCoco")):
        M[name] = sum(t(r["owned"]) for r in rows if t(r["cell"]) and r["model_key"] == mk and r["dataset"] == "coco")
    # ---- validation (Table cf-validation)
    ag = V["answer_direction"]["aggregate"]
    for ds, D in (("nih", "Nih"), ("chexpert", "Chex")):
        M[f"cfValAurocAns{D}"] = fx(ag[f"{ds}_median_auroc_answer_dir"], 2); M[f"cfValAurocProbe{D}"] = fx(ag[f"{ds}_median_auroc_probe"], 2)
    M["cfValAurocAnsClean"] = fx(ag["median_auroc_answer_dir_vs_clean_answer"], 2); M["cfValAurocProbeClean"] = fx(ag["median_auroc_probe_vs_clean_answer"], 2)
    cs = [ag[f"{ds}_median_cos_sigma_projected"] for ds in ci.CHEST]
    M["cfValCosSigmaChestMin"] = fx(min(cs), 2); M["cfValCosSigmaChestMax"] = fx(max(cs), 2); M["cfValCosSigmaCoco"] = fx(ag["coco_median_cos_sigma_projected"], 2)
    csel = V["column_selectivity"]["per_dataset"]; kn = V["known_label"]["per_dataset"]
    for ds, tag in (("nih", "Nih"), ("chexpert", "Chex"), ("coco", "Coco")):
        M[f"cfValColSel{tag}"] = fx(csel[ds]["median_S_d"], 2); M[f"cfValColSelGeHalf{tag}"] = str(csel[ds]["n_S_ge_0.5"]); M[f"cfValColSelN{tag}"] = str(csel[ds]["n_cells"])
    M["cfValColSelGeHalfChest"] = str(csel["nih"]["n_S_ge_0.5"] + csel["chexpert"]["n_S_ge_0.5"]); M["cfValColSelNChest"] = str(csel["nih"]["n_cells"] + csel["chexpert"]["n_cells"])
    M["cfValKnownSignAgree"] = str(kn["chexpert"]["n_sign_agrees"]); M["cfValKnownN"] = str(kn["chexpert"]["n_cells"])
    M["cfValKnownOwnedSummary"] = str(kn["chexpert"]["n_owned_summary"]); M["cfValKnownOwnedKnown"] = str(kn["chexpert"]["n_owned_known"])
    col = V["column_selectivity"]
    for ds, D in DS_MACRO.items():
        M[f"cfColSel{D}"] = fx(col["per_dataset"][ds]["median_S_d"], 2)
    M["cfColSelChestGe"] = col["chest"]["n_S_ge_0.5"]; M["cfColSelChestN"] = col["chest"]["n_cells"]
    M["cfColSelCocoGe"] = col["per_dataset"]["coco"]["n_S_ge_0.5"]; M["cfColSelCocoN"] = col["per_dataset"]["coco"]["n_cells"]
    kl = V["known_label"]["per_dataset"]["chexpert"]
    M["cfKnownSignAgree"] = kl["n_sign_agrees"]; M["cfKnownN"] = kl["n_cells"]; M["cfKnownOwned"] = kl["n_owned_summary"]
    # owned cells (paper verdict) that stay owned under the percentile verdict on the known-label rows
    kc = [c for b in V["known_label"]["blocks"] if b["dataset"] == "chexpert" for c in b["per_question"].values() if c["summary_owned"]]
    M["cfKnownOwnedKept"] = sum(bool(c["owned_known"]) for c in kc)
    if len(kc) != M["cfKnownOwned"]:
        warn.append(f"validation: {len(kc)} summary-owned CheXpert cells in known_label blocks vs {M['cfKnownOwned']} in per_dataset")
    # ---- PRECISION (Table cf-precision), from the included blocks' summaries
    prec = [(k, b["s"]["precision"]) for k, b in wb.items() if b["s"].get("precision")]
    done = [(k, st, p[st]) for k, p in prec for st in p["settings"] if p.get(st, {}).get("status") == "COMPLETE"]
    M["cfPrecisionBlocks"] = len(prec); M["cfPrecisionMaxDW"] = fx(max(r["max_abs_dW"] for _, _, r in done), 3) if done else "--"
    M["cfPrecisionVerdictChanges"] = sum(12 - r["n_agree_competitor"] - r["n_agree_random_p95"] for _, _, r in done)
    if M["cfPrecisionVerdictChanges"]:
        warn.append(f"precision: {M['cfPrecisionVerdictChanges']} point-verdict change(s) (prose says no verdict changes)")
    # ---- label gap (t3 of the included blocks)
    gaps = {ds: [] for ds in ci.DATASETS}
    for (mk, ds), b in wb.items():
        v = (b["s"].get("t3") or {}).get("all|median_label_gap")
        if isinstance(v, dict) and v.get("estimate") is not None:
            gaps[ds].append(v["estimate"])
    coco = [g for g in gaps["coco"] if g <= -3]
    M["cfLabelGapCocoBlocks"] = len(coco); M["cfLabelGapCocoHi"] = f"{max(coco):.0f}"; M["cfLabelGapCocoLo"] = f"{min(coco):.0f}"
    M["cfLabelGapNihBlocks"] = sum(abs(g) <= 1 for g in gaps["nih"])
    if len(gaps["coco"]) != M["cfCocoBlocks"] or len(gaps["nih"]) != M["cfNihBlocks"]:
        warn.append(f"label gap defined for {len(gaps['coco'])} COCO / {len(gaps['nih'])} NIH blocks, not every included block")
    # ---- coverage (Table cf-coverage): modules completed per block, from the manifest
    PLANNED = ("CORE", "CALIBRATION", "PROMPT", "DOSE", "REFIT", "LOCUS", "LOCUS_CALIBRATION"); CHEX_EXT = ("PROMPT", "DOSE", "REFIT", "LOCUS", "LOCUS_CALIBRATION")
    done = {}
    for r in rows:
        done.setdefault((r["model_key"], r["dataset"]), set(m for m in r["completed_modules"].split("|") if m))
    M["cfCoverageBlocksAllSeven"] = sum(all(m in s for m in PLANNED) for s in done.values())
    M["cfCoverageChexFull"] = sum(ds == "chexpert" and all(m in s for m in CHEX_EXT) for (mk, ds), s in done.items())
    # ---- refit full-grade transitions (Table cf-refit), runs/robustness/refit.json
    Rf = json.loads((ROB / "refit.json").read_text())
    for ds, D in DS_MACRO.items():
        g = Rf["transitions"].get(ds)
        if not g:
            continue
        s = g["survival"]
        M[f"cfRefitBlocks{D}"] = g["n_blocks"]; M[f"cfRefitGradeOwned{D}"] = s["owned"]["n"]
        M[f"cfRefitGradeStable{D}"] = s["owned"]["both"]; M[f"cfRefitGradeStableAny{D}"] = s["owned"]["any"]; M[f"cfRefitGradeLost{D}"] = s["owned"]["none"]
        M[f"cfRefitGradeComp{D}"] = s["stronger_competitor"]["n"]; M[f"cfRefitGradeCompStable{D}"] = s["stronger_competitor"]["both"]
        M[f"cfRefitGradeUnres{D}"] = s["unresolved"]["n"]; M[f"cfRefitGradeUnresStable{D}"] = s["unresolved"]["both"]
        if ds in ("nih", "coco") and s["owned"]["n"] != M[f"cf{D}Owned"]:
            warn.append(f"refit: {s['owned']['n']} owned cells among the REFIT blocks on {ds} vs {M[f'cf{D}Owned']} in the manifest")
        if g["seed0_regrade_agrees"] != g["n_cells"]:
            warn.append(f"refit: seed-0 regrade with {Rf['meta']['draws']} draws agrees with the paper's grade in {g['seed0_regrade_agrees']} of {g['n_cells']} {ds} cells")
    # ---- alignment versus ownership (Table cf-validation row), validation.json
    al = V["answer_direction"]["aggregate"]["alignment_vs_ownership"]["all"]
    M["cfValAlignN"] = al["n_cells"]; M["cfValAlignSpearman"] = fx(al["spearman_cos_sigma_vs_O_q"]["rho"], 2); M["cfValAlignP"] = fp(al["spearman_cos_sigma_vs_O_q"]["p"])
    M["cfValAlignOwnedSpearman"] = fx(al["spearman_cos_sigma_vs_owned"]["rho"], 2); M["cfValAlignOwnedP"] = fp(al["spearman_cos_sigma_vs_owned"]["p"])
    # ---- ANSDIR (Table cf-ansdir)
    ans = {k: b["s"]["ansdir"]["per_question"] for k, b in wb.items() if b["s"].get("ansdir")}
    def owned_n(pq):
        return sum(ci.owned(v) for v in pq.values())
    def medcos(pq):
        return statistics.median(v["cos_to_logistic_model"] for v in pq.values())
    M["cfAnsQwenOwnedNih"] = owned_n(ans[("q25-7", "nih")]); M["cfAnsQwenOwnedChex"] = owned_n(ans[("q25-7", "chexpert")])
    M["cfAnsLingOwnedNih"] = owned_n(ans[("lingshu-32", "nih")]); M["cfAnsLingOwnedChex"] = owned_n(ans[("lingshu-32", "chexpert")])
    e = ans[("q25-7", "nih")]["Effusion"]; M["cfAnsQwenEffO"] = fx(e["O_q"], 2, True); M["cfAnsQwenEffW"] = fx(e["W_qq"], 2)
    for mk, D in (("q25-7", "Qwen"), ("lingshu-32", "Ling")):
        c = [medcos(ans[(mk, ds)]) for ds in ci.CHEST]; M[f"cfAnsCos{D}Min"] = fx(min(c), 2); M[f"cfAnsCos{D}Max"] = fx(max(c), 2)
    c = [medcos(pq) for (mk, ds), pq in ans.items() if ds == "coco"]; M["cfAnsCosCocoMin"] = fx(min(c), 2); M["cfAnsCosCocoMax"] = fx(max(c), 2)
    M["cfAnsGemOwnedNih"] = owned_n(ans[("gemma3-12", "nih")]); M["cfAnsGemOwnedChex"] = owned_n(ans[("gemma3-12", "chexpert")])
    # aggregate over every ANSDIR block: cells, a_q owned, label direction owned (paper grade, validation.json), and
    # cells whose a_q write beats the strongest logistic competitor (simultaneous lower bound above zero)
    vb = V["answer_direction"]["blocks"]
    def beats_n(pq):
        return sum(v.get("own_minus_max_logistic_competitor_ci95_percentile", [0, 0])[0] > 0 for v in pq.values())
    for tag, sel in (("Chest", ci.CHEST), ("Coco", ("coco",))):
        keys = [k for k in ans if k[1] in sel]
        M[f"cfAns{tag}N"] = 6 * len(keys); M[f"cfAns{tag}Blocks"] = len(keys)
        M[f"cfAns{tag}OwnedA"] = sum(owned_n(ans[k]) for k in keys)
        M[f"cfAns{tag}OwnedLabel"] = sum(sum(c["core_owned"] for c in vb[f"{k[0]}/{k[1]}"]["per_question"].values()) for k in keys)
        M[f"cfAns{tag}Beats"] = sum(beats_n(ans[k]) for k in keys)
    M["cfAnsBlocks"] = len(ans)


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
    warn = []
    robustness_macros(M, rows, wb, warn)
    for w in warn:
        print("  WARNING:", w)
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
