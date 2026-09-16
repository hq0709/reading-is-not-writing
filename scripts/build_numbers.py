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
  round 2                runs/robustness/round2.json (VALID, ALTDIRD, ANSDIRT, ATTR, full-grade PRECISION); each section's block set must
                         equal the included blocks carrying that module (ANSDIRT and ATTR: blocks scored on every test row).
  ATTR                   per dataset and pooled over the chest sets: owned attribute cells and owned clinical cells of the same
                         nine-direction family, the median probe and answer AUROC range over the three attributes, and the largest
                         median |cosine| between an attribute direction and a clinical normal.
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


def sci(x, nd=1) -> str:
    """Math-mode scientific notation: 1.7\\times10^{-4} (same formatter as build_robustness_tables.sci)."""
    m, e = f"{x:.{nd}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


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
    ufg = S["item2_ceiling"]["unsaturated_full_grade"]["chest"]
    M["cfScaleUnsatGradeN"] = ufg["owned_p_evaluable"]; M["cfScaleUnsatGradeKept"] = ufg["owned_p_owned_unsaturated"]
    M["cfScaleUnsatGradeRef"] = ufg["owned_p_reference_kept"]; M["cfScaleUnsatGradeAdv"] = ufg["owned_p_advantage_kept"]
    M["cfScaleUnsatGradeGained"] = ufg["not_owned_owned_unsaturated"]; M["cfScaleUnsatGradeOtherN"] = ufg["not_owned_evaluable"]
    if ufg["owned_p_evaluable"] != ufg["owned_p"]:
        warn.append(f"scale: {ufg['owned_p'] - ufg['owned_p_evaluable']} owned chest cell(s) have too few unsaturated rows to regrade")
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
    M["cfPairsRatioN"] = len(ratios); M["cfPairsRatioAboveOne"] = sum(r >= 1 for r in ratios)
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
    M["cfValKnownOwnedFull"] = str(kn["chexpert"]["n_owned_summary_and_owned_known_full_grade"])
    M["cfValKnownOwnedFullAll"] = str(kn["chexpert"]["n_owned_known_full_grade"])
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
    # point columns (36 clinical cells, two point verdicts); the full-grade macros cfPrecisionMaxDW / cfPrecisionVerdictChanges
    # come from runs/robustness/round2.json in round2_macros()
    prec = [(k, b["s"]["precision"]) for k, b in wb.items() if b["s"].get("precision")]
    done = [(k, st, p[st]) for k, p in prec for st in p["settings"] if p.get(st, {}).get("status") == "COMPLETE"]
    M["cfPrecisionBlocks"] = len(prec); M["cfPrecisionPointMaxDW"] = fx(max(r["max_abs_dW"] for _, _, r in done), 3) if done else "--"
    M["cfPrecisionPointVerdictChanges"] = sum(12 - r["n_agree_competitor"] - r["n_agree_random_p95"] for _, _, r in done)
    if M["cfPrecisionPointVerdictChanges"]:
        warn.append(f"precision: {M['cfPrecisionPointVerdictChanges']} point-verdict change(s) over the 36 clinical cells")
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
    ch = [medcos(pq) for (mk, ds), pq in ans.items() if ds in ci.CHEST]
    M["cfAnsCosChestMin"] = fx(min(ch), 2); M["cfAnsCosChestMax"] = fx(max(ch), 2); M["cfAnsCosChestMed"] = fx(statistics.median(ch), 2)
    M["cfAnsCosCocoMed"] = fx(statistics.median(c), 2)
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


def prose_macros(M: dict, rows: list[dict], wb: dict, warn: list) -> None:
    """Numbers the prose quotes directly and that used to be typed into the text: the seed cell of Qwen2.5-VL-7B on NIH,
    the dose response of Figure 2(c), the largest Qwen2.5-VL-72B chest ownership, the token-weighted writes of the two
    Qwen blocks, the Gemma 3 4B clean-yes share, the shared-tower direction cosine, and the two gate bounds."""
    import math
    # ---- seed cell (Table cf-seed is built from this same block)
    e = wb[("q25-7", "nih")]["s"]["core"]["per_question"]["Effusion"]
    M["cfSeedEffRandP"] = fx(e["random_p95"], 3); M["cfSeedEffSham"] = fx(e["abs_sham"], 3)
    M["cfSeedEffO"] = fx(e["O_q"], 3, True); M["cfSeedEffOLo"] = fx(e["O_q_ci95_percentile"][0], 3, True)
    M["cfSeedEffOHi"] = fx(e["O_q_ci95_percentile"][1], 3, True)
    # the seed row of the write matrix (Figure 1 draws every bar from these macros): the three competitors the prose does not name
    Wrow = wb[("q25-7", "nih")]["s"]["core"]["W"]["Effusion"]
    for c, name in (("Pneumothorax", "cfSeedWPneu"), ("Cardiomegaly", "cfSeedWCard"), ("Mass", "cfSeedWMass")):
        M[name] = fx(Wrow[f"concept:{c}"], 3)
    for c, name in (("Effusion", "cfSeedWEff"), ("Nodule", "cfSeedWNodule"), ("Atelectasis", "cfSeedWAtel")):
        if name in M and fx(Wrow[f"concept:{c}"], 3) != M[name]:
            warn.append(f"seed: summary W[Effusion, {c}] {fx(Wrow[f'concept:{c}'], 3)} differs from geometry.json {M[name]}")
    if e["argmax_other"] != "Nodule" or e["rank_in_random_family"] != 2:
        warn.append(f"seed: Effusion's strongest competitor is {e['argmax_other']} and its rank {e['rank_in_random_family']} "
                    f"(prose says Nodule and second)")
    # ---- dose response, from the cache Figure 2(c) plots, under the same gate
    dose = json.loads((ci.RUNS / "figures" / "dose_curves.json").read_text())
    dose = {k: v for k, v in dose.items() if v}
    done = {}
    for r in ci.read_manifest():
        done.setdefault((r["model_key"], r["dataset"]), (t(r["block_included"]), set(m for m in r["completed_modules"].split("|") if m)))
    expect = {f"{mk}/{ds}" for (mk, ds), (inc, mods) in done.items() if inc and "DOSE" in mods}
    if set(dose) != expect:
        raise RuntimeError(f"runs/figures/dose_curves.json differs from the included blocks with DOSE by {sorted(set(dose) ^ expect)}; "
                           f"delete it and rerun cftransfer.figures.dose_curves")
    def med(ds, a):
        xs = [v[str(a)] for k, v in dose.items() if k.endswith("/" + ds) and str(a) in v]
        return float(np.nanmedian(xs))
    M["cfDoseCocoLow"] = fx(med("coco", -0.1), 2, True); M["cfDoseCocoPeak"] = fx(med("coco", 0.25), 2, True)
    M["cfDoseCocoHalf"] = fx(med("coco", 0.5), 2, True)
    alphas = sorted({float(a) for v in dose.values() for a in v})
    nih = [med("nih", a) for a in alphas]
    M["cfDoseNihMin"] = fx(min(nih), 3, True); M["cfDoseNihMax"] = fx(max(nih), 3, True)
    if med("coco", 0.5) <= 0:
        warn.append("dose: the COCO median at alpha +0.5 is not positive (prose says it remains positive)")
    if max(nih) >= 0:
        warn.append("dose: a NIH dose median is not negative (prose gives a negative range)")
    # ---- largest Qwen2.5-VL-72B chest ownership (Section 5.5 names Atelectasis)
    q72 = wb[("q25-72", "nih")]["s"]["core"]["per_question"]
    M["cfOQwenSeventyTwoAtel"] = fx(q72["Atelectasis"]["O_q"], 2)
    if max(q72, key=lambda q: q72[q]["O_q"]) != "Atelectasis":
        warn.append("families: Atelectasis is no longer Qwen2.5-VL-72B's largest NIH ownership (prose names it)")
    # ---- token-weighted writes of the two Qwen blocks the prose names
    tw_chest = tw_chest_n = tw_coco = tw_coco_n = 0
    for (mk, ds), b in wb.items():
        a = b["s"].get("tokenw")
        if not a or mk not in ("q25-7", "q3-8"):
            continue
        for v in ("tokenw", "topq"):                       # the two variants of Table cf-tokenw
            pq = (a.get(v) or {}).get("per_question") or {}
            n = len(pq); own = sum(ci.owned(c) for c in pq.values())
            if ds in ci.CHEST:
                tw_chest += own; tw_chest_n += n
            else:
                tw_coco += own; tw_coco_n += n
    M["cfTokenwChestOwned"] = tw_chest; M["cfTokenwChestN"] = tw_chest_n
    M["cfTokenwCocoOwned"] = tw_coco; M["cfTokenwCocoN"] = tw_coco_n
    # ---- Gemma 3 4B clean-yes share on the chest sets (scale.json saturation, high side)
    S = json.loads((ROB / "scale.json").read_text())
    sat = [b["saturation"]["sat_hi"] for b in S["blocks"] if b.get("status") == "OK" and b["model"] == "gemma3-4" and b["dataset"] in ci.CHEST]
    M["cfGemmaFourYesMin"] = pct(min(sat), 1); M["cfGemmaFourYesMax"] = pct(max(sat), 1)
    # ---- shared-tower write-vector cosine over the bf16-equal Gemma 3 / MedGemma pairs (floored, the prose says "above")
    P = json.loads((ROB / "pairs.json").read_text())
    cs = [pr["write_vectors"]["min_cos"] for fam in ("gemma3", "medgemma") for d in P["task3_tower_sharing"][fam]["per_dataset"].values()
          for pr in d["pairs"] if (pr.get("write_vectors") or {}).get("min_cos") is not None and not pr["write_vectors"]["identical"]]
    M["cfPairsTowerCosMin"] = f"{math.floor(min(cs) * 1e4) / 1e4:.4f}"
    # ---- gate bounds: fp32-versus-model logit difference (ceiled, the prose says "below") and the declared batch deviation
    g = [((b.get("preflight") or {}).get("vis.last") or {}).get("G_fp32_vs_model_max_abs_diff") for b in S["blocks"] if b.get("status") == "OK"]
    g = [x for x in g if x is not None]
    M["cfGateFpMax"] = f"{math.ceil(max(g) * 100) / 100:.2f}"
    # ---- COCO ownership margins of the Qwen / Lingshu / Gemma families (Section 5.3)
    fams = ("q25", "q3-", "lingshu", "gemma3")
    co = [f(r["O_q"]) for r in rows if t(r["cell"]) and t(r["block_included"]) and r["dataset"] == "coco" and t(r["owned"])
          and r["model_key"].startswith(fams)]
    M["cfCocoMarginMin"] = fx(math.floor(min(co) * 100) / 100, 2); M["cfCocoMarginMax"] = fx(math.ceil(max(co) * 100) / 100, 2)
    # ---- InternVL3.5 chest writes (Section 5.5)
    iv = [abs(f(r["W_qq"])) for r in rows if t(r["cell"]) and t(r["block_included"]) and r["dataset"] in ci.CHEST
          and r["model_key"].startswith("iv35")]
    M["cfInternvlWqqMax"] = fx(math.ceil(max(iv) * 100) / 100, 2)
    # ---- |cos| of the difference-of-means and pattern directions with the logistic normal (Section 5.6): per block the
    #      median over concepts, from fits/vis.last/altdir_seed0.npz (cos_model over cftransfer.altdir.FAMILY_ORDER)
    FAM_ORDER = ("logistic", "dom", "pattern", "orth", "resid")
    li = FAM_ORDER.index("logistic")
    cosb = []
    for (mk, ds) in wb:
        f_npz = ci.RUNS / mk / ds / "fits" / "vis.last" / "altdir_seed0.npz"
        if not f_npz.exists():
            continue
        cm = np.load(f_npz, allow_pickle=False)["cos_model"]
        for fam in ("dom", "pattern"):
            fi = FAM_ORDER.index(fam)
            cosb.append(float(np.median([abs(cm[c, li, fi]) for c in range(cm.shape[0])])))
    M["cfAltdirCosMin"] = fx(math.floor(min(cosb) * 100) / 100, 2); M["cfAltdirCosMax"] = fx(math.ceil(max(cosb) * 100) / 100, 2)
    M["cfAltdirCosBlocks"] = len(cosb) // 2
    if M["cfAltdirCosBlocks"] != M["cfAltdirBlocks"]:
        warn.append(f"altdir: {M['cfAltdirCosBlocks']} blocks carry a fitted-direction file vs {M['cfAltdirBlocks']} scored ALTDIR blocks "
                    f"(the prose reports the cosines across the scored blocks)")
    b5 = S["item5_batch"]["distribution"]["vis.last"]
    M["cfGateBatchMax"] = fx(b5["max_abs_candidate_logit_diff"]["max"], 2)
    if b5.get("max_abs_candidate_logit_diff", {}).get("argmax_block", "gemma3-12/nih").split("/")[0] != "gemma3-12":
        warn.append("gates: the largest declared batch deviation is no longer a Gemma 3 12B block (prose names it)")


def extcomp_macros(M: dict, wb: dict, warn: list) -> None:
    """EXTCOMP (Table cf-extcomp): blocks scored, cells owned under the six-direction family, and how many of them
    keep ownership when every extra dataset label with enough support joins the competitor family. Same source and
    same per-cell rule as scripts/build_robustness_tables.py extcomp()."""
    blocks = owned = ext = retained = 0
    for k, b in sorted(wb.items()):
        a = b["s"].get("extcomp")
        if not a:
            continue
        blocks += 1
        core = b["s"]["core"]["per_question"]
        for q, v in a["per_question"].items():
            lo = ci.owned(core[q])
            eo = bool(v["W_qq"] > 0 and v["W_qq"] > v.get("random_p95", 0) and v["W_qq"] > v.get("abs_sham", 0)
                      and v.get("verdict") == "fixed_family_advantage")
            owned += lo; ext += eo; retained += lo and eo
    M["cfExtcompBlocks"] = blocks; M["cfExtcompOwned"] = owned; M["cfExtcompRetained"] = retained; M["cfExtcompExtOwned"] = ext
    if ext != retained:
        warn.append(f"extcomp: {ext - retained} cell(s) owned under the extended family but not under the six directions "
                    f"(prose reports retention of the {owned} owned cells)")


def support_macros(M: dict, rows: list[dict], warn: list) -> None:
    """Support eligibility of the readability / answerability denominators. A cell needs at least ten known positives
    and ten known negatives in the grading cohort (Section 3.3); a cell below that support can never be graded
    readable or answerable, so it enters the published denominators as a failure. These macros give the counts and
    the eligible-only shares."""
    cells = [r for r in rows if t(r["cell"])]
    def inel(r):
        return r["n_pos"] == "" or r["n_neg"] == "" or int(r["n_pos"]) < 10 or int(r["n_neg"]) < 10
    groups = {"Nih": ("nih",), "Chex": ("chexpert",), "Coco": ("coco",), "Chest": ci.CHEST}
    for D, dss in groups.items():
        probe = [r for r in cells if r["dataset"] in dss and t(r["probe_graded"]) and r["readable"] != ""]
        wm = [r for r in cells if r["dataset"] in dss and t(r["block_included"])]
        M[f"cfSupportInel{D}"] = sum(inel(r) for r in wm)
        pe = [r for r in probe if not inel(r)]; we = [r for r in wm if not inel(r)]
        M[f"cfSupportReadable{D}"] = sum(t(r["readable"]) for r in pe); M[f"cfSupportReadableN{D}"] = len(pe)
        M[f"cfSupportReadablePct{D}"] = pct(M[f"cfSupportReadable{D}"], len(pe))
        M[f"cfSupportAnswerable{D}"] = sum(t(r["answer_capable"]) for r in we); M[f"cfSupportAnswerableN{D}"] = len(we)
        M[f"cfSupportAnswerablePct{D}"] = pct(M[f"cfSupportAnswerable{D}"], len(we))
    if M["cfSupportReadableChest"] != M["cfChestReadable"] or M["cfSupportAnswerableChest"] != M["cfChestAnswerable"]:
        warn.append("support: an eligible-only chest cell was graded readable or answerable below the support rule")
    if M["cfSupportReadablePctChest"] < M["cfChestReadablePct"] or M["cfSupportAnswerablePctChest"] < M["cfChestAnswerablePct"]:
        warn.append("support: the eligible-only chest share is below the all-cell share (prose says it is higher)")


def dose_macros(M: dict, rows: list[dict]) -> None:
    """Blocks entering Figure 2(c) and the D column of Table cf-coverage: included blocks with DOSE completed."""
    done = {}
    for r in rows:
        done.setdefault((r["model_key"], r["dataset"]), (t(r["block_included"]), set(m for m in r["completed_modules"].split("|") if m)))
    for ds, D in DS_MACRO.items():
        M[f"cfDoseBlocks{D}"] = sum(inc and "DOSE" in mods for (mk, d), (inc, mods) in done.items() if d == ds)


def round2_macros(M: dict, rows: list[dict], wb: dict, warn: list) -> None:
    """Macros for the round-2 modules (Tables cf-valid, cf-altdird, cf-ansdirt, cf-attr and the full-grade columns of cf-precision), from
    runs/robustness/round2.json (scripts/mayo/robustness_round2.py). Each section's block set is checked against the included
    blocks that carry the module; owned counts on the test side are checked against the manifest."""
    R2 = json.loads((ROB / "round2.json").read_text())
    owned_cell = {(r["model_key"], r["dataset"], r["concept"]): t(r["owned"]) for r in rows if t(r["cell"]) and t(r["block_included"])}
    def key(block):
        mk, ds = block.split("/"); return (mk, ds)
    def done(k, mod):
        return mod in set(wb[k]["run"].get("completed_modules") or [])
    # ---- VALID (Table cf-valid)
    V = R2["valid"]; a = V["aggregate"]
    expect_blocks = {k for k in wb if k[1] == "chexpert" and wb[k]["s"].get("valid") and done(k, "VALID")}
    got = {key(b["block"]) for b in V["blocks"]}
    if got != expect_blocks:
        raise RuntimeError(f"round2.json valid blocks differ from the included CheXpert blocks with VALID by {sorted(got ^ expect_blocks)}; rerun scripts/mayo/robustness_round2.py")
    M["cfValidBlocks"] = a["blocks"]; M["cfValidCells"] = a["cells"]
    if len(a["valid_rows"]) != 1:
        warn.append(f"valid: blocks differ in valid rows {a['valid_rows']}")
    M["cfValidRows"] = a["valid_rows"][0] if a["valid_rows"] else "--"
    M["cfValidOwnedAgree"] = a["owned_agree"]; M["cfValidOwnedAgreePct"] = pct(a["owned_agree"], a["cells"])
    M["cfValidVerdictAgree"] = a["verdict_agree"]; M["cfValidVerdictAgreePct"] = pct(a["verdict_agree"], a["cells"])
    M["cfValidOwnedTest"] = a["owned_test"]; M["cfValidOwnedValid"] = a["owned_valid"]
    M["cfValidMedianDO"] = fx(a["median_abs_dO_q"], 3); M["cfValidNinetyDO"] = fx(a["p90_abs_dO_q"], 3)
    M["cfValidSupported"] = a["supported_cells"]
    M["cfValidReadableAgree"] = a["readable_agree_supported"]; M["cfValidReadableN"] = a["readable_compared_supported"]
    M["cfValidAnswerAgree"] = a["answer_capable_agree_supported"]; M["cfValidAnswerN"] = a["answer_capable_compared_supported"]
    man = sum(owned_cell[(b["model"], b["dataset"], q)] for b in V["blocks"] for q in b["per_question"])
    if a["test_regrade_vs_paper_mismatches"] or a["owned_test"] != man:
        warn.append(f"valid: test regrade owns {a['owned_test']} cells vs {man} in the manifest on the same blocks "
                    f"({a['test_regrade_vs_paper_mismatches']} verdict/ownership mismatches)")
    # ---- ALTDIRD (Table cf-altdird)
    A = R2["altdird"]; P = A["per_dataset"]
    expect_blocks = {k for k in wb if wb[k]["s"].get("altdird") and done(k, "ALTDIRD")}
    got = {key(b["block"]) for b in A["blocks"]}
    if got != expect_blocks:
        raise RuntimeError(f"round2.json altdird blocks differ from the included blocks with ALTDIRD by {sorted(got ^ expect_blocks)}; rerun scripts/mayo/robustness_round2.py")
    M["cfAltdirdBlocks"] = P["all"]["blocks"]
    for g, G in (("nih", "Nih"), ("chexpert", "Chex"), ("coco", "Coco"), ("chest", "Chest")):
        d = P[g]
        M[f"cfAltdird{G}Blocks"] = d["blocks"]; M[f"cfAltdird{G}N"] = d["cells"]
        for fam, F in (("dom_disp", "Dom"), ("pattern_disp", "Pattern"), ("logistic", "Logistic")):
            M[f"cfAltdird{G}{F}"] = d[fam]["owned"]; M[f"cfAltdird{G}{F}Pct"] = pct(d[fam]["owned"], d["cells"])
        for fam, F in (("dom_disp", "Dom"), ("pattern_disp", "Pattern")):
            M[f"cfAltdird{G}Cos{F}"] = fx(d[fam]["median_cos_to_logistic"], 2) if d["blocks"] else "--"
    gs = P["all"]["gram_spectrum"]
    M["cfAltdirdCondMin"] = fx(gs["condition_min"], 1); M["cfAltdirdCondMax"] = fx(gs["condition_max"], 1)
    M["cfAltdirdCondMedian"] = fx(gs["condition_median"], 1)
    M["cfAltdirdEigMin"] = sci(gs["min_eigenvalue"]); M["cfAltdirdEigMax"] = sci(gs["max_eigenvalue"])
    M["cfAltdirdGramDim"] = gs["n"][0] if isinstance(gs["n"], list) else gs["n"]
    if not gs["all_full_rank"]:
        warn.append("altdird: an R^T R / D spectrum is rank deficient (Table cf-altdird caption says full rank)")
    man = {g: sum(owned_cell[(b["model"], b["dataset"], q)] for b in A["blocks"] if b["dataset"] in dss for q in b["per_question"])
           for g, dss in (("chest", ci.CHEST), ("coco", ("coco",)))}
    for g in man:
        if man[g] != P[g]["logistic"]["owned"]:
            warn.append(f"altdird: logistic owned on the {g} ALTDIRD blocks {P[g]['logistic']['owned']} vs {man[g]} in the manifest")
    # ---- ANSDIRT (Table cf-ansdirt)
    T = R2["ansdirt"]; ag = T["aggregate"]
    M["cfAnsdirtBlocks"] = ag["all"]["blocks"]
    for g, G in (("chest", "Chest"), ("coco", "Coco")):
        M[f"cfAnsdirt{G}Blocks"] = ag[g]["blocks"]; M[f"cfAnsdirt{G}IYOwned"] = ag[g]["iy_owned"]
        M[f"cfAnsdirt{G}Kept"] = ag[g]["kept_pairs"]; M[f"cfAnsdirt{G}KeptN"] = ag[g]["pairs"]; M[f"cfAnsdirt{G}KeptPct"] = pct(ag[g]["kept_pairs"], ag[g]["pairs"])
        # unconditional: owned (cell, held-out template) pairs over every such pair, not only the pairs owned under the primary template
        M[f"cfAnsdirt{G}OwnedPairs"] = ag[g]["owned_pairs"]; M[f"cfAnsdirt{G}PairsAll"] = ag[g]["all_pairs"]
        M[f"cfAnsdirt{G}OwnedPairsPct"] = pct(ag[g]["owned_pairs"], ag[g]["all_pairs"])
    if T.get("partial_blocks"):
        print("  NOTE: ANSDIRT blocks scored on fewer than all test rows are not counted: "
              + ", ".join(f"{b['block']} ({b['status']}, {min(b['n_scored_rows_min'].values())}/{b['n_rows']} rows)" for b in T["partial_blocks"]))
    for b in T["blocks"]:
        if b.get("iy_owned_matches_summary_ansdir") is False:
            warn.append(f"ansdirt: {b['block']} IY-owned set differs from summary.json ansdir")
    # ---- ATTR (Table cf-attr): three non-clinical attributes of the same radiographs written inside one nine-direction family
    B = R2["attr"]; Pa = B["per_dataset"]
    expect_blocks = {k for k in wb if wb[k]["s"].get("attr") and done(k, "ATTR")
                     and wb[k]["s"]["attr"].get("n_scored_rows_min") == wb[k]["s"]["attr"].get("n_rows")}
    got = {key(b["block"]) for b in B["blocks"]}
    if got != expect_blocks:
        raise RuntimeError(f"round2.json attr blocks differ from the included blocks with ATTR scored on every row by "
                           f"{sorted(got ^ expect_blocks)}; rerun scripts/mayo/robustness_round2.py")
    M["cfAttrBlocks"] = Pa["chest"]["blocks"]
    for g, G in (("nih", "Nih"), ("chexpert", "Chex"), ("chest", "Chest")):
        d = Pa[g]
        M[f"cfAttr{G}Blocks"] = d["blocks"]
        M[f"cfAttr{G}AttrOwned"] = d["attr_owned"]; M[f"cfAttr{G}AttrN"] = d["attr_cells"]
        M[f"cfAttr{G}ClinOwned"] = d["clin_owned"]; M[f"cfAttr{G}ClinN"] = d["clin_cells"]
        M[f"cfAttr{G}AttrPct"] = pct(d["attr_owned"], d["attr_cells"]); M[f"cfAttr{G}ClinPct"] = pct(d["clin_owned"], d["clin_cells"])
    pa_ = Pa["chest"]["per_attribute"]
    au = [p["median_auroc_real"] for p in pa_.values() if p["blocks"]]
    an = [p["median_answer_auroc"] for p in pa_.values() if p["blocks"]]
    M["cfAttrAurocMin"] = fx(min(au), 3) if au else "--"; M["cfAttrAurocMax"] = fx(max(au), 3) if au else "--"
    M["cfAttrAnswerAurocMin"] = fx(min(an), 2) if an else "--"; M["cfAttrAnswerAurocMax"] = fx(max(an), 2) if an else "--"
    mm = Pa["chest"]["max_median_abs_cos_pair"] or {}
    M["cfAttrCosMax"] = fx(mm["median_abs_cos"], 2) if mm.get("median_abs_cos") is not None else "--"
    M["cfAttrReadable"] = Pa["chest"]["attr_readable"]
    # the same blocks' clinical probes, for "the attributes are read at least as well as the findings"
    ab = {tuple(b["block"].split("/")) for b in B["blocks"]}
    cl = [v["auroc_real"] for k in ab for v in ci.calibration(wb[k]).values() if v.get("auroc_real") is not None]
    M["cfAttrClinAuroc"] = fx(statistics.median(cl), 3) if cl else "--"
    if au and min(au) < statistics.median(cl):
        warn.append(f"attr: an attribute's median probe AUROC {min(au):.3f} is below the clinical median {statistics.median(cl):.3f} "
                    f"(prose says the attributes are read at least as well)")
    if Pa["chest"]["attribute_random_reference"]:
        warn.append("attr: an attribute question carries a random family (Table cf-attr caption says the sham alone is its reference)")
    if Pa["chest"]["clinical_random_reference"] is False:
        warn.append("attr: a clinical question in the nine-direction family lacks its random family")
    if M["cfAttrReadable"] != M["cfAttrChestAttrN"]:
        warn.append(f"attr: {M['cfAttrReadable']} of {M['cfAttrChestAttrN']} attribute probes are readable (prose says the attributes are read at least as well as the findings)")
    # ---- PRECISION full grade (Table cf-precision, full-grade columns)
    Pr = R2["precision"]; pa = Pr["aggregate"]
    M["cfPrecisionGradeBlocks"] = pa["blocks"]; M["cfPrecisionGradeCells"] = pa["graded_cells"]
    M["cfPrecisionVerdictChanges"] = pa["verdict_changes"]; M["cfPrecisionRefChanges"] = pa["steering_reference_changes"]
    M["cfPrecisionGradeChanges"] = pa["grade_changes"]
    M["cfPrecisionMaxDW"] = fx(pa["max_abs_dW_grid"], 3) if pa["max_abs_dW_grid"] is not None else "--"
    M["cfPrecisionMaxDC"] = fx(pa["max_abs_dcontrast"], 3) if pa["max_abs_dcontrast"] is not None else "--"
    if pa["blocks"] != M["cfPrecisionBlocks"]:
        warn.append(f"precision: {pa['blocks']} blocks carry the full grade vs {M['cfPrecisionBlocks']} with the module")


def main() -> Path:
    rows = ci.read_manifest()
    all_blocks = ci.load_runs()
    ci.check_manifest_agrees(all_blocks, rows)
    wb = ci.write_blocks(all_blocks)
    C = counts(rows); S = own_share(wb); A = altdir(rows)
    ch, co = C["chest"], C["coco"]
    cells = [r for r in rows if t(r["cell"])]
    M = {}
    warn = []
    # campaign
    M["cfCheckpoints"] = len({r["model_key"] for r in rows})
    M["cfBlocks"] = C["all"]["blocks"]; M["cfWriteBlocks"] = C["all"]["write_blocks"]; M["cfProbeBlocks"] = C["all"]["probe_blocks"]
    M["cfCells"] = C["all"]["write_cells"]; M["cfProbeCells"] = C["all"]["probe_cells"]
    M["cfChestBlocks"] = ch["write_blocks"]; M["cfChestProbeBlocks"] = ch["probe_blocks"]
    M["cfNihBlocks"] = C["nih"]["write_blocks"]; M["cfChexBlocks"] = C["chexpert"]["write_blocks"]; M["cfCocoBlocks"] = C["coco"]["write_blocks"]
    mks = {k[0] for k in wb}
    M["cfCheckpointsComplete"] = sum(all((m, d) in wb for d in ci.DATASETS) for m in mks)
    M["cfCheckpointsCompleteWord"] = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
                                      "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty",
                                      "Twenty-one", "Twenty-two"][M["cfCheckpointsComplete"]]
    # chest
    M["cfChestReadable"] = ch["readable"]; M["cfChestReadableN"] = ch["probe_cells"]; M["cfChestReadablePct"] = pct(ch["readable"], ch["probe_cells"])
    M["cfChestAnswerable"] = ch["answerable"]; M["cfChestAnswerableN"] = ch["write_cells"]; M["cfChestAnswerablePct"] = pct(ch["answerable"], ch["write_cells"])
    M["cfChestOwned"] = ch["owned"]; M["cfChestOwnedPct"] = pct(ch["owned"], ch["write_cells"])
    M["cfChestCompetitor"] = ch["stronger_competitor"]; M["cfChestCompetitorPct"] = pct(ch["stronger_competitor"], ch["write_cells"])
    M["cfChestRefMet"] = ch["reference_met"]; M["cfChestRefBelow"] = ch["write_cells"] - ch["reference_met"]; M["cfChestRefNotOwned"] = ch["reference_met_not_owned"]
    # ---- steering reference x verdict (Table cf-contingency, same row selection as build_cf_transfer_tables.table_contingency):
    #      chest cells, the readable-and-answerable chest subset, and COCO cells, each split into reference met / not met
    import csv as _csv, collections as _co
    _tf = lambda x: x.lower() in ("true", "1")
    prim = [r for r in _csv.DictReader(open(ci.RUNS / "manifest.csv"))
            if r["module"] == "CORE" and r["verdict"] and _tf(r["block_included"]) and _tf(r["is_primary_template"])]
    groups = {"Chest": [r for r in prim if r["dataset"] in ci.CHEST],
              "ReadAns": [r for r in prim if r["dataset"] in ci.CHEST and _tf(r["readable"]) and _tf(r["answer_capable"])],
              "Coco": [r for r in prim if r["dataset"] == "coco"]}
    for g, sel in groups.items():
        cc = _co.Counter((_tf(r["steering_reference"]), r["verdict"]) for r in sel)
        if g == "Chest" and (cc[(True, "fixed_family_advantage")] != ch["owned"]
                             or cc[(True, "stronger_competitor")] + cc[(True, "unresolved")] != ch["reference_met_not_owned"]):
            warn.append(f"contingency: manifest counts {dict(cc)} disagree with the chest summary (owned {ch['owned']}, ref-met-not-owned {ch['reference_met_not_owned']})")
        if g == "ReadAns":
            M["cfReadAnsRefMet"] = sum(cc[(True, v)] for v in ("fixed_family_advantage", "stronger_competitor", "unresolved"))
            M["cfReadAnsRefBelow"] = sum(cc[(False, v)] for v in ("fixed_family_advantage", "stronger_competitor", "unresolved"))
            M["cfReadAnsCompetitor"] = cc[(True, "stronger_competitor")] + cc[(False, "stronger_competitor")]
        M[f"cf{g}RefStrong"] = cc[(True, "stronger_competitor")]; M[f"cf{g}RefUnres"] = cc[(True, "unresolved")]
        M[f"cf{g}BelowAdv"] = cc[(False, "fixed_family_advantage")]; M[f"cf{g}BelowStrong"] = cc[(False, "stronger_competitor")]
        M[f"cf{g}BelowUnres"] = cc[(False, "unresolved")]
    M["cfChestBelowStrongPct"] = pct(M["cfChestBelowStrong"], ch["stronger_competitor"])
    if M["cfChestRefStrong"] + M["cfChestBelowStrong"] != ch["stronger_competitor"]:
        warn.append("contingency: stronger-competitor verdicts above and below the reference do not add up to cfChestCompetitor")
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
    robustness_macros(M, rows, wb, warn)
    round2_macros(M, rows, wb, warn)
    extcomp_macros(M, wb, warn)
    prose_macros(M, rows, wb, warn)
    support_macros(M, rows, warn)
    dose_macros(M, rows)
    for w in warn:
        print("  WARNING:", w)
    # write
    L = ["% Generated by scripts/build_numbers.py from runs/manifest.csv; do not edit.",
         "% Rule: a block enters the write-matrix counts iff run.json completed_modules contains CORE and CALIBRATION;",
         "% readability counts run over probe-graded cells (CALIBRATION completed). See scripts/cf_inclusion.py.",
         f"% {M['cfBlocks']} blocks, {M['cfWriteBlocks']} included, {M['cfCells']} write-matrix cells, {M['cfProbeCells']} probe-graded cells."]
    bad = [k for k in M if not k.isalpha()]
    if bad:
        raise RuntimeError(f"macro names must be letters only (TeX control sequences take no digits): {bad}")
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
