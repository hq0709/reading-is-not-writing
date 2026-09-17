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
import cf_ledger as cl   # noqa: E402
import cf_round3 as r3   # noqa: E402

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
    # the random reference is a mean ABSOLUTE cosine, so the paper compares it with mean absolute cosines and never
    # with the signed means above; both are kept because the table prints both
    M["cfGeoAbsCosNih"] = fx(dg["nih"]["offdiag_abs_cos_model_mean"], 3)
    M["cfGeoAbsCosChex"] = fx(dg["chexpert"]["offdiag_abs_cos_model_mean"], 2)
    M["cfGeoAbsCosCoco"] = fx(dg["coco"]["offdiag_abs_cos_model_mean"], 3)
    cx = G["labels"]["chexpert"]["concepts"]; i, j = cx.index("Effusion"), cx.index("Consolidation")
    M["cfGeoCosEffCons"] = fx(dg["chexpert"]["cos_model_mean_matrix"][i][j], 2); M["cfGeoPhiEffCons"] = fx(G["labels"]["chexpert"]["phi"][i][j], 2)
    M["cfGeoPhiNih"] = fx(lg["nih"]["offdiag_phi_mean"], 3)
    M["cfGeoOwnWinsNihPct"] = pct(lo["nih"]["O_q_positive"], lo["nih"]["n_cells"]); M["cfGeoOwnWinsChexPct"] = pct(lo["chexpert"]["O_q_positive"], lo["chexpert"]["n_cells"])
    cos_ch = [pct(dg[ds]["coincide_cos_model"]["n_true"], dg[ds]["coincide_cos_model"]["n"]) for ds in ci.CHEST]
    M["cfGeoChanceCosMin"], M["cfGeoChanceCosMax"] = min(cos_ch), max(cos_ch)
    lab = [pct(lg[ds][k]["n_true"], lg[ds][k]["n"]) for ds in ci.CHEST for k in ("coincide_phi", "coincide_cooccurrence")]
    M["cfGeoChanceLabelMin"], M["cfGeoChanceLabelMax"] = min(lab), max(lab)
    M["cfGeoOffdiagCells"] = f"{asc['chest']['n_cells']:,}".replace(",", "{,}")
    M["cfGeoRhoAdvCos"] = fx(asc["chest"]["rho_adv_vs_cos_model"]["rho"], 3); M["cfGeoRhoAdvCosP"] = fx(asc["chest"]["rho_adv_vs_cos_model"]["p"], 2)
    M["cfGeoRhoAdvPhi"] = fx(asc["chest"]["rho_adv_vs_phi"]["rho"], 3)
    M["cfGeoTwoAbove"] = lo["chest"]["n_above_ge2"]; M["cfGeoTwoAboveN"] = lo["chest"]["n_cells"]; M["cfGeoTwoAbovePct"] = pct(lo["chest"]["n_above_ge2"], lo["chest"]["n_cells"])
    fam = [pct(v["O_family_positive"], v["n_cells"]) for v in G["coco_families"].values()]
    M["cfGeoCocoFamMinPct"], M["cfGeoCocoFamMaxPct"] = min(fam), max(fam)
    full = [pct(v["O_full_positive"], v["n_cells"]) for v in G["coco_families"].values()]
    M["cfGeoCocoFamFullMinPct"], M["cfGeoCocoFamFullMaxPct"] = min(full), max(full)
    # the five restricted COCO families, all named in the Appendix A.7 prose with their mean within-family phi
    for key, name in (("person+bicycle", "PersonBicycle"), ("person+dog", "PersonDog"), ("chair+bottle", "ChairBottle"),
                      ("car+bicycle", "CarBicycle"), ("person+chair+bottle", "PersonChairBottle")):
        M[f"cfGeoFamPhi{name}"] = fx(G["coco_families"][key]["phi_within_family_mean"], 2)
    if len(G["coco_families"]) != 5:
        warn.append(f"geometry: {len(G['coco_families'])} restricted COCO families (the prose names five)")
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
    # ---- the three numbers Figure 3 annotates on each panel, from the sidecar the figure script writes beside it,
    #      so the caption states the plotted values and never a transcription of them
    exp = HERE.parent / "figures" / "fig3_example.json"
    if exp.exists():
        ex = json.loads(exp.read_text())
        for ds, K in (("nih", "Chest"), ("coco", "Coco")):
            d = ex[ds]
            M[f"cfEx{K}Clean"] = fx(d["clean"], 2); M[f"cfEx{K}Own"] = fx(d["concept_write"], 2)
            M[f"cfEx{K}Comp"] = fx(d["competitor_write"], 2); M[f"cfEx{K}MaxOther"] = fx(d["max_other_answer"], 2)
            M[f"cfEx{K}Question"] = d["question"]; M[f"cfEx{K}Competitor"] = d["competitor"]
        if ex["coco"]["max_other_answer"] > 0.05:
            warn.append(f"figure 3: the COCO concept write moves another answer to {ex['coco']['max_other_answer']:.3f}; "
                        f"the caption says it moves nothing else")
        if ex["nih"]["competitor_write"] <= ex["nih"]["concept_write"]:
            warn.append("figure 3: the chest competitor no longer beats the concept write; the caption says it does")
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
    # ---- token-weighted writes (the token-weight panel of Table cf-altdir): every included block with TOKENW, chest sets
    #      pooled and COCO; owned cells under the uniform (protocol) write and under each of the two weighted variants
    tw = {g: {"blocks": 0, "cells": 0, "uniform": 0, "tokenw": 0, "topq": 0} for g in ("Chest", "Coco")}
    for (mk, ds), b in wb.items():
        a = b["s"].get("tokenw")
        if not a:
            continue
        g = tw["Chest" if ds in ci.CHEST else "Coco"]
        core = b["s"]["core"]["per_question"]
        g["blocks"] += 1; g["cells"] += len(core); g["uniform"] += sum(ci.owned(c) for c in core.values())
        for v in ("tokenw", "topq"):
            pq = (a.get(v) or {}).get("per_question") or {}
            if len(pq) != len(core):
                warn.append(f"tokenw: {mk}/{ds} {v} covers {len(pq)} of {len(core)} concepts")
            g[v] += sum(ci.owned(c) for c in pq.values())
    for G, g in tw.items():
        M[f"cfTokenw{G}Blocks"] = g["blocks"]; M[f"cfTokenw{G}Cells"] = g["cells"]; M[f"cfTokenwUniform{G}Owned"] = g["uniform"]
        M[f"cfTokenwSoft{G}Owned"] = g["tokenw"]; M[f"cfTokenwTopq{G}Owned"] = g["topq"]
    if tw["Chest"]["tokenw"] > tw["Chest"]["uniform"] or tw["Chest"]["topq"] > tw["Chest"]["uniform"]:
        warn.append("tokenw: a weighted write owns more chest cells than the uniform write (prose: weighting creates no clinical handle)")
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
    M["cfInternvlWqqMax"] = fx(math.ceil(max(iv) * 1000) / 1000, 3)   # a 3-dp ceiling: the 2-dp one lands on the effect floor
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


def seed_macros(M: dict, warn: list) -> None:
    """Controlled decodability of the three seed-study cells (Appendix C, first subsection), from data/accepted_results.json (hash-pinned in
    MANIFEST.sha256; the registered table_decoding.tex carries the same values). Four decimals, as registered."""
    cells = {c["key"]: c for c in json.loads((HERE.parent / "data" / "accepted_results.json").read_text())["cells"]}
    for key, name in (("qwen_effusion", "QwenEff"), ("llava_effusion", "LlavaEff"), ("llava_edema", "LlavaEdema")):
        c = cells[key]
        M[f"cfSeedDec{name}Auroc"] = fx(c["auroc"], 4)
        M[f"cfSeedDec{name}AurocLo"], M[f"cfSeedDec{name}AurocHi"] = (fx(x, 4) for x in c["auroc_ci95"])
        M[f"cfSeedDec{name}Ctrl"] = fx(c["control_mean"], 4)
        M[f"cfSeedDec{name}CtrlLo"] = fx(c["control_spread"]["p05"], 4); M[f"cfSeedDec{name}CtrlHi"] = fx(c["control_spread"]["p95"], 4)
        M[f"cfSeedDec{name}Sel"] = fx(c["selectivity"], 4)
        M[f"cfSeedDec{name}SelLo"], M[f"cfSeedDec{name}SelHi"] = (fx(x, 4) for x in c["selectivity_ci95"])
    for k in ("Ctrl", "CtrlLo", "CtrlHi"):
        if M[f"cfSeedDecLlavaEff{k}"] != M[f"cfSeedDecLlavaEdema{k}"]:
            warn.append(f"seed: the two LLaVA cells differ in control {k} (the prose quotes one control AUROC for both)")
    # the two-model NIH study's own Effusion ownership, which Section 5.2 says the campaign reproduces: read from the
    # registered seed artefact rather than typed into the prose
    sp = json.loads((HERE.parent / "data" / "accepted_results.json").read_text())["specificity"]["primary"]
    M["cfSeedPriorO"] = fx(sp["margin"], 3, signed=True)
    M["cfSeedPriorOLo"] = fx(sp["ci95"][0], 3, signed=True); M["cfSeedPriorOHi"] = fx(sp["ci95"][1], 3, signed=True)
    M["cfSeedPriorOFull"] = fx(sp["margin"], 4, signed=True)
    M["cfSeedPriorOLoFull"] = fx(sp["ci95"][0], 4, signed=True); M["cfSeedPriorOHiFull"] = fx(sp["ci95"][1], 4, signed=True)


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
    and ten known negatives in the cohort that grades it, the 400-row CALIBRATION cohort (cftransfer.analysis.calibration
    loads role `calibration`; Section 3.4); a cell below that support can never be graded readable or answerable, so it
    enters the published denominators as a failure. These macros give the counts and the eligible-only shares. The
    manifest's n_pos / n_neg are checked against the calibration counts of the frozen data manifests, and the named
    ineligible cells (CheXpert Atelectasis, COCO bicycle in the prose) get their calibration and test counts."""
    cells = [r for r in rows if t(r["cell"])]
    cal = {ds: cohort_counts(ds, "calibration") for ds in ci.DATASETS}
    test = {ds: cohort_counts(ds, "test") for ds in ci.DATASETS}
    off = sorted({(r["dataset"], r["concept"]) for r in cells if r["n_pos"] != ""
                  and (int(r["n_pos"]), int(r["n_neg"])) != cal[r["dataset"]][r["concept"]]})
    if off:
        raise RuntimeError(f"manifest n_pos / n_neg differ from the calibration cohort counts of the data manifests for {off}")
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
    # the ineligible cells the prose names, with their calibration (grading) and test (write) counts
    for D, ds, named in (("Chex", "chexpert", "Atelectasis"), ("Coco", "coco", "bicycle")):
        wm = [r for r in cells if r["dataset"] == ds and t(r["block_included"])]
        names = sorted({r["concept"] for r in wm if inel(r)})
        if names != [named]:
            warn.append(f"support: the support-ineligible {ds} concepts are {names} (prose names {named})")
        M[f"cfSupportInel{D}Blocks"] = len({r["model_key"] for r in wm if r["concept"] == named and inel(r)})
        M[f"cfCal{D}InelPos"], M[f"cfCal{D}InelNeg"] = cal[ds][named]
        M[f"cfTest{D}InelPos"], M[f"cfTest{D}InelNeg"] = test[ds][named]
    M["cfSupportInelNihBlocks"] = len({r["model_key"] for r in cells if r["dataset"] == "nih" and t(r["block_included"]) and inel(r)})
    nih = {c: cal["nih"][c] for r in cells if r["dataset"] == "nih" for c in (r["concept"],) if c in cal["nih"]}
    thin = min(nih, key=lambda c: min(nih[c]))
    M["cfCalNihMinPos"] = min(nih[thin])
    if thin != "Pneumothorax":
        warn.append(f"support: the thinnest NIH calibration class is {thin} ({nih[thin]}); the appendix names Pneumothorax")


def cohort_counts(dataset_id: str, role: str) -> dict:
    """concept -> (known positives, known negatives) of one cohort role, from the frozen data manifests the evaluation
    code reads (cohort.csv roles, labels.csv label_known / label)."""
    import csv as _csv
    root = ci.RUNS.parent / "data" / dataset_id / "manifests"
    ids = {r["row_id"] for r in _csv.DictReader((root / "cohort.csv").open(newline="", encoding="utf-8")) if r["role"] == role}
    out = {}
    for r in _csv.DictReader((root / "labels.csv").open(newline="", encoding="utf-8")):
        if r["row_id"] in ids and r["label_known"] == "true":
            p, n = out.get(r["concept"], (0, 0))
            out[r["concept"]] = (p + (r["label"] == "1"), n + (r["label"] == "0"))
    return out


def cohort_units(dataset_id: str, row_ids) -> int:
    """Distinct units (patients on the chest sets, images on COCO) behind a list of row ids."""
    import csv as _csv
    root = ci.RUNS.parent / "data" / dataset_id / "manifests"
    unit = {r["row_id"]: r["unit_id"] for r in _csv.DictReader((root / "cohort.csv").open(newline="", encoding="utf-8"))}
    return len({unit[r] for r in row_ids})


def answer_direction_macros(M: dict, wb: dict, warn: list) -> None:
    """Section 5.6 and Appendix A: what the answer direction a_q is fitted to and what it reads.
      label / clean-answer AUROC   runs/robustness/validation.json answer_direction: per-block medians over the six
                                   questions (Table cf-validation rows) and medians over the chest cells
      cosines                      the same file: raw model-space and covariance-whitened, per block
      ridge penalty and rows       fits/vis.last/ansdir_seed0.npz of every ANSDIR block: the CV-chosen penalty per question,
                                   the training rows used and the units (patients / images) behind them
      column selectivity           S_d = W_dd / sum_q |W_qd| of the answer directions (summary.json ansdir W) next to the
                                   label directions' S_d of the same blocks (validation.json column_selectivity cells,
                                   recomputed here from summary.json core W as a cross-check)"""
    V = json.loads((ROB / "validation.json").read_text())
    B = V["answer_direction"]["blocks"]
    ans_keys = {k for k, b in wb.items() if b["s"].get("ansdir")}
    if {tuple(k.split("/")) for k in B} != ans_keys:
        raise RuntimeError(f"validation.json answer_direction blocks differ from the included ANSDIR blocks by {sorted({tuple(k.split('/')) for k in B} ^ ans_keys)}")
    q = B["q25-7/nih"]["summary"]
    M["cfValAurocAnsQwenNih"] = fx(q["median_auroc_answer_dir"], 2); M["cfValAurocProbeQwenNih"] = fx(q["median_auroc_probe"], 2)
    M["cfValAurocAnsCleanQwenNih"] = fx(q["median_auroc_answer_dir_vs_clean_answer"], 2)
    M["cfValAurocProbeCleanQwenNih"] = fx(q["median_auroc_probe_vs_clean_answer"], 2)
    l = B["lingshu-32/chexpert"]["summary"]
    M["cfValCosRawLingChex"] = fx(l["median_cos_raw_model"], 2); M["cfValCosSigmaLingChex"] = fx(l["median_cos_sigma_projected"], 2)
    gap = max(B, key=lambda k: B[k]["summary"]["median_cos_sigma_projected"] - B[k]["summary"]["median_cos_raw_model"])
    if gap != "lingshu-32/chexpert":
        warn.append(f"ansdir: the largest whitened-minus-raw cosine gap is now {gap} (Section 5.6 names the Lingshu 32B CheXpert block)")
    chest = [c for k, b in B.items() if k.split("/")[1] in ci.CHEST for c in b["per_question"].values()]
    def med(key):
        xs = [c[key] for c in chest if c.get(key) is not None and c[key] == c[key]]
        return statistics.median(xs), len(xs)
    for key, name in (("auroc_answer_dir", "cfValAurocAnsChest"), ("auroc_probe", "cfValAurocProbeChest"),
                      ("auroc_answer_dir_vs_clean_answer", "cfValAurocAnsCleanChest"), ("auroc_probe_vs_clean_answer", "cfValAurocProbeCleanChest")):
        v, n = med(key); M[name] = fx(v, 2); M[f"{name}N"] = n
    if not (float(M["cfValAurocAnsChest"]) < float(M["cfValAurocProbeChest"]) and float(M["cfValAurocAnsCleanChest"]) > float(M["cfValAurocProbeCleanChest"])):
        warn.append("ansdir: on the chest cells the answer direction no longer reads the label worse and the clean answer better than the probe (prose says it does)")
    # ridge penalty chosen by cross-validation, training rows, and the units behind them
    alphas, units, nrows = [], {}, set()
    for (mk, ds) in sorted(ans_keys):
        z = np.load(ci.RUNS / mk / ds / "fits" / "vis.last" / "ansdir_seed0.npz", allow_pickle=False)
        alphas += [float(a) for a in z["ridge_alpha"]]; nrows |= {int(z["n_train_rows"])} | {int(x) for x in z["n_rows_used"]}
        units.setdefault(ds, set()).add(cohort_units(ds, [str(x) for x in z["train_row_ids"]]))
        if [float(a) for a in z["cv_alphas"]] != [0.1, 1.0, 10.0, 100.0] or int(z["cv_folds"]) != 5:
            warn.append(f"ansdir: {mk}/{ds} penalty grid {z['cv_alphas'].tolist()} / {int(z['cv_folds'])} folds (prose gives 0.1, 1, 10, 100 and five folds)")
    if nrows != {3000}:
        warn.append(f"ansdir: training rows used {sorted(nrows)} (prose says 3,000)")
    M["cfAnsAlphaCells"] = len(alphas)
    for a, name in ((0.1, "Tenth"), (1.0, "One"), (10.0, "Ten"), (100.0, "Hundred")):
        M[f"cfAnsAlpha{name}"] = sum(x == a for x in alphas)
    if M["cfAnsAlphaTenth"]:
        warn.append(f"ansdir: the penalty 0.1 is chosen in {M['cfAnsAlphaTenth']} cells (Section 5.6 lists only 1, 10, and 100)")
    for ds, D in DS_MACRO.items():
        if len(units.get(ds, ())) != 1:
            warn.append(f"ansdir: {ds} blocks use different training rows ({units.get(ds)})")
        M[f"cfAnsTrainUnits{D}"] = f"{min(units[ds]):,}".replace(",", "{,}") if units.get(ds) else "--"
    # column selectivity of the answer directions against the label directions of the same blocks
    lab = {(c["block"], c["d"]): c["S_d"] for c in V["column_selectivity"]["cells"]}
    Sa, Sl = {"chest": [], "coco": []}, {"chest": [], "coco": []}
    for (mk, ds) in sorted(ans_keys):
        s = wb[(mk, ds)]["s"]; Wa = s["ansdir"]["W"]; Wl = s["core"]["W"]; qs = list(s["ansdir"]["per_question"])
        g = "coco" if ds == "coco" else "chest"
        for d in qs:
            sa = Wa[d][d] / sum(abs(Wa[x][d]) for x in qs)
            sl = Wl[d][f"concept:{d}"] / sum(abs(Wl[x][f"concept:{d}"]) for x in qs)
            if abs(sl - lab[(f"{mk}/{ds}", d)]) > 1e-9:
                raise RuntimeError(f"column selectivity of {mk}/{ds}/{d} recomputed {sl} differs from validation.json {lab[(f'{mk}/{ds}', d)]}")
            Sa[g].append(sa); Sl[g].append(sl)
    for g, G in (("chest", "Chest"), ("coco", "Coco")):
        M[f"cfAnsColSel{G}"] = fx(statistics.median(Sa[g]), 2); M[f"cfAnsColSelLabel{G}"] = fx(statistics.median(Sl[g]), 2)
        M[f"cfAnsColSel{G}N"] = len(Sa[g])
        M[f"cfAnsColSelGe{G}"] = sum(x >= 0.5 for x in Sa[g]); M[f"cfAnsColSelLabelGe{G}"] = sum(x >= 0.5 for x in Sl[g])


def validfit_macros(M: dict, wb: dict, warn: list) -> None:
    """Appendix A and Section 5.8: the expert-label refit (summary.json `validfit`). The six CheXpert directions refitted on
    the 200 radiologist-labelled valid rows with the protocol's probe settings, compared with the report-label directions
    of the same block: raw model-space and covariance-whitened cosines, the cross-fitted held-out AUROC of the refit against
    the radiologist labels next to the report-label direction's AUROC on the same rows, and ownership of both on the 600
    test rows under the same references. Blocks: included CheXpert blocks with VALIDFIT completed."""
    keys = sorted(k for k, b in wb.items() if k[1] == "chexpert" and b["s"].get("validfit")
                  and "VALIDFIT" in set(b["run"].get("completed_modules") or []))
    cells = [(k, q, v, wb[k]["s"]["core"]["per_question"][q]) for k in keys for q, v in wb[k]["s"]["validfit"]["per_question"].items()]
    M["cfValidfitBlocks"] = len(keys); M["cfValidfitCells"] = len(cells)
    if not cells:
        for name in ("CosRaw", "CosWhitened", "AurocExpert", "AurocReportDir", "OwnedExpert", "OwnedReport", "FitRows", "Folds"):
            M[f"cfValidfit{name}"] = "--"
        return
    fr = {wb[k]["s"]["validfit"]["fit_rows"] for k in keys}; fo = {wb[k]["s"]["validfit"]["n_folds"] for k in keys}
    if len(fr) != 1 or len(fo) != 1:
        warn.append(f"validfit: blocks differ in fit rows {fr} or folds {fo}")
    M["cfValidfitFitRows"] = min(fr); M["cfValidfitFolds"] = min(fo)
    def med(key):
        xs = [v[key] for _, _, v, _ in cells if v.get(key) is not None and v[key] == v[key]]
        return fx(statistics.median(xs), 2)
    M["cfValidfitCosRaw"] = med("cos_to_report_model"); M["cfValidfitCosWhitened"] = med("cos_to_report_whitened")
    M["cfValidfitAurocExpert"] = med("auroc_expert_heldout"); M["cfValidfitAurocReportDir"] = med("report_direction_auroc_expert")
    M["cfValidfitOwnedExpert"] = sum(bool(v["owned"]) for _, _, v, _ in cells)
    M["cfValidfitOwnedReport"] = sum(ci.owned(c) for _, _, _, c in cells)
    if M["cfValidfitOwnedExpert"] > M["cfValidfitOwnedReport"]:
        warn.append("validfit: the expert-label refits own more cells than the report-label directions (prose says fewer)")
    validfit_arm_macros(M, warn)


# the arms that separate the LABEL SOURCE of a direction from the NUMBER OF ROWS it was estimated on. The expert-label
# refit uses 200 films and the shipped direction about 20,000, so the two differ in both at once; the size-matched arms
# hold the label source fixed at 200 rows and hold the sample fixed at one label source.
VALIDFIT_ARMS = (("expert_valid200", "Expert"), ("report_valid200", "ReportSameFilms"),
                 ("report_train_sub", "ReportSubCohort"), ("report_train_sub_known", "ReportSub"),
                 ("report_train_full", "ReportFull"))


def validfit_arm_macros(M: dict, warn: list) -> None:
    """Appendix A: five arms scored on the same radiologist-labelled films, each with its cross-fitted AUROC against
    the radiologist labels, against the report-derived labels of those films, and its raw and whitened cosine to the
    shipped direction (runs/robustness/validfit.json, scripts/mayo/robustness_validfit.py)."""
    V = r3.validfit_arms()
    A = V["aggregate"]["per_arm"]
    M["cfVfArmBlocks"] = V["meta"]["n_blocks"]; M["cfVfArmDraws"] = V["meta"]["draws"]
    M["cfVfArmFolds"] = V["meta"]["folds"]; M["cfVfArmRows"] = V["blocks"][0]["n_valid_rows"]
    M["cfVfArmTrainRows"] = f"{V['blocks'][0]['n_train_rows']:,}".replace(",", "{,}")
    for arm, K in VALIDFIT_ARMS:
        d = A[arm]
        M[f"cfVfArm{K}Auroc"] = fx(d["auroc_expert_labels"], 2)
        M[f"cfVfArm{K}AurocReport"] = fx(d["auroc_report_labels"], 2)
        M[f"cfVfArm{K}Cos"] = fx(d["cos_model"], 2)
        M[f"cfVfArm{K}CosWhite"] = fx(d["cos_whitened"], 2)
        M[f"cfVfArm{K}Rows"] = int(round(d["median_fit_rows"]))
        M[f"cfVfArm{K}Cells"] = d["cells"]
    for key, K in (("label_source_at_matched_size", "Label"), ("label_source_on_the_same_films", "SameFilms"),
                   ("sample_size_at_one_label_source", "Size"),
                   ("expert_refit_against_the_shipped_direction", "Mixed")):
        M[f"cfVfArmDelta{K}"] = fx(V["aggregate"][key]["delta_auroc_expert_labels"], 2, signed=True)
    lab = abs(V["aggregate"]["label_source_at_matched_size"]["delta_auroc_expert_labels"])
    size = V["aggregate"]["sample_size_at_one_label_source"]["delta_auroc_expert_labels"]
    if lab >= size:
        warn.append(f"validfit arms: the label-source contrast at a matched sample ({lab:+.3f}) is at least as large as "
                    f"the sample-size contrast ({size:+.3f}); Appendix A says the sample size carries the difference")
    if not V["aggregate"]["size_match_exact"]:
        warn.append(f"validfit arms: the size-matched arm fits {A['report_train_sub_known']['median_fit_rows']:.0f} rows "
                    f"against {A['expert_valid200']['median_fit_rows']:.0f} for the expert refit; they must match")


def dose_macros(M: dict, rows: list[dict]) -> None:
    """Blocks entering Figure 2(c) and the D column of Table cf-coverage: included blocks with DOSE completed."""
    done = {}
    for r in rows:
        done.setdefault((r["model_key"], r["dataset"]), (t(r["block_included"]), set(m for m in r["completed_modules"].split("|") if m)))
    for ds, D in DS_MACRO.items():
        M[f"cfDoseBlocks{D}"] = sum(inc and "DOSE" in mods for (mk, d), (inc, mods) in done.items() if d == ds)


WORD = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def thou(n) -> str:
    """An integer with the paper's thousands separator, so 18212 prints as 18{,}212."""
    return f"{int(n):,}".replace(",", "{,}")

ROLE_MACRO = {"train": "Train", "preflight": "Preflight", "calibration": "Calibration", "test": "Test", "valid": "Valid"}


def ledger_macros(M: dict, wb: dict, warn: list) -> None:
    """Headline counts of the reference-and-rule ledger (Table cf-ledger), through scripts/cf_ledger.py, which checks
    every declared reference and sham against the packaged per-block statistics before the row is written."""
    rows = cl.ledger(wb)
    L = cl.totals(rows)
    M["cfLedgerAnalyses"] = L["n_analyses"]; M["cfLedgerCells"] = thou(L["cells"])
    M["cfLedgerFullReference"] = L["n_full_reference"]
    M["cfLedgerSeedSham"] = L["n_seed0_sham"]; M["cfLedgerSeedShamWord"] = WORD[L["n_seed0_sham"]]
    M["cfLedgerOwnSham"] = L["n_own_sham"]; M["cfLedgerMaxT"] = L["n_max_t"]
    M["cfLedgerSeedShamCells"] = thou(L["cells_seed0_sham"])
    M["cfLedgerPercentile"] = L["n_percentile"]; M["cfLedgerPercentileCells"] = L["cells_percentile"]
    M["cfAnsdirtRefPairs"] = L["ansdirt_random"]; M["cfAnsdirtShamPairs"] = L["ansdirt_sham"]
    M["cfAnsdirtLedgerPairs"] = L["ansdirt_cells"]
    M["cfSemLedgerCells"] = L["semend_cells"]; M["cfSemLedgerRandom"] = L["semend_random"]
    if L["n_seed0_sham"] + L["n_percentile"] == 0:
        warn.append("ledger: no analysis departs from the write matrix's reference and rule, so Appendix A.7's sentence "
                    "about the exceptions has nothing to point at")


def cohort_macros(M: dict, warn: list) -> None:
    """Rows, independent units and role overlaps of the fixed cohort manifests (Section 4, Table cf-roles).

    One cohort manifest per dataset is copied into every block; this reads them and refuses to go on if two blocks of a
    dataset disagree, so the counts in the text are the counts the campaign actually scored."""
    import csv, collections, hashlib
    seen, counts = {}, {}
    for d in sorted(ci.RUNS.glob("*/*/manifests/cohort.csv")):
        ds = d.parent.parent.name
        h = hashlib.sha256(d.read_bytes()).hexdigest()
        if ds in seen and seen[ds] != h:
            raise RuntimeError(f"{ds}: blocks carry different cohort manifests; the cohort counts are not one cohort")
        if ds in seen:
            continue
        seen[ds] = h
        by = collections.defaultdict(list)
        for r in csv.DictReader(d.open()):
            by[r["role"]].append(r["unit_id"])
        counts[ds] = {role: collections.Counter(u) for role, u in by.items()}
    for ds, D in DS_MACRO.items():
        roles = counts[ds]
        for role, R in ROLE_MACRO.items():
            c = roles.get(role)
            if not c:
                continue
            M[f"cfCohort{D}{R}Rows"] = thou(sum(c.values())); M[f"cfCohort{D}{R}Units"] = thou(len(c))
            M[f"cfCohort{D}{R}Max"] = max(c.values())
            M[f"cfCohort{D}{R}Multi"] = thou(sum(1 for v in c.values() if v > 1))
        names = [r for r in ROLE_MACRO if r in roles]
        overlap = 0
        for i, ra in enumerate(names):
            for rb in names[i + 1:]:
                n = len(set(roles[ra]) & set(roles[rb]))
                M[f"cfOverlap{D}{ROLE_MACRO[ra]}{ROLE_MACRO[rb]}"] = n
                overlap = max(overlap, n)
        M[f"cfCohort{D}MaxOverlap"] = overlap
        M[f"cfCohort{D}Pairs"] = len(names) * (len(names) - 1) // 2
        if overlap:
            warn.append(f"cohorts: two roles of {ds} share {overlap} units (Appendix F.5 says the roles are disjoint)")
    M["cfCohortMaxOverlap"] = max(M[f"cfCohort{D}MaxOverlap"] for D in DS_MACRO.values())
    multi = [(ds, role) for ds in counts for role, c in counts[ds].items() if max(c.values()) > 1]
    M["cfCohortMultiRoles"] = len(multi)
    if multi != [("nih", "train")]:
        warn.append(f"cohorts: the roles holding more than one row per unit are {multi}; Section 4 names the NIH "
                    f"training role alone")


COVERAGE_LEGEND = {"CORE": "C", "CALIBRATION": "Cal", "PROMPT": "P", "DOSE": "D", "REFIT": "R", "LOCUS": "L",
                   "LOCUS_CALIBRATION": "Lc"}
CHEX_FIRST_WAVE = ("CORE", "CALIBRATION")
CHEX_ADDED = ("PROMPT", "DOSE", "REFIT", "LOCUS", "LOCUS_CALIBRATION")


def coverage_macros(M: dict, rows: list[dict], warn: list) -> None:
    """The planned, completed and later coverage of Table cf-coverage, so Section 4 and Appendix A.4 can state each once.

    The seven modules of the planned campaign are CORE, CALIBRATION, PROMPT, DOSE, REFIT, LOCUS and LOCUS_CALIBRATION.
    CheXpert blocks carried the first two in the first wave; amendment A2 added the other five."""
    done = {}
    for r in rows:
        done.setdefault((r["model_key"], r["dataset"]),
                        (t(r["block_included"]), {m for m in r["completed_modules"].split("|") if m}))
    all_ = {k: mods for k, (i, mods) in done.items() if True}
    planned = tuple(COVERAGE_LEGEND)
    M["cfCoveragePlannedModules"] = len(planned)
    M["cfCoveragePlannedWord"] = WORD[len(planned)]
    if M.get("cfCoverageBlocksAllSeven") != sum(all(m in mods for m in planned) for mods in all_.values()):
        warn.append("coverage: the seven planned modules of this helper are not the seven the coverage table counts")
    for ds, D in DS_MACRO.items():
        b = {k: v for k, v in all_.items() if k[1] == ds}
        M[f"cfCoverage{D}Blocks"] = len(b)
        M[f"cfCoverage{D}AllSeven"] = sum(all(m in mods for m in planned) for mods in b.values())
    chex = {k: v for k, v in all_.items() if k[1] == "chexpert"}
    M["cfCoverageChexFirstWave"] = sum(all(m in mods for m in CHEX_FIRST_WAVE) for mods in chex.values())
    if M.get("cfCoverageChexFull") != sum(all(m in mods for m in CHEX_ADDED) for mods in chex.values()):
        warn.append("coverage: the five modules CheXpert gained after the first wave are not the five the table counts")
    M["cfCoverageChexAddedN"] = len(CHEX_ADDED)
    M["cfCoverageChexAddedWord"] = WORD[len(CHEX_ADDED)]
    M["cfCoverageChexPartial"] = len(chex) - M["cfCoverageChexFull"]
    # the CheXpert blocks short of the five: the paper names the reason, so the reason has to be the same one every time
    short = {k[0]: sorted(set(CHEX_ADDED) - mods) for k, mods in chex.items() if not set(CHEX_ADDED) <= mods}
    if sorted({m for v in short.values() for m in v}) not in ([], ["PROMPT"]):
        warn.append(f"coverage: the CheXpert blocks short of the five added modules are short of {short}, not of the "
                    f"five additional templates alone, which is what Appendix A.4 says")


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
    if not Pa["chest"]["attribute_random_reference"]:
        warn.append("attr: an attribute question carries no random family, so it is not on the clinical questions' steering "
                    "reference (Table cf-attr caption says both kinds carry the same one)")
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


DS_KEY = {"nih": "Nih", "chexpert": "Chex", "coco": "Coco"}
_CKPT_ORDER = ("q25-3", "q25-7", "q25-32", "q25-72", "q3-4", "q3-8", "q3-32", "iv35-8", "iv35-14",
               "iv35-38", "gemma3-4", "gemma3-12", "gemma3-27", "medgemma-4", "medgemma-27", "lingshu-7",
               "lingshu-32", "llava15-7", "llava15-13", "llavamed-7", "llama32-11")


def round3_macros(M: dict, rows: list[dict], wb: dict, warn: list) -> None:
    """Macros for the four crossed experiments of Section 5 (Tables cf-towerswap, cf-replay, cf-semend, cf-fgobj) and for the
    three ways the attribute comparison is read (Table cf-attr, bottom panel), through scripts/cf_round3.py. That helper
    raises when the blocks a module claims in run.json differ from the blocks whose summary carries its results, so no macro
    here can be built from a block set the manifest does not say exists. The native arms of the crossover and the chest
    ownership counts are checked against the manifest on top of that."""
    owned_cell = {(r["model_key"], r["dataset"], r["concept"]): t(r["owned"]) for r in rows if t(r["cell"]) and t(r["block_included"])}

    # ---- TOWERSWAP (Table cf-towerswap)
    T = r3.towerswap(wb)
    rec = T["receipt"]
    for name, key in (("Tensors", "n_tensors_replaced"), ("TowerTensors", "n_tower_tensors"),
                      ("Outside", "n_tensors_outside_tower"), ("OutsideChanged", "n_tensors_outside_tower_changed")):
        if len(rec[key]) != 1:
            warn.append(f"towerswap: blocks differ in {key} {rec[key]}")
        M[f"cfSwap{name}"] = rec[key][0]
    M["cfSwapParamsM"] = f"{rec['n_params_replaced'][0] / 1e6:.0f}"
    if not (rec["verified_bitwise_equal_to_donor"] and rec["verified_rest_unchanged"]):
        warn.append("towerswap: a swap is not verified bitwise equal to the donor with the rest unchanged "
                    "(Table cf-towerswap caption says every one is)")
    M["cfSwapBlocks"] = len(T["blocks"]); M["cfSwapCrossovers"] = len(T["crossover_blocks"])
    M["cfSwapRows"] = T["n_rows"][0] if len(T["n_rows"]) == 1 else "--"
    M["cfSwapCombos"] = max(d["n_combinations"] for d in T["per_dataset"].values())
    M["cfSwapConcepts"] = max(d["n_concepts"] for d in T["per_dataset"].values())
    for g, G in (("chest", "Chest"), ("coco", "Coco")):
        d = T["pooled"].get(g)
        if not d:
            continue
        M[f"cfSwap{G}Owned"] = d["owned"]; M[f"cfSwap{G}Cells"] = d["cells"]; M[f"cfSwap{G}Combos"] = d["combinations"]
        M[f"cfSwap{G}NativeMin"] = min(d["native_owned"]); M[f"cfSwap{G}NativeMax"] = max(d["native_owned"])
        M[f"cfSwap{G}CrossMin"] = min(d["crossed_owned"]); M[f"cfSwap{G}CrossMax"] = max(d["crossed_owned"])
    # the two swapped COCO combinations by name, so the prose does not depend on which of them happens to be the larger
    hosts = sorted({b["reader"] for b in T["blocks"]}, key=lambda m: list(_CKPT_ORDER).index(m) if m in _CKPT_ORDER else 99)
    if T["per_dataset"].get("coco") and len(hosts) == 2:
        if {hosts[0], hosts[1]} != {"gemma3-4", "medgemma-4"}:
            warn.append(f"towerswap: the crossover hosts are {hosts} (Section 5.5 names Gemma 3 4B and MedGemma 4B)")
        by = {(r["tower"], r["reader"]): r["n_owned"] for r in T["per_dataset"]["coco"]["combinations"]}
        M["cfSwapCocoGemmaTowerCross"] = by.get((hosts[0], hosts[1]), "--")
        M["cfSwapCocoMedTowerCross"] = by.get((hosts[1], hosts[0]), "--")
    if T["pooled"].get("chest", {}).get("owned"):
        warn.append(f"towerswap: {T['pooled']['chest']['owned']} chest cells are owned in some combination "
                    f"(Section 5.5 says no combination owns a finding)")
    # the crossover's factor effects on three quantities of the same six questions, from one pass over the four arms'
    # current outcomes (runs/robustness/crossover.json): the own write, the ownership contrast, and the margin over the
    # strongest competitor in the whole reference family. The four arms of a dataset are the same four files whichever
    # host block indexes them, so the two host blocks of a dataset must give the same crossover.
    X = r3.crossover(wb)
    QK = {"W_qq": "", "O_q": "O", "M_q": "M"}
    for ds, D in DS_KEY.items():
        xs = [b["crossover"] for b in X["blocks"] if b["dataset"] == ds]
        if not xs:
            continue
        for q, qk in QK.items():
            for name, factor in (("Reader", "reader"), ("Tower", "tower")):
                key = f"mean_abs_{factor}_effect_{q}"
                M[f"cfSwap{name}{D}{qk}Min"] = fx(min(x[key] for x in xs), 3)
                M[f"cfSwap{name}{D}{qk}Max"] = fx(max(x[key] for x in xs), 3)
                lo = min(x[f"{key}_ci95"][0] for x in xs); hi = max(x[f"{key}_ci95"][1] for x in xs)
                M[f"cfSwap{name}{D}{qk}Lo"] = fx(lo, 3); M[f"cfSwap{name}{D}{qk}Hi"] = fx(hi, 3)
        if len({fx(x["mean_abs_reader_effect_O_q"], 3) for x in xs}) != 1:
            warn.append(f"crossover: the {ds} host blocks disagree on the reader effect on the ownership contrast")
    if X["stale_packaged"]:
        warn.append(f"crossover: the packaged crossover of {X['stale_packaged']} predates a rescored arm; the tables and "
                    f"macros use the recomputation in runs/robustness/crossover.json")
    # the native arms are this block's own grade recomputed on the crossover's rows: the point estimates must track the
    # published grade, and a cell may only lose the owned grade to the smaller cohort's resolution, never to a competitor
    nv = T["native"]
    M["cfSwapNativeCells"] = nv["cells"]; M["cfSwapNativeOwnedFull"] = nv["owned_full"]; M["cfSwapNativeOwned"] = nv["owned_rows"]
    M["cfSwapNativeMaxDO"] = fx(nv["max_abs_dO"], 3) if nv["max_abs_dO"] is not None else "--"
    M["cfSwapNativeUnresolved"] = nv["lost_to_resolution"]
    man = sum(owned_cell[(b["model"], b["dataset"], q)] for b in T["blocks"] for q in ci.core(wb[(b["model"], b["dataset"])]))
    if man != nv["owned_full"]:
        warn.append(f"towerswap: the blocks' published grade owns {nv['owned_full']} cells against {man} in the manifest")
    if nv["sign_flips"] or nv["lost_to_competitor"]:
        warn.append(f"towerswap: a native arm flips the sign of the ownership contrast in {nv['sign_flips']} cells and loses "
                    f"{nv['lost_to_competitor']} published owned cells to a competitor (Section 5.5 says the crossover's rows "
                    f"only cost resolution)")
    if T["arms_disagreeing_across_hosts"]:
        warn.append(f"towerswap: arms graded in both host blocks disagree: {T['arms_disagreeing_across_hosts']}")
    # how well each crossed arm answers with no write, on the crossing's own rows: the hybrid arms against the
    # native arm of the same block, median over the block's six concepts
    TA = r3.towerswap_answer(wb)
    SHORT = {"gemma3-4": "Gem", "medgemma-4": "Med"}
    drops = []
    for b in TA["blocks"]:
        nm = SHORT.get(b["model"])
        if nm is None:
            warn.append(f"towerswap: the crossed host {b['model']} has no macro name (Appendix A.8 names Gemma 3 4B and MedGemma 4B)")
            continue
        key = f"{nm}{DS_KEY[b['dataset']]}"
        M[f"cfSwapAnsNative{key}"] = fx(b["native"], 3); M[f"cfSwapAnsHybrid{key}"] = fx(b["hybrid"], 3)
        drops.append(round(float(M[f"cfSwapAnsNative{key}"]) - float(M[f"cfSwapAnsHybrid{key}"]), 3))
    M["cfSwapAnsArms"] = TA["arms"]; M["cfSwapAnsWorse"] = TA["worse"]; M["cfSwapAnsRows"] = min(b["rows"] for b in TA["blocks"])
    M["cfSwapAnsDropMin"] = fx(min(drops), 3); M["cfSwapAnsDropMax"] = fx(max(drops), 3)
    if TA["worse"] != TA["arms"]:
        warn.append(f"towerswap: {TA['arms'] - TA['worse']} hybrid arm(s) answer at least as well as the native arm of "
                    f"the same block (Appendix A.8 says every hybrid arm answers worse)")

    # ---- REPLAY (Table cf-replay)
    R = r3.replay(wb)
    M["cfReplayBlocks"] = len(R["blocks"]); M["cfReplaySelf"] = R["n_self"]; M["cfReplayCross"] = R["n_cross"]
    M["cfReplaySelfExact"] = R["n_self_exact"]; M["cfReplayCells"] = R["cells"]; M["cfReplayMoved"] = R["n_nonzero"]
    M["cfReplayVerdictChanges"] = R["verdict_changes"]; M["cfReplayOwnedChanges"] = R["owned_changes"]
    M["cfReplayMaxDW"] = sci(R["max_abs_dW"]); M["cfReplayMaxDO"] = sci(R["max_abs_dO"])
    M["cfReplayMaxDWSelf"] = sci(R["max_abs_dW_self"]) if R["max_abs_dW_self"] else "0"
    M["cfReplayDriftZero"] = R["n_drift_zero"]; M["cfReplayDriftMax"] = fx(R["drift_max_abs"], 2)
    M["cfReplayTensorMin"] = fx(min(b["block_mean_abs"] for b in R["blocks"]), 2)
    M["cfReplayTensorMax"] = fx(max(b["block_mean_abs"] for b in R["blocks"]), 2)
    M["cfReplayPairs"] = len(R["pairs"]); M["cfReplayPairMoved"] = R["pair_nonzero"]; M["cfReplayPairCells"] = R["pair_cells"]
    M["cfReplayPairOwnMin"] = fx(R["pair_own_min"], 3); M["cfReplayPairOwnMax"] = fx(R["pair_own_max"], 3)
    M["cfReplayChangeMax"] = fx(R["pair_change_max"], 4)
    M["cfReplayRelMax"] = fx(max(abs(100 * p["mean_abs_change"] / p["mean_abs_own_towers"]) for p in R["pairs"]), 1)
    if R["verdict_changes"] or R["owned_changes"]:
        warn.append(f"replay: {R['verdict_changes']} verdict and {R['owned_changes']} ownership changes "
                    f"(Section 5.5 says the replay changes neither)")
    if R["n_self_exact"] < R["n_self"]:
        print(f"  NOTE: {R['n_self'] - R['n_self_exact']} of {R['n_self']} self-replays do not reproduce the block's own grade "
              f"to the last bit; the largest difference over them is {R['max_abs_dW_self']:.1e} in a write magnitude")

    # ---- SEMEND (Table cf-semend)
    S = r3.semend(wb)
    M["cfSemBlocks"] = S["n_blocks"]; M["cfSemEndpoints"] = S["n_endpoints"]; M["cfSemCells"] = S["cells"]
    M["cfSemRows"] = S["blocks"][0]["n_rows"] if S["blocks"] else "--"
    M["cfSemRandom"] = S["blocks"][0]["n_random"] if S["blocks"] else "--"
    M["cfSemLabelOwned"] = S["owned"]["label"]; M["cfSemAnswerOwned"] = S["owned"]["answer"]
    M["cfSemLabelRef"] = S["reference_met"]["label"]; M["cfSemAnswerRef"] = S["reference_met"]["answer"]
    M["cfSemNegOpposite"] = S["negation_opposite"]; M["cfSemNegN"] = S["negation_n"]
    M["cfSemForcedOwned"] = S["forced_both_owned"]; M["cfSemForcedN"] = S["negation_n"]
    M["cfSemGapMin"] = fx(S["forced_gap_min"], 3); M["cfSemGapMax"] = fx(S["forced_gap_max"], 3)
    M["cfSemReportOwned"] = S["report_owned"]; M["cfSemReportCells"] = S["report_cells"]
    M["cfSemIVN"] = S["iv_n"]; M["cfSemIVExcl"] = S["iv_excludes_zero"]
    M["cfSemIVMin"] = fx(S["iv_delta_min"], 3); M["cfSemIVMax"] = fx(S["iv_delta_max"], 3)
    for fam, F in (("label", "Label"), ("answer", "Answer")):
        d = S["iv_per_family"][fam]
        M[f"cfSemIV{F}Min"] = fx(d["delta_min"], 3); M[f"cfSemIV{F}Max"] = fx(d["delta_max"], 3)
        M[f"cfSemIV{F}BaseMin"] = fx(d["base_min"], 3); M[f"cfSemIV{F}BaseMax"] = fx(d["base_max"], 3)
    M["cfSemIVBaseMin"] = fx(min(S["iv_per_family"][f]["base_min"] for f in r3.FAMS), 3)
    M["cfSemIVBaseMax"] = fx(max(S["iv_per_family"][f]["base_max"] for f in r3.FAMS), 3)
    # ---- the same question out of sample (Table cf-semend, bottom panel). An in-sample increment is not evidence,
    # so no macro above may be quoted as one; these are the macros the prose reads.
    CV = r3.semend_cv(wb)
    M["cfSemCvPerms"] = f"{CV['permutations']:,}".replace(",", "{,}")
    M["cfSemCvTests"] = CV["pooled_leave_one_concept_out"]["n_tests"]
    M["cfSemCvBlocks"] = CV["n_blocks"]
    for key, K in (("pooled_leave_one_concept_out", "Loco"), ("pooled_leave_one_block_out", "Lobo")):
        d = CV[key]
        M[f"cfSemCv{K}Delta"] = fx(d["mean_delta_heldout_r2"], 3, signed=True)
        M[f"cfSemCv{K}DeltaMin"] = fx(d["delta_min"], 3, signed=True)
        M[f"cfSemCv{K}DeltaMax"] = fx(d["delta_max"], 3, signed=True)
        M[f"cfSemCv{K}P"] = fx(d["permutation_p"], 2)
        M[f"cfSemCv{K}Positive"] = d["n_tests_delta_positive"]; M[f"cfSemCv{K}Tests"] = d["n_tests"]
        M[f"cfSemCv{K}MinP"] = fx(d["min_permutation_p"], 2)
    M["cfSemCvLocoFolds"] = min(r["n_folds"] for r in CV["leave_one_concept_out"])
    M["cfSemCvLoboFolds"] = min(r["n_folds"] for r in CV["leave_one_block_out"])
    if S["iv_delta_min"] < 0:
        warn.append(f"semend: an in-sample increment is negative ({S['iv_delta_min']:.4f}); adding predictors cannot "
                    f"lower a training R^2, so the regression is not the one Table cf-semend describes")
    if CV["ownership_adds_out_of_sample"]:
        warn.append("semend: ownership now adds predictive value out of sample; Section 5.6 states that the held-out "
                    "test finds none, so the sentence and this guard both have to change")
    if min(CV["pooled_leave_one_concept_out"]["min_permutation_p"],
           CV["pooled_leave_one_block_out"]["min_permutation_p"]) < 0.05:
        warn.append("semend: one held-out test reaches p < 0.05 under the permutation null; Section 5.6 says none does")
    if S["owned"]["label"] >= S["owned"]["answer"]:
        warn.append(f"semend: the label direction is owned in {S['owned']['label']} endpoint cells against "
                    f"{S['owned']['answer']} for the answer direction (Section 5.6 says fewer)")

    # ---- FGOBJ (Table cf-fgobj)
    F = r3.fgobj(wb)
    P = F["pool"]
    M["cfFgBlocks"] = len(F["blocks"]); M["cfFgConcepts"] = len(F["concepts"]); M["cfFgWindow"] = fx(F["window"], 2)
    # Easy / EasySame are two different pools and the prose may never swap them: `Easy` is the grid-wide COCO number
    # over every COCO block, `EasySame` is the easy objects of the SIX blocks the fine-grained cells come from, which
    # is the only pool the fine-grained rate may be compared with.
    for g, G in (("chest", "Chest"), ("coco_easy", "Easy"), ("coco_fine", "Fine"), ("coco_easy_same_blocks", "EasySame")):
        d = P[g]
        M[f"cfFg{G}Owned"] = d["owned"]; M[f"cfFg{G}Cells"] = d["cells"]; M[f"cfFg{G}Rate"] = fx(d["owned_rate"], 3)
        M[f"cfFg{G}Auroc"] = fx(d["median_answer_auroc"], 3); M[f"cfFg{G}Sel"] = fx(d["median_selectivity"], 3)
        M[f"cfFg{G}Blocks"] = d["blocks"]
    SB = F["same_block"]
    M["cfFgSameBlocks"] = SB["n_blocks"]
    M["cfFgSameDiffMin"] = SB["paired_difference_min"]; M["cfFgSameDiffMax"] = SB["paired_difference_max"]
    M["cfFgSameSignP"] = fx(SB["sign_test_p"], 2)
    M["cfFgSameFineHigher"] = SB["n_blocks_fine_higher"]; M["cfFgSameEasyHigher"] = SB["n_blocks_easy_higher"]
    M["cfFgSameTied"] = SB["n_blocks_tied"]
    if (SB["fine_owned"], SB["fine_cells"]) != (P["coco_fine"]["owned"], P["coco_fine"]["cells"]) or \
            (SB["easy_owned"], SB["easy_cells"]) != (P["coco_easy_same_blocks"]["owned"], P["coco_easy_same_blocks"]["cells"]):
        warn.append("fgobj: the same-block comparison and the same-block pool disagree; rerun robustness_round2.py")
    if P["coco_easy_same_blocks"]["blocks"] != len(F["blocks"]):
        warn.append(f"fgobj: the same-block easy pool spans {P['coco_easy_same_blocks']['blocks']} blocks against "
                    f"{len(F['blocks'])} FGOBJ blocks; they must be the same checkpoints")
    # the grid-wide easy pool IS the COCO number of Section 5.1; it belongs there and not in the fine-grained comparison
    if (P["coco_easy"]["owned"], P["coco_easy"]["cells"]) != (M["cfCocoOwned"], M["cfCocoOwnedN"]):
        warn.append(f"fgobj: the grid-wide easy pool is {P['coco_easy']['owned']}/{P['coco_easy']['cells']} against "
                    f"{M['cfCocoOwned']}/{M['cfCocoOwnedN']} COCO cells in the manifest; they are the same cells")
    for m, Mk in (("selectivity", "Sel"), ("answer_auroc", "Ans"), ("both", "Both")):
        d = F["matched"][m]
        M[f"cfFgMatch{Mk}N"] = d["n_matched_chest_cells"]; M[f"cfFgMatch{Mk}Cells"] = d["n_chest_cells"]
        M[f"cfFgMatch{Mk}Partners"] = d["n_natural_cells_used"]; M[f"cfFgMatch{Mk}PartnersN"] = d["n_natural_cells"]
        M[f"cfFgMatch{Mk}Chest"] = fx(d["chest_owned_rate_matched"], 3); M[f"cfFgMatch{Mk}Nat"] = fx(d["natural_owned_rate_used"], 3)
        M[f"cfFgMatch{Mk}Diff"] = fx(d["matched_ownership_difference"], 3, signed=True)
        M[f"cfFgMatch{Mk}Lo"] = fx(d["matched_ownership_difference_ci95"][0], 3, signed=True)
        M[f"cfFgMatch{Mk}Hi"] = fx(d["matched_ownership_difference_ci95"][1], 3, signed=True)
        M[f"cfFgMatch{Mk}Pooled"] = fx(d["pooled_ownership_difference"], 3, signed=True)
        M[f"cfFgMatch{Mk}PooledLo"] = fx(d["pooled_ownership_difference_ci95"][0], 3, signed=True)
        M[f"cfFgMatch{Mk}PooledHi"] = fx(d["pooled_ownership_difference_ci95"][1], 3, signed=True)
        # the pooled estimator is the difference of the two rates; the matched one is a different quantity and the
        # two are never reported as one. Both guards fire on any regression of that.
        if not d["pooled_equals_rate_difference"]:
            warn.append(f"fgobj {m}: the pooled difference is not the difference of the two reported rates")
        if abs(d["matched_ownership_difference"] - d["rate_difference"]) < 5e-4:
            warn.append(f"fgobj {m}: the matched and pooled differences round to the same number, so the table must "
                        f"still label them (matched {d['matched_ownership_difference']:.4f}, "
                        f"pooled {d['rate_difference']:.4f})")
    if abs(P["coco_fine"]["owned_rate"] - P["coco_easy_same_blocks"]["owned_rate"]) > 0.15:
        warn.append(f"fgobj: the fine-grained ownership rate {P['coco_fine']['owned_rate']:.3f} differs from the "
                    f"same-block easy rate {P['coco_easy_same_blocks']['owned_rate']:.3f} by more than the paired "
                    f"per-block spread supports (Section 5.4 says fine-grainedness does not reduce ownership)")
    if SB["sign_test_p"] is not None and SB["sign_test_p"] < 0.05:
        warn.append(f"fgobj: the paired sign test over the {SB['n_blocks']} FGOBJ blocks gives p = {SB['sign_test_p']:.3f}, "
                    f"so the same-block difference is not null (Section 5.4 says fine-grainedness does not reduce ownership)")
    if F["matched"]["selectivity"]["n_matched_chest_cells"] != P["chest"]["cells"]:
        warn.append("fgobj: matching on probe selectivity alone leaves chest cells unmatched (Section 5.4 says every one matches)")

    # ---- ownership stratified on clean-answer AUROC (Table cf-fgobj bottom panel, Table cf-attr bottom panel)
    # The stratification is what Sections 5.4 and 5.8 lead with: it keeps every cell, where the matched estimator can
    # pair only part of the chest pool. Each group's bins must sum to its total, which cf_round3.stratified checks.
    ST = r3.stratified()
    M["cfStratTopEdge"] = fx(ST["bin_edges"][ST["top_bin_index"]], 2)
    M["cfStratBins"] = len(ST["bin_labels"])
    inner = [fx(e, 2) for e in ST["bin_edges"][1:-1]]
    M["cfStratEdges"] = ", ".join(inner[:-1]) + " and " + inner[-1]
    for fam, keys in (("chest_vs_natural", (("chest", "Chest"), ("natural", "Nat"), ("nih", "Nih"),
                                            ("chexpert", "Chex"), ("coco_easy", "Easy"), ("coco_fine", "Fine"))),
                      ("attribute_vs_finding", (("attribute_chest", "Attr"), ("finding_chest", "Find"),
                                                ("attribute_nih", "AttrNih"), ("attribute_chexpert", "AttrChex"),
                                                ("finding_nih", "FindNih"), ("finding_chexpert", "FindChex")))):
        for g, G in keys:
            d = ST[fam]["groups"][g]
            top = d["bins"][ST["top_bin_index"]]
            M[f"cfStrat{G}TopOwned"] = top["owned"]; M[f"cfStrat{G}TopN"] = top["cells"]
            M[f"cfStrat{G}TopPct"] = pct(top["owned"], top["cells"]) if top["cells"] else 0
            M[f"cfStrat{G}Owned"] = d["owned"]; M[f"cfStrat{G}Cells"] = d["cells"]
            for i, b in enumerate(d["bins"]):
                M[f"cfStrat{G}Bin{'ABCDE'[i]}Owned"] = b["owned"]; M[f"cfStrat{G}Bin{'ABCDE'[i]}N"] = b["cells"]
    for key, K in (("contrast", "Cvn"), ("contrast_easy_partners_only", "CvnEasy")):
        c = ST["chest_vs_natural"][key]
        M[f"cfStrat{K}Diff"] = fx(c["difference"], 3, signed=True)
        M[f"cfStrat{K}Lo"] = fx(c["difference_ci95"][0], 3, signed=True)
        M[f"cfStrat{K}Hi"] = fx(c["difference_ci95"][1], 3, signed=True)
    c = ST["attribute_vs_finding"]["contrast"]
    M["cfStratAvfDiff"] = fx(c["difference"], 3, signed=True)
    M["cfStratAvfLo"] = fx(c["difference_ci95"][0], 3, signed=True)
    M["cfStratAvfHi"] = fx(c["difference_ci95"][1], 3, signed=True)
    # the stratified pools ARE the pools of the two tables above them; a drift between them is a regenerate, not a
    # rounding difference, and the prose of Sections 5.4 and 5.8 reads both
    if (ST["chest_vs_natural"]["groups"]["chest"]["cells"], ST["chest_vs_natural"]["groups"]["chest"]["owned"]) != \
            (P["chest"]["cells"], P["chest"]["owned"]):
        warn.append("stratified: the chest pool differs from the FGOBJ chest pool; they are the same cells")
    if ST["attribute_vs_finding"]["groups"]["attribute_chest"]["owned"] != M["cfAttrChestAttrOwned"] or \
            ST["attribute_vs_finding"]["groups"]["finding_chest"]["owned"] != M["cfAttrChestClinOwned"]:
        warn.append("stratified: the attribute / finding owned counts differ from the ATTR section's; they are the same cells")
    if M["cfStratAttrTopOwned"]:
        warn.append(f"stratified: {M['cfStratAttrTopOwned']} of the {M['cfStratAttrTopN']} well-answered attribute cells is "
                    f"owned; Sections 5.4 and 5.8 say none is, so the sentence has to change")
    if M["cfStratChestTopPct"] >= M["cfStratNatTopPct"]:
        warn.append("stratified: chest cells are owned at least as often as natural-image cells in the top answer bin; "
                    "the abstract and Section 5.4 say the opposite")

    # ---- the attribute comparison read three ways (Table cf-attr, bottom panel) and the phrasing selection
    A = r3.attr_modes(wb)
    M["cfAttrModeBlocks"] = A["blocks"]; M["cfAttrPairWindow"] = fx(A["window"], 2)
    M["cfAttrRefBlocks"] = A["matched_reference_blocks"]
    for mode, K in (("all_cells", "All"), ("answer_capable_cells", "Cap"), ("answerability_matched", "Pair")):
        c = A["modes"][mode]
        M[f"cfAttr{K}AttrOwned"] = c["attr_owned"]; M[f"cfAttr{K}AttrN"] = c["attr_cells"]
        M[f"cfAttr{K}ClinOwned"] = c["clin_owned"]; M[f"cfAttr{K}ClinN"] = c["clin_cells"]
        M[f"cfAttr{K}Diff"] = fx(c["attr_owned_share"] - c["clin_owned_share"], 3, signed=True)
    pm = A["modes"]["answerability_matched"]
    M["cfAttrPairMatched"] = pm["n_attr_cells_matched"]; M["cfAttrPairUnmatched"] = pm["n_attr_cells_unmatched"]
    M["cfAttrPairCellsN"] = pm["n_attr_cells"]
    if A["matched_reference_blocks"] != A["blocks"]:
        warn.append(f"attr: only {A['matched_reference_blocks']} of {A['blocks']} blocks grade attribute and clinical cells "
                    f"against the same steering reference (Table cf-attr caption says every one does)")
    Q = r3.attrq(wb)
    M["cfAttrqBlocks"] = Q["n_blocks"]; M["cfAttrqCells"] = Q["n_cells"]; M["cfAttrqPhrasings"] = Q["n_phrasings"]
    M["cfAttrqChanged"] = Q["n_changes"]; M["cfAttrqSelected"] = Q["n_selected"]; M["cfAttrqCapable"] = Q["n_written_capable"]
    M["cfAttrqWrittenAuroc"] = fx(Q["median_written_auroc"], 3); M["cfAttrqSelectedAuroc"] = fx(Q["median_selected_auroc"], 3)
    M["cfAttrqGain"] = fx(Q["median_gain"], 3); M["cfAttrqGainMax"] = fx(Q["max_gain"], 3)
    if Q["n_blocks"] != A["blocks"]:
        warn.append(f"attrq: {Q['n_blocks']} blocks score the phrasings against {A['blocks']} with the attribute family")


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
    M["cfChestAnswerable"] = ch["answerable"]; M["cfChestAnswerableN"] = ch["write_cells"]; M["cfChestCellsN"] = ch["write_cells"]; M["cfChestAnswerablePct"] = pct(ch["answerable"], ch["write_cells"])
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
    # the spread of chest probe selectivity, so Section 5.1 states a measured quartile range and not an impression
    schest = sorted(f(r["selectivity"]) for r in cells
                    if r["dataset"] in ci.CHEST and t(r["probe_graded"]) and r["selectivity"] != "")
    qs = statistics.quantiles(schest, n=4)
    M["cfSelChestMed"] = f"{statistics.median(schest):.2f}"
    M["cfSelChestQOne"] = f"{qs[0]:.2f}"; M["cfSelChestQThree"] = f"{qs[2]:.2f}"; M["cfSelChestN"] = len(schest)
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
    round3_macros(M, rows, wb, warn)
    extcomp_macros(M, wb, warn)
    seed_macros(M, warn)
    prose_macros(M, rows, wb, warn)
    support_macros(M, rows, warn)
    answer_direction_macros(M, wb, warn)
    validfit_macros(M, wb, warn)
    dose_macros(M, rows)
    ledger_macros(M, wb, warn)
    cohort_macros(M, warn)
    coverage_macros(M, rows, warn)
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
