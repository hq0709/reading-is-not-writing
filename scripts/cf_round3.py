"""Aggregates for the four crossed experiments of Section 5, shared by the table generator, the macro generator and the
consistency check so that a table and the macro beside it can never drift apart.

    towerswap(wb)   crossed tower and reader: the vision tower of one checkpoint behind the other's connector and
                    language model, against the two native arms, on the same rows.
    replay(wb)      identical-tensor replay: the consumed block's stored token tensors fed to both readers of a
                    shared-tower pair, so the input is identical bit for bit rather than equal up to rounding.
    semend(wb)      semantic endpoints: both direction families written at four endpoints no direction was fitted
                    against, with the in-sample regression fit of the ownership contrast.
    semend_cv(wb)   the same question answered OUT OF SAMPLE: leave-one-concept-out and leave-one-block-out
                    cross-validation with a concept-level permutation null (runs/robustness/round2.json).
    fgobj(wb)       fine-grained object family and the difficulty-matched comparison (runs/robustness/round2.json).
    validfit_arms() label source against sample size for the expert-label refits (runs/robustness/validfit.json).
    attrq(wb)       phrasing selection for the three non-clinical attributes.

Every function takes the block table of cf_inclusion.write_blocks and raises when the set of blocks carrying the
module in run.json differs from the set of blocks whose summary.json carries its results: a macro must never be
built from a block set the manifest does not say exists.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import cf_inclusion as ci

ROB = ci.RUNS / "robustness"
DATASETS = ("nih", "chexpert", "coco")
CHEST = ci.CHEST


def _blocks(wb: dict, module: str, key: str) -> list[tuple]:
    """Included blocks carrying `module` in run.json, checked against the ones whose summary carries `key`."""
    run = {k for k, b in wb.items() if module in set(b["run"].get("completed_modules") or [])}
    got = {k for k, b in wb.items() if b["s"].get(key)}
    if run != got:
        raise RuntimeError(f"{module}: run.json lists {sorted(run)} but summary.json carries {key} for {sorted(got)}; "
                           f"they differ by {sorted(run ^ got)}")
    return sorted(run, key=lambda k: (k[0], DATASETS.index(k[1])))


def _med(xs):
    xs = [x for x in xs if x is not None and x == x]
    return statistics.median(xs) if xs else None


# --------------------------------------------------------------------------------------------- crossed tower and reader

def _combo(tower: str, reader: str) -> str:
    return f"tower={tower}|reader={reader}"


def towerswap(wb: dict) -> dict:
    """The 2x2 of tower and reader per block, the swap receipt, and the four crossover contrasts.

    A block is one host reader on one dataset. Its four arms are the two native grades (each reader on its own
    tower) and the two swapped grades (each reader behind the other's tower). The written directions travel with
    the tower, because they are fitted in the representation the tower produces. Ownership is the campaign's rule.
    The crossover is complete only where all four arms are graded."""
    keys = _blocks(wb, "TOWERSWAP", "towerswap")
    blocks, arms_by_ds = [], {}
    for k in keys:
        t = wb[k]["s"]["towerswap"]
        host, partner = t["reader"], t["partner"]
        combos = []
        for tower, reader in ((host, host), (partner, host), (partner, partner), (host, partner)):
            c = t["combinations"].get(_combo(tower, reader), {})
            done = c.get("status") == "COMPLETE"
            combos.append({"tower": tower, "reader": reader, "native": tower == reader, "complete": done,
                           "owned": sorted(c.get("owned") or []) if done else None,
                           "n_owned": len(c.get("owned") or []) if done else None,
                           "n_concepts": len(c.get("grade") or {}) if done else None,
                           "directions_from": c.get("directions_from") if done else None})
            if done:
                arms_by_ds.setdefault(k[1], {}).setdefault((tower, reader), {})[k[0]] = len(c.get("owned") or [])
        cx = t.get("crossover") or {}
        # the host's native arm is this block's own grade recomputed on the crossover's rows; the full grade lives on the
        # block's whole test cohort, so the two can differ in resolution. Both are recorded here.
        native = t["combinations"].get(_combo(host, host), {})
        core = ci.core(wb[k])
        nvc = None
        if native.get("status") == "COMPLETE" and core:
            g = native["grade"]
            nvc = {"cells": len(core), "owned_full": sum(ci.owned(v) for v in core.values()),
                   "owned_rows": sum(bool(g[q]["owned"]) for q in core if q in g),
                   "max_abs_dO": max(abs(g[q]["O_q"] - core[q]["O_q"]) for q in core if q in g),
                   # a sign flip counts only where at least one of the two cohorts puts the contrast above the campaign's
                   # effect floor; below it both cohorts call the cell unresolved and the sign carries no decision
                   "sign_flips": sum((g[q]["O_q"] > 0) != (core[q]["O_q"] > 0)
                                     and max(abs(g[q]["O_q"]), abs(core[q]["O_q"])) >= ci.EFFECT_FLOOR
                                     for q in core if q in g),
                   "lost_to_competitor": sum(ci.owned(core[q]) and g[q]["verdict"] == "stronger_competitor"
                                             for q in core if q in g),
                   "lost_to_resolution": sum(ci.owned(core[q]) and not g[q]["owned"]
                                             and g[q]["verdict"] != "stronger_competitor" for q in core if q in g)}
        blocks.append({
            "block": f"{k[0]}/{k[1]}", "model": k[0], "dataset": k[1], "reader": host, "partner": partner,
            "n_rows": t["n_rows"], "alpha": t["alpha"], "template_id": t["template_id"], "draws": t["draws"],
            "receipt": t["swap_receipt"], "combinations": combos, "native_vs_full": nvc,
            "crossover_complete": bool(t.get("crossover_complete")),
            "crossover": ({name: {"mean_abs_effect": cx[name]["mean_abs_effect"],
                                  "n_simultaneously_nonzero": cx[name]["n_simultaneously_nonzero"],
                                  "n_concepts": len([x for x in cx[name].values() if isinstance(x, dict)])}
                           for name in ("reader_effect_at_own_tower", "reader_effect_at_partner_tower",
                                        "tower_effect_at_own_reader", "tower_effect_at_partner_reader")}
                          | {"mean_abs_reader_effect": cx["mean_abs_reader_effect"],
                             "mean_abs_tower_effect": cx["mean_abs_tower_effect"],
                             "reader_over_tower": cx["reader_over_tower"]}) if cx else None})
    # the four combinations of a dataset, deduplicated across the two host blocks: an arm graded in both blocks is
    # counted once, taking the block whose own reader it is, so a dataset contributes 4 combinations and 24 cells
    per_ds = {}
    for ds, arms in arms_by_ds.items():
        rows = []
        for (tower, reader), by_host in sorted(arms.items()):
            n = by_host.get(reader, sorted(by_host.values())[0])
            rows.append({"tower": tower, "reader": reader, "native": tower == reader, "n_owned": n,
                         "n_blocks": len(by_host), "agrees": len(set(by_host.values())) == 1})
        n_concepts = {b["dataset"]: max((c["n_concepts"] or 0) for c in b["combinations"]) for b in blocks}[ds]
        per_ds[ds] = {"combinations": rows, "n_combinations": len(rows), "n_concepts": n_concepts,
                      "cells": len(rows) * n_concepts, "owned": sum(r["n_owned"] for r in rows),
                      "native_owned": [r["n_owned"] for r in rows if r["native"]],
                      "crossed_owned": [r["n_owned"] for r in rows if not r["native"]]}
    groups = {"chest": [ds for ds in CHEST if ds in per_ds], "coco": ["coco"] if "coco" in per_ds else []}
    pooled = {g: {"cells": sum(per_ds[ds]["cells"] for ds in dss), "owned": sum(per_ds[ds]["owned"] for ds in dss),
                  "combinations": sum(per_ds[ds]["n_combinations"] for ds in dss),
                  "native_owned": [n for ds in dss for n in per_ds[ds]["native_owned"]],
                  "crossed_owned": [n for ds in dss for n in per_ds[ds]["crossed_owned"]]}
              for g, dss in groups.items() if dss}
    nv = [b["native_vs_full"] for b in blocks if b["native_vs_full"]]
    native = {"blocks": len(nv), "cells": sum(x["cells"] for x in nv),
              "owned_full": sum(x["owned_full"] for x in nv), "owned_rows": sum(x["owned_rows"] for x in nv),
              "max_abs_dO": max((x["max_abs_dO"] for x in nv), default=None),
              "sign_flips": sum(x["sign_flips"] for x in nv),
              "lost_to_competitor": sum(x["lost_to_competitor"] for x in nv),
              "lost_to_resolution": sum(x["lost_to_resolution"] for x in nv)}
    disagree = [f"{ds} {t}/{r}" for ds, d in per_ds.items() for rr in d["combinations"] if not rr["agrees"]
                for t, r in [(rr["tower"], rr["reader"])]]
    rec = [b["receipt"] for b in blocks]
    receipt = {"n_tensors_replaced": sorted({r["n_tensors_replaced"] for r in rec}),
               "n_tower_tensors": sorted({r["n_tower_tensors"] for r in rec}),
               "n_params_replaced": sorted({r["n_params_replaced"] for r in rec}),
               "n_tensors_outside_tower": sorted({r["n_tensors_outside_tower"] for r in rec}),
               "n_tensors_outside_tower_changed": sorted({r["n_tensors_outside_tower_changed"] for r in rec}),
               "verified_bitwise_equal_to_donor": all(r["verified_bitwise_equal_to_donor"] for r in rec),
               "verified_rest_unchanged": all(r["verified_rest_unchanged"] for r in rec)}
    return {"blocks": blocks, "per_dataset": per_ds, "pooled": pooled, "receipt": receipt, "native": native,
            "arms_disagreeing_across_hosts": disagree,
            "n_arms_in_two_hosts": sum(1 for d in per_ds.values() for r in d["combinations"] if r["n_blocks"] > 1),
            "n_rows": sorted({b["n_rows"] for b in blocks}), "alpha": sorted({b["alpha"] for b in blocks}),
            "draws": sorted({b["draws"] for b in blocks}),
            "crossover_blocks": [b for b in blocks if b["crossover"]]}


# ------------------------------------------------------------------------------------------ identical-tensor replay

def replay(wb: dict) -> dict:
    """Per block: the replayed grade against the block's own grade, the drift of this reader's own tower output from
    the stored tensor, and, per shared-tower pair, how much the between-reader difference changes when the two readers
    are fed the same tensors bit for bit instead of each running its own tower."""
    keys = _blocks(wb, "REPLAY", "replay")
    blocks, pairs = [], {}
    for k in keys:
        r = wb[k]["s"]["replay"]
        d = {q: v for q, v in r["replay_minus_core"].items() if isinstance(v, dict)}
        blocks.append({
            "block": f"{k[0]}/{k[1]}", "model": k[0], "dataset": k[1], "reader": r["reader"],
            "source_block": r["source_block"], "self_replay": r["reader"] == r["source_block"],
            "directions_from": r["directions_from"], "n_rows": r["n_rows"], "draws": r["draws"],
            "n_concepts": len(d), "max_abs_dW": max(abs(v["estimate"]) for v in d.values()),
            "max_abs_dO": max(abs(v["dO_q"]) for v in d.values()),
            "n_nonzero": sum(bool(v["nonzero_simultaneous"]) for v in d.values()),
            "verdict_changes": sum(bool(v["verdict_changed"]) for v in d.values()),
            "owned_changes": sum(bool(v["owned_changed"]) for v in d.values()),
            "exact": all(v["estimate"] == 0.0 for v in d.values()),
            "drift_max_abs": r["drift"]["max_abs"], "drift_mean_abs": r["drift"]["mean_abs"],
            "block_mean_abs": r["drift"]["block_mean_abs"], "rows_replayed": r["drift"]["rows_replayed"]})
        for other, gc in r["group_comparison"].items():
            if gc.get("status") != "COMPLETE":
                continue
            pairs.setdefault(frozenset((r["reader"], other)), {
                "readers": tuple(sorted((r["reader"], other))), "dataset": k[1],
                "mean_abs_own_towers": gc["mean_abs_own_towers"], "mean_abs_exact": gc["mean_abs_exact"],
                "mean_abs_change": gc["mean_abs_change"],
                "n_nonzero_exact": sum(bool(v["nonzero_simultaneous"]) for v in gc["exact"].values() if isinstance(v, dict)),
                "n_concepts": len([v for v in gc["exact"].values() if isinstance(v, dict)])})
    ps = sorted(pairs.values(), key=lambda p: p["readers"])
    selves = [b for b in blocks if b["self_replay"]]
    cross = [b for b in blocks if not b["self_replay"]]
    return {"blocks": blocks, "pairs": ps,
            "n_self": len(selves), "n_cross": len(cross), "n_self_exact": sum(b["exact"] for b in selves),
            "cells": sum(b["n_concepts"] for b in blocks),
            "n_nonzero": sum(b["n_nonzero"] for b in blocks),
            "verdict_changes": sum(b["verdict_changes"] for b in blocks),
            "owned_changes": sum(b["owned_changes"] for b in blocks),
            "max_abs_dW": max(b["max_abs_dW"] for b in blocks),
            "max_abs_dW_self": max(b["max_abs_dW"] for b in selves) if selves else None,
            "max_abs_dO": max(b["max_abs_dO"] for b in blocks),
            "n_drift_zero": sum(b["drift_max_abs"] == 0.0 for b in blocks),
            "drift_max_abs": max(b["drift_max_abs"] for b in blocks),
            "pair_change_max": max(abs(p["mean_abs_change"]) for p in ps) if ps else None,
            "pair_own_min": min(p["mean_abs_own_towers"] for p in ps) if ps else None,
            "pair_own_max": max(p["mean_abs_own_towers"] for p in ps) if ps else None,
            "pair_nonzero": sum(p["n_nonzero_exact"] for p in ps), "pair_cells": sum(p["n_concepts"] for p in ps)}


# ------------------------------------------------------------------------------------------------ semantic endpoints

FAMS = ("label", "answer")


def semend(wb: dict) -> dict:
    """Per block: ownership of both direction families at each of the four endpoints, the negation sign test, the
    counterbalanced forced choice, the report continuation, and the incremental validity of the ownership contrast
    over the write magnitude, the probe selectivity and the clean-answer AUROC."""
    keys = _blocks(wb, "SEMEND", "semend")
    blocks = []
    for k in keys:
        s = wb[k]["s"]["semend"]
        eps = list(s["endpoints"])
        per_ep = {ep: {fam: {"owned": len(s["per_endpoint"][ep]["families"][fam]["owned"]),
                             "reference_met": len(s["per_endpoint"][ep]["families"][fam]["reference_met"]),
                             "n_concepts": len(s["per_endpoint"][ep]["families"][fam]["per_question"])}
                       for fam in FAMS} for ep in eps}
        neg, fc, rep = s["negation"], s["forced_choice"], s["report"]
        rep_cells = {q: v for q, v in rep.items() if isinstance(v, dict)}
        iv = {fam: {"n_cells": s["incremental_validity"][fam]["n_cells"],
                    "base_r2": s["incremental_validity"][fam]["base_model"]["r2"],
                    "full_r2": s["incremental_validity"][fam]["full_model"]["r2"],
                    "delta_r2": s["incremental_validity"][fam]["ownership_adds"]["delta_r2"],
                    "delta_r2_ci95": s["incremental_validity"][fam]["ownership_adds"]["delta_r2_ci95"],
                    "excludes_zero": bool(s["incremental_validity"][fam]["ownership_adds"]["delta_r2_interval_excludes_zero"]),
                    "coef_O_q": s["incremental_validity"][fam]["ownership_adds"]["coef_O_q_core"],
                    "coef_O_q_ci95": s["incremental_validity"][fam]["ownership_adds"]["coef_O_q_core_ci95"],
                    "coef_excludes_zero": bool(s["incremental_validity"][fam]["ownership_adds"]["O_q_core_interval_excludes_zero"]),
                    "base_predictors": s["incremental_validity"][fam]["base_predictors"]}
              for fam in FAMS}
        blocks.append({
            "block": f"{k[0]}/{k[1]}", "model": k[0], "dataset": k[1], "n_rows": s["n_rows"], "alpha": s["alpha"],
            "draws": s["draws"], "endpoints": eps, "n_random": s["n_random"], "primary_template": s["primary_template"],
            "per_endpoint": per_ep, "owned": dict(s["summary"]["n_owned"]), "reference_met": dict(s["summary"]["n_reference_met"]),
            "n_cells": s["summary"]["n_cells"], "owned_core": list(s["summary"]["owned_core"]),
            "negation_opposite": neg["n_opposite_sign"], "negation_n": len(neg["raw_negated_effect"]),
            "negation_corr": neg["correlation_affirmative_vs_raw_negated"],
            "forced_both_owned": fc["n_both_owned"], "forced_order_gap": fc["mean_abs_order_gap"],
            "report_cells": len(rep_cells), "report_owned": sum(bool(v["owned"]) for v in rep_cells.values()),
            "incremental_validity": iv})
    ivs = [(b, fam, b["incremental_validity"][fam]) for b in blocks for fam in FAMS]
    return {"blocks": blocks, "n_blocks": len(blocks),
            "endpoints": sorted({ep for b in blocks for ep in b["endpoints"]}),
            "n_endpoints": max(len(b["endpoints"]) for b in blocks) if blocks else 0,
            "cells": sum(b["n_cells"] for b in blocks),
            "owned": {fam: sum(b["owned"][fam] for b in blocks) for fam in FAMS},
            "reference_met": {fam: sum(b["reference_met"][fam] for b in blocks) for fam in FAMS},
            "negation_opposite": sum(b["negation_opposite"] for b in blocks),
            "negation_n": sum(b["negation_n"] for b in blocks),
            "forced_both_owned": sum(b["forced_both_owned"] for b in blocks),
            "forced_gap_min": min(b["forced_order_gap"] for b in blocks) if blocks else None,
            "forced_gap_max": max(b["forced_order_gap"] for b in blocks) if blocks else None,
            "report_cells": sum(b["report_cells"] for b in blocks),
            "report_owned": sum(b["report_owned"] for b in blocks),
            "iv_n": len(ivs), "iv_excludes_zero": sum(v["excludes_zero"] for _, _, v in ivs),
            "iv_coef_excludes_zero": sum(v["coef_excludes_zero"] for _, _, v in ivs),
            "iv_delta_min": min(v["delta_r2"] for _, _, v in ivs) if ivs else None,
            "iv_delta_max": max(v["delta_r2"] for _, _, v in ivs) if ivs else None,
            "iv_per_family": {fam: {"delta_min": min(b["incremental_validity"][fam]["delta_r2"] for b in blocks),
                                    "delta_max": max(b["incremental_validity"][fam]["delta_r2"] for b in blocks),
                                    "base_min": min(b["incremental_validity"][fam]["base_r2"] for b in blocks),
                                    "base_max": max(b["incremental_validity"][fam]["base_r2"] for b in blocks)}
                              for fam in FAMS} if blocks else {},
            "base_predictors": blocks[0]["incremental_validity"]["label"]["base_predictors"] if blocks else []}


# --------------------------------------------------------------------------------------- fine-grained object family

MATCHINGS = ("selectivity", "answer_auroc", "both")


def fgobj(wb: dict) -> dict:
    """The six fine-grained COCO categories against the six easy ones and against the chest cells, with the
    difficulty-matched comparison (runs/robustness/round2.json fgobj section). The block set of the section is
    checked against the included blocks carrying FGOBJ."""
    F = json.loads((ROB / "round2.json").read_text())["fgobj"]
    keys = _blocks(wb, "FGOBJ", "fgobj")
    got = {tuple(b["block"].split("/")) for b in F["blocks"]}
    if got != set(keys):
        raise RuntimeError(f"round2.json fgobj blocks differ from the included blocks with FGOBJ by "
                           f"{sorted(got ^ set(keys))}; rerun scripts/mayo/robustness_round2.py")
    for m in MATCHINGS + ("both_easy_partners_only",):
        d = F["matched"][m]
        if d.get("pooled_ownership_difference") is not None and not d.get("pooled_equals_rate_difference"):
            raise RuntimeError(f"round2.json fgobj {m}: the pooled difference is not the difference of the two reported "
                               f"rates; rerun scripts/mayo/robustness_round2.py")
    if "coco_easy_same_blocks" not in F["pool"] or "same_block_comparison" not in F:
        raise RuntimeError("round2.json fgobj carries no same-block easy pool; rerun scripts/mayo/robustness_round2.py")
    return {"concepts": list(F["concepts"]), "blocks": F["blocks"], "pool": F["pool"],
            "matched": {m: F["matched"][m] for m in MATCHINGS},
            "matched_easy_only": F["matched"]["both_easy_partners_only"],
            "same_block": F["same_block_comparison"],
            "prespecification": F["prespecification"], "matching_variables": F["matching_variables"],
            "window": F["matched"]["both"]["window_answer_auroc"]}


# ------------------------------------------------------------------------ held-out incremental validity of ownership

def semend_cv(wb: dict) -> dict:
    """The out-of-sample test of Section 5.6: does ownership add anything about endpoint behaviour beyond write
    magnitude, probe selectivity and clean-answer AUROC once the fit never sees the fold it predicts?
    (runs/robustness/round2.json semend section). The block set is checked against the included blocks with SEMEND."""
    S = json.loads((ROB / "round2.json").read_text()).get("semend")
    if not S or not S.get("leave_one_concept_out"):
        raise RuntimeError("round2.json carries no semend held-out section; rerun scripts/mayo/robustness_round2.py")
    keys = _blocks(wb, "SEMEND", "semend")
    got = {tuple(b.split("/")) for b in S["blocks"]}
    if got != set(keys):
        raise RuntimeError(f"round2.json semend blocks differ from the included blocks with SEMEND by "
                           f"{sorted(got ^ set(keys))}; rerun scripts/mayo/robustness_round2.py")
    return S


def validfit_arms() -> dict:
    """The label-source-against-sample-size arms of Appendix A (runs/robustness/validfit.json,
    scripts/mayo/robustness_validfit.py). Every arm is scored on the same radiologist-labelled films with the same
    projection, the same stored train-only scaler, the same probe settings and the same cross-fitting."""
    path = ROB / "validfit.json"
    if not path.exists():
        raise RuntimeError(f"{path} is missing; run scripts/mayo/robustness_validfit.py (CPU, minutes)")
    V = json.loads(path.read_text())
    if not V.get("blocks"):
        raise RuntimeError("runs/robustness/validfit.json carries no blocks; rerun scripts/mayo/robustness_validfit.py")
    return V


# ------------------------------------------------------------------------------------------------ phrasing selection

def attrq(wb: dict) -> dict:
    """The three phrasings of each attribute question scored on the calibration rows: whether the phrasing that was
    written is the one with the highest one-sided lower bound of the clean-answer AUROC, and by how much it is not."""
    keys = _blocks(wb, "ATTRQ", "attrq")
    cells, phr = [], set()
    for k in keys:
        a = wb[k]["s"]["attrq"]
        phr.update(a["phrasings"])
        for q, v in a["per_attribute"].items():
            w = v["phrasings"][v["written_phrasing"]]
            cells.append({"block": f"{k[0]}/{k[1]}", "dataset": k[1], "attribute": q,
                          "status": v["selection_status"], "written": v["written_phrasing"],
                          "written_auroc": w["answer_auroc"], "written_capable": bool(w["answer_capable"]),
                          "selected": v["selected"], "selected_auroc": v["selected_answer_auroc"],
                          "n_eligible": len(v["eligible_phrasings"]),
                          "changes": bool(v["selection_changes_written_phrasing"])})
    sel = [c for c in cells if c["status"] == "selected"]
    gains = [c["selected_auroc"] - c["written_auroc"] for c in sel]
    return {"blocks": [f"{k[0]}/{k[1]}" for k in keys], "n_blocks": len(keys), "cells": cells,
            "n_cells": len(cells), "n_phrasings": len(phr), "phrasings": sorted(phr),
            "n_changes": sum(c["changes"] for c in cells), "n_selected": len(sel),
            "n_written_capable": sum(c["written_capable"] for c in cells),
            "median_written_auroc": _med([c["written_auroc"] for c in cells]),
            "median_selected_auroc": _med([c["selected_auroc"] for c in sel]),
            "median_gain": _med(gains), "max_gain": max(gains) if gains else None}


# ------------------------------------------------------------------------------------------------- attribute control

def attr_modes(wb: dict) -> dict:
    """The three ways the attribute comparison is reported (runs/robustness/round2.json attr section), pooled over the
    chest blocks, together with the block set that grades attribute and clinical cells against one steering
    reference."""
    A = json.loads((ROB / "round2.json").read_text())["attr"]
    ch = A["per_dataset"]["chest"]
    c = ch["comparison"]
    return {"blocks": ch["blocks"], "attributes": list(A["attributes"]), "window": A["match_window"],
            "modes": {m: c[m] for m in ("all_cells", "answer_capable_cells", "answerability_matched")},
            "matched_reference_blocks": len(ch["blocks_matched_reference"]),
            "attribute_random_reference": bool(ch["attribute_random_reference"]),
            "clinical_random_reference": bool(ch["clinical_random_reference"]),
            "per_dataset": {ds: A["per_dataset"][ds]["comparison"] for ds in CHEST if A["per_dataset"][ds]["blocks"]}}


# ------------------------------------------------------- crossed tower and reader: the factor effects on ownership

CROSSOVER_QUANTITIES = ("W_qq", "O_q", "M_q")
CROSSOVER_CONTRASTS = ("reader_effect_at_own_tower", "reader_effect_at_partner_tower",
                       "tower_effect_at_own_reader", "tower_effect_at_partner_reader")


def crossover(wb: dict) -> dict:
    """The four crossover contrasts of the tower-and-reader experiment on three quantities, from
    runs/robustness/crossover.json: the own-question write magnitude, the ownership contrast, and the margin over the
    strongest competitor in the full reference family. One pass over the current outcomes computes all three, so the
    columns of one row cannot come from different states of the same arm. The block set is checked against the blocks
    whose packaged crossover is complete."""
    C = json.loads((ROB / "crossover.json").read_text())
    got = {b["block"] for b in C["blocks"]}
    want = {f"{k[0]}/{k[1]}" for k in _blocks(wb, "TOWERSWAP", "towerswap")
            if (wb[k]["s"]["towerswap"] or {}).get("crossover_complete")}
    if got != want:
        raise RuntimeError(f"crossover.json covers {sorted(got)}, the summaries say the crossover is complete for "
                           f"{sorted(want)}; rerun scripts/mayo/robustness_crossover.py")
    return {"meta": C["meta"], "blocks": C["blocks"], "n_blocks": len(C["blocks"]),
            "quantities": list(CROSSOVER_QUANTITIES), "contrasts": list(CROSSOVER_CONTRASTS),
            "by_block": {b["block"]: b for b in C["blocks"]},
            "stale_packaged": list(C["meta"]["blocks_whose_packaged_W_qq_crossover_predates_the_current_outcomes"])}
