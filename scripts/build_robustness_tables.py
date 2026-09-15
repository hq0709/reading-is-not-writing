"""Robustness tables for the paper from runs/robustness/*.json (geometry now; scale when present)."""
import json
from pathlib import Path

ROB = Path("/rodata/azradonc_dev/m253405/cf-transfer/runs/robustness")
OUT = Path(__file__).resolve().parents[1] / "tables"
NAMES = {"nih": "NIH ChestX-ray14", "chexpert": "CheXpert Plus", "coco": "COCO"}


def f2(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def geometry():
    d = json.loads((ROB / "geometry.json").read_text())
    dg, lg, lo, assoc = d["direction_geometry"], d["label_geometry"], d["leave_one_out"], d["association"]
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Direction and label geometry against off-diagonal dominance.} Per dataset: blocks with a complete write "
         r"matrix; mean off-diagonal cosine between the six write directions in model space (random pairs of unit vectors give "
         r"$0.023$); mean off-diagonal $\phi$ between the six test labels; the fraction of cells whose strongest competitor is the "
         r"most similar direction, and the most co-occurring label (chance $0.20$); Spearman correlation over off-diagonal cells "
         r"between the competitor advantage $W_{q,d}-W_{q,q}$ and cosine; the fraction of cells with $O_q>0$, the same after "
         r"removing the strongest competitor, and the fraction with at least two competitors above the own write. Bottom: "
         r"COCO ownership when the competitor set is restricted to a co-occurring family (mean within-family $\phi$ in parentheses).}",
         r"\label{tab:cf-geometry}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r"Dataset & blocks & cos & $\phi$ & comp.\ = max cos & comp.\ = max $\phi$ & $\rho$(adv., cos) & $O_q>0$ & $O_q^{-d^*}>0$ & $\geq2$ above \\", r"\midrule"]
    for ds in ("nih", "chexpert", "coco"):
        g, l, o, a = dg[ds], lg[ds], lo[ds], assoc[ds]
        L.append(f"{NAMES[ds]} & {g['n_blocks']} & {f2(g['offdiag_cos_model_mean'],3)} & {f2(l['offdiag_phi_mean'],3)} & "
                 f"{f2(g['coincide_cos_model']['fraction'])} & {f2(l['coincide_phi']['fraction'])} & {f2(a['rho_adv_vs_cos_model']['rho'])} & "
                 f"{o['O_q_positive']}/{o['n_cells']} & {o['O_loo_positive']}/{o['n_cells']} & {f2(o['n_above_ge2']/o['n_cells'])} \\\\")
    L += [r"\midrule", r"\multicolumn{10}{l}{\textit{COCO restricted competitor families: cells with $O_q>0$ within the family / with all five competitors}} \\"]
    for fam, v in d["coco_families"].items():
        phi = v.get("phi_within_family_mean")
        L.append(f"\\quad {fam.replace('+', ' + ')} ($\\phi={f2(phi)}$) & \\multicolumn{{9}}{{l}}{{{v['O_family_positive']}/{v['n_cells']} within the family; {v['O_full_positive']}/{v['n_cells']} with all five competitors}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table_cf_geometry.tex").write_text("\n".join(L) + "\n")
    print("table_cf_geometry.tex")


def scale():
    d = json.loads((ROB / "scale.json").read_text())
    m, st, rf, cn, bt = d["item1_margin_scale"], d["item2_ceiling"]["strata"], d["item3_refit"]["per_dataset"], d["item4_connector"]["per_dataset"], d["item5_batch"]
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Ownership under alternative scales, strata, refits, and loci.} Per dataset: owned cells on the "
         r"probability scale (the paper's verdict) and on the logit-margin scale (same rules; percentile interval for $O^m_q$), with "
         r"the cells owned on both; cells with $O_q>0$ among owned and among not-owned cells when every cell is restricted to its "
         r"unsaturated rows ($0.01<P(\mathrm{yes})<0.99$ on the clean pass); median $|O_q(\text{seed }k)-O_q(\text{seed }0)|$ over the "
         r"two refit seeds and the owned cells that keep $O_q>0$ under both (REFIT exists for NIH and COCO); median $|W_{q,q}|$ at the "
         r"primary and connector loci and connector cells whose write beats the connector random family.}",
         r"\label{tab:cf-scale}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
         r"Dataset & owned ($p$) & owned ($m$) & both & unsat.\ owned $O>0$ & unsat.\ other $O>0$ & refit med.\ $|\Delta O|$ & owned stable & $|W_{q,q}|$ prim.\ & $|W_{q,q}|$ conn.\ & conn.\ $>$ p95 \\", r"\midrule"]
    for ds in ("nih", "chexpert", "coco"):
        mm = m[ds]; ag = mm["agreement_owned_p_vs_owned_m"]; u = st[ds]["unsaturated"]
        r = rf.get(ds); c = cn.get(ds, {}).get("p_scale", {})
        refit = f"{r['median_abs_dO_pooled']:.3f} & {r['owned_p_O_pos_both_refits']}/{r['owned_p']}" if r else "-- & --"
        conn = f"{c['median_abs_W_qq_primary']:.3f} & {c['median_abs_W_qq_connector']:.3f} & {c['connector_W_qq_gt_random_p95']}/{cn[ds]['n_cells']}" if c else "-- & -- & --"
        L.append(f"{NAMES[ds]} & {mm['owned_p']}/{mm['n_cells']} & {mm['owned_m_(CI>0&ref)']}/{mm['n_cells']} & {ag['a_yes_b_yes']} & "
                 f"{u['owned_p_O_pos']}/{u['owned_p_evaluable']} & {u['not_owned_O_pos']}/{u['not_owned_evaluable']} & {refit} & {conn} \\\\")
    dist = bt["distribution"]["vis.last"]
    n_dis = len(bt["sign_disagreement"]) if isinstance(bt["sign_disagreement"], list) else len(bt["sign_disagreement"].get("blocks", []))
    L += [r"\midrule", r"\multicolumn{11}{l}{\textit{Batch effects at the gates (61 blocks): batched-vs-single candidate-logit difference "
          + f"{dist['max_abs_candidate_logit_diff']['min']:.2f}--{dist['max_abs_candidate_logit_diff']['max']:.2f} (median {dist['max_abs_candidate_logit_diff']['median']:.2f}); "
          + f"margin sign agreement in {61 - n_dis} of 61 blocks; owned cells in disagreeing blocks: {bt['owned_cells_in_sign_disagreement_blocks']}.}}}} \\\\",
          r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table_cf_scale.tex").write_text("\n".join(L) + "\n")
    print("table_cf_scale.tex")


def pairs():
    d = json.loads((ROB / "pairs.json").read_text())
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Same write, different reader: paired differences.} For every pair of checkpoints that share a vision "
         r"tower (identical features and write vectors, or equal to one bf16 rounding step), the difference in ownership "
         r"$\Delta O_q=O_q(\text{larger})-O_q(\text{smaller})$ on the same 600 test rows, with a paired patient-bootstrap 95\% "
         r"interval (2{,}000 shared draws, strongest competitor recomputed in every draw): concepts whose interval excludes zero, "
         r"the median and maximum $|\Delta O_q|$, and their ratio to the mean refit standard deviation of the two blocks.}",
         r"\label{tab:cf-pairs}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{llcrrrr}", r"\toprule",
         r"Pair & Dataset & relation & $\Delta O_q$ CI $\not\ni 0$ & median $|\Delta O_q|$ & max $|\Delta O_q|$ & median / max ratio to refit SD \\", r"\midrule"]
    names = {"gemma3-4": "Gemma 3 4B", "gemma3-12": "Gemma 3 12B", "gemma3-27": "Gemma 3 27B", "medgemma-4": "MedGemma 4B", "medgemma-27": "MedGemma 27B"}
    for t in d["task1_paired_deltas"]:
        rel = "bit-identical" if t["relationship"] == "bit-identical" else "bf16-equal"
        ratio = f"{t['median_abs_delta_O_over_refit_sd']:.1f} / {t['max_abs_delta_O_over_refit_sd']:.1f}" if t.get("median_abs_delta_O_over_refit_sd") else "--"
        L.append(f"{names.get(t['small'], t['small'])} $\\to$ {names.get(t['large'], t['large'])} & {NAMES[t['dataset']]} & {rel} & "
                 f"{t['n_concepts_delta_O_ci_excludes_zero']}/6 & {t['median_abs_delta_O']:.2f} & {t['max_abs_delta_O']:.2f} & {ratio} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table_cf_pairs.tex").write_text("\n".join(L) + "\n")
    print("table_cf_pairs.tex")


RUNS = Path("/rodata/azradonc_dev/m253405/cf-transfer/runs")
FAMILIES = ("dom", "pattern", "orth", "resid")
FAM_LABEL = {"logistic": "logistic normal", "dom": "difference of means", "pattern": "Haufe pattern", "orth": "orthogonalised normal", "resid": "residualised normal"}


def altdir():
    """Per dataset and direction family: owned cells (fixed-family advantage with the steering reference), cells
    owned under any family, and the logistic reference, over every block whose summary carries ALTDIR."""
    import statistics
    rows = {ds: {f: {"owned": 0, "n": 0, "O": []} for f in ("logistic",) + FAMILIES} for ds in NAMES}
    anyf = {ds: [0, 0] for ds in NAMES}; blocks = {ds: 0 for ds in NAMES}
    for sp in sorted(RUNS.glob("*/*/summary.json")):
        j = json.loads(sp.read_text()); a = j.get("altdir")
        if not a or not all(f in a for f in FAMILIES):
            continue
        ds = sp.parent.name; blocks[ds] += 1
        core = j["core"]["per_question"]
        for q, v in core.items():
            ownedlog = bool(v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage")
            rows[ds]["logistic"]["owned"] += ownedlog; rows[ds]["logistic"]["n"] += 1; rows[ds]["logistic"]["O"].append(v["O_q"])
            anyo = ownedlog
            for f in FAMILIES:
                c = a[f]["per_question"][q]
                o = bool(c.get("steering_reference") and c.get("verdict") == "fixed_family_advantage")
                rows[ds][f]["owned"] += o; rows[ds][f]["n"] += 1; rows[ds][f]["O"].append(c["O_q"]); anyo = anyo or o
            anyf[ds][0] += anyo; anyf[ds][1] += 1
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Ownership under five direction constructions.} For every block with the alternative-direction module, "
         r"cells owned (fixed-family advantage with the steering reference, the paper's rule applied within each family) and the "
         r"median $O_q$, per dataset and family; the last column counts cells owned under at least one family. Difference-of-means "
         r"and pattern directions are built from the same projected, train-scaled features as the logistic normal; the "
         r"orthogonalised normal has the other five normals' span removed; the residualised normal is refitted on features "
         r"residualised against the other five labels.}",
         r"\label{tab:cf-altdir}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{lr" + "rr" * 5 + r"r}", r"\toprule",
         r"Dataset & blocks & \multicolumn{2}{c}{logistic} & \multicolumn{2}{c}{diff.\ of means} & \multicolumn{2}{c}{pattern} & \multicolumn{2}{c}{orthogonalised} & \multicolumn{2}{c}{residualised} & any \\",
         r" & & owned & med.\ $O$ & owned & med.\ $O$ & owned & med.\ $O$ & owned & med.\ $O$ & owned & med.\ $O$ & \\", r"\midrule"]
    for ds in ("nih", "chexpert", "coco"):
        if blocks[ds] == 0:
            continue
        cells = [f"{NAMES[ds]} & {blocks[ds]}"]
        for f in ("logistic",) + FAMILIES:
            r = rows[ds][f]; cells.append(f"{r['owned']}/{r['n']} & {statistics.median(r['O']):+.2f}")
        cells.append(f"{anyf[ds][0]}/{anyf[ds][1]}")
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table_cf_altdir.tex").write_text("\n".join(L) + "\n")
    print("table_cf_altdir.tex", {ds: blocks[ds] for ds in blocks})


if __name__ == "__main__":
    geometry()
    scale()
    pairs()
    altdir()
