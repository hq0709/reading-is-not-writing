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


if __name__ == "__main__":
    geometry()
    scale()
