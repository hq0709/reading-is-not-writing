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


if __name__ == "__main__":
    geometry()
