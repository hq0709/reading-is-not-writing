"""Robustness tables for the paper from runs/robustness/*.json (geometry, scale, pairs) and, for the addendum modules
(ALTDIR, ANSDIR, EXTCOMP, TOKENW), from the summaries of the blocks under the paper's one inclusion rule
(cf_inclusion.block_included: CORE and CALIBRATION in run.json completed_modules)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cf_inclusion as ci   # noqa: E402

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


def included_summaries(key=None):
    """(summary path, summary) of every block that enters the paper's counts (cf_inclusion.block_included)."""
    paths = [sp for sp in RUNS.glob("*/*/summary.json")
             if (sp.parent / "run.json").exists() and ci.block_included(json.loads((sp.parent / "run.json").read_text()))]
    return [(sp, json.loads(sp.read_text())) for sp in sorted(paths, key=key)]

FAM_LABEL = {"logistic": "logistic normal", "dom": "difference of means", "pattern": "Haufe pattern", "orth": "orthogonalised normal", "resid": "residualised normal"}


def altdir():
    """Per dataset and direction family: owned cells (fixed-family advantage with the steering reference), cells
    owned under any family, and the logistic reference, over every block whose summary carries ALTDIR."""
    import statistics
    rows = {ds: {f: {"owned": 0, "n": 0, "O": []} for f in ("logistic",) + FAMILIES} for ds in NAMES}
    anyf = {ds: [0, 0] for ds in NAMES}; blocks = {ds: 0 for ds in NAMES}
    for sp, j in included_summaries():
        a = j.get("altdir")
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


def ansdir():
    """Answer-direction oracle: per block, cells where the answer direction a_q is owned, where a_q beats the strongest
    logistic competitor, median W^a_qq, median cos(a_q, w_q), median CV R^2; plus the logistic owned count."""
    import statistics
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{The direction the reader responds to.} For each block, the answer direction $a_q$ is a ridge "
         r"regression of the model's own clean answer margin on the same projected, train-scaled features the probes use "
         r"(3{,}000 training rows), lifted like the probe normal and written at the same dose against the same reference "
         r"family. Columns: concepts whose $a_q$ is owned (steering reference and fixed-family advantage over the other five "
         r"$a_d$); concepts whose $a_q$ write beats the strongest logistic competitor of the label direction; median own write "
         r"$W^a_{q,q}$; median cosine between $a_q$ and the label direction $\hat w_q$; median cross-validated $R^2$ of the "
         r"regression; and, for reference, the concepts owned by the label direction.}",
         r"\label{tab:cf-ansdir}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Checkpoint & Dataset & $a_q$ owned & $a_q>$ max logistic comp. & med.\ $W^a_{q,q}$ & med.\ $\cos(a_q,\hat w_q)$ & med.\ $R^2$ & $\hat w_q$ owned \\", r"\midrule"]
    names = {"q25-7": "Qwen2.5-VL-7B", "lingshu-32": "Lingshu 32B", "gemma3-12": "Gemma 3 12B"}
    for sp, j in included_summaries(key=lambda p: (list(names).index(p.parent.parent.name) if p.parent.parent.name in names else 9, ["nih", "chexpert", "coco"].index(p.parent.name))):
        a = j.get("ansdir")
        if not a:
            continue
        mk, ds = sp.parent.parent.name, sp.parent.name
        pq = a["per_question"]; core = j["core"]["per_question"]
        owned = sum(bool(v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage") for v in pq.values())
        beats = sum(v.get("own_minus_max_logistic_competitor_ci95_percentile", [0, 0])[0] > 0 for v in pq.values())
        wqq = statistics.median(v["W_qq"] for v in pq.values()); cos = statistics.median(v.get("cos_to_logistic_model", float("nan")) for v in pq.values())
        r2 = statistics.median(v.get("cv_r2", v.get("r2", float("nan"))) for v in pq.values())
        lowned = sum(bool(v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage") for v in core.values())
        L.append(f"{names.get(mk, mk)} & {NAMES[ds]} & {owned}/6 & {beats}/6 & {wqq:.2f} & {cos:.2f} & {r2:.2f} & {lowned}/6 \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table_cf_ansdir.tex").write_text("\n".join(L) + "\n")
    print("table_cf_ansdir.tex")


def extcomp():
    """Extended competitor family: per block, owned under the six-direction family vs under the extended family."""
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Ownership under an extended competitor family.} Extra clinical directions fitted like the protocol "
         r"normals for every additional dataset label with at least 100 known positives and negatives (NIH: Consolidation, "
         r"Edema, Infiltration; CheXpert: Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, Pneumonia, Support "
         r"Devices) and written at the same dose; ownership is then the margin over the five protocol competitors and every "
         r"extra direction.}",
         r"\label{tab:cf-extcomp}",
         r"\begin{tabular}{llrrr}", r"\toprule",
         r"Checkpoint & Dataset & extra directions & owned (six directions) & owned (extended family) \\", r"\midrule"]
    names = {"q25-7": "Qwen2.5-VL-7B", "lingshu-32": "Lingshu 32B", "gemma3-12": "Gemma 3 12B", "q3-8": "Qwen3-VL-8B", "iv35-8": "InternVL3.5-8B", "medgemma-4": "MedGemma 4B"}
    for sp, j in included_summaries():
        a = j.get("extcomp")
        if not a:
            continue
        mk, ds = sp.parent.parent.name, sp.parent.name
        pq = a["per_question"]; core = j["core"]["per_question"]
        owned_ext = sum(bool(v["W_qq"] > 0 and v["W_qq"] > v.get("random_p95", 0) and v["W_qq"] > v.get("abs_sham", 0) and v.get("verdict") == "fixed_family_advantage") for v in pq.values())
        lowned = sum(bool(v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage") for v in core.values())
        L.append(f"{names.get(mk, mk)} & {NAMES[ds]} & {len(a.get('extra_directions', []))} & {lowned}/6 & {owned_ext}/6 \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_extcomp.tex").write_text("\n".join(L) + "\n")
    print("table_cf_extcomp.tex")


def tokenw():
    """Token-weighted writes: per block and variant, owned cells and median own write, beside the uniform write."""
    import statistics
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Token-weighted writes.} The six logistic directions written with a per-token weight instead of "
         r"uniformly: \emph{softmax} weights each consumed token by the softmax of its probe score (mean weight one, so the same L1 "
         r"dose), \emph{top quarter} writes only the 25\% of tokens with the highest probe score at four times the weight (twice the Frobenius norm of the uniform write). "
         r"Cells owned and median own write $W_{q,q}$ per variant, beside the uniform write of the main protocol.}",
         r"\label{tab:cf-tokenw}",
         r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Checkpoint & Dataset & \multicolumn{2}{c}{uniform} & \multicolumn{2}{c}{softmax} & \multicolumn{2}{c}{top quarter} \\",
         r" & & owned & med.\ $W_{q,q}$ & owned & med.\ $W_{q,q}$ & owned & med.\ $W_{q,q}$ \\", r"\midrule"]
    names = {"q25-7": "Qwen2.5-VL-7B", "lingshu-32": "Lingshu 32B", "gemma3-12": "Gemma 3 12B", "q3-8": "Qwen3-VL-8B"}
    for sp, j in included_summaries():
        t = j.get("tokenw")
        if not t:
            continue
        mk, ds = sp.parent.parent.name, sp.parent.name
        core = j["core"]["per_question"]
        cells = [f"{names.get(mk, mk)} & {NAMES[ds]}",
                 f"{sum(bool(v.get('steering_reference') and v.get('verdict') == 'fixed_family_advantage') for v in core.values())}/6 & {statistics.median(v['W_qq'] for v in core.values()):.2f}"]
        for var in ("tokenw", "topq"):
            pq = t.get(var, {}).get("per_question", {})
            if not pq:
                cells.append("-- & --"); continue
            owned = sum(bool(v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage") for v in pq.values())
            cells.append(f"{owned}/6 & {statistics.median(v['W_qq'] for v in pq.values()):.2f}")
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_tokenw.tex").write_text("\n".join(L) + "\n")
    print("table_cf_tokenw.tex")


def precision():
    """PRECISION: the CORE grid on the first 200 test rows rescored with fp32 weights and forward, and in bf16 at batch
    size one, against the CORE values on the same rows (summary.json["precision"] of every included block that has it)."""
    names = {"q25-7": "Qwen2.5-VL-7B", "q3-8": "Qwen3-VL-8B", "lingshu-32": "Lingshu 32B", "gemma3-12": "Gemma 3 12B", "iv35-8": "InternVL3.5-8B", "medgemma-4": "MedGemma 4B"}
    rows = []
    for sp, j in included_summaries():
        p = j.get("precision")
        if not p:
            continue
        mk, ds = sp.parent.parent.name, sp.parent.name
        cells = [f"{names.get(mk, mk)} & {NAMES[ds]} & {p['n_rows']}"]
        for st in p["settings"]:
            r = p.get(st, {})
            if r.get("status") != "COMPLETE":
                cells.append("-- & -- & -- & --"); continue
            cells.append(f"{r['max_abs_dW']:.4f} & {r['max_abs_dO']:.4f} & {r['n_agree_competitor']}/6 & {r['n_agree_random_p95']}/6")
        rows.append((mk, ds, " & ".join(cells) + r" \\"))
    which = ("One block has completed the module so far (" + ", ".join(f"{names.get(mk, mk)} on {NAMES[ds]}" for mk, ds, _ in rows) + ")."
             if len(rows) == 1 else f"{len(rows)} blocks have completed the module.")
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Numerics.} The CORE grid (six clinical directions, 119 random directions, and the sham at $\alpha=+0.25$) "
         r"on the first 200 test rows rescored with fp32 weights and forward pass, and in bf16 at batch size one, against the CORE "
         r"values (bf16, batched) on the same rows: the largest change over the 36 clinical cells $\max|\Delta W_{q,d}|$, the largest "
         r"change in ownership $\max|\Delta O_q|$, and the concepts whose two point verdicts (own write above the strongest competitor; "
         r"own write above the random 95th percentile) agree with CORE's. " + which + "}",
         r"\label{tab:cf-precision}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrrrr}", r"\toprule",
         r"Checkpoint & Dataset & rows & \multicolumn{4}{c}{fp32 weights and forward} & \multicolumn{4}{c}{bf16, batch size one} \\",
         r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
         r" & & & $\max|\Delta W|$ & $\max|\Delta O|$ & agree comp. & agree p95 & $\max|\Delta W|$ & $\max|\Delta O|$ & agree comp. & agree p95 \\", r"\midrule"]
    L += [r for _, _, r in rows]
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table_cf_precision.tex").write_text("\n".join(L) + "\n")
    print("table_cf_precision.tex", len(rows), "block(s)")


def validation():
    """Answer-direction validation, column selectivity, and known-label ownership from runs/robustness/validation.json."""
    d = json.loads((ROB / "validation.json").read_text())
    names = {"q25-7": "Qwen2.5-VL-7B", "lingshu-32": "Lingshu 32B", "gemma3-12": "Gemma 3 12B"}
    order = list(names)
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{What the answer direction reads, and how selective the columns of $W$ are.} Top: for every block with the "
         r"answer-direction module, the median over the six questions of the AUROC of the answer direction $a_q$ and of the probe normal "
         r"$\hat w_q$ against the test label (known-label rows), of the AUROC of both scores against the model's own clean answer, of the "
         r"Pearson correlation between the two scores over the 600 test rows, and of their cosine, raw (model space) and whitened by the "
         r"covariance of the training features ($\cos_\Sigma=a^{\top}\Sigma w/\sqrt{a^{\top}\Sigma a\,w^{\top}\Sigma w}$). Bottom, per "
         r"dataset over every block with a write matrix: column selectivity $S_d=W_{d,d}/\sum_q|W_{q,d}|$ (median; directions with "
         r"$S_d\geq0.5$), and ownership recomputed on the rows whose label for $q$ is known: cells whose $O_q$ keeps its sign and owned "
         r"cells (paper verdict) that stay owned under the percentile verdict on those rows. NIH and COCO have every test label known, so "
         r"their known-row numbers reproduce the all-row numbers.}",
         r"\label{tab:cf-validation}",
         r"\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Checkpoint & Dataset & AUROC $a_q$ & AUROC $\hat w_q$ & $a_q$ vs clean & $\hat w_q$ vs clean & Pearson & cos raw & $\cos_\Sigma$ \\", r"\midrule"]
    blocks = d["answer_direction"]["blocks"]
    for key in sorted(blocks, key=lambda k: (order.index(k.split("/")[0]) if k.split("/")[0] in order else 9, ["nih", "chexpert", "coco"].index(k.split("/")[1]))):
        b = blocks[key]; s = b["summary"]; mk, ds = key.split("/")
        L.append(f"{names.get(mk, mk)} & {NAMES[ds]} & {f2(s['median_auroc_answer_dir'])} & {f2(s['median_auroc_probe'])} & "
                 f"{f2(s.get('median_auroc_answer_dir_vs_clean_answer'))} & {f2(s.get('median_auroc_probe_vs_clean_answer'))} & "
                 f"{f2(s['median_pearson_scores'])} & {f2(s['median_cos_raw_model'])} & {f2(s['median_cos_sigma_projected'])} \\\\")
    L += [r"\midrule", r"\multicolumn{9}{l}{\textit{Column selectivity and known-label ownership, per dataset}} \\",
          r"Dataset & blocks & cells & median $S_d$ & $S_d\geq0.5$ & known rows (median) & sign of $O_q$ kept & owned kept & \\", r"\midrule"]
    col, kl = d["column_selectivity"]["per_dataset"], d["known_label"]["per_dataset"]
    for ds in ("nih", "chexpert", "coco"):
        c, k = col[ds], kl[ds]
        owned_cells = [q for b in d["known_label"]["blocks"] if b["dataset"] == ds for q in b["per_question"].values() if q["summary_owned"]]
        kept = sum(bool(q["owned_known"]) for q in owned_cells)      # paper-owned cells still owned by the known-row percentile verdict
        L.append(f"{NAMES[ds]} & {c['n_blocks']} & {c['n_cells']} & {f2(c['median_S_d'])} & {c['n_S_ge_0.5']}/{c['n_cells']} & {k['median_n_rows_known']:.0f} & "
                 f"{k['n_sign_agrees']}/{k['n_cells']} & {kept}/{len(owned_cells)} & \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table_cf_validation.tex").write_text("\n".join(L) + "\n")
    print("table_cf_validation.tex", len(blocks), "ansdir block(s)")


if __name__ == "__main__":
    geometry()
    scale()
    pairs()
    altdir()
    ansdir()
    extcomp()
    tokenw()
    precision()
    validation()
