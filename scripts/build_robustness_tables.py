"""Robustness tables for the paper from runs/robustness/*.json (geometry, scale, pairs, validation, refit, round2) and, for
the addendum modules (ALTDIR, ANSDIR, EXTCOMP, TOKENW), from the summaries of the blocks under the paper's one inclusion rule
(cf_inclusion.block_included: CORE and CALIBRATION in run.json completed_modules). The round-2 tables (VALID, ALTDIRD,
ANSDIRT, ATTR, and the full-grade PRECISION columns) read runs/robustness/round2.json (scripts/mayo/robustness_round2.py in the
code repository, which applies the same rule and admits a module only when it covers every row).

Merged tables: table_cf_altdir.tex carries ALTDIR, ALTDIRD, TOKENW and EXTCOMP (per dataset); table_cf_ansdir.tex carries
ANSDIR, ANSDIRT and the per-block rescue counts; table_cf_scale.tex carries the scale, saturation, refit and locus panel
above the PRECISION panel; table_cf_valid.tex carries the radiologist-label grade above the VALIDFIT arms. REPLAY has no
table: it changes no grade, so Section 5.5 and Appendix A.8 state it in prose and check_numbers.py reads the module's own
records. No table is scaled with \\resizebox: a table that does not fit the text width at \\scriptsize loses columns or is
split."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cf_inclusion as ci   # noqa: E402
import cf_ledger as cl   # noqa: E402
import cf_round3 as r3   # noqa: E402

ROB = Path("/rodata/azradonc_dev/m253405/cf-transfer/runs/robustness")
OUT = Path(__file__).resolve().parents[1] / "tables"
NAMES = {"nih": "NIH ChestX-ray14", "chexpert": "CheXpert Plus", "coco": "COCO"}
CKPT = {"q25-3": "Qwen2.5-VL-3B", "q25-7": "Qwen2.5-VL-7B", "q25-32": "Qwen2.5-VL-32B", "q25-72": "Qwen2.5-VL-72B",
        "q3-4": "Qwen3-VL-4B", "q3-8": "Qwen3-VL-8B", "q3-32": "Qwen3-VL-32B", "iv35-8": "InternVL3.5-8B", "iv35-14": "InternVL3.5-14B",
        "iv35-38": "InternVL3.5-38B", "gemma3-4": "Gemma 3 4B", "gemma3-12": "Gemma 3 12B", "gemma3-27": "Gemma 3 27B",
        "medgemma-4": "MedGemma 4B", "medgemma-27": "MedGemma 27B", "lingshu-7": "Lingshu 7B", "lingshu-32": "Lingshu 32B",
        "llava15-7": "LLaVA-1.5-7B", "llava15-13": "LLaVA-1.5-13B", "llavamed-7": "LLaVA-Med 1.5 7B", "llama32-11": "Llama 3.2 Vision 11B"}


def f2(x, nd=2):
    return "--" if x is None else f"{x:.{nd}f}"


def fp(p):
    """p-value for math mode: two decimals, or a power of ten below 0.001 (e.g. 8\\times10^{-11})."""
    if p is None or p != p:
        return "--"
    if p >= 0.001:
        return f"{p:.2f}"
    m, e = f"{p:.0e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"




def geometry():
    """Direction and label geometry against off-diagonal dominance (runs/robustness/geometry.json), one row per quantity and
    one column per dataset plus the two chest sets pooled. Definitions live in the Appendix A.7 prose; the COCO restricted
    competitor families are quoted there too (build_numbers.py cfGeoFam* macros)."""
    d = json.loads((ROB / "geometry.json").read_text())
    dg, lg, lo, asc = d["direction_geometry"], d["label_geometry"], d["leave_one_out"], d["association"]
    groups = ("nih", "chexpert", "chest", "coco")
    members = {"nih": ("nih",), "chexpert": ("chexpert",), "chest": ("nih", "chexpert"), "coco": ("coco",)}

    def mean_per_ds(src, key):
        return lambda g: "--" if g == "chest" else f2(src[g][key], 3)

    def coincide(src, key):
        return lambda g: f"{sum(src[ds][key]['n_true'] for ds in members[g])}/{sum(src[ds][key]['n'] for ds in members[g])}"

    def rho(key):
        return lambda g: f2(asc[g][key]["rho"], 3)

    def count(key):
        return lambda g: f"{lo[g][key]}/{lo[g]['n_cells']}"

    sections = [
        ("Geometry of the six directions and the six labels", [
            ("mean off-diagonal cosine between the directions", mean_per_ds(dg, "offdiag_cos_model_mean")),
            ("mean $|\\cos|$ between random unit directions", mean_per_ds(dg, "random_pair_abs_cos_mean")),
            ("mean off-diagonal $\\phi$ between the labels", mean_per_ds(lg, "offdiag_phi_mean"))]),
        ("Cells whose strongest competitor is (chance: one in five)", [
            ("the most similar direction", coincide(dg, "coincide_cos_model")),
            ("the label with the largest $\\phi$", coincide(lg, "coincide_phi")),
            ("the most co-occurring label", coincide(lg, "coincide_cooccurrence"))]),
        ("Spearman $\\rho$ over off-diagonal cells of $W_{q,d}-W_{q,q}$ with", [
            ("the cosine between the two directions", rho("rho_adv_vs_cos_model")),
            ("the $\\phi$ between the two labels", rho("rho_adv_vs_phi"))]),
        ("Sign of $O_q$", [
            ("cells with $O_q>0$", count("O_q_positive")),
            ("cells with $O_q>0$ without the strongest competitor", count("O_loo_positive")),
            ("cells with at least two competitors above the own write", count("n_above_ge2"))])]
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{5pt}",
         r"\caption{\textbf{Direction and label geometry against off-diagonal dominance.} Per dataset and pooled over the two chest "
         r"sets (``--'': defined per dataset only); Appendix~\ref{app:cf-robustness} defines each row. The last three rows count the "
         r"sign of $O_q$, not the owned grade.}",
         r"\label{tab:cf-geometry}",
         r"\begin{tabular}{lrrrr}", r"\toprule",
         r" & NIH ChestX-ray14 & CheXpert Plus & both chest sets & COCO \\"]
    for title, rows in sections:
        L += [r"\midrule", f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        for label, cell in rows:
            L.append(f"\\quad {label} & " + " & ".join(cell(g) for g in groups) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_geometry.tex").write_text("\n".join(L) + "\n")
    print("table_cf_geometry.tex")


def scale():
    d = json.loads((ROB / "scale.json").read_text())
    m, st, rf, cn, bt = d["item1_margin_scale"], d["item2_ceiling"]["strata"], d["item3_refit"]["per_dataset"], d["item4_connector"]["per_dataset"], d["item5_batch"]
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Ownership under alternative scales, strata, refits, loci, and numerics.} Per dataset: cells with the complete owned "
         r"grade on the probability scale (the paper's grade) and on the logit-margin scale (the steering reference recomputed on "
         r"that scale and a positive percentile interval for $O^m_q$), with the cells owned on both; cells with $O_q>0$ (the sign "
         r"alone) among owned and among not-owned cells when every cell is restricted to its "
         r"unsaturated rows ($0.01<P(\mathrm{yes})<0.99$ on the clean pass); median $|O_q(\text{seed }k)-O_q(\text{seed }0)|$ over the "
         r"two refit seeds and the owned cells that keep $O_q>0$ under both (the refits exist for NIH and COCO); median $|W_{q,q}|$ at the "
         r"primary and connector loci and connector cells whose write beats the connector random family. The lower panel rescores the "
         r"write grid on the first 200 test rows of each block with fp32 weights and forward pass, and in bf16 at batch size one, against "
         r"the bf16 batched grid on the same rows: the largest change over the $6\times126$ written cells, $\max|\Delta W|$, and in a "
         r"clinical contrast, $\max|\Delta C_{q,d}|$. Regraded under either setting, verdicts change in \cfPrecisionVerdictChanges{} and "
         r"steering references in \cfPrecisionRefChanges{} of the \cfPrecisionGradeCells{} grades, and the two point verdicts change in "
         r"\cfPrecisionPointVerdictChanges{}.}",
         r"\label{tab:cf-scale}",
         r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
         r" & \multicolumn{3}{c}{owned cells} & \multicolumn{2}{c}{unsaturated, $O_q>0$} & \multicolumn{2}{c}{refits} & "
         r"\multicolumn{2}{c}{median $|W_{q,q}|$} & connector \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
         r"Dataset & $p$ & $m$ & both & owned & other & med.\ $|\Delta O|$ & stable & primary & connector & $>$ p95 \\", r"\midrule"]
    for ds in ("nih", "chexpert", "coco"):
        mm = m[ds]; ag = mm["agreement_owned_p_vs_owned_m"]; u = st[ds]["unsaturated"]
        r = rf.get(ds); c = cn.get(ds, {}).get("p_scale", {})
        refit = f"{r['median_abs_dO_pooled']:.3f} & {r['owned_p_O_pos_both_refits']}/{r['owned_p']}" if r else "-- & --"
        conn = f"{c['median_abs_W_qq_primary']:.3f} & {c['median_abs_W_qq_connector']:.3f} & {c['connector_W_qq_gt_random_p95']}/{cn[ds]['n_cells']}" if c else "-- & -- & --"
        L.append(f"{NAMES[ds]} & {mm['owned_p']}/{mm['n_cells']} & {mm['owned_m_(CI>0&ref)']}/{mm['n_cells']} & {ag['a_yes_b_yes']} & "
                 f"{u['owned_p_O_pos']}/{u['owned_p_evaluable']} & {u['not_owned_O_pos']}/{u['not_owned_evaluable']} & {refit} & {conn} \\\\")
    dist = bt["distribution"]["vis.last"]
    n_dis = len(bt["sign_disagreement"]) if isinstance(bt["sign_disagreement"], list) else len(bt["sign_disagreement"].get("blocks", []))
    L += [r"\midrule", r"\multicolumn{11}{p{0.9\linewidth}}{\textit{Batch effects at the gates (61 blocks): batched-vs-single candidate-logit difference "
          + f"{dist['max_abs_candidate_logit_diff']['min']:.2f}--{dist['max_abs_candidate_logit_diff']['max']:.2f} (median {dist['max_abs_candidate_logit_diff']['median']:.2f}); "
          + f"margin sign agreement in {61 - n_dis} of 61 blocks; owned cells in disagreeing blocks: {bt['owned_cells_in_sign_disagreement_blocks']}.}}}} \\\\",
          r"\bottomrule", r"\end{tabular}"]
    L += [r"", r"\medskip", r"\setlength{\tabcolsep}{5pt}"] + precision_panel() + [r"\end{table}"]
    (OUT / "table_cf_scale.tex").write_text("\n".join(L) + "\n")
    print("table_cf_scale.tex (with the numerics panel)")


def pairs():
    d = json.loads((ROB / "pairs.json").read_text())
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Same write, different reader: paired differences.} For every pair of checkpoints that share a vision "
         r"tower (identical features and write vectors, or equal to one bf16 rounding step), the difference in ownership "
         r"$\Delta O_q=O_q(\text{larger})-O_q(\text{smaller})$ on the same 600 test rows, with a paired patient-bootstrap 95\% "
         r"interval (2{,}000 shared draws, strongest competitor recomputed in every draw): concepts whose interval excludes zero, "
         r"the median and maximum $|\Delta O_q|$, and their ratio to the mean refit standard deviation of the two blocks.}",
         r"\label{tab:cf-pairs}",
         r"\begin{tabular}{llcrrrrr}", r"\toprule",
         r" & & & $\Delta O_q$ CI & \multicolumn{2}{c}{$|\Delta O_q|$} & \multicolumn{2}{c}{ratio to refit SD} \\",
         r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Pair & Dataset & relation & $\not\ni 0$ & median & max & median & max \\", r"\midrule"]
    names = {"gemma3-4": "Gemma 3 4B", "gemma3-12": "Gemma 3 12B", "gemma3-27": "Gemma 3 27B", "medgemma-4": "MedGemma 4B", "medgemma-27": "MedGemma 27B"}
    for t in d["task1_paired_deltas"]:
        rel = "bit-identical" if t["relationship"] == "bit-identical" else "bf16-equal"
        ratio = f"{t['median_abs_delta_O_over_refit_sd']:.1f} & {t['max_abs_delta_O_over_refit_sd']:.1f}" if t.get("median_abs_delta_O_over_refit_sd") else "-- & --"
        L.append(f"{names.get(t['small'], t['small'])} $\\to$ {names.get(t['large'], t['large'])} & {NAMES[t['dataset']]} & {rel} & "
                 f"{t['n_concepts_delta_O_ci_excludes_zero']}/6 & {t['median_abs_delta_O']:.2f} & {t['max_abs_delta_O']:.2f} & {ratio} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
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



def sci(x, nd=1):
    """Math-mode scientific notation: 1.7\\times10^{-4}."""
    m, e = f"{x:.{nd}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def _r2():
    return json.loads((ROB / "round2.json").read_text())


def _ckpt_key(block: str):
    mk, ds = block.split("/")
    order = list(CKPT)
    return (order.index(mk) if mk in order else len(order), ["nih", "chexpert", "coco"].index(ds))


def valid():
    """VALID: the full grade on the radiologist-labelled CheXpert valid rows against the labeler-labelled test grade, per block
    (runs/robustness/round2.json valid section)."""
    V = _r2()["valid"]; a = V["aggregate"]
    rows_txt = "/".join(str(n) for n in a["valid_rows"])
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{The grade on radiologist labels, and what refitting on them changes.} For every CheXpert block with the valid module, the complete grade is "
         r"recomputed on " + rows_txt + r" frontal films of the CheXpert validation set (one per patient, radiologist consensus labels, no "
         r"patient shared with any campaign role) with the campaign rules: the steering reference against the random 95th percentile and "
         r"the sham on those rows, and the $6\times5$ max-$T$ verdict over 2{,}000 unit-bootstrap draws. The test grade is the same "
         r"computation on the 600 labeler-labelled test rows. Columns: concepts of six with the complete owned grade on the test rows and "
         r"on the valid rows, concepts of six whose clinical-comparison verdict agrees across the two cohorts, concepts of six whose "
         r"owned grade agrees across the two cohorts, and the median $|O_q(\text{valid})-O_q(\text{test})|$ (90th percentile over all cells " + f2(a["p90_abs_dO_q"], 3)
         + r"). On the " + str(a["supported_cells"]) + r" cells with at least 10 positives and 10 negatives on both cohorts, readability agrees in "
         + f"{a['readable_agree_supported']} of {a['readable_compared_supported']}" + r" and answer capability in "
         + f"{a['answer_capable_agree_supported']} of {a['answer_capable_compared_supported']}" + r". The lower panel refits the six "
         r"directions on the radiologist labels themselves and separates the two things that change at once when the refit is compared "
         r"with the shipped direction: which labels were used, and how many rows were available to fit them.}",
         r"\label{tab:cf-valid}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Checkpoint & owned (test) & owned (valid) & verdict agrees & owned grade agrees & median $|\Delta O_q|$ \\", r"\midrule"]
    for b in sorted(V["blocks"], key=lambda b: _ckpt_key(b["block"])):
        L.append(f"{CKPT.get(b['model'], b['model'])} & {b['owned_test']} & {b['owned_valid']} & {b['verdict_agree']}/{b['n_cells']} & "
                 f"{b['owned_agree']}/{b['n_cells']} & {f2(b['median_abs_dO_q'], 3)} \\\\")
    L += [r"\midrule",
          f"\\textit{{all}} ({a['blocks']} blocks) & {a['owned_test']} & {a['owned_valid']} & {a['verdict_agree']}/{a['cells']} & {a['owned_agree']}/{a['cells']} & {f2(a['median_abs_dO_q'], 3)} \\\\",
          r"\bottomrule", r"\end{tabular}"]
    L += [r"", r"\medskip"] + validfit_arms_panel() + [r"\end{table}"]
    (OUT / "table_cf_valid.tex").write_text("\n".join(L) + "\n")
    print("table_cf_valid.tex", a["blocks"], "block(s), with the refit-arms panel")


ARM_ROW = (("expert_valid200", "radiologist", "200 valid films"),
           ("report_valid200", "report-derived", "the same 200 valid films"),
           ("report_train_sub", "report-derived", "200 training rows"),
           ("report_train_sub_known", "report-derived", "200 labelled training rows"),
           ("report_train_full", "report-derived", "every training row"))


def validfit_arms_panel():
    """VALIDFIT arms: the label source of a direction separated from the number of rows it was estimated on
    (runs/robustness/validfit.json, scripts/mayo/robustness_validfit.py). This is the lower panel of Table cf-valid:
    both report the radiologist-labelled cohort, so the paper gives them one number and one caption."""
    V = r3.validfit_arms()
    A = V["aggregate"]["per_arm"]
    m = V["meta"]
    b0 = V["blocks"][0]
    L = [r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"\multicolumn{7}{p{0.95\linewidth}}{\textit{The label source of a refitted direction against the number of rows it was "
         r"fitted on. The expert-label refit is estimated on " + str(b0["n_valid_rows"]) + r" radiologist-labelled films and the "
         r"shipped direction on about " + f"{b0['n_train_rows']:,}".replace(",", "{,}") + r" report-labelled rows, so comparing "
         r"only those two confounds the two. Every arm keeps the block's own projection, its stored train-only scaler, the "
         r"protocol's probe settings and the same " + str(m["folds"]) + r"-fold cross-fitting over patients, and is scored on the "
         r"same films by a direction not fitted on them. The two training subsample arms redraw the fitting rows "
         + str(m["draws"]) + r" times and report the median; the second draws enough rows to leave the concept with as many "
         r"labelled rows as the expert refit has. Cosines are to the shipped direction. Medians over the "
         + str(A["expert_valid200"]["cells"]) + r" cells of " + str(m["n_blocks"]) + r" CheXpert blocks.}} \\",
         r"\midrule",
         r" & & & \multicolumn{2}{c}{cross-fitted AUROC against} & \multicolumn{2}{c}{cosine to shipped} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Labels & Fitted on & rows & radiologist & report & raw & whitened \\", r"\midrule"]
    for arm, lab, where in ARM_ROW:
        d = A[arm]
        L.append(f"{lab} & {where} & {d['median_fit_rows']:.0f} & {f2(d['auroc_expert_labels'], 2)} & "
                 f"{f2(d['auroc_report_labels'], 2)} & {f2(d['cos_model'], 2)} & {f2(d['cos_whitened'], 2)} \\\\")
    G = V["aggregate"]
    L += [r"\midrule", r"\multicolumn{7}{l}{\textit{What each contrast isolates, in AUROC against the radiologist labels}} \\",
          r"\midrule",
          r"\multicolumn{6}{l}{label source, both arms fitting the same number of rows} & "
          + f2(G["label_source_at_matched_size"]["delta_auroc_expert_labels"], 2) + r" \\",
          r"\multicolumn{6}{l}{sample size, both arms on report-derived labels} & "
          + f2(G["sample_size_at_one_label_source"]["delta_auroc_expert_labels"], 2) + r" \\",
          r"\multicolumn{6}{l}{the expert refit against the shipped direction, which mixes the two} & "
          + f2(G["expert_refit_against_the_shipped_direction"]["delta_auroc_expert_labels"], 2) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_cf_validfit_arms.tex").unlink(missing_ok=True)
    print(f"refit-arms panel of table_cf_valid.tex: {m['n_blocks']} block(s), {m['draws']} draws")
    return L



DSS = ("nih", "chexpert", "coco")
SHORT = {"nih": "NIH", "chexpert": "CheXpert", "coco": "COCO"}


def _owned(v) -> bool:
    return bool(v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage")


def _blocks_txt(b: dict) -> str:
    n = [b[ds] for ds in DSS if b[ds]]
    if len(n) == len(DSS) and len(set(n)) == 1:
        return f"{n[0]} blocks per dataset"
    return ", ".join(f"{b[ds]} {SHORT[ds]}" for ds in DSS if b[ds]) + " blocks"


def constructions():
    """One table over the ways of building and writing the write direction, per dataset (owned cells over cells, median O_q):
    the five direction constructions under the coefficient lift (ALTDIR), the displacement lift of the difference-of-means and
    pattern vectors (ALTDIRD, runs/robustness/round2.json), the token-weighted writes of the logistic normal (TOKENW), and the
    logistic normal against the competitor family extended by every extra dataset label (EXTCOMP). Each panel repeats the
    logistic normal on its own blocks. Owned = the paper's rule applied within each family (steering reference and the
    simultaneous fixed-family advantage); the extended-family grade applies the steering reference as W_qq > 0, above the
    random p95 and the absolute sham, as the EXTCOMP summaries record it."""
    import statistics

    def cell():
        return {ds: {"owned": 0, "n": 0, "O": []} for ds in DSS}

    def add(c, o, O):
        c["owned"] += bool(o); c["n"] += 1; c["O"].append(O)

    p1 = {f: cell() for f in ("logistic",) + FAMILIES}; any1 = cell(); b1 = {ds: 0 for ds in DSS}
    p3 = {v: cell() for v in ("uniform", "tokenw", "topq")}; b3 = {ds: 0 for ds in DSS}
    p4 = {v: cell() for v in ("six", "ext")}; b4 = {ds: 0 for ds in DSS}
    for sp, j in included_summaries():
        ds = sp.parent.name; core = j["core"]["per_question"]
        a = j.get("altdir")
        if a and all(f in a for f in FAMILIES):
            b1[ds] += 1
            for q, v in core.items():
                add(p1["logistic"][ds], _owned(v), v["O_q"]); o_any = _owned(v)
                for f in FAMILIES:
                    c = a[f]["per_question"][q]
                    add(p1[f][ds], _owned(c), c["O_q"]); o_any = o_any or _owned(c)
                add(any1[ds], o_any, None)
        t = j.get("tokenw")
        if t:
            b3[ds] += 1
            for q, v in core.items():
                add(p3["uniform"][ds], _owned(v), v["O_q"])
            for var in ("tokenw", "topq"):
                pq = (t.get(var) or {}).get("per_question") or {}
                if set(pq) != set(core):
                    raise RuntimeError(f"{sp.parent}: TOKENW {var} does not cover the six concepts")
                for q, c in pq.items():
                    add(p3[var][ds], _owned(c), c["O_q"])
        e = j.get("extcomp")
        if e:
            b4[ds] += 1
            for q, c in e["per_question"].items():
                add(p4["six"][ds], _owned(core[q]), core[q]["O_q"])
                eo = c["W_qq"] > 0 and c["W_qq"] > c.get("random_p95", 0) and c["W_qq"] > c.get("abs_sham", 0) and c.get("verdict") == "fixed_family_advantage"
                add(p4["ext"][ds], eo, c["O_q"])
    A = _r2()["altdird"]["per_dataset"]
    b2 = {ds: A[ds]["blocks"] for ds in DSS}
    p2 = {fam: {ds: {"owned": A[ds][fam]["owned"], "n": A[ds]["cells"], "median": A[ds][fam]["median_O_q"]} for ds in DSS}
          for fam in ("logistic", "dom_disp", "pattern_disp")}

    def cells(c, with_median=True):
        out = []
        for ds in DSS:
            x = c[ds]
            if not x["n"]:
                out += ["--", "--"]; continue
            med = x["median"] if "median" in x else (statistics.median(x["O"]) if with_median else None)
            out += [f"{x['owned']}/{x['n']}", "--" if med is None else f"{med:+.2f}"]
        return " & ".join(out)

    panels = [
        (f"Direction construction ({_blocks_txt(b1)})", [
            ("logistic normal", "coefficient", cells(p1["logistic"])),
            ("difference of means", "coefficient", cells(p1["dom"])),
            ("Haufe pattern", "coefficient", cells(p1["pattern"])),
            ("orthogonalised normal", "coefficient", cells(p1["orth"])),
            ("residualised normal", "coefficient", cells(p1["resid"])),
            ("owned under at least one", "", cells(any1, with_median=False))]),
        (f"Displacement lift ({_blocks_txt(b2)})", [
            ("logistic normal", "coefficient", cells(p2["logistic"])),
            ("difference of means", "displacement", cells(p2["dom_disp"])),
            ("Haufe pattern", "displacement", cells(p2["pattern_disp"]))]),
        (f"Token weights of the logistic normal ({_blocks_txt(b3)})", [
            ("uniform", "coefficient", cells(p3["uniform"])),
            ("softmax of the probe score", "coefficient", cells(p3["tokenw"])),
            ("top quarter of tokens", "coefficient", cells(p3["topq"]))]),
        (f"Competitor family of the logistic normal ({_blocks_txt(b4)})", [
            ("six clinical directions", "coefficient", cells(p4["six"])),
            ("with every extra dataset label", "coefficient", cells(p4["ext"]))])]
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Ownership under alternative directions, lifts, token weights, and competitor families.} Cells owned under "
         r"the paper's rule within each family, and the median $O_q$, per dataset; each panel repeats the reference write on its own "
         r"blocks. Appendix~\ref{app:cf-robustness} defines every construction.}",
         r"\label{tab:cf-altdir}",
         r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r" & & \multicolumn{2}{c}{NIH ChestX-ray14} & \multicolumn{2}{c}{CheXpert Plus} & \multicolumn{2}{c}{COCO} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Direction & Lift & owned & med.\ $O_q$ & owned & med.\ $O_q$ & owned & med.\ $O_q$ \\"]
    for title, rows in panels:
        L += [r"\midrule", f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        L += [f"\\quad {name} & {lift} & {c} \\\\" for name, lift, c in rows]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_altdir.tex").write_text("\n".join(L) + "\n")
    print("table_cf_altdir.tex", {"altdir": b1, "altdird": b2, "tokenw": b3, "extcomp": b4})


def answer_direction():
    """The answer direction in one table. Top: per block with ANSDIR on the primary template, concepts owned by the label
    direction and by the answer direction, split into kept (both), rescued (answer direction only) and lost (label direction
    only); concepts whose a_q write beats the strongest logistic competitor; medians over concepts of W^a_qq, the raw
    model-space cosine, the whitened cosine (validation.json) and the cross-validated R^2; one aggregate row per dataset and
    one for the chest sets (medians over cells). Bottom: ANSDIRT per block and template (runs/robustness/round2.json)."""
    import statistics
    V = json.loads((ROB / "validation.json").read_text())
    VB, VA = V["answer_direction"]["blocks"], V["answer_direction"]["aggregate"]
    T = _r2()["ansdirt"]; tpl = T["templates"]; TB = {b["block"]: b for b in T["blocks"]}
    summ = {f"{sp.parent.parent.name}/{sp.parent.name}": j for sp, j in included_summaries() if j.get("ansdir")}
    keys = sorted(summ, key=_ckpt_key)
    if set(keys) != set(VB) or set(keys) != set(TB):
        raise RuntimeError(f"answer-direction block sets differ: summaries {sorted(summ)}, validation {sorted(VB)}, ansdirt {sorted(TB)}")
    groups = {"nih": ("nih",), "chexpert": ("chexpert",), "chest": ("nih", "chexpert"), "coco": ("coco",)}
    agg = {g: {"blocks": 0, "n": 0, "w": 0, "a": 0, "kept": 0, "rescued": 0, "lost": 0, "beats": 0, "W": [], "cos": [], "cs": [], "r2": []} for g in groups}
    top, bottom = [], []
    for key in keys:
        mk, ds = key.split("/"); j = summ[key]
        pq, core, vq = j["ansdir"]["per_question"], j["core"]["per_question"], VB[key]["per_question"]
        r = {"n": 0, "w": 0, "a": 0, "kept": 0, "rescued": 0, "lost": 0, "beats": 0, "W": [], "cos": [], "cs": [], "r2": []}
        for q, v in pq.items():
            w_own, a_own = _owned(core[q]), _owned(v)
            if w_own != bool(vq[q]["core_owned"]):
                raise RuntimeError(f"{key} {q}: label-direction ownership differs between summary.json and validation.json")
            r["n"] += 1; r["w"] += w_own; r["a"] += a_own
            r["kept"] += w_own and a_own; r["rescued"] += a_own and not w_own; r["lost"] += w_own and not a_own
            r["beats"] += v.get("own_minus_max_logistic_competitor_ci95_percentile", [0, 0])[0] > 0
            r["W"].append(v["W_qq"]); r["cos"].append(v.get("cos_to_logistic_model", float("nan")))
            r["cs"].append(vq[q]["cos_sigma_projected"]); r["r2"].append(v.get("cv_r2", v.get("r2", float("nan"))))
        if abs(statistics.median(r["cs"]) - VB[key]["summary"]["median_cos_sigma_projected"]) > 1e-9:
            raise RuntimeError(f"{key}: median whitened cosine differs from validation.json")
        tb = TB[key]
        if tb["n_iy_owned"] != r["a"]:
            raise RuntimeError(f"{key}: ANSDIRT IY-owned {tb['n_iy_owned']} differs from ANSDIR owned {r['a']}")
        for g, members in groups.items():
            if ds in members:
                G = agg[g]; G["blocks"] += 1
                for k in ("n", "w", "a", "kept", "rescued", "lost", "beats"):
                    G[k] += r[k]
                for k in ("W", "cos", "cs", "r2"):
                    G[k] += r[k]
        med = {k: statistics.median(r[k]) for k in ("W", "cos", "cs", "r2")}
        top.append(f"{CKPT.get(mk, mk)} & {NAMES[ds]} & {r['w']}/{r['n']} & {r['a']}/{r['n']} & {r['kept']} & {r['rescued']} & {r['lost']} & "
                   f"{r['beats']}/{r['n']} & {med['W']:.2f} & {med['cos']:.2f} & {med['cs']:.2f} & {med['r2']:.2f} \\\\")
        bottom.append(f"{CKPT.get(mk, mk)} & {NAMES[ds]} & "
                      + " & ".join(f"{tb['templates'][t]['n_owned']} ({tb['templates'][t]['kept_of_iy']})" if t in tb["templates"] else "--" for t in tpl)
                      + f" & {tb['kept_pairs']}/{tb['pairs']} \\\\")
    for ds in DSS:
        if abs(statistics.median(agg[ds]["cs"]) - VA[f"{ds}_median_cos_sigma_projected"]) > 1e-9:
            raise RuntimeError(f"{ds}: median whitened cosine over cells differs from validation.json")
    top.append(r"\midrule")
    for g, name in (("nih", NAMES["nih"]), ("chexpert", NAMES["chexpert"]), ("chest", r"\textit{chest}"), ("coco", NAMES["coco"])):
        G = agg[g]
        med = {k: statistics.median(G[k]) for k in ("W", "cos", "cs", "r2")}
        top.append(f"{name} & {G['blocks']} blocks & {G['w']}/{G['n']} & {G['a']}/{G['n']} & {G['kept']} & {G['rescued']} & {G['lost']} & "
                   f"{G['beats']}/{G['n']} & {med['W']:.2f} & {med['cos']:.2f} & {med['cs']:.2f} & {med['r2']:.2f} \\\\")
    bottom.append(r"\midrule")
    for g, name in (("chest", r"\textit{chest}"), ("coco", NAMES["coco"])):
        d = T["aggregate"][g]
        if d["blocks"] != agg[g]["blocks"] or d["iy_owned"] != agg[g]["a"]:
            raise RuntimeError(f"ansdirt aggregate {g} differs from the ANSDIR blocks")
        bottom.append(f"{name} & {d['blocks']} blocks & "
                      + " & ".join(f"{d['per_template'][t]['owned']} ({d['per_template'][t]['kept_of_iy']})" for t in tpl)
                      + f" & {d['kept_pairs']}/{d['pairs']} \\\\")
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\caption{\textbf{The answer direction.} Top, on the primary template: concepts owned by the label direction $\hat w_q$ and by "
         r"the answer direction $a_q$, split into kept (both), rescued ($a_q$ only), and lost ($\hat w_q$ only); concepts whose $a_q$ "
         r"write beats the strongest logistic competitor; and medians of $W^a_{q,q}$, of the raw and whitened cosines between $a_q$ and "
         r"$\hat w_q$, and of the cross-validated $R^2$. Bottom: $a_q$ written under each held-out template, with the concepts owned and, "
         r"in parentheses, the IY-owned concepts that stay owned, and the kept (concept, template) pairs. A held-out template scores no "
         r"random family of its own, so a cell there meets the steering reference against the permutation sham of the $a_q$ under test "
         r"and, where the additional-template module scored a random family under that template, against its 95th percentile as well "
         r"(Table~\ref{tab:cf-ledger}); the verdict is the same $6\times5$ max-$T$ verdict.}",
         r"\label{tab:cf-ansdir}",
         r"\begin{tabular}{llrrrrrrrrrr}", r"\toprule",
         r" & & \multicolumn{5}{c}{owned concepts} & & \multicolumn{4}{c}{median} \\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){9-12}",
         r"Checkpoint & Dataset & $\hat w_q$ & $a_q$ & kept & rescued & lost & $a_q>$ comp. & $W^a_{q,q}$ & $\cos$ & $\cos_\Sigma$ & $R^2$ \\",
         r"\midrule"] + top + [r"\bottomrule", r"\end{tabular}", "", r"\medskip",
         r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r" & & \multicolumn{5}{c}{held-out template: owned (IY-owned kept)} & \\", r"\cmidrule(lr){3-7}",
         r"Checkpoint & Dataset & " + " & ".join(tpl) + r" & kept pairs \\", r"\midrule"] + bottom + [
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_ansdir.tex").write_text("\n".join(L) + "\n")
    print("table_cf_ansdir.tex", len(keys), "block(s)")


def precision_panel():
    """PRECISION, one row per block: the largest change over the 6x126 written cells and in a clinical contrast, with fp32
    weights and forward and in bf16 at batch size one, against the bf16 batched grid on the same first 200 test rows
    (runs/robustness/round2.json). The facts that do not vary across rows (no verdict, reference or point-verdict change)
    are stated through the build_numbers.py macros in the caption of Table cf-scale, whose lower panel this is: a control
    that changes no grade does not earn a numbered table of its own."""
    R2 = {b["block"]: b for b in _r2()["precision"]["blocks"]}
    keys = sorted((f"{sp.parent.parent.name}/{sp.parent.name}" for sp, j in included_summaries() if j.get("precision")), key=_ckpt_key)
    if set(keys) != set(R2):
        raise RuntimeError(f"precision blocks differ: summaries {keys}, round2.json {sorted(R2)}")
    rows = []
    for key in keys:
        mk, ds = key.split("/"); s = R2[key]["settings"]
        rows.append(f"{CKPT.get(mk, mk)} & {NAMES[ds]} & " + " & ".join(
            f"{s[st]['max_abs_dW_grid']:.4f} & {s[st]['max_abs_dcontrast']:.4f}" for st in ("fp32", "batch1")) + r" \\")
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"\multicolumn{6}{l}{\textit{Numerics: the same grid rescored in fp32 and at batch size one}} \\",
         r"\midrule",
         r" & & \multicolumn{2}{c}{fp32 weights and forward} & \multicolumn{2}{c}{bf16, batch size one} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Checkpoint & Dataset & $\max|\Delta W|$ & $\max|\Delta C|$ & $\max|\Delta W|$ & $\max|\Delta C|$ \\", r"\midrule"]
    L += rows
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_cf_precision.tex").unlink(missing_ok=True)
    print(f"numerics panel of table_cf_scale.tex: {len(keys)} block(s)")
    return L


def validation():
    """Answer-direction validation, column selectivity, and known-label ownership from runs/robustness/validation.json."""
    d = json.loads((ROB / "validation.json").read_text())
    names = {"q25-7": "Qwen2.5-VL-7B", "lingshu-32": "Lingshu 32B", "gemma3-12": "Gemma 3 12B"}
    order = list(names)
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{What the answer direction reads, and how selective the columns of $W$ are.} Top: for every block with the "
         r"answer-direction module, the median over the six questions of the AUROC of the answer direction $a_q$ and of the probe normal "
         r"$\hat w_q$ against the test label (known-label rows), of the AUROC of both scores against the model's own clean answer, of the "
         r"Pearson correlation between the two scores over the 600 test rows, and of their cosine in two metrics: raw, between the "
         r"lifted vectors in model space (cos raw), and whitened, between the coefficient vectors in the projected, train-scaled space "
         r"under the covariance $\Sigma$ of the training features ($\cos_\Sigma=a^{\top}\Sigma w/\sqrt{a^{\top}\Sigma a\,w^{\top}\Sigma w}$). "
         r"The alignment row pools the answer-direction cells and reports $\cos_\Sigma$ per cell and its median over cells. Bottom, per "
         r"dataset over every block with a write matrix: the signed column-selectivity index $S_d=W_{d,d}/\sum_q|W_{q,d}|$ of each "
         r"direction in each block, as the median over cells and the number of cells with $S_d\geq0.5$. $S_d$ keeps the sign of the "
         r"direction's effect on its own question, counts negative changes in the denominator, and weights every cell equally. It "
         r"differs from the write-flow share of Figure~\ref{fig:cf-w-structure}, which keeps only positive changes, weights each "
         r"write by the size of its effect, and pools all checkpoints and directions into one ratio per dataset. The remaining columns give the median number of rows whose label for $q$ is known and ownership recomputed on those rows: cells whose $O_q$ keeps its sign and owned "
         r"cells (paper verdict) that stay owned under the full grade on those rows---the steering reference recomputed there and the "
         r"simultaneous max-$T$ advantage over the same six-direction family. NIH and COCO have every test label known, so "
         r"their known-row numbers reproduce the all-row numbers.}",
         r"\label{tab:cf-validation}",
         r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r" & & \multicolumn{2}{c}{AUROC, label} & \multicolumn{2}{c}{AUROC, clean answer} & & \multicolumn{2}{c}{cosine} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){8-9}",
         r"Checkpoint & Dataset & $a_q$ & $\hat w_q$ & $a_q$ & $\hat w_q$ & Pearson & raw & $\Sigma$ \\", r"\midrule"]
    blocks = d["answer_direction"]["blocks"]
    for key in sorted(blocks, key=lambda k: (order.index(k.split("/")[0]) if k.split("/")[0] in order else 9, ["nih", "chexpert", "coco"].index(k.split("/")[1]))):
        b = blocks[key]; s = b["summary"]; mk, ds = key.split("/")
        L.append(f"{names.get(mk, mk)} & {NAMES[ds]} & {f2(s['median_auroc_answer_dir'])} & {f2(s['median_auroc_probe'])} & "
                 f"{f2(s.get('median_auroc_answer_dir_vs_clean_answer'))} & {f2(s.get('median_auroc_probe_vs_clean_answer'))} & "
                 f"{f2(s['median_pearson_scores'])} & {f2(s['median_cos_raw_model'])} & {f2(s['median_cos_sigma_projected'])} \\\\")
    al = d["answer_direction"]["aggregate"].get("alignment_vs_ownership", {}).get("all")
    alc = d["answer_direction"]["aggregate"].get("alignment_vs_ownership", {}).get("chest")
    if al:
        L += [r"\midrule", r"\multicolumn{9}{p{0.92\linewidth}}{\textit{Alignment against ownership over the " + str(al["n_cells"]) + r" answer-direction cells: Spearman correlation between $\cos_\Sigma(a_q,\hat w_q)$ and the label direction's $O_q$ $\rho="
              + f2(al["spearman_cos_sigma_vs_O_q"]["rho"]) + r"$ ($p=" + fp(al["spearman_cos_sigma_vs_O_q"]["p"]) + r"$); with the label direction's owned indicator $\rho="
              + f2(al["spearman_cos_sigma_vs_owned"]["rho"]) + r"$ ($p=" + fp(al["spearman_cos_sigma_vs_owned"]["p"]) + r"$); median $\cos_\Sigma$ "
              + f2(al["median_cos_sigma_owned"]) + r" in owned against " + f2(al["median_cos_sigma_not_owned"]) + r" in not-owned cells"
              + (r"; within the chest sets alone $\rho=" + f2(alc["spearman_cos_sigma_vs_O_q"]["rho"]) + r"$ ($p=" + fp(alc["spearman_cos_sigma_vs_O_q"]["p"]) + r"$, "
                 + str(alc["n_cells"]) + " cells)" if alc else "") + r".}} \\"]
    L += [r"\midrule", r"\multicolumn{9}{l}{\textit{Column selectivity and known-label ownership, per dataset}} \\",
          r"Dataset & blocks & cells & med.\ $S_d$ & $S_d\geq0.5$ & known rows & sign kept & owned kept & \\", r"\midrule"]
    col, kl = d["column_selectivity"]["per_dataset"], d["known_label"]["per_dataset"]
    for ds in ("nih", "chexpert", "coco"):
        c, k = col[ds], kl[ds]
        owned_cells = [q for b in d["known_label"]["blocks"] if b["dataset"] == ds for q in b["per_question"].values() if q["summary_owned"]]
        kept = sum(bool(q["owned_known_full_grade"]) for q in owned_cells)   # paper-owned cells still owned by the FULL grade on the known rows
        L.append(f"{NAMES[ds]} & {c['n_blocks']} & {c['n_cells']} & {f2(c['median_S_d'])} & {c['n_S_ge_0.5']}/{c['n_cells']} & {k['median_n_rows_known']:.0f} & "
                 f"{k['n_sign_agrees']}/{k['n_cells']} & {kept}/{len(owned_cells)} & \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_validation.tex").write_text("\n".join(L) + "\n")
    print("table_cf_validation.tex", len(blocks), "ansdir block(s)")



def refit():
    """Full ownership grade under the refit seeds (runs/robustness/refit.json): survival of the seed-0 grade per dataset."""
    d = json.loads((ROB / "refit.json").read_text()); m = d["meta"]
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{The full grade under refitted probes.} For every block with the refit module, the six probes are refitted on "
         r"two resamples of the training patients (fit seeds 1 and 2) and the complete grade is recomputed with the campaign rules: the "
         r"steering reference against the seed-0 random 95th percentile and the seed-0 sham of the same block, question and rows --- the "
         r"permutation of the seed-0 normal, not of the refitted direction (Table~\ref{tab:cf-ledger}) --- and the $6\times5$ max-$T$ "
         r"verdict over the shared unit-bootstrap draws (" + f"{m['draws']:,}".replace(",", "{,}") + r" draws). Per dataset: cells whose seed-0 "
         r"grade is owned, stronger competitor, unresolved, or fixed-family advantage without the reference, and how many keep that grade "
         r"under both refits, under at least one, or under none; the last column regrades seed 0 with the same draws and counts agreement "
         r"with the paper's grade.}",
         r"\label{tab:cf-refit}",
         r"\begin{tabular}{lrr" + "rrrr" + "rrr" + "rrr" + "rr" + r"r}", r"\toprule",
         r"Dataset & blocks & cells & \multicolumn{4}{c}{owned} & \multicolumn{3}{c}{stronger competitor} & \multicolumn{3}{c}{unresolved} & \multicolumn{2}{c}{adv.\ no ref.} & seed 0 \\",
         r"\cmidrule(lr){4-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}\cmidrule(lr){14-15}",
         r" & & & $n$ & both & $\geq1$ & none & $n$ & both & $\geq1$ & $n$ & both & $\geq1$ & $n$ & both & agrees \\", r"\midrule"]
    for ds in ("nih", "chexpert", "coco", "chest", "all"):
        g = d["transitions"].get(ds)
        if not g or g["n_blocks"] == 0:
            continue
        s = g["survival"]; name = {"chest": r"\textit{chest}", "all": r"\textit{all}"}.get(ds, NAMES.get(ds, ds))
        L.append(f"{name} & {g['n_blocks']} & {g['n_cells']} & {s['owned']['n']} & {s['owned']['both']} & {s['owned']['any']} & {s['owned']['none']} & "
                 f"{s['stronger_competitor']['n']} & {s['stronger_competitor']['both']} & {s['stronger_competitor']['any']} & "
                 f"{s['unresolved']['n']} & {s['unresolved']['both']} & {s['unresolved']['any']} & "
                 f"{s['advantage_no_reference']['n']} & {s['advantage_no_reference']['both']} & {g['seed0_regrade_agrees']}/{g['n_cells']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_refit.tex").write_text("\n".join(L) + "\n")
    print("table_cf_refit.tex", m["n_blocks_by_dataset"])



ATTR_LABEL = {"view_AP": "AP (portable) projection", "sex_F": "female sex", "age_60": r"age $\geq60$"}
ATTR_SHORT = {"view_AP": "view AP", "sex_F": "sex F", "age_60": r"age $\geq60$"}


def _pair_label(pair):
    """'sex_F~Nodule' -> 'female sex with Nodule'."""
    if not pair:
        return "--"
    a, _, c = pair.partition("~")
    return f"{ATTR_LABEL.get(a, a)} with {c}"


def attr():
    """ATTR: three non-clinical attributes of the same radiographs fitted, lifted and written like the six findings inside one
    nine-direction family. Per checkpoint and dataset the owned attribute cells beside the owned clinical cells of the same
    family, then per attribute and dataset the probe AUROC, selectivity, answer AUROC and owned count
    (runs/robustness/round2.json attr section)."""
    B = _r2()["attr"]; P = B["per_dataset"]; attrs = B["attributes"]
    dss = [ds for ds in ("nih", "chexpert", "coco") if P[ds]["blocks"]]
    ch = P["chest"]
    groups = [(ds, NAMES[ds]) for ds in dss] + ([("chest", r"\textit{chest}")] if len(dss) > 1 else [])
    mm = ch["max_median_abs_cos_pair"] or {}
    L = [r"\begin{table}[p]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Non-clinical attributes of the same radiographs.} Three attributes of each film---an anteroposterior "
         r"(portable) projection, female sex, and age at least 60---are read from the dataset's own metadata and fitted on the same "
         r"images, features, projection, and probe settings as the clinical probes. The three attribute directions and the six "
         r"clinical directions are lifted and written together as one nine-direction family at the primary dose and template, and "
         r"graded with the paper's rule ($9\times8$ max-$T$ verdict over 2{,}000 shared unit-bootstrap draws). Attribute and clinical "
         r"questions carry the same steering reference in every block of this table: the 119-direction random family at the same dose "
         r"and the coordinate-permutation sham. Top: per checkpoint and dataset, owned attribute cells and owned clinical cells inside the same family, and "
         r"which attributes are owned. Bottom: per attribute and dataset, medians over blocks of the probe AUROC against the "
         r"attribute label, the probe selectivity against the stratum controls, the AUROC of the probe score against the model's own "
         r"clean answer, and the absolute raw model-space cosine between the attribute direction and the six clinical normals "
         r"(median over blocks and the six normals); the last column counts the blocks where the attribute is owned. Taken per "
         r"attribute--finding pair as the median over blocks, the largest absolute raw model-space cosine over the "
         + f"{len(ch['per_pair_median_abs_cos'])}" + r" pairs is " + f2(mm.get("median_abs_cos")) + r" ("
         + _pair_label(mm.get("pair")) + r").}",
         r"\label{tab:cf-attr}",
         r"\begin{tabular}{llrrccc}", r"\toprule",
         r"Checkpoint & Dataset & attributes owned & findings owned & " + " & ".join(ATTR_SHORT[a] for a in attrs) + r" \\", r"\midrule"]
    for b in sorted(B["blocks"], key=lambda b: _ckpt_key(b["block"])):
        L.append(f"{CKPT.get(b['model'], b['model'])} & {NAMES[b['dataset']]} & {b['attr_owned']}/{b['attr_cells']} & {b['clin_owned']}/{b['clin_cells']} & "
                 + " & ".join("yes" if b["attributes"][a]["owned"] else "--" for a in attrs) + r" \\")
    if B["blocks"]:
        L.append(r"\midrule")
    for g, name in groups:
        d = P[g]
        L.append(f"{name} & {d['blocks']} block{'s' if d['blocks'] != 1 else ''} & {d['attr_owned']}/{d['attr_cells']} & {d['clin_owned']}/{d['clin_cells']} & "
                 + " & ".join(f"{d['per_attribute'][a]['owned']}/{d['per_attribute'][a]['blocks']}" for a in attrs) + r" \\")
    L += [r"\midrule", r"\multicolumn{7}{l}{\textit{Attribute probes and directions: medians over the blocks of each dataset}} \\",
          r"Attribute & Dataset & probe AUROC & selectivity & answer AUROC & med.\ $|\cos|$ & owned \\", r"\midrule"]
    for a in attrs:
        for g, name in groups:
            d = P[g]["per_attribute"][a]
            if not d["blocks"]:
                continue
            L.append(f"{ATTR_LABEL[a]} & {name} & {f2(d['median_auroc_real'], 3)} & {f2(d['median_selectivity'])} & {f2(d['median_answer_auroc'])} & "
                     f"{f2(d['median_abs_cos_to_clinical'])} & {d['owned']}/{d['blocks']} \\\\")
    cmp_ = ch["comparison"]
    L += [r"\midrule", r"\multicolumn{7}{l}{\textit{The same comparison read three ways, pooled over the "
          + str(ch["blocks"]) + r" chest blocks}} \\",
          r"Comparison & unit & attribute cells & owned & finding cells & owned & difference \\", r"\midrule"]
    for mode, label, unit in (("all_cells", "every cell", "cell"),
                              ("answer_capable_cells", "answer-capable cells only", "cell"),
                              ("answerability_matched", "answerability-matched pairs", "pair")):
        c = cmp_[mode]
        L.append(f"{label} & {unit} & {c['attr_cells']} & {c['attr_owned']} & {c['clin_cells']} & {c['clin_owned']} & "
                 f"{f2(c['attr_owned_share'] - c['clin_owned_share'], 3)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_attr.tex").write_text("\n".join(L) + "\n")
    print("table_cf_attr.tex", {g: P[g]["blocks"] for g, _ in groups})


# --------------------------------------------------------------------------- the four crossed experiments of Section 5

def _short(model_key: str) -> str:
    """Single-letter tag for the two checkpoints of the crossover, defined in the table's caption."""
    return CKPT.get(model_key, model_key)[0]


def towerswap():
    """TOWERSWAP: the vision tower of one checkpoint loaded behind the other's connector and language model, against the two
    native arms, on the same rows. Top: owned concepts of six in each of the four (tower, reader) combinations, per host block
    and then per dataset with the two hosts' shared arms counted once. Bottom: the four crossover contrasts."""
    wb = ci.write_blocks(ci.load_runs())
    T = r3.towerswap(wb)
    X = r3.crossover(wb)
    rec, order = T["receipt"], ("nih", "chexpert", "coco")
    readers = sorted({b["reader"] for b in T["blocks"]}, key=lambda m: list(CKPT).index(m) if m in CKPT else 99)
    a, b_ = readers[0], readers[-1]
    cols = [(a, a), (b_, b_), (b_, a), (a, b_)]      # native, native, swapped, swapped
    heads = [f"{_short(t)}/{_short(r)}" for t, r in cols]
    twice = sum(1 for d in T["per_dataset"].values() for r in d["combinations"] if r["n_blocks"] > 1)
    L = [r"\begin{table}[p]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Crossing the tower and the reader.} The vision tower of one checkpoint is loaded behind the other's "
         r"connector and language model. Of the " + str(rec["n_tower_tensors"][0]) + r" tensors of the host tower, "
         + str(rec["n_tensors_replaced"][0]) + r" carry weights, and every one of them is replaced and verified bitwise equal to "
         r"the donor's (" + f"{rec['n_params_replaced'][0] / 1e6:.0f}" + r" million parameters); the remaining tensor is a buffer "
         r"the module rebuilds. None of the " + str(rec["n_tensors_outside_tower"][0]) + r" tensors outside the tower changes. The "
         r"written directions travel with the tower, because they are fitted in the representation that tower produces; everything "
         r"else---rows, dose, template, reference family, and decision rule---is the campaign's. " + _short(a) + r" is "
         + CKPT.get(a, a).replace(" 3 ", "~3 ") + r" and " + _short(b_) + r" is " + CKPT.get(b_, b_) + r", and each column of the "
         r"top two panels is tower/reader. A dash marks an arm that is not graded in that host block. The middle panel counts the "
         r"four combinations of each dataset once; an arm graded in both host blocks agrees in all " + str(twice)
         + r" cases where it appears twice. The bottom panel holds one factor and "
         r"changes the other, and gives the paired effect on three quantities of the same six questions: the own write "
         r"$W_{q,q}$, the ownership contrast $O_q=W_{q,q}-\max_{d\neq q}W_{q,d}$ over the five competing concept directions, "
         r"and the margin $M_q$ of the own write over the strongest competitor in the whole reference family, "
         r"$M_q=W_{q,q}-\max(\max_{d\neq q}W_{q,d},\,\text{random }p_{95},\,|\text{sham}|)$. Each entry is the mean over the "
         r"six questions of the absolute paired change, with its 95\% percentile interval over the same "
         + f"{X['meta']['draws']:,}".replace(",", "{,}") + r" draws of the "
         r"rows and, in brackets, the number of questions of six whose change is simultaneously nonzero; the last two rows of "
         r"each block average each factor over its two levels. The three quantities come from one pass over the packaged "
         r"outcomes of the four arms, so the columns of a row are the same arms read three ways.}",
         r"\label{tab:cf-towerswap}",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{native} & \multicolumn{2}{c}{swapped} & & \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Host checkpoint & Dataset & " + " & ".join(heads) + r" & concepts & crossover \\", r"\midrule"]
    for b in T["blocks"]:
        by = {(c["tower"], c["reader"]): c for c in b["combinations"]}
        L.append(f"{CKPT.get(b['model'], b['model'])} & {NAMES[b['dataset']]} & "
                 + " & ".join(str(by[c]["n_owned"]) if by[c]["complete"] else "--" for c in cols)
                 + f" & {max((c['n_concepts'] or 0) for c in b['combinations'])} & "
                 + ("complete" if b["crossover_complete"] else "--") + r" \\")
    L += [r"\midrule", r"\multicolumn{8}{l}{\textit{The four combinations of each dataset, shared arms counted once}} \\",
          r"Combinations & Dataset & " + " & ".join(heads) + r" & concepts & owned of cells \\", r"\midrule"]
    for ds in order:
        d = T["per_dataset"].get(ds)
        if not d:
            continue
        by = {(r["tower"], r["reader"]): r["n_owned"] for r in d["combinations"]}
        L.append(f"all four & {NAMES[ds]} & " + " & ".join("--" if by.get(c) is None else str(by[c]) for c in cols)
                 + f" & {d['n_concepts']} & {d['owned']}/{d['cells']}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\vspace{4pt}", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{llccc}", r"\toprule",
          r"\multicolumn{5}{l}{\textit{Crossover: the effect of replacing one factor with the other held fixed}} \\",
          r"\midrule",
          r"Block & Changed, held & own write $W_{q,q}$ & ownership $O_q$ & family margin $M_q$ \\",
          r"\midrule"]
    rows_ = [("reader_effect_at_own_tower", "reader, own tower"),
             ("reader_effect_at_partner_tower", "reader, partner tower"),
             ("tower_effect_at_own_reader", "tower, own reader"),
             ("tower_effect_at_partner_reader", "tower, partner reader")]
    short_ds = {"nih": "NIH", "chexpert": "CheXpert", "coco": "COCO"}
    xb = sorted(X["blocks"], key=lambda b: (list(CKPT).index(b["model"]) if b["model"] in CKPT else 99,
                                            order.index(b["dataset"])))
    for i, b in enumerate(xb):
        label = f"{CKPT.get(b['model'], b['model'])}, {short_ds[b['dataset']]}"
        cx = b["crossover"]
        for j, (name, what) in enumerate(rows_):
            L.append((label if j == 0 else "") + f" & {what} & " + " & ".join(
                f"{f2(cx[name][q]['mean_abs_effect'], 3)} "
                f"[{f2(cx[name][q]['mean_abs_effect_ci95'][0], 3)}, {f2(cx[name][q]['mean_abs_effect_ci95'][1], 3)}] "
                f"({cx[name][q]['n_simultaneously_nonzero']})" for q in r3.CROSSOVER_QUANTITIES) + r" \\")
        for factor in ("reader", "tower"):
            L.append(f" & \\quad mean {factor} effect & " + " & ".join(
                f"{f2(cx[f'mean_abs_{factor}_effect_{q}'], 3)} "
                f"[{f2(cx[f'mean_abs_{factor}_effect_{q}_ci95'][0], 3)}, {f2(cx[f'mean_abs_{factor}_effect_{q}_ci95'][1], 3)}]"
                for q in r3.CROSSOVER_QUANTITIES) + r" \\")
        if i < len(xb) - 1:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_towerswap.tex").write_text("\n".join(L) + "\n")
    print("table_cf_towerswap.tex", len(T["blocks"]), "block(s),", len(T["crossover_blocks"]), "crossover(s)")


ENDPOINT_LABEL = {"NY": "negated question", "DA": "forced choice, finding first",
                  "DB": "forced choice, competitor first", "RF": "report continuation"}
FAM_NAME = {"label": "label direction", "answer": "answer direction"}


def semend():
    """SEMEND: both direction families written at four endpoints no direction was fitted against. Top: owned and
    reference-meeting concepts of six per block and endpoint. Middle: the sign test on the negated question, the order gap of
    the counterbalanced forced choice, and the report continuation. Bottom: the incremental validity of the ownership contrast
    over the write magnitude, the probe selectivity and the clean-answer AUROC."""
    wb = ci.write_blocks(ci.load_runs())
    S = r3.semend(wb)
    CV = r3.semend_cv(wb)
    eps = [e for e in ("NY", "DA", "DB", "RF") if e in S["endpoints"]]
    L = [r"\begin{table}[p]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{Four endpoints no direction was fitted against.} Both direction families of Section~"
         r"\ref{sec:ansdir} are written at the primary dose into the same blocks and read at four endpoints: the negated form of "
         r"the concept's yes/no question, a two-alternative forced choice against the concept's strongest competitor in each of "
         r"the two presentation orders, and the log-probability of the finding word in a report continuation. Owned and "
         r"reference-meeting counts use the endpoint's own clean baseline, a permutation sham of the direction under test, and the "
         r"endpoint's own family of \cfSemLedgerRandom{} random directions read at its maximum rather than the campaign's 119 read at their 95th "
         r"percentile (Table~\ref{tab:cf-ledger}); the verdict is the campaign's $6\times5$ max-$T$ verdict. \emph{Opposite} counts "
         r"concepts of six whose raw effect on the negated question has the sign "
         r"opposite to its effect on the affirmative one. \emph{Order gap} is the mean absolute difference between the two "
         r"presentation orders of the forced choice. The bottom panel asks whether ownership carries information about endpoint "
         r"behaviour beyond the write magnitude, the probe selectivity and the clean-answer AUROC, and answers it out of sample. "
         r"Each endpoint effect is regressed on those three quantities with endpoint dummies, then on the same model plus the "
         r"ownership contrast and its owned indicator. Both models are ridge regressions whose penalty and standardisation are "
         r"fitted inside the training fold alone, and they are cross-validated twice: leaving out one concept at a time inside a "
         r"block, and leaving out one block at a time over the pooled blocks. $\Delta R^2$ is the change in held-out $R^2$ when "
         r"ownership is added, against a null that permutes ownership across the concepts of a block over "
         + f"{CV['permutations']:,}".replace(",", "{,}") + r" draws. Adding a predictor cannot lower a training $R^2$, so the "
         r"in-sample increment is reported only as the reference it is.}",
         r"\label{tab:cf-semend}",
         r"\begin{tabular}{lllcccc}", r"\toprule",
         r" & & & \multicolumn{2}{c}{label direction} & \multicolumn{2}{c}{answer direction} \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Checkpoint & Dataset & Endpoint & owned & reference & owned & reference \\", r"\midrule"]
    for b in S["blocks"]:
        for ep in eps:
            d = b["per_endpoint"][ep]
            L.append(f"{CKPT.get(b['model'], b['model'])} & {NAMES[b['dataset']]} & {ENDPOINT_LABEL[ep]} & "
                     + " & ".join(f"{d[fam]['owned']}/{d[fam]['n_concepts']} & {d[fam]['reference_met']}/{d[fam]['n_concepts']}"
                                  for fam in r3.FAMS) + r" \\")
    L += [r"\midrule",
          f"\\textit{{all}} & \\textit{{{S['n_blocks']} blocks}} & \\textit{{{len(eps)} endpoints}} & "
          f"{S['owned']['label']}/{S['cells']} & {S['reference_met']['label']}/{S['cells']} & "
          f"{S['owned']['answer']}/{S['cells']} & {S['reference_met']['answer']}/{S['cells']} \\\\",
          r"\midrule", r"\multicolumn{7}{l}{\textit{The three endpoint-specific tests}} \\",
          r"Checkpoint & Dataset & opposite on the negation & order gap & forced choice owned & report owned & core owned \\",
          r"\midrule"]
    for b in S["blocks"]:
        L.append(f"{CKPT.get(b['model'], b['model'])} & {NAMES[b['dataset']]} & {b['negation_opposite']}/{b['negation_n']} & "
                 f"{f2(b['forced_order_gap'], 3)} & {b['forced_both_owned']}/{b['negation_n']} & "
                 f"{b['report_owned']}/{b['report_cells']} & {len(b['owned_core'])}/{b['negation_n']} \\\\")
    L += [r"\midrule", r"\multicolumn{7}{l}{\textit{What ownership adds, held out: leaving out one concept at a time}} \\",
          r"Checkpoint & Dataset & Direction & held-out $R^2$ & with ownership & $\Delta R^2$ & permutation $p$ \\", r"\midrule"]
    for r in CV["leave_one_concept_out"]:
        mk = r["block"].split("/")[0]
        L.append(f"{CKPT.get(mk, mk)} & {NAMES[r['dataset']]} & {FAM_NAME[r['family']]} & "
                 f"{f2(r['heldout_r2_base'], 3)} & {f2(r['heldout_r2_full'], 3)} & {f2(r['delta_heldout_r2'], 3)} & "
                 f"{f2(r['permutation_p'], 2)} \\\\")
    d = CV["pooled_leave_one_concept_out"]
    L.append(f"\\textit{{pooled}} & \\textit{{{d['n_tests']} tests}} & \\textit{{both}} & -- & -- & "
             f"{f2(d['mean_delta_heldout_r2'], 3)} & {f2(d['permutation_p'], 2)} \\\\")
    L += [r"\midrule", r"\multicolumn{7}{l}{\textit{The same, leaving out one block at a time}} \\", r"\midrule"]
    for r in CV["leave_one_block_out"]:
        L.append(f"\\textit{{all {r['n_folds']}}} & \\textit{{pooled}} & {FAM_NAME[r['family']]} & "
                 f"{f2(r['heldout_r2_base'], 3)} & {f2(r['heldout_r2_full'], 3)} & {f2(r['delta_heldout_r2'], 3)} & "
                 f"{f2(r['permutation_p'], 2)} \\\\")
    d = CV["pooled_leave_one_block_out"]
    L.append(f"\\textit{{pooled}} & \\textit{{{d['n_tests']} tests}} & \\textit{{both}} & -- & -- & "
             f"{f2(d['mean_delta_heldout_r2'], 3)} & {f2(d['permutation_p'], 2)} \\\\")
    L += [r"\midrule", r"\multicolumn{7}{l}{\textit{In-sample reference only, since adding a predictor cannot lower a "
          r"training $R^2$}} \\",
          r"Checkpoint & Dataset & Direction & training $R^2$ & with ownership & $\Delta R^2$ & \\", r"\midrule"]
    for b in S["blocks"]:
        for fam in r3.FAMS:
            v = b["incremental_validity"][fam]
            L.append(f"{CKPT.get(b['model'], b['model'])} & {NAMES[b['dataset']]} & {FAM_NAME[fam]} & {f2(v['base_r2'], 3)} & "
                     f"{f2(v['full_r2'], 3)} & {f2(v['delta_r2'], 3)} & \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_semend.tex").write_text("\n".join(L) + "\n")
    print("table_cf_semend.tex", S["n_blocks"], "block(s),", len(eps), "endpoint(s)")


POOL_LABEL = {"chest": "chest findings", "coco_easy": "easy objects, every COCO block",
              "coco_easy_same_blocks": "easy objects, these blocks",
              "coco_fine": "fine-grained objects, these blocks"}
POOL_ORDER = ("chest", "coco_easy", "coco_easy_same_blocks", "coco_fine")
MATCH_LABEL = {"selectivity": "probe selectivity alone", "answer_auroc": "clean-answer AUROC alone",
               "both": "both variables", "both_easy_partners_only": "both variables, easy partners only"}


def fgobj():
    """FGOBJ: six small or fine-grained COCO categories fitted, written and graded exactly like the six easy ones, and the
    difficulty-matched comparison of chest cells against natural-image cells (runs/robustness/round2.json fgobj section)."""
    F = r3.fgobj(ci.write_blocks(ci.load_runs()))
    P = F["pool"]
    L = [r"\begin{table}[p]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Small and fine-grained objects, and the difficulty-matched comparison.} The six categories---"
         + ", ".join(F["concepts"][:-1]) + ", and " + F["concepts"][-1] + r"---were fixed by a rule on the label counts and the mean "
         r"instance areas of the fixed cohorts alone, registered before any of them was scored, as were the two matching windows; "
         r"Appendix~\ref{app:repro-protocol} states the rule in full. They are fitted, lifted, written and graded "
         r"exactly like the six easy objects, in the same blocks and rows, and their clean-answer AUROC is measured on the same "
         r"calibration rows and by the same rule as every other cell. The middle panel separates the easy objects of every COCO "
         r"block, which is the grid-wide number, from the easy objects of the "
         + str(len(F["blocks"])) + r" blocks the fine-grained cells come from, which is the pool the fine-grained rate is "
         r"compared with. The bottom panel pairs each chest cell with every natural-image cell whose matching variables lie "
         r"within " + f2(F["window"], 2) + r" of it and reports two estimators of the ownership difference, each with its own "
         r"interval. The \emph{pooled} difference is the ownership rate among the matched chest cells minus the rate among the "
         r"natural-image cells that served as a partner at least once: it is the difference of the two rates printed to its "
         r"left. The \emph{matched} difference is the mean over matched chest cells of that cell's own ownership minus the "
         r"ownership rate of its own partner set, so a chest cell with many partners and one with few count equally. The two "
         r"answer different questions and neither can be obtained from the other. \emph{Chest cells} and \emph{partners} give "
         r"the effective sample of both: matched chest cells of all chest cells, and distinct natural-image cells used as a "
         r"partner of all of them. Both intervals are percentile intervals over 5{,}000 draws of a cluster bootstrap whose unit "
         r"is the model. Probe selectivity is the probe AUROC minus the mean of the twenty random-label control AUROCs on the "
         r"calibration rows.}",
         r"\label{tab:cf-fgobj}",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{owned of six} & \multicolumn{2}{c}{median answer AUROC} & \multicolumn{2}{c}{median selectivity} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Checkpoint & fine & easy & fine & easy & fine & easy \\", r"\midrule"]
    for b in sorted(F["blocks"], key=lambda b: _ckpt_key(b["block"])):
        L.append(f"{CKPT.get(b['model'], b['model'])} & {b['fine_owned']} & {b['easy_owned']} & "
                 f"{f2(b['median_answer_auroc_fine'], 3)} & {f2(b['median_answer_auroc_easy'], 3)} & "
                 f"{f2(b['median_selectivity_fine'], 3)} & {f2(b['median_selectivity_easy'], 3)} \\\\")
    SB = F["same_block"]
    L += [r"\midrule",
          r"\multicolumn{7}{l}{\textit{Paired within these blocks:} fine-grained minus easy is "
          + ", ".join(f"{p['difference']:+d}" for p in SB["per_block"]) + r" cells of six; fine higher in "
          + str(SB["n_blocks_fine_higher"]) + r",} \\",
          r"\multicolumn{7}{l}{easy higher in " + str(SB["n_blocks_easy_higher"]) + r", tied in " + str(SB["n_blocks_tied"])
          + r"; exact two-sided sign test $p=" + f2(SB["sign_test_p"], 2) + r"$} \\"]
    L += [r"\midrule", r"\multicolumn{7}{l}{\textit{The four pools of cells}} \\",
          r"Cells & blocks & owned & rate & median answer AUROC & median selectivity & readable \\", r"\midrule"]
    for g in POOL_ORDER:
        d = P[g]
        L.append(f"{POOL_LABEL[g]} & {d['blocks']} & {d['owned']}/{d['cells']} & {f2(d['owned_rate'], 3)} & "
                 f"{f2(d['median_answer_auroc'], 3)} & {f2(d['median_selectivity'], 3)} & {d['readable']}/{d['cells']} \\\\")
    L += [r"\midrule", r"\multicolumn{7}{l}{\textit{Chest cells matched to natural-image cells on difficulty, under both "
          r"estimators}} \\",
          r"Matching & chest cells & partners & chest rate & natural rate & difference & 95\% interval \\", r"\midrule"]
    for m in ("selectivity", "answer_auroc", "both", "both_easy_partners_only"):
        d = F["matched_easy_only"] if m == "both_easy_partners_only" else F["matched"][m]
        L.append(f"{MATCH_LABEL[m]} & {d['n_matched_chest_cells']}/{d['n_chest_cells']} & "
                 f"{d['n_natural_cells_used']}/{d['n_natural_cells']} & {f2(d['chest_owned_rate_matched'], 3)} & "
                 f"{f2(d['natural_owned_rate_used'], 3)} &  &  \\\\")
        for est, key in (("pooled", "pooled_ownership_difference"), ("matched", "matched_ownership_difference")):
            L.append(f"\\quad {est} &  &  &  &  & {f2(d[key], 3)} & "
                     f"[{f2(d[key + '_ci95'][0], 3)}, {f2(d[key + '_ci95'][1], 3)}] \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_fgobj.tex").write_text("\n".join(L) + "\n")
    print("table_cf_fgobj.tex", len(F["blocks"]), "block(s)")


def ledger():
    """The reference-and-rule ledger: one row per graded analysis, populated from the packaged artefacts."""
    rows = cl.ledger(ci.write_blocks(ci.load_runs()))
    L = [r"\begin{table}[p]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{\textbf{What each analysis scores its writes against.} One row per graded analysis. "
         r"\emph{Rows} is the cohort each analysis scores and its size. \emph{Reference} is the steering test the "
         r"own write must pass: the 95th percentile of the 119-direction random family and the absolute sham, a "
         r"smaller random family read at its maximum, or the sham alone. \emph{Rule} is what decides the verdict: "
         r"simultaneous max-$T$ lower bounds on the fixed contrasts between the own write and each competitor, or a "
         r"percentile interval on the ownership contrast with the maximum recomputed in every draw. \emph{Sham} names "
         r"the direction the sham permutes: \emph{the direction written} means the coordinate-permutation sham of the "
         r"very direction the analysis writes, and \emph{the seed-0 normal} means the permutation of the seed-0 "
         r"logistic normal, which four analyses reuse for a direction they did not fit. \emph{Cells} counts the graded "
         r"(question, arm) cells. The token-weighted writes compare against the uniformly written permutation of the "
         r"normal they weight. Every entry is read from the packaged per-block statistics, and each declared "
         r"reference and sham is checked against them.}",
         r"\label{tab:cf-ledger}",
         r"\begin{tabular}{>{\raggedright\arraybackslash}p{0.155\textwidth}>{\raggedright\arraybackslash}p{0.088\textwidth}"
         r">{\raggedright\arraybackslash}p{0.215\textwidth}>{\raggedright\arraybackslash}p{0.175\textwidth}"
         r">{\raggedright\arraybackslash}p{0.125\textwidth}r}", r"\toprule",
         r"Analysis & Rows & Reference & Rule & Sham & Cells \\", r"\midrule"]
    for r in rows:
        L.append(" & ".join([r["analysis"], r["rows"], r["reference"], r["rule"], r["sham"], f"{r['cells']:,}".replace(",", "{,}")]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_ledger.tex").write_text("\n".join(L) + "\n")
    print("table_cf_ledger.tex", len(rows), "analyses,", sum(r["cells"] for r in rows), "cells")


if __name__ == "__main__":
    ledger()
    geometry()
    scale()
    pairs()
    constructions()
    answer_direction()
    validation()
    refit()
    valid()
    attr()
    towerswap()
    semend()
    fgobj()
    for merged in ("altdird", "ansdirt", "extcomp", "tokenw", "rescue", "replay"):   # now panels of another table, or dropped for prose
        (OUT / f"table_cf_{merged}.tex").unlink(missing_ok=True)
