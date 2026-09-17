"""Consistency check: manifest counts == Table 1 "All cells" row == Figure 2(a) counts == cf_numbers macros == prose.

    python scripts/check_numbers.py        (exit 1 on any disagreement)

Reads runs/manifest.csv (cf_inclusion.read_manifest), tables/table_cf_main.tex, tables/table_cf_{altdir,ansdir,scale,geometry}.tex,
tables/table_seed_mass.tex with the registered seed tables (table_decoding, table_encoding_mass, table_mass_confirmation), figures/fig2_counts.json (written by
plot_paper_figures.py fig2_overview), figures/figA1_own_share.json (figA1_write_structure), tables/cf_numbers.json
(build_numbers.py), tables/table_cf_contingency.tex (the Section 5.1 decomposition macros), runs/robustness/round2.json with tables/table_cf_valid.tex and tables/table_cf_attr.tex (round-2 and ATTR macros), and, when main.pdf exists and pdftotext is available, the rendered abstract and Section 5.1.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cf_inclusion as ci   # noqa: E402
from build_numbers import counts, pct, sci   # noqa: E402

ROOT = HERE.parent
DS = ("nih", "chexpert", "coco")
MATCHINGS_IN_TABLE = ("probe selectivity alone", "clean-answer AUROC alone", "both variables",
                      "both variables, easy partners only")
bad = 0


def round_half_up(x):
    return int(x + 0.5)


def altdir_panels(tex):
    """[(panel title, {row label: [(owned, cells) per dataset, or (0, 0) for '--']})] of Table cf-altdir."""
    out = []
    for chunk in tex.split(r"\multicolumn{8}{l}{\textit{")[1:]:
        title = chunk.split("}} \\\\", 1)[0]
        rows = {}
        for m in re.finditer(r"^\\quad (.+?) & (\w*) & (.+?) \\\\$", chunk, flags=re.M):
            cells = m.group(3).split(" & ")
            rows[m.group(1)] = [tuple(map(int, cells[i].split("/"))) if "/" in cells[i] else (0, 0) for i in (0, 2, 4)]
        out.append((title, rows))
    return out


def expect(name, a, b):
    global bad
    ok = a == b
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'} {name}: {a} {'==' if ok else '!='} {b}")


def main():
    rows = ci.read_manifest(); C = counts(rows)
    print("manifest per-dataset (probe cells / readable | write cells / answerable / owned):")
    for ds in DS:
        d = C[ds]
        print(f"  {ds:8s} {d['probe_cells']}/{d['readable']} | {d['write_cells']}/{d['answerable']}/{d['owned']}")
    # Table 1 "All cells" rows: means and owned shares, then the nine (count, denominator) pairs on the row below
    tex = (ROOT / "tables" / "table_cf_main.tex").read_text()
    row = re.search(r"\\textbf\{All cells\} & (.*?) \\\\", tex).group(1)
    sup = re.findall(r"(\d+)/(\d+)", re.search(r"count / cells & (.*?) \\\\", tex).group(1))   # Read/Ans/Own x 3 datasets
    print("Table 1 'All cells' row vs manifest:")
    for i, ds in enumerate(DS):
        d = C[ds]
        expect(f"{ds} Read", tuple(map(int, sup[3 * i])), (d["readable"], d["probe_cells"]))
        expect(f"{ds} Ans", tuple(map(int, sup[3 * i + 1])), (d["answerable"], d["write_cells"]))
        expect(f"{ds} Own", tuple(map(int, sup[3 * i + 2])), (d["owned"], d["write_cells"]))
    shares = re.findall(r"(\d+)\\%", row)
    expect("Table 1 clinical owned share", int(shares[0]), pct(C["chest"]["owned"], C["chest"]["write_cells"]))
    expect("Table 1 COCO owned share", int(shares[1]), pct(C["coco"]["owned"], C["coco"]["write_cells"]))
    # Figure 2(a)
    f2 = ROOT / "figures" / "fig2_counts.json"
    if f2.exists():
        g = json.loads(f2.read_text())
        print("Figure 2(a) counts vs manifest:")
        for ds in DS:
            d = C[ds]
            expect(f"{ds} readable", tuple(g[ds]["readable"]), (d["readable"], d["probe_cells"]))
            expect(f"{ds} answerable", tuple(g[ds]["answerable"]), (d["answerable"], d["write_cells"]))
            expect(f"{ds} owned", tuple(g[ds]["owned"]), (d["owned"], d["write_cells"]))
    else:
        print("  (figures/fig2_counts.json missing: run plot_paper_figures.py)")
    # macros
    M = json.loads((ROOT / "tables" / "cf_numbers.json").read_text())["macros"]
    print("cf_numbers macros vs manifest:")
    expect("cfChestReadable/N", (M["cfChestReadable"], M["cfChestReadableN"]), (C["chest"]["readable"], C["chest"]["probe_cells"]))
    expect("cfChestAnswerable/N", (M["cfChestAnswerable"], M["cfChestAnswerableN"]), (C["chest"]["answerable"], C["chest"]["write_cells"]))
    expect("cfChestOwned", M["cfChestOwned"], C["chest"]["owned"])
    expect("cfCocoOwned/N", (M["cfCocoOwned"], M["cfCocoOwnedN"]), (C["coco"]["owned"], C["coco"]["write_cells"]))
    expect("cfCells", M["cfCells"], C["all"]["write_cells"])
    # Table cf-prevalence (label counts per cohort) against the frozen data manifests, and the support macros against it
    from build_numbers import cohort_counts   # noqa: E402
    pv = (ROOT / "tables" / "table_cf_prevalence.tex").read_text()
    dsname = {"NIH ChestX-ray14": "nih", "CheXpert Plus": "chexpert", "COCO": "coco"}
    print("Table cf-prevalence vs the data manifests (train / calibration / test):")
    ds = None
    for m in re.finditer(r"^\s*([A-Za-z0-9 .-]*?)\s*&\s*([A-Za-z]+)\s*&\s*([\d &]+?)\s*\\\\$", pv, flags=re.M):
        ds = dsname.get(m.group(1).strip(), ds)
        if ds is None:
            continue
        got = [int(x) for x in m.group(3).split("&")]
        want = []
        for role in ("train", "calibration", "test"):
            p, n = cohort_counts(ds, role)[m.group(2)]
            want += [p, n]
        expect(f"{ds} {m.group(2)} pos/neg", [got[0], got[1], got[3], got[4], got[6], got[7]], want)
    expect("cfCal/cfTest ineligible cells vs Table cf-prevalence",
           (M["cfCalChexInelPos"], M["cfCalChexInelNeg"], M["cfTestChexInelPos"], M["cfTestChexInelNeg"],
            M["cfCalCocoInelPos"], M["cfTestCocoInelPos"]),
           (cohort_counts("chexpert", "calibration")["Atelectasis"] + cohort_counts("chexpert", "test")["Atelectasis"]
            + (cohort_counts("coco", "calibration")["bicycle"][0], cohort_counts("coco", "test")["bicycle"][0])))
    # contingency macros (Section 5.1 decomposition) vs Table cf-contingency
    ct = (ROOT / "tables" / "table_cf_contingency.tex").read_text()
    crow = re.findall(r"^(.*?) & (met|not met) & (\d+) & (\d+) & (\d+) & (\d+) \\\\$", ct, flags=re.M)
    tab, grp = {}, None
    for name, met, adv, strong, unres, tot in crow:
        grp = name.strip() or grp
        tab[(grp, met)] = tuple(map(int, (adv, strong, unres, tot)))
    print("contingency macros vs Table cf-contingency:")
    for g, G in (("Chest, all cells", "Chest"), ("Chest, readable and answerable", "ReadAns"), ("COCO, all cells", "Coco")):
        own = {"Chest": "cfChestOwned", "ReadAns": "cfReadAnsOwned", "Coco": "cfCocoOwned"}[G]
        refmet = {"Chest": "cfChestRefMet", "ReadAns": "cfReadAnsRefMet", "Coco": "cfCocoRefMet"}[G]
        expect(f"{G} reference met (owned, strong, unresolved, total)", (M[own], M[f"cf{G}RefStrong"], M[f"cf{G}RefUnres"], M[refmet]), tab.get((g, "met")))
        expect(f"{G} reference not met (advantage, strong, unresolved)", (M[f"cf{G}BelowAdv"], M[f"cf{G}BelowStrong"], M[f"cf{G}BelowUnres"]), tab.get((g, "not met"))[:3])
    expect("cfChestRefBelow", M["cfChestRefBelow"], tab[("Chest, all cells", "not met")][3])
    expect("cfReadAnsRefBelow", M["cfReadAnsRefBelow"], tab[("Chest, readable and answerable", "not met")][3])
    expect("cfChestCompetitor = strong above + below", M["cfChestCompetitor"], M["cfChestRefStrong"] + M["cfChestBelowStrong"])
    expect("cfReadAnsCompetitor = strong above + below", M["cfReadAnsCompetitor"], M["cfReadAnsRefStrong"] + M["cfReadAnsBelowStrong"])
    expect("cfCocoCompetitor = strong above + below", M["cfCocoCompetitor"], M["cfCocoRefStrong"] + M["cfCocoBelowStrong"])
    expect("cfReadAns = met + not met", M["cfReadAns"], M["cfReadAnsRefMet"] + M["cfReadAnsRefBelow"])
    # round-2 macros and tables vs runs/robustness/round2.json
    r2p = ci.RUNS / "robustness" / "round2.json"
    if r2p.exists() and "cfValidCells" in M:
        R2 = json.loads(r2p.read_text())
        va, al, an, pr = R2["valid"]["aggregate"], R2["altdird"]["per_dataset"], R2["ansdirt"]["aggregate"], R2["precision"]["aggregate"]
        print("round-2 macros vs runs/robustness/round2.json:")
        expect("cfValidOwnedAgree/Cells", (M["cfValidOwnedAgree"], M["cfValidCells"]), (va["owned_agree"], va["cells"]))
        expect("cfValidOwnedTest/Valid", (M["cfValidOwnedTest"], M["cfValidOwnedValid"]), (va["owned_test"], va["owned_valid"]))
        expect("cfAltdirdChest Dom/Pattern/N", (M["cfAltdirdChestDom"], M["cfAltdirdChestPattern"], M["cfAltdirdChestN"]),
               (al["chest"]["dom_disp"]["owned"], al["chest"]["pattern_disp"]["owned"], al["chest"]["cells"]))
        expect("cfAltdirdCoco Dom/Pattern/N", (M["cfAltdirdCocoDom"], M["cfAltdirdCocoPattern"], M["cfAltdirdCocoN"]),
               (al["coco"]["dom_disp"]["owned"], al["coco"]["pattern_disp"]["owned"], al["coco"]["cells"]))
        expect("cfAnsdirtChestKept/N", (M["cfAnsdirtChestKept"], M["cfAnsdirtChestKeptN"]), (an["chest"]["kept_pairs"], an["chest"]["pairs"]))
        expect("cfAnsdirtCocoKept/N", (M["cfAnsdirtCocoKept"], M["cfAnsdirtCocoKeptN"]), (an["coco"]["kept_pairs"], an["coco"]["pairs"]))
        expect("cfPrecisionGradeChanges/Cells", (M["cfPrecisionGradeChanges"], M["cfPrecisionGradeCells"]), (pr["grade_changes"], pr["graded_cells"]))
        vt = re.search(r"\\textit\{all\} \((\d+) blocks\) & (\d+) & (\d+) & (\d+)/(\d+) & (\d+)/(\d+)", (ROOT / "tables" / "table_cf_valid.tex").read_text())
        expect("Table cf-valid totals row", tuple(map(int, vt.groups())) if vt else None,
               (va["blocks"], va["owned_test"], va["owned_valid"], va["verdict_agree"], va["cells"], va["owned_agree"], va["cells"]))
    # ATTR macros and Table cf-attr against runs/robustness/round2.json
    if r2p.exists() and M.get("cfAttrBlocks"):
        at = json.loads(r2p.read_text())["attr"]["per_dataset"]
        print("ATTR macros vs runs/robustness/round2.json:")
        expect("cfAttrBlocks", M["cfAttrBlocks"], at["chest"]["blocks"])
        for g, G in (("nih", "Nih"), ("chexpert", "Chex"), ("chest", "Chest")):
            expect(f"cfAttr{G} attribute/clinical owned of N",
                   (M[f"cfAttr{G}AttrOwned"], M[f"cfAttr{G}AttrN"], M[f"cfAttr{G}ClinOwned"], M[f"cfAttr{G}ClinN"]),
                   (at[g]["attr_owned"], at[g]["attr_cells"], at[g]["clin_owned"], at[g]["clin_cells"]))
        pa = [p for p in at["chest"]["per_attribute"].values() if p["blocks"]]
        expect("cfAttrAurocMin/Max", (M["cfAttrAurocMin"], M["cfAttrAurocMax"]),
               (f"{min(p['median_auroc_real'] for p in pa):.3f}", f"{max(p['median_auroc_real'] for p in pa):.3f}"))
        expect("cfAttrAnswerAurocMin/Max", (M["cfAttrAnswerAurocMin"], M["cfAttrAnswerAurocMax"]),
               (f"{min(p['median_answer_auroc'] for p in pa):.2f}", f"{max(p['median_answer_auroc'] for p in pa):.2f}"))
        expect("cfAttrCosMax", M["cfAttrCosMax"], f"{at['chest']['max_median_abs_cos_pair']['median_abs_cos']:.2f}")
        tt = re.findall(r"\\textit\{chest\} & (\d+) blocks? & (\d+)/(\d+) & (\d+)/(\d+)", (ROOT / "tables" / "table_cf_attr.tex").read_text())
        expect("Table cf-attr chest totals row", tuple(map(int, tt[0])) if tt else None,
               (at["chest"]["blocks"], at["chest"]["attr_owned"], at["chest"]["attr_cells"], at["chest"]["clin_owned"], at["chest"]["clin_cells"]))
    # Figure 2(c) dose blocks against the manifest (included blocks with DOSE completed) and the macros
    f2d = ROOT / "figures" / "fig2_dose_blocks.json"
    done = {}
    for r in rows:
        done.setdefault((r["model_key"], r["dataset"]), (r["block_included"] == "true", set(m for m in r["completed_modules"].split("|") if m)))
    man_dose = {ds: sum(inc and "DOSE" in mods for (mk, d), (inc, mods) in done.items() if d == ds) for ds in DS}
    print("Figure 2(c) dose blocks vs manifest:")
    for ds, D in (("nih", "Nih"), ("chexpert", "Chex"), ("coco", "Coco")):
        expect(f"macro cfDoseBlocks{D}", M[f"cfDoseBlocks{D}"], man_dose[ds])
    if f2d.exists():
        g = json.loads(f2d.read_text())
        for ds in g:
            expect(f"figure 2(c) {ds} blocks", g[ds], man_dose[ds])
    else:
        print("  (figures/fig2_dose_blocks.json missing: run plot_paper_figures.py)")
    # Table cf-altdir (constructions, displacement lift, token weights, extended competitor family): every panel vs the macros
    panels = altdir_panels((ROOT / "tables" / "table_cf_altdir.tex").read_text())
    print("ALTDIR / ALTDIRD / TOKENW / EXTCOMP macros vs Table cf-altdir:")
    (t1, r1), (t2, r2), (t3, r3), (t4, r4) = panels
    for ds, D, i in (("nih", "Nih", 0), ("chexpert", "Chex", 1), ("coco", "Coco", 2)):
        for fam, F in (("logistic normal", "Logistic"), ("difference of means", "Dom"), ("Haufe pattern", "Pattern"),
                       ("orthogonalised normal", "Orth"), ("residualised normal", "Resid"), ("owned under at least one", "Any")):
            expect(f"cfAltdir{D}{F}/N", (M[f"cfAltdir{D}{F}"], M[f"cfAltdir{D}N"]), r1[fam][i])
    chest_pct = [round_half_up(100 * a / n) for fam in list(r1)[:5] for (a, n) in r1[fam][:2]]
    coco_pct = [round_half_up(100 * a / n) for fam in list(r1)[:5] for (a, n) in r1[fam][2:3]]
    expect("cfAltdirChestPctMin/Max", (M["cfAltdirChestPctMin"], M["cfAltdirChestPctMax"]), (min(chest_pct), max(chest_pct)))
    expect("cfAltdirCocoPctMin/Max", (M["cfAltdirCocoPctMin"], M["cfAltdirCocoPctMax"]), (min(coco_pct), max(coco_pct)))
    expect("cfAltdirBlocks (panel header)", t1, f"Direction construction ({M['cfAltdirBlocksPerDs']} blocks per dataset)")
    ch = lambda row: tuple(map(sum, zip(*row[:2])))
    expect("cfAltdirdChest Dom/Pattern of N", (M["cfAltdirdChestDom"], M["cfAltdirdChestPattern"], M["cfAltdirdChestN"]),
           (ch(r2["difference of means"])[0], ch(r2["Haufe pattern"])[0], ch(r2["difference of means"])[1]))
    expect("cfAltdirdCoco Dom/Pattern of N", (M["cfAltdirdCocoDom"], M["cfAltdirdCocoPattern"], M["cfAltdirdCocoN"]),
           (r2["difference of means"][2][0], r2["Haufe pattern"][2][0], r2["difference of means"][2][1]))
    expect("cfTokenwChest uniform/softmax/top quarter of cells",
           (M["cfTokenwUniformChestOwned"], M["cfTokenwSoftChestOwned"], M["cfTokenwTopqChestOwned"], M["cfTokenwChestCells"]),
           (ch(r3["uniform"])[0], ch(r3["softmax of the probe score"])[0], ch(r3["top quarter of tokens"])[0], ch(r3["uniform"])[1]))
    expect("cfTokenwCoco softmax/top quarter of cells", (M["cfTokenwSoftCocoOwned"], M["cfTokenwTopqCocoOwned"], M["cfTokenwCocoCells"]),
           (r3["softmax of the probe score"][2][0], r3["top quarter of tokens"][2][0], r3["uniform"][2][1]))
    blocks3 = dict((k, int(n)) for n, k in re.findall(r"(\d+) (NIH|CheXpert|COCO)", t3))
    expect("cfTokenwChestBlocks (panel header)", M["cfTokenwChestBlocks"], blocks3.get("NIH", 0) + blocks3.get("CheXpert", 0))
    blocks4 = dict((k, int(n)) for n, k in re.findall(r"(\d+) (NIH|CheXpert|COCO)", t4))
    expect("cfExtcompBlocks (panel header)", M["cfExtcompBlocks"], sum(blocks4.values()))
    expect("cfExtcompOwned (six directions)", M["cfExtcompOwned"], sum(a for a, n in r4["six clinical directions"] if n))
    expect("cfExtcompExtOwned (extended family)", M["cfExtcompExtOwned"], sum(a for a, n in r4["with every extra dataset label"] if n))
    expect("cfExtcompRetained within both counts", M["cfExtcompRetained"] <= min(M["cfExtcompOwned"], M["cfExtcompExtOwned"]), True)
    # Table cf-ansdir aggregate rows (primary template) and the ANSDIRT aggregates vs the macros
    at = (ROOT / "tables" / "table_cf_ansdir.tex").read_text()
    agg = {m.group(1): m.groups() for m in re.finditer(
        r"^(NIH ChestX-ray14|CheXpert Plus|\\textit\{chest\}|COCO) & (\d+) blocks & (\d+)/(\d+) & (\d+)/(\d+) & (\d+) & (\d+) & (\d+) & (\d+)/(\d+) & "
        r"([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\$", at, flags=re.M)}
    tpl = {m.group(1): m.groups() for m in re.finditer(r"^(\\textit\{chest\}|COCO) & (\d+) blocks & ((?:\d+ \(\d+\) & ){5})(\d+)/(\d+) \\\\$", at, flags=re.M)}
    print("answer-direction macros vs Table cf-ansdir:")
    c = agg["\\textit{chest}"]
    expect("cfAnsChest blocks, N, label owned, a_q owned, beats", (M["cfAnsChestBlocks"], M["cfAnsChestN"], M["cfAnsChestOwnedLabel"], M["cfAnsChestOwnedA"], M["cfAnsChestBeats"]),
           (int(c[1]), int(c[3]), int(c[2]), int(c[4]), int(c[9])))
    expect("cfValCosSigmaChestMin/Max and Coco (whitened cosine rows)", (M["cfValCosSigmaChestMin"], M["cfValCosSigmaChestMax"], M["cfValCosSigmaCoco"]),
           (min(agg["NIH ChestX-ray14"][13], agg["CheXpert Plus"][13]), max(agg["NIH ChestX-ray14"][13], agg["CheXpert Plus"][13]), agg["COCO"][13]))
    for g, G in (("\\textit{chest}", "Chest"), ("COCO", "Coco")):
        row = tpl[g]; owned_pairs = sum(int(x) for x in re.findall(r"(\d+) \(", row[2]))
        expect(f"cfAnsdirt{G}Kept/KeptN, OwnedPairs/PairsAll", (M[f"cfAnsdirt{G}Kept"], M[f"cfAnsdirt{G}KeptN"], M[f"cfAnsdirt{G}OwnedPairs"], M[f"cfAnsdirt{G}PairsAll"]),
               (int(row[3]), int(row[4]), owned_pairs, 30 * int(row[1])))
    # the numerics panel of Table cf-scale vs the macros quoted in Section 5.8 and in the Appendix A.7 numerics paragraph
    pt = (ROOT / "tables" / "table_cf_scale.tex").read_text()
    numerics_prose = (ROOT / "sections" / "app_campaign.tex").read_text()
    pr = re.findall(r"& ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\$", pt, flags=re.M)
    print("PRECISION macros vs Table cf-precision:")
    expect("cfPrecisionGradeBlocks = rows", M["cfPrecisionGradeBlocks"], len(pr))
    expect("cfPrecisionGradeCells = 6 x rows x 2 settings", M["cfPrecisionGradeCells"], 12 * len(pr))
    expect("cfPrecisionMaxDW = largest max|dW| in the table", M["cfPrecisionMaxDW"], f"{max(float(r[i]) for r in pr for i in (0, 2)):.3f}")
    expect("precision macros present in the Appendix A.7 numerics paragraph",
           [k for k in ("cfPrecisionVerdictChanges", "cfPrecisionRefChanges", "cfPrecisionPointVerdictChanges",
                        "cfPrecisionGradeCells") if f"\\{k}" not in numerics_prose], [])
    # Table cf-geometry vs the Appendix A.7 geometry macros
    gt = (ROOT / "tables" / "table_cf_geometry.tex").read_text()
    grow = {m.group(1): m.group(2).split(" & ") for m in re.finditer(r"^\\quad (.+?) & (.+?) \\\\$", gt, flags=re.M)}
    print("geometry macros vs Table cf-geometry:")
    expect("cfGeoCosNih/Chex", (M["cfGeoCosNih"], M["cfGeoCosChex"]), (grow["mean off-diagonal cosine between the directions"][0], f"{float(grow['mean off-diagonal cosine between the directions'][1]):.2f}"))
    expect("cfGeoCosRandom", M["cfGeoCosRandom"], grow["mean $|\\cos|$ between random unit directions"][0])
    expect("cfGeoPhiNih", M["cfGeoPhiNih"], grow["mean off-diagonal $\\phi$ between the labels"][0])
    frac = lambda label, i: tuple(map(int, grow[label][i].split("/")))
    pc = lambda label, i: round_half_up(100 * frac(label, i)[0] / frac(label, i)[1])
    cos_rates = [pc("the most similar direction", i) for i in (0, 1)]
    lab_rates = [pc(l, i) for l in ("the label with the largest $\\phi$", "the most co-occurring label") for i in (0, 1)]
    expect("cfGeoChanceCosMin/Max", (M["cfGeoChanceCosMin"], M["cfGeoChanceCosMax"]), (min(cos_rates), max(cos_rates)))
    expect("cfGeoChanceLabelMin/Max", (M["cfGeoChanceLabelMin"], M["cfGeoChanceLabelMax"]), (min(lab_rates), max(lab_rates)))
    expect("cfGeoRhoAdvCos/Phi (chest)", (M["cfGeoRhoAdvCos"], M["cfGeoRhoAdvPhi"]),
           (grow["the cosine between the two directions"][2], grow["the $\\phi$ between the two labels"][2]))
    expect("cfGeoOwnWins Nih/Chex pct", (M["cfGeoOwnWinsNihPct"], M["cfGeoOwnWinsChexPct"]), (pc("cells with $O_q>0$", 0), pc("cells with $O_q>0$", 1)))
    expect("cfNihCells/cfChexCells", (M["cfNihCells"], M["cfChexCells"]), (frac("cells with $O_q>0$", 0)[1], frac("cells with $O_q>0$", 1)[1]))
    expect("cfGeoTwoAbove/N (chest)", (M["cfGeoTwoAbove"], M["cfGeoTwoAboveN"]), frac("cells with at least two competitors above the own write", 2))
    # seed study: decodability macros (Appendix C.1) vs the registered table, and the merged Mass table vs the registered tables
    print("seed-study macros and table vs the registered seed tables:")
    dec = (ROOT / "tables" / "table_decoding.tex").read_text()
    for name, model, concept in (("QwenEff", "Qwen2.5-VL-7B", "Effusion"), ("LlavaEff", "LLaVA-1.5-7B", "Effusion"), ("LlavaEdema", "LLaVA-1.5-7B", "Edema")):
        r = re.search(rf"^{re.escape(model)} & {concept} & .*$", dec, flags=re.M).group(0)
        nums = re.findall(r"\d\.\d{4}", r)
        expect(f"cfSeedDec{name} (AUROC, CI, control, spread, selectivity, CI)",
               tuple(M[f"cfSeedDec{name}{k}"] for k in ("Auroc", "AurocLo", "AurocHi", "Ctrl", "CtrlLo", "CtrlHi", "Sel", "SelLo", "SelHi")), tuple(nums))
    mass = (ROOT / "tables" / "table_seed_mass.tex").read_text()
    for reg in ("table_encoding_mass.tex", "table_mass_confirmation.tex"):
        body = (ROOT / "tables" / reg).read_text().split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
        for ln in (x.strip().rstrip("\\").strip() for x in body.strip().splitlines() if x.strip()):
            expect(f"seed-mass row {ln.split(' & ')[0]}", ln in mass, True)
    a1 = ROOT / "figures" / "figA1_own_share.json"
    if a1.exists():
        s = json.loads(a1.read_text())
        for ds, D in (("nih", "Nih"), ("chexpert", "Chex"), ("coco", "Coco")):
            expect(f"own share {ds} (figA1 vs macro)", pct(s[ds]["own_share"], 1), M[f"cfOwnShare{D}"])
    # ---- the four crossed experiments: every macro against the table it is printed in, and each table against the summaries
    print("crossed-experiment macros vs their tables and the block summaries:")
    import cf_round3 as r3   # noqa: E402
    wb = ci.write_blocks(ci.load_runs())      # cf_round3 raises if a module's run.json set differs from its summary set
    tw = (ROOT / "tables" / "table_cf_towerswap.tex").read_text()
    T = r3.towerswap(wb)
    tw_rows = re.findall(r"^(.+?) & (NIH ChestX-ray14|CheXpert Plus|COCO) & ([\d-]+) & ([\d-]+) & ([\d-]+) & ([\d-]+) & (\d+) & (.+?) \\\\$",
                         tw, flags=re.M)
    host = [r for r in tw_rows if r[0] != "all four"]
    allf = [r for r in tw_rows if r[0] == "all four"]
    expect("Table cf-towerswap host rows = cfSwapBlocks", len(host), M["cfSwapBlocks"])
    expect("Table cf-towerswap dataset rows = datasets with a swap", len(allf), len(T["per_dataset"]))
    expect("cfSwapCrossovers = rows saying complete", M["cfSwapCrossovers"], sum(r[7] == "complete" for r in host))
    chest = [r for r in allf if r[1] != "COCO"]
    coco = [r for r in allf if r[1] == "COCO"]
    owned_of = lambda rs: (sum(int(x) for r in rs for x in r[2:6] if x != "-" and x != "--"),
                           sum(int(r[6]) for r in rs for x in r[2:6] if x not in ("-", "--")))
    expect("cfSwapChestOwned/Cells = the chest dataset rows", (M["cfSwapChestOwned"], M["cfSwapChestCells"]), owned_of(chest))
    expect("cfSwapCocoOwned/Cells = the COCO dataset row", (M["cfSwapCocoOwned"], M["cfSwapCocoCells"]), owned_of(coco))
    expect("cfSwapCocoNativeMin/Max = the two native columns", (M["cfSwapCocoNativeMin"], M["cfSwapCocoNativeMax"]),
           (min(int(x) for r in coco for x in r[2:4]), max(int(x) for r in coco for x in r[2:4])))
    expect("cfSwapCocoCrossMin/Max = the two swapped columns", (M["cfSwapCocoCrossMin"], M["cfSwapCocoCrossMax"]),
           (min(int(x) for r in coco for x in r[4:6]), max(int(x) for r in coco for x in r[4:6])))
    tw_head = re.search(r"^Combinations & Dataset & (\S+) & (\S+) & (\S+) & (\S+) & ", tw, flags=re.M)
    expect("Table cf-towerswap swapped columns are M/G then G/M", tw_head.groups()[2:], ("M/G", "G/M"))
    expect("cfSwapCocoMedTowerCross/GemmaTowerCross = those two columns of the COCO row",
           (M["cfSwapCocoMedTowerCross"], M["cfSwapCocoGemmaTowerCross"]), (int(coco[0][4]), int(coco[0][5])))
    # the crossover panel: one block per "reader, own tower" row, each followed by its two mean-effect rows; every
    # printed mean is checked against the macro of its own dataset and quantity
    cx_blocks = re.findall(r"^(.+?), (NIH|CheXpert|COCO) & reader, own tower & ", tw, flags=re.M)
    expect("Table cf-towerswap crossover blocks = cfSwapCrossovers", len(cx_blocks), M["cfSwapCrossovers"])
    ds_of = {"NIH": "Nih", "CheXpert": "Chex", "COCO": "Coco"}
    cur, printed = None, {}
    for ln in tw.splitlines():
        m = re.match(r"^(.+?), (NIH|CheXpert|COCO) & reader, own tower & ", ln)
        if m:
            cur = ds_of[m.group(2)]
        m = re.match(r"^ & \\quad mean (reader|tower) effect &(.+?)\\\\$", ln)
        if m and cur:
            printed.setdefault((cur, m.group(1)), []).append([c.strip().split(" ")[0] for c in m.group(2).split("&")])
    expect("Table cf-towerswap mean-effect rows", sorted({k[0] for k in printed}), sorted(ds_of.values()))
    for (D, factor), rows_ in printed.items():
        F = factor.capitalize()
        for qi, qk in enumerate(("", "O", "M")):
            expect(f"cfSwap{F}{D}{qk}Min/Max = the {D} mean {factor} effect",
                   (M[f"cfSwap{F}{D}{qk}Min"], M[f"cfSwap{F}{D}{qk}Max"]),
                   (min(r[qi] for r in rows_), max(r[qi] for r in rows_)))
    swap_prose = (ROOT / "sections" / "app_campaign.tex").read_text()
    for name, key in (("cfSwapTensors", "n_tensors_replaced"), ("cfSwapTowerTensors", "n_tower_tensors"),
                      ("cfSwapOutside", "n_tensors_outside_tower")):
        expect(f"{name} in the Appendix A.10 crossed-tower paragraph", f"\\{name}" in swap_prose, True)

    # The crossed arms' clean-answer ability has no table: Appendix A.8 carries it in prose, one native and one hybrid
    # figure per crossed block, and the range the prose quotes is recomputed from those twelve macros.
    TA = r3.towerswap_answer(wb)
    pairs = [(f"cfSwapAnsNative{h}{d}", f"cfSwapAnsHybrid{h}{d}") for h in ("Gem", "Med") for d in ("Nih", "Chex", "Coco")]
    expect("cfSwapAnsArms = the crossed blocks", M["cfSwapAnsArms"], len(TA["blocks"]))
    expect("cfSwapAnsArms = cfSwapBlocks (one hybrid arm per crossed block)", M["cfSwapAnsArms"], M["cfSwapBlocks"])
    expect("one native and one hybrid figure per crossed block", len(pairs), M["cfSwapAnsArms"])
    drops = [round(float(M[n]) - float(M[y]), 3) for n, y in pairs]
    expect("cfSwapAnsWorse = hybrid arms below the native arm of their block", sum(d > 0 for d in drops), M["cfSwapAnsWorse"])
    expect("cfSwapAnsWorse covers every crossed arm", M["cfSwapAnsWorse"], M["cfSwapAnsArms"])
    expect("cfSwapAnsDropMin/Max = the range of the twelve printed figures",
           (M["cfSwapAnsDropMin"], M["cfSwapAnsDropMax"]), (f"{min(drops):.3f}", f"{max(drops):.3f}"))

    # The replay is a control that changes no grade, so it has no table: Section 5.5 and Appendix A.8 carry it in prose.
    # Every replay macro is therefore checked against the per-replay and per-pair records the module produces.
    R = r3.replay(wb)
    rp_rows, pr_rows = R["blocks"], R["pairs"]
    expect("cfReplayBlocks = the replay records", M["cfReplayBlocks"], len(rp_rows))
    expect("cfReplaySelf = replays whose tensors are their own", M["cfReplaySelf"], sum(b["self_replay"] for b in rp_rows))
    expect("cfReplayMoved/Cells = the moved concepts", (M["cfReplayMoved"], M["cfReplayCells"]),
           (sum(b["n_nonzero"] for b in rp_rows), sum(b["n_concepts"] for b in rp_rows)))
    expect("cfReplayVerdictChanges + cfReplayOwnedChanges = the grade changes",
           M["cfReplayVerdictChanges"] + M["cfReplayOwnedChanges"],
           sum(b["verdict_changes"] + b["owned_changes"] for b in rp_rows))
    expect("cfReplayMaxDW = the largest write change over the replays", M["cfReplayMaxDW"],
           sci(max(b["max_abs_dW"] for b in rp_rows)))
    expect("cfReplayDriftMax = the largest drift over the replays", M["cfReplayDriftMax"],
           f"{max(b['drift_max_abs'] for b in rp_rows):.2f}")
    expect("cfReplayDriftZero = replays with zero drift", M["cfReplayDriftZero"],
           sum(b["drift_max_abs"] == 0.0 for b in rp_rows))
    expect("cfReplayTensorMin/Max = the mean absolute tensor entry", (M["cfReplayTensorMin"], M["cfReplayTensorMax"]),
           (f"{min(b['block_mean_abs'] for b in rp_rows):.2f}", f"{max(b['block_mean_abs'] for b in rp_rows):.2f}"))
    expect("cfReplayPairs = the reader-pair records", M["cfReplayPairs"], len(pr_rows))
    expect("cfReplayPairOwnMin/Max = the own-tower differences", (M["cfReplayPairOwnMin"], M["cfReplayPairOwnMax"]),
           (f"{min(p['mean_abs_own_towers'] for p in pr_rows):.3f}", f"{max(p['mean_abs_own_towers'] for p in pr_rows):.3f}"))
    expect("cfReplayChangeMax = the largest absolute change", M["cfReplayChangeMax"],
           f"{max(abs(p['mean_abs_change']) for p in pr_rows):.4f}")
    expect("cfReplayRelMax = the largest absolute relative change", M["cfReplayRelMax"],
           f"{max(abs(100 * p['mean_abs_change'] / p['mean_abs_own_towers']) for p in pr_rows):.1f}")
    expect("cfReplayPairMoved/Cells = the pair moved counts", (M["cfReplayPairMoved"], M["cfReplayPairCells"]),
           (sum(p["n_nonzero_exact"] for p in pr_rows), sum(p["n_concepts"] for p in pr_rows)))

    se = (ROOT / "tables" / "table_cf_semend.tex").read_text()
    S = r3.semend(wb)
    tot = re.search(r"^\\textit\{all\} & \\textit\{(\d+) blocks\} & \\textit\{(\d+) endpoints\} & (\d+)/(\d+) & (\d+)/(\d+) & "
                    r"(\d+)/(\d+) & (\d+)/(\d+) \\\\$", se, flags=re.M)
    expect("Table cf-semend totals row present", bool(tot), True)
    if tot:
        g = tot.groups()
        expect("cfSemBlocks/Endpoints/Cells", (M["cfSemBlocks"], M["cfSemEndpoints"], M["cfSemCells"]),
               (int(g[0]), int(g[1]), int(g[3])))
        expect("cfSemLabelOwned/Ref and cfSemAnswerOwned/Ref",
               (M["cfSemLabelOwned"], M["cfSemLabelRef"], M["cfSemAnswerOwned"], M["cfSemAnswerRef"]),
               (int(g[2]), int(g[4]), int(g[6]), int(g[8])))
    ep_rows = re.findall(r"^(.+?) & (?:NIH ChestX-ray14|CheXpert Plus) & (?:negated question|forced choice, finding first|"
                         r"forced choice, competitor first|report continuation) & (\d+)/(\d+) & (\d+)/(\d+) & (\d+)/(\d+) & "
                         r"(\d+)/(\d+) \\\\$", se, flags=re.M)
    expect("Table cf-semend endpoint rows = blocks x endpoints", len(ep_rows), M["cfSemBlocks"] * M["cfSemEndpoints"])
    expect("cfSemLabelOwned = the endpoint rows", M["cfSemLabelOwned"], sum(int(r[1]) for r in ep_rows))
    expect("cfSemAnswerOwned = the endpoint rows", M["cfSemAnswerOwned"], sum(int(r[5]) for r in ep_rows))
    t_rows = re.findall(r"^(.+?) & (?:NIH ChestX-ray14|CheXpert Plus) & (\d+)/(\d+) & ([\d.]+) & (\d+)/(\d+) & (\d+)/(\d+) & "
                        r"(\d+)/(\d+) \\\\$", se, flags=re.M)
    expect("Table cf-semend test rows = cfSemBlocks", len(t_rows), M["cfSemBlocks"])
    expect("cfSemNegOpposite/N = the negation column", (M["cfSemNegOpposite"], M["cfSemNegN"]),
           (sum(int(r[1]) for r in t_rows), sum(int(r[2]) for r in t_rows)))
    expect("cfSemForcedOwned/N = the forced-choice column", (M["cfSemForcedOwned"], M["cfSemForcedN"]),
           (sum(int(r[4]) for r in t_rows), sum(int(r[5]) for r in t_rows)))
    expect("cfSemGapMin/Max = the order-gap column", (M["cfSemGapMin"], M["cfSemGapMax"]),
           (min(r[3] for r in t_rows), max(r[3] for r in t_rows)))
    expect("cfSemReportOwned/Cells = the report column", (M["cfSemReportOwned"], M["cfSemReportCells"]),
           (sum(int(r[6]) for r in t_rows), sum(int(r[7]) for r in t_rows)))
    # the in-sample panel is a reference only: it carries no interval column and the prose may not quote it as evidence
    iv_rows = re.findall(r"^(.+?) & (?:NIH ChestX-ray14|CheXpert Plus) & (label|answer) direction & (-?[\d.]+) & "
                         r"(-?[\d.]+) & (-?[\d.]+) & \\\\$", se, flags=re.M)
    expect("Table cf-semend in-sample rows = cfSemIVN", len(iv_rows), M["cfSemIVN"])
    expect("cfSemIVMin/Max = the in-sample added-variance column", (M["cfSemIVMin"], M["cfSemIVMax"]),
           (min(r[4] for r in iv_rows), max(r[4] for r in iv_rows)))
    expect("cfSemIVBaseMin/Max = the in-sample base column", (M["cfSemIVBaseMin"], M["cfSemIVBaseMax"]),
           (min(r[2] for r in iv_rows), max(r[2] for r in iv_rows)))
    for fam, F in (("label", "Label"), ("answer", "Answer")):
        xs = [r for r in iv_rows if r[1] == fam]
        expect(f"cfSemIV{F}Min/Max", (M[f"cfSemIV{F}Min"], M[f"cfSemIV{F}Max"]), (min(r[4] for r in xs), max(r[4] for r in xs)))
    # the held-out panel: the two splits, their pooled rows, and the macros the prose actually quotes
    cv_rows = re.findall(r"^(.+?) & (?:NIH ChestX-ray14|CheXpert Plus) & (label|answer) direction & (-?[\d.]+) & "
                         r"(-?[\d.]+) & (-?[\d.]+) & ([\d.]+) \\\\$", se, flags=re.M)
    lb_rows = re.findall(r"^\\textit\{all (\d+)\} & \\textit\{pooled\} & (label|answer) direction & (-?[\d.]+) & "
                         r"(-?[\d.]+) & (-?[\d.]+) & ([\d.]+) \\\\$", se, flags=re.M)
    expect("Table cf-semend held-out concept rows = cfSemCvLocoTests", len(cv_rows), M["cfSemCvLocoTests"])
    expect("Table cf-semend held-out block rows = cfSemCvLoboTests", len(lb_rows), M["cfSemCvLoboTests"])
    expect("cfSemCvLoboFolds = cfSemCvBlocks (the block split has one fold per block)",
           M["cfSemCvLoboFolds"], M["cfSemCvBlocks"])
    expect("the block split pools every block", {int(r[0]) for r in lb_rows}, {M["cfSemCvBlocks"]})
    pooled = [m.groups() for m in re.finditer(
        r"^\\textit\{pooled\} & \\textit\{(\d+) tests\} & \\textit\{both\} & -- & -- & (-?[\d.]+) & ([\d.]+) \\\\$",
        se, flags=re.M)]
    expect("Table cf-semend carries a pooled row for each split", len(pooled), 2)
    for row, K in zip(pooled, ("Loco", "Lobo")):
        signed = lambda x: ("+" + x) if not x.startswith("-") else x
        expect(f"cfSemCv{K}Delta/P/Tests = the pooled row of that split",
               (M[f"cfSemCv{K}Delta"], M[f"cfSemCv{K}P"], M[f"cfSemCv{K}Tests"]),
               (signed(row[1]), row[2], int(row[0])))
    per_p = [float(r[5]) for r in cv_rows] + [float(r[5]) for r in lb_rows]
    expect("every held-out test clears the permutation null (Section 5.6 says none reaches p < 0.05)",
           [p for p in per_p if p < 0.05], [])

    fg = (ROOT / "tables" / "table_cf_fgobj.tex").read_text()
    F = r3.fgobj(wb)
    fg_rows = re.findall(r"^(.+?) & (\d) & (\d) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\$", fg, flags=re.M)
    expect("Table cf-fgobj block rows = cfFgBlocks", len(fg_rows), M["cfFgBlocks"])
    expect("cfFgFineOwned = the per-block fine column", M["cfFgFineOwned"], sum(int(r[1]) for r in fg_rows))
    expect("cfFgEasySameOwned = the per-block easy column of the same blocks", M["cfFgEasySameOwned"],
           sum(int(r[2]) for r in fg_rows))
    pool = {m.group(1): m.groups() for m in re.finditer(
        r"^(chest findings|easy objects, every COCO block|easy objects, these blocks|"
        r"fine-grained objects, these blocks) & (\d+) & (\d+)/(\d+) & ([\d.]+) & ([\d.]+) & "
        r"([\d.]+) & (\d+)/(\d+) \\\\$", fg, flags=re.M)}
    for g, G in (("chest findings", "Chest"), ("easy objects, every COCO block", "Easy"),
                 ("easy objects, these blocks", "EasySame"),
                 ("fine-grained objects, these blocks", "Fine")):
        row = pool[g]
        expect(f"cfFg{G} blocks/owned/cells/rate/AUROC/selectivity",
               (M[f"cfFg{G}Blocks"], M[f"cfFg{G}Owned"], M[f"cfFg{G}Cells"], M[f"cfFg{G}Rate"], M[f"cfFg{G}Auroc"], M[f"cfFg{G}Sel"]),
               (int(row[1]), int(row[2]), int(row[3]), row[4], row[5], row[6]))
    expect("cfFgEasySameBlocks = cfFgBlocks (the same checkpoints as the fine-grained cells)",
           M["cfFgEasySameBlocks"], M["cfFgBlocks"])
    expect("cfFgSameSignP present in the paired line of Table cf-fgobj",
           bool(re.search(r"sign test \$p=" + re.escape(str(M["cfFgSameSignP"])) + r"\$", fg)), True)
    # the matched panel carries TWO labelled estimators per matching, each with its own interval
    mt = {}
    for m in re.finditer(
            r"^(probe selectivity alone|clean-answer AUROC alone|both variables|both variables, easy partners only) & "
            r"(\d+)/(\d+) & (\d+)/(\d+) & ([\d.]+) & ([\d.]+) &  &  \\\\\n"
            r"\\quad pooled &  &  &  &  & (-?[\d.]+) & \[(-?[\d.]+), (-?[\d.]+)\] \\\\\n"
            r"\\quad matched &  &  &  &  & (-?[\d.]+) & \[(-?[\d.]+), (-?[\d.]+)\] \\\\$", fg, flags=re.M):
        mt[m.group(1)] = m.groups()
    expect("Table cf-fgobj reports both estimators for every matching", sorted(mt), sorted(MATCHINGS_IN_TABLE))
    for lab, K in (("probe selectivity alone", "Sel"), ("clean-answer AUROC alone", "Ans"), ("both variables", "Both")):
        row = mt[lab]
        signed = lambda x: ("+" + x) if not x.startswith("-") else x
        expect(f"cfFgMatch{K} chest cells / partners / the two rates",
               (M[f"cfFgMatch{K}N"], M[f"cfFgMatch{K}Cells"], M[f"cfFgMatch{K}Partners"], M[f"cfFgMatch{K}PartnersN"],
                M[f"cfFgMatch{K}Chest"], M[f"cfFgMatch{K}Nat"]),
               (int(row[1]), int(row[2]), int(row[3]), int(row[4]), row[5], row[6]))
        expect(f"cfFgMatch{K} pooled difference and interval",
               (M[f"cfFgMatch{K}Pooled"], M[f"cfFgMatch{K}PooledLo"], M[f"cfFgMatch{K}PooledHi"]),
               (signed(row[7]), signed(row[8]), signed(row[9])))
        expect(f"cfFgMatch{K} matched difference and interval",
               (M[f"cfFgMatch{K}Diff"], M[f"cfFgMatch{K}Lo"], M[f"cfFgMatch{K}Hi"]),
               (signed(row[10]), signed(row[11]), signed(row[12])))
        # the pooled estimand IS the difference of the two printed rates; round2.json guarantees that exactly
        # (cf_round3.fgobj refuses to build otherwise), so the table only has to agree to the last printed place
        expect(f"cfFgMatch{K}Pooled = chest rate minus natural rate, to the last printed place",
               abs(round(float(M[f"cfFgMatch{K}Pooled"]) - float(row[5]) + float(row[6]), 6)) <= 0.001, True)
    expect("cfFgMatchSelN = every chest cell (the prose says all of them match)", M["cfFgMatchSelN"], M["cfFgChestCells"])

    # ---- the stratified panels of Table cf-fgobj and Table cf-attr, recomputed here from the raw cells
    # The bins are re-derived from runs/robustness/round2.json's own cell lists rather than read from its stratified
    # section, so a reviewer's recount of the artefacts and the panel have to meet.
    ST = r3.stratified()
    edges, top = ST["bin_edges"], ST["top_bin_index"]

    def rebin(cells):
        """(owned, cells) per bin, recomputed from a raw cell list; the last bin is closed on the right."""
        out = []
        for i in range(len(edges) - 1):
            sel = [c for c in cells if c["answer_auroc"] is not None
                   and (edges[i] <= c["answer_auroc"] < edges[i + 1]
                        or (i == len(edges) - 2 and c["answer_auroc"] >= edges[i]))]
            out.append((sum(bool(c["owned"]) for c in sel), len(sel)))
        return out

    raw = json.loads((ci.RUNS / "robustness" / "round2.json").read_text())
    fcells = raw["fgobj"]["cells"]
    acells = [{"dataset": b["dataset"], "owned": c["owned"], "answer_auroc": (c.get("answerability") or {}).get("answer_auroc")}
              for b in raw["attr"]["blocks"] for c in b["attributes"].values()]
    ccells = [{"dataset": b["dataset"], "owned": c["owned"], "answer_auroc": (c.get("answerability") or {}).get("answer_auroc")}
              for b in raw["attr"]["blocks"] for c in b["clinical"].values()]
    recomputed = {
        "nih": rebin([c for c in fcells if c["dataset"] == "nih"]),
        "chexpert": rebin([c for c in fcells if c["dataset"] == "chexpert"]),
        "chest": rebin([c for c in fcells if c["group"] == "chest"]),
        "coco_easy": rebin([c for c in fcells if c["group"] == "coco_easy"]),
        "coco_fine": rebin([c for c in fcells if c["group"] == "coco_fine"]),
        "natural": rebin([c for c in fcells if c["group"] in ("coco_easy", "coco_fine")]),
        "attribute_nih": rebin([c for c in acells if c["dataset"] == "nih"]),
        "attribute_chexpert": rebin([c for c in acells if c["dataset"] == "chexpert"]),
        "attribute_chest": rebin(acells),
        "finding_nih": rebin([c for c in ccells if c["dataset"] == "nih"]),
        "finding_chexpert": rebin([c for c in ccells if c["dataset"] == "chexpert"]),
        "finding_chest": rebin(ccells)}
    STRAT_ROW = re.compile(r"^(.+?) & (\d+)/(\d+) & (\d+)/(\d+) & (\d+)/(\d+) & (\d+)/(\d+) & (\d+)/(\d+) & (\d+)/(\d+) \\\\$", re.M)
    strat_src = fg + "".join((ROOT / "tables" / f"{t}.tex").read_text()
                             for t in ("table_cf_attr", "table_cf_attr_strat"))   # the attribute stratification is its own table
    panel = {m.group(1): [(int(m.group(2 * i + 2)), int(m.group(2 * i + 3))) for i in range(6)]
             for m in STRAT_ROW.finditer(strat_src)}
    print("stratified panels vs a recount of the raw cells:")
    for g, lab, G in (("nih", "NIH ChestX-ray14", "Nih"), ("chexpert", "CheXpert Plus", "Chex"),
                      ("chest", r"\textit{chest findings}", "Chest"),
                      ("coco_easy", "easy objects, every COCO block", "Easy"),
                      ("coco_fine", "fine-grained objects, these blocks", "Fine"),
                      ("natural", r"\textit{natural-image objects}", "Nat"),
                      ("attribute_nih", "attributes, NIH ChestX-ray14", "AttrNih"),
                      ("attribute_chexpert", "attributes, CheXpert Plus", "AttrChex"),
                      ("attribute_chest", r"\textit{attributes, chest}", "Attr"),
                      ("finding_nih", "findings, NIH ChestX-ray14", "FindNih"),
                      ("finding_chexpert", "findings, CheXpert Plus", "FindChex"),
                      ("finding_chest", r"\textit{findings, chest}", "Find")):
        want = recomputed[g] + [(sum(o for o, _ in recomputed[g]), sum(n for _, n in recomputed[g]))]
        expect(f"stratified panel row {g}", panel.get(lab), want)
        expect(f"cfStrat{G} top bin and totals",
               (M[f"cfStrat{G}TopOwned"], M[f"cfStrat{G}TopN"], M[f"cfStrat{G}Owned"], M[f"cfStrat{G}Cells"]),
               (want[top][0], want[top][1], want[-1][0], want[-1][1]))
        if want[top][1]:
            expect(f"cfStrat{G}TopPct = the top-bin rate", M[f"cfStrat{G}TopPct"], pct(want[top][0], want[top][1]))
    # the stratified pools are the pools of the panels above them, and of the ATTR comparison
    expect("stratified chest pool = the FGOBJ chest pool", (M["cfStratChestOwned"], M["cfStratChestCells"]),
           (M["cfFgChestOwned"], M["cfFgChestCells"]))
    expect("stratified natural pool = the two COCO pools together", (M["cfStratNatOwned"], M["cfStratNatCells"]),
           (M["cfFgEasyOwned"] + M["cfFgFineOwned"], M["cfFgEasyCells"] + M["cfFgFineCells"]))
    expect("stratified attribute / finding pools = the ATTR pools",
           (M["cfStratAttrOwned"], M["cfStratAttrCells"], M["cfStratFindOwned"], M["cfStratFindCells"]),
           (M["cfAttrAllAttrOwned"], M["cfAttrAllAttrN"], M["cfAttrAllClinOwned"], M["cfAttrAllClinN"]))
    expect("stratified NIH + CheXpert = chest", (M["cfStratNihCells"] + M["cfStratChexCells"]), M["cfStratChestCells"])
    expect("no well-answered attribute cell is owned (Sections 5.4, 5.8 and the conclusion say none is)",
           M["cfStratAttrTopOwned"], 0)
    expect("the top-bin contrast line of Table cf-fgobj carries cfStratCvnDiff and its interval",
           bool(re.search(re.escape(M["cfStratCvnDiff"].lstrip("+")) + r", 95\\% interval \["
                          + re.escape(M["cfStratCvnLo"].lstrip("+")) + ", " + re.escape(M["cfStratCvnHi"].lstrip("+")) + r"\]", fg)), True)
    expect("cfStratTopEdge is the top bin edge of the fixed ladder", M["cfStratTopEdge"], f"{edges[top]:.2f}")

    # ---- Section 5.3 states the three numbers Figure 3 annotates, from the figure's own sidecar
    f3 = ROOT / "figures" / "fig3_example.json"
    if f3.exists():
        ex = json.loads(f3.read_text())
        print("Figure 3 macros vs the figure's sidecar:")
        for ds, K in (("nih", "Chest"), ("coco", "Coco")):
            d = ex[ds]
            expect(f"cfEx{K} clean / own write / competing write",
                   (M[f"cfEx{K}Clean"], M[f"cfEx{K}Own"], M[f"cfEx{K}Comp"]),
                   (f"{d['clean']:.2f}", f"{d['concept_write']:.2f}", f"{d['competitor_write']:.2f}"))
            expect(f"cfEx{K} question and competitor", (M[f"cfEx{K}Question"], M[f"cfEx{K}Competitor"]),
                   (d["question"], d["competitor"]))
        expect("the COCO concept write moves nothing else (Section 5.3 says so)",
               ex["coco"]["max_other_answer"] <= 0.05, True)
    # ---- the two-model seed study's own ownership contrast, quoted in Section 5.2 and in the registered gate table
    sp = json.loads((ROOT / "data" / "accepted_results.json").read_text())["specificity"]["primary"]
    expect("cfSeedPrior* = data/accepted_results.json specificity.primary",
           (M["cfSeedPriorO"], M["cfSeedPriorOLo"], M["cfSeedPriorOHi"]),
           (f"{sp['margin']:+.3f}", f"{sp['ci95'][0]:+.3f}", f"{sp['ci95'][1]:+.3f}"))
    gates = (ROOT / "tables" / "table_gates.tex").read_text()
    expect("the registered gate table carries the same seed ownership contrast",
           bool(re.search(r"-?" + re.escape(f"{sp['margin']:.4f}") + r" \[" + re.escape(f"{sp['ci95'][0]:.4f}")
                          + r", " + re.escape(f"{sp['ci95'][1]:.4f}") + r"\]", gates)), True)

    # the refit-arms panel of Table cf-valid: the label source separated from the sample size
    va = (ROOT / "tables" / "table_cf_valid.tex").read_text()
    arm_rows = {m.group(1) + "|" + m.group(2): m.groups() for m in re.finditer(
        r"^(radiologist|report-derived) & (200 valid films|the same 200 valid films|200 training rows|"
        r"200 labelled training rows|every training row) & (\d+) & ([\d.]+) & ([\d.]+) & (-?[\d.]+) & (-?[\d.]+) \\\\$",
        va, flags=re.M)}
    expect("the refit-arms panel has five arms", len(arm_rows), 5)
    for key, K in (("radiologist|200 valid films", "Expert"),
                   ("report-derived|the same 200 valid films", "ReportSameFilms"),
                   ("report-derived|200 training rows", "ReportSubCohort"),
                   ("report-derived|200 labelled training rows", "ReportSub"),
                   ("report-derived|every training row", "ReportFull")):
        row = arm_rows[key]
        expect(f"cfVfArm{K} rows / AUROC against both label sets / the two cosines",
               (M[f"cfVfArm{K}Rows"], M[f"cfVfArm{K}Auroc"], M[f"cfVfArm{K}AurocReport"],
                M[f"cfVfArm{K}Cos"], M[f"cfVfArm{K}CosWhite"]),
               (int(row[2]), row[3], row[4], row[5], row[6]))
    expect("cfVfArmReportSubRows = cfVfArmExpertRows (the size-matched arm fits as many rows as the expert refit)",
           M["cfVfArmReportSubRows"], M["cfVfArmExpertRows"])
    # the arms script recomputes the campaign's own two arms from the block fits; they must land on the numbers the
    # summaries already carry, or the added arms are not measuring the same thing
    expect("cfVfArmExpert* = cfValidfit* (the recomputed expert arm reproduces the shipped one)",
           (M["cfVfArmExpertAuroc"], M["cfVfArmExpertCos"], M["cfVfArmExpertCosWhite"], M["cfVfArmExpertCells"]),
           (M["cfValidfitAurocExpert"], M["cfValidfitCosRaw"], M["cfValidfitCosWhitened"], M["cfValidfitCells"]))
    expect("cfVfArmReportFullAuroc = cfValidfitAurocReportDir (the shipped direction, both ways)",
           M["cfVfArmReportFullAuroc"], M["cfValidfitAurocReportDir"])
    expect("cfVfArmRows = cfValidfitFitRows and cfVfArmFolds = cfValidfitFolds",
           (M["cfVfArmRows"], M["cfVfArmFolds"]), (M["cfValidfitFitRows"], M["cfValidfitFolds"]))

    # GUARD: the grid-wide easy-object pool and the same-block pool answer different questions, so no sentence may
    # quote a grid-wide easy macro beside a fine-grained one as if the two were the same comparison.
    grid_wide = ("cfFgEasyOwned", "cfFgEasyCells", "cfFgEasyRate", "cfFgEasyAuroc", "cfFgEasySel", "cfFgEasyBlocks")
    fine = ("cfFgFineOwned", "cfFgFineCells", "cfFgFineRate", "cfFgFineAuroc", "cfFgFineSel")
    same = tuple(n.replace("cfFgEasy", "cfFgEasySame") for n in grid_wide)
    clashes = []
    for path in sorted((ROOT / "sections").glob("*.tex")):
        src = path.read_text()
        for sent in re.split(r"(?<=[.;:])\s", src):
            has = lambda names: any(re.search(rf"\\{n}(?![A-Za-z])", sent) for n in names)
            if has(grid_wide) and has(fine):
                clashes.append(f"{path.name}: a grid-wide easy macro beside a fine-grained one")
            if has(grid_wide) and has(same):
                clashes.append(f"{path.name}: the grid-wide and the same-block easy macros in one sentence")
    expect("no sentence mixes the grid-wide easy-object pool with the fine-grained comparison", clashes, [])

    # the attribute comparison read three ways, in the bottom panel of Table cf-attr
    am = {m.group(1): m.groups() for m in re.finditer(
        r"^(every cell|answer-capable cells only|answerability-matched pairs) & (cell|pair) & (\d+) & (\d+) & (\d+) & (\d+) & "
        r"(-?[\d.]+) \\\\$", (ROOT / "tables" / "table_cf_attr.tex").read_text(), flags=re.M)}
    for lab, K in (("every cell", "All"), ("answer-capable cells only", "Cap"), ("answerability-matched pairs", "Pair")):
        row = am[lab]
        expect(f"cfAttr{K} attribute/clinical cells and owned",
               (M[f"cfAttr{K}AttrN"], M[f"cfAttr{K}AttrOwned"], M[f"cfAttr{K}ClinN"], M[f"cfAttr{K}ClinOwned"]),
               (int(row[2]), int(row[3]), int(row[4]), int(row[5])))
    expect("cfAttrAllAttrOwned/N = cfAttrChestAttrOwned/N (the same pool, two macros)",
           (M["cfAttrAllAttrOwned"], M["cfAttrAllAttrN"], M["cfAttrAllClinOwned"], M["cfAttrAllClinN"]),
           (M["cfAttrChestAttrOwned"], M["cfAttrChestAttrN"], M["cfAttrChestClinOwned"], M["cfAttrChestClinN"]))
    expect("cfAttrqBlocks = cfAttrBlocks (the phrasings are scored in every attribute block)", M["cfAttrqBlocks"], M["cfAttrBlocks"])
    expect("cfAttrRefBlocks = cfAttrBlocks (one steering reference in every block)", M["cfAttrRefBlocks"], M["cfAttrBlocks"])
    # the abstract and the results section state the reference decomposition in order, from macros: reference met, of
    # how many cells, owned, stronger competitor, unresolved, and the share of stronger-competitor verdicts arising
    # below the reference. The decomposition is stated once, in the results; the abstract carries the two counts that
    # open it, and the introduction no longer restates either.
    print("abstract and results decomposition (macros, in order):")
    order = ["cfChestCellsN", "cfChestRefMet", "cfChestOwned", "cfChestRefStrong", "cfChestRefUnres",
             "cfChestBelowStrong", "cfChestCompetitor"]
    src = (ROOT / "sections" / "0_abstract.tex").read_text()
    expect("0_abstract: reference count and owned count present",
           [m for m in ("cfChestRefMet", "cfChestOwned") if not re.search(rf"\\{m}(?![A-Za-z])", src)], [])
    intro = (ROOT / "sections" / "1_introduction.tex").read_text()
    expect("1_introduction: the decomposition is not restated before the method",
           [m for m in ("cfChestRefMet", "cfChestRefStrong", "cfChestRefUnres", "cfChestBelowStrong")
            if re.search(rf"\\{m}(?![A-Za-z])", intro)], [])
    src = (ROOT / "sections" / "5_results.tex").read_text()
    para = [p for p in src.split("\n\n") if re.search(r"\\cfChestRefMet(?![A-Za-z])", p)]
    expect("5_results: the decomposition is stated in exactly one paragraph", len(para), 1)
    if len(para) == 1:
        at = [(lambda g: g.start() if g else -1)(re.search(rf"\\{m}(?![A-Za-z])", para[0])) for m in order]
        expect("5_results: every decomposition macro present", [m for m, p in zip(order, at) if p < 0], [])
        expect("5_results: cells before reference before ownership before the below-reference share",
               [p for p in at if p >= 0] == sorted(p for p in at if p >= 0), True)
    # ---- the reference-and-rule ledger (Table cf-ledger): the table's own rows against the macros the prose reads,
    #      and both against the ledger the artefacts produce, so a row cannot say one thing and the prose another
    led = (ROOT / "tables" / "table_cf_ledger.tex").read_text()
    body = led.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    lrows = [ln for ln in body.splitlines() if ln.strip().endswith(r"\\")]
    print("ledger table vs macros:")
    expect("ledger: rows", len(lrows), int(M["cfLedgerAnalyses"]))
    cells = [int(ln.rstrip("\\ ").rsplit("&", 1)[1].strip().replace("{,}", "")) for ln in lrows]
    expect("ledger: cells", sum(cells), int(str(M["cfLedgerCells"]).replace("{,}", "")))
    expect("ledger: analyses with the seed-0 sham", sum("the seed-0 normal" in ln for ln in lrows), int(M["cfLedgerSeedSham"]))
    expect("ledger: cells graded against the seed-0 sham",
           sum(c for c, ln in zip(cells, lrows) if "the seed-0 normal" in ln), int(str(M["cfLedgerSeedShamCells"]).replace("{,}", "")))
    expect("ledger: analyses decided by a percentile interval",
           sum("percentile interval on" in ln for ln in lrows), int(M["cfLedgerPercentile"]))
    expect("ledger: analyses decided by simultaneous bounds",
           sum("simultaneous max-$T$," in ln for ln in lrows), int(M["cfLedgerMaxT"]))
    expect("ledger: analyses with a permutation sham",
           sum("the direction written" in ln for ln in lrows), int(M["cfLedgerOwnSham"]))
    expect("ledger: the held-out-template split",
           f"{M['cfAnsdirtRefPairs']} pairs" in led and f"in {M['cfAnsdirtShamPairs']}" in led, True)

    # ---- the two host blocks of a dataset index the same four arms, so their crossovers must agree
    print("crossover, the two host blocks of a dataset:")
    expect("crossover: the two host blocks of a dataset give the same crossover",
           [(F, D, qk) for F in ("Reader", "Tower") for D in ("Nih", "Chex", "Coco") for qk in ("", "O", "M")
            if M[f"cfSwap{F}{D}{qk}Min"] != M[f"cfSwap{F}{D}{qk}Max"]], [])

    # rendered prose
    pdf = ROOT / "main.pdf"
    if pdf.exists() and shutil.which("pdftotext"):
        txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
        # The preprint layout prints no margin line numbers, so nothing is stripped from the start of a
        # line: a leading number there is a table value, not a line label.
        flat = re.sub(r"\s+", " ", txt)
        print("rendered PDF vs macros:")
        for phrase in (
                       f"readable in {M['cfChestReadable']} of {M['cfChestReadableN']} cells ({M['cfChestReadablePct']}%)",
                       f"{M['cfChestAnswerable']} of {M['cfChestAnswerableN']} cells ({M['cfChestAnswerablePct']}%)",
                       f"Yet only {M['cfChestOwned']} cells ({M['cfChestOwnedPct']}%) are owned",
                       f"In {M['cfChestCompetitor']} cells ({M['cfChestCompetitorPct']}%), the verdict",
                       f"owns {M['cfCocoOwnedPct']}% of cells on COCO",
                       f"{M['cfCocoOwned']} of {M['cfCocoOwnedN']} cells ({M['cfCocoOwnedPct']}%)",
                       ):
            found = phrase in flat or phrase.replace("–", "-") in flat
            expect(f"prose: {phrase[:60]}...", found, True)
        # The decomposition sentences are rewritten by the prose pass, so they are checked as an ordered number
        # sequence inside a bounded window instead of as a fixed string.
        for name, nums in (("abstract readable / answerable / owned",
                            (M["cfChestReadablePct"], M["cfChestAnswerablePct"], M["cfChestOwnedPct"])),
                           ("chest decomposition", (M["cfChestAnswerableN"], M["cfChestRefMet"], M["cfChestRefMet"], M["cfChestOwned"],
                                                    M["cfChestRefStrong"], M["cfChestRefUnres"])),
                           ("below-reference share", (M["cfChestBelowStrong"], M["cfChestCompetitor"], M["cfChestBelowStrongPct"])),
                           ("readable-and-answerable decomposition", (M["cfReadAns"], M["cfReadAnsRefMet"], M["cfReadAnsOwned"],
                                                                      M["cfReadAnsRefStrong"], M["cfReadAnsRefUnres"],
                                                                      M["cfReadAnsCompetitor"], M["cfReadAnsBelowStrong"], M["cfReadAnsRefBelow"]))):
            pat = r"[^0-9]{0,90}".join(rf"\b{re.escape(str(n))}\b" for n in nums)
            expect(f"prose: {name} {nums}", bool(re.search(pat, flat)), True)
    print("ALL CONSISTENT" if bad == 0 else f"{bad} DISAGREEMENT(S)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
