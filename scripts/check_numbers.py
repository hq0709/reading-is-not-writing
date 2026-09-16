"""Consistency check: manifest counts == Table 1 "All cells" row == Figure 2(a) counts == cf_numbers macros == prose.

    python scripts/check_numbers.py        (exit 1 on any disagreement)

Reads runs/manifest.csv (cf_inclusion.read_manifest), tables/table_cf_main.tex, figures/fig2_counts.json (written by
plot_paper_figures.py fig2_overview), figures/figA1_own_share.json (figA1_write_structure), tables/cf_numbers.json
(build_numbers.py), runs/robustness/round2.json with tables/table_cf_valid.tex and tables/table_cf_attr.tex (round-2 and ATTR macros), and, when main.pdf exists and pdftotext is available, the rendered abstract and Section 5.1.
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
from build_numbers import counts, pct   # noqa: E402

ROOT = HERE.parent
DS = ("nih", "chexpert", "coco")
bad = 0


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
    # Table 1 "All cells" row
    tex = (ROOT / "tables" / "table_cf_main.tex").read_text()
    row = re.search(r"\\textbf\{All cells\} & (.*?) \\\\", tex).group(1)
    sup = re.findall(r"\^\{(\d+)/(\d+)\}", row)          # nine (count, denominator) pairs: Read/Ans/Own x 3 datasets
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
    # EXTCOMP macros against Table cf-extcomp
    et = (ROOT / "tables" / "table_cf_extcomp.tex").read_text()
    er = re.findall(r"& (\d+)/6 & (\d+)/6 \\\\", et)
    if er:
        print("EXTCOMP macros vs Table cf-extcomp:")
        expect("cfExtcompBlocks", M["cfExtcompBlocks"], len(er))
        expect("cfExtcompOwned", M["cfExtcompOwned"], sum(int(a) for a, _ in er))
        expect("cfExtcompExtOwned", M["cfExtcompExtOwned"], sum(int(b) for _, b in er))
        expect("cfExtcompRetained within both counts", M["cfExtcompRetained"] <= min(M["cfExtcompOwned"], M["cfExtcompExtOwned"]), True)
    a1 = ROOT / "figures" / "figA1_own_share.json"
    if a1.exists():
        s = json.loads(a1.read_text())
        for ds, D in (("nih", "Nih"), ("chexpert", "Chex"), ("coco", "Coco")):
            expect(f"own share {ds} (figA1 vs macro)", pct(s[ds]["own_share"], 1), M[f"cfOwnShare{D}"])
    # rendered prose
    pdf = ROOT / "main.pdf"
    if pdf.exists() and shutil.which("pdftotext"):
        txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
        txt = "\n".join(re.sub(r"^\s*\d{1,4}(?=\s{2,}|$)", "", ln) for ln in txt.splitlines())   # ICLR margin line numbers
        flat = re.sub(r"\s+", " ", txt)
        print("rendered PDF vs macros:")
        for phrase in (f"readable in {M['cfChestReadablePct']}% of checkpoint–concept cells, answerable in {M['cfChestAnswerablePct']}%, and owned in only {M['cfChestOwnedPct']}%",
                       f"readable in {M['cfChestReadable']} of {M['cfChestReadableN']} cells ({M['cfChestReadablePct']}%)",
                       f"{M['cfChestAnswerable']} of the {M['cfChestAnswerableN']} cells with a scored write matrix ({M['cfChestAnswerablePct']}%)",
                       f"Yet only {M['cfChestOwned']} cells ({M['cfChestOwnedPct']}%) are owned",
                       f"In {M['cfChestCompetitor']} cells ({M['cfChestCompetitorPct']}%), the verdict",
                       f"owns {M['cfCocoOwnedPct']}% of cells",
                       f"Objects are owned in {M['cfCocoOwned']} of {M['cfCocoOwnedN']} cells ({M['cfCocoOwnedPct']}%)"):
            found = phrase in flat or phrase.replace("–", "-") in flat
            expect(f"prose: {phrase[:60]}...", found, True)
    print("ALL CONSISTENT" if bad == 0 else f"{bad} DISAGREEMENT(S)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
