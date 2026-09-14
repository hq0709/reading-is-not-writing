"""Build the cf-transfer-v1 LaTeX tables for the paper from the packaged campaign results.

Inputs (read-only): the campaign run root (leaderboard.csv, runs/<model>/<dataset>/summary.json, run.json).
Outputs: tables/table_cf_leaderboard.tex (main text) and tables/table_cf_<dataset>.tex (appendix, one row per
checkpoint x concept with S, answer AUROC, W_qq, O_q and the verdict). Numbers are copied, never recomputed.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/rodata/azradonc_dev/m253405/cf-transfer/runs")
OUT = Path(__file__).resolve().parents[1] / "tables"
DATASETS = [("nih", "NIH ChestX-ray14"), ("chexpert", "CheXpert Plus"), ("coco", "COCO")]
NAMES = {"q25-3": "Qwen2.5-VL-3B", "q25-7": "Qwen2.5-VL-7B", "q25-32": "Qwen2.5-VL-32B", "q25-72": "Qwen2.5-VL-72B",
         "q3-4": "Qwen3-VL-4B", "q3-8": "Qwen3-VL-8B", "q3-32": "Qwen3-VL-32B",
         "iv35-8": "InternVL3.5-8B", "iv35-14": "InternVL3.5-14B", "iv35-38": "InternVL3.5-38B",
         "gemma3-4": "Gemma 3 4B", "gemma3-12": "Gemma 3 12B", "gemma3-27": "Gemma 3 27B",
         "medgemma-4": "MedGemma 4B", "medgemma-27": "MedGemma 27B",
         "lingshu-7": "Lingshu 7B", "lingshu-32": "Lingshu 32B",
         "llava15-7": "LLaVA-1.5-7B", "llava15-13": "LLaVA-1.5-13B", "llavamed-7": "LLaVA-Med 1.5 7B",
         "llama32-11": "Llama 3.2 Vision 11B", "llama32-90": "Llama 3.2 Vision 90B"}
ORDER = list(NAMES)


def tex(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%")


def leaderboard():
    rows = {(r["model_key"], r["dataset_id"]): r for r in csv.DictReader((RUNS / "leaderboard.csv").open())}
    models = [m for m in ORDER if any((m, d) in rows for d, _ in DATASETS)]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{\textbf{Read, answer, write.} For every checkpoint and dataset: number of the six concepts that are "
             r"readable at the consumed visual block (probe selectivity lower bound $>0$), answer-capable on the clean yes/no "
             r"question (AUROC lower bound $>0.5$), and owned by their concept write (steering reference and positive "
             r"simultaneous ownership bounds). ``inel.'' marks checkpoints whose yes/no template fails the image-free mapping "
             r"preflight, where ownership is undefined; ``--'' marks blocks not run.}",
             r"\label{tab:cf-leaderboard}",
             r"\begin{tabular}{l" + "ccc" * 3 + "}", r"\toprule",
             r"& \multicolumn{3}{c}{NIH ChestX-ray14} & \multicolumn{3}{c}{CheXpert Plus} & \multicolumn{3}{c}{COCO (control)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Checkpoint & read & answer & own & read & answer & own & read & answer & own \\", r"\midrule"]
    tot = {d: [0, 0, 0, 0] for d, _ in DATASETS}
    for m in models:
        cells = []
        for d, _ in DATASETS:
            r = rows.get((m, d))
            if not r:
                cells += ["--", "--", "--"]; continue
            if r["owned"] == "":
                cells += [r["readable"], "--", "inel."]
            else:
                own = int(r["owned"]); cells += [r["readable"], r["answer_capable"] or "--", (r"\textbf{%d}" % own) if own else "0"]
                tot[d][0] += int(r["readable"]); tot[d][1] += int(r["answer_capable"] or 0); tot[d][2] += own; tot[d][3] += 6
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", "Cells (share) & " + " & ".join(
        f"{tot[d][0]} ({100 * tot[d][0] / max(tot[d][3], 1):.0f}\\%) & {tot[d][1]} ({100 * tot[d][1] / max(tot[d][3], 1):.0f}\\%) & {tot[d][2]} ({100 * tot[d][2] / max(tot[d][3], 1):.0f}\\%)"
        for d, _ in DATASETS) + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_cf_leaderboard.tex").write_text("\n".join(lines) + "\n")
    return tot


def per_dataset(ds: str, label: str):
    lines = [r"\begin{longtable}{llrrrrrl}",
             r"\caption{\textbf{" + label + r": per-concept results of the external replication.} $S$: probe selectivity; "
             r"ans.\ AUROC: clean-answer AUROC; $W_{qq}$: mean change in $P(\mathrm{yes})$ under the concept write; "
             r"$O_q$: ownership contrast; p95: 95th percentile of the 119 random-direction writes; verdict from the "
             r"simultaneous max-$T$ bounds (own = steering reference met and every lower bound $>0$).}\\",
             r"\label{tab:cf-" + ds + r"}\\", r"\toprule",
             r"Checkpoint & Concept & $S$ & ans.\ AUROC & $W_{qq}$ & $O_q$ & p95 & Verdict \\", r"\midrule", r"\endfirsthead",
             r"\toprule", r"Checkpoint & Concept & $S$ & ans.\ AUROC & $W_{qq}$ & $O_q$ & p95 & Verdict \\", r"\midrule", r"\endhead"]
    for m in ORDER:
        s = RUNS / m / ds / "summary.json"
        if not s.exists():
            continue
        j = json.loads(s.read_text()); cal = j.get("calibration", {}); core = (j.get("core") or {}).get("per_question")
        run = json.loads((RUNS / m / ds / "run.json").read_text()) if (RUNS / m / ds / "run.json").exists() else {}
        if not core:
            if cal:
                for c, v in cal.items():
                    if isinstance(v, dict):
                        lines.append(f"{NAMES[m]} & {c} & {v.get('selectivity', float('nan')):.3f} & -- & -- & -- & -- & ineligible \\\\")
            continue
        for c, v in core.items():
            cc = cal.get(c, {}) if isinstance(cal.get(c, {}), dict) else {}
            own = v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage"
            verdict = "own" if own else {"stronger_competitor": "competitor", "unresolved": "unresolved", "fixed_family_advantage": "no ref."}.get(v.get("verdict"), "--")
            aa = cc.get("answer_auroc"); aa = f"{aa:.3f}" if isinstance(aa, (int, float)) and aa == aa else "--"
            lines.append(f"{NAMES[m]} & {c} & {cc.get('selectivity', float('nan')):.3f} & {aa} & {v['W_qq']:.3f} & "
                         f"{v['O_q']:.3f} & {v['random_p95']:.3f} & {verdict} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    (OUT / f"table_cf_{ds}.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    tot = leaderboard()
    for ds, label in DATASETS:
        per_dataset(ds, label)
    print({d: t for d, t in tot.items()})
