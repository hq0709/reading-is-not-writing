"""The one block-inclusion rule of the paper, shared by every table, figure, and number script.

Mirrors cftransfer.manifest in the code repository (src/cftransfer/manifest.py), which writes runs/manifest.csv
with the same rule; build_numbers.py cross-checks the two.

    block_included(run)  =  "CORE" in run.json completed_modules  and  "CALIBRATION" in run.json completed_modules
    probe_graded(run)    =  "CALIBRATION" in run.json completed_modules

Only included blocks enter the write-matrix counts (answerable, owned, stronger competitor, steering reference,
median W_qq, write matrices, dose, refit, altdir, ansdir, extcomp, tokenw). Probe-graded blocks that are not
included (CORE ineligible or incomplete) contribute readability only where the paper says "probe-graded": the
Read column of Table 1 and the readable count of Figure 2a. Their partial core statistics are never read.
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("/rodata/azradonc_dev/m253405/cf-transfer/runs")
INCLUSION_MODULES = ("CORE", "CALIBRATION")
PROBE_MODULE = "CALIBRATION"
DATASETS = ("nih", "chexpert", "coco")
CHEST = ("nih", "chexpert")
EFFECT_FLOOR = 0.05


def block_included(run: dict) -> bool:
    done = set(run.get("completed_modules") or [])
    return all(m in done for m in INCLUSION_MODULES)


def probe_graded(run: dict) -> bool:
    return PROBE_MODULE in set(run.get("completed_modules") or [])


def owned(v: dict) -> bool:
    return bool(v.get("steering_reference") and v.get("verdict") == "fixed_family_advantage")


def _json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def load_runs(runs: Path = RUNS, order=None) -> dict:
    """Every block with a run.json: (model_key, dataset) -> dict(run, s, elig, pf, dir, included, probe).
    `s` is the summary.json (may lack core); `included` and `probe` are the two flags above."""
    out = {}
    for rj in sorted(runs.glob("*/*/run.json")):
        d = rj.parent; run = _json(rj)
        mk, ds = run.get("model_key", d.parent.name), run.get("dataset_id", d.name)
        out[(mk, ds)] = {"run": run, "s": _json(d / "summary.json"), "elig": _json(d / "template_eligibility.json"),
                         "pf": _json(d / "preflight.vis.last.json"), "dir": d,
                         "included": block_included(run), "probe": probe_graded(run)}
    if order is not None:
        rank = {m: i for i, m in enumerate(order)}
        out = dict(sorted(out.items(), key=lambda kv: (rank.get(kv[0][0], len(rank)), DATASETS.index(kv[0][1]))))
    return out


def write_blocks(all_blocks: dict) -> dict:
    """Blocks with a scored write matrix under the rule; every such block carries core.per_question."""
    out = {k: b for k, b in all_blocks.items() if b["included"]}
    for k, b in out.items():
        if not (b["s"].get("core") or {}).get("per_question"):
            raise RuntimeError(f"{k}: included by run.json but summary.json carries no core.per_question")
    return out


def probe_blocks(all_blocks: dict) -> dict:
    return {k: b for k, b in all_blocks.items() if b["probe"]}


def calibration(b: dict) -> dict:
    return {c: v for c, v in (b["s"].get("calibration") or {}).items() if isinstance(v, dict)}


def core(b: dict) -> dict:
    """Core verdict cells of an included block; empty for every other block (their partial core is never read)."""
    return (b["s"]["core"]["per_question"] if b["included"] else {})


def read_manifest(path: Path = RUNS / "manifest.csv") -> list[dict]:
    import csv
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m cftransfer.manifest` in the code repository first")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def check_manifest_agrees(all_blocks: dict, rows: list[dict]) -> None:
    """The manifest's included / probe-graded block sets must equal the ones this helper derives."""
    m_inc = {(r["model_key"], r["dataset"]) for r in rows if r["block_included"] == "true"}
    m_probe = {(r["model_key"], r["dataset"]) for r in rows if r["probe_graded"] == "true"}
    h_inc = {k for k, b in all_blocks.items() if b["included"]}
    h_probe = {k for k, b in all_blocks.items() if b["probe"]}
    if m_inc != h_inc or m_probe != h_probe:
        raise RuntimeError(f"manifest.csv disagrees with the summaries: included {sorted(m_inc ^ h_inc)}, "
                           f"probe-graded {sorted(m_probe ^ h_probe)}; regenerate with `python -m cftransfer.manifest`")
