"""Reference-and-rule ledger: what every graded analysis of the campaign actually scores it against.

One row per analysis. Each row carries the cohort and row count it scores, the reference its steering test applies
(the 119-direction random family's 95th percentile and the sham, a smaller random family, the sham alone, or none),
the inferential rule that decides its verdict (simultaneous max-T bounds on the fixed contrasts, or a percentile
interval), the construction of the sham it compares against (a coordinate permutation of the direction under test,
or the seed-0 sham of the logistic normal), and the number of scored cells.

Everything is read from the packaged artefacts -- each block's summary.json and run.json under the paper's one
inclusion rule (cf_inclusion) -- and every declared reference and sham is CHECKED against those artefacts:

    random family      the cell's own random_n / random_p95 / random_max fields
    sham               the cell's abs_sham against the same block and question's CORE abs_sham; equal means the
                       analysis reused the seed-0 sham value, unequal means it carries a sham of its own
    rows               the section's n_rows

A declared string that the artefacts contradict raises, so the table and the artefacts cannot drift apart.
"""
from __future__ import annotations

import json

import cf_inclusion as ci

ROB = ci.RUNS / "robustness"

# reference / rule / sham vocabulary, written once so every row spells them the same way
R_FULL = "119 random $p_{95}$, sham"
R_SMALL = "18 random, maximum; sham"
R_DOSE = "20 random, 95th percentile; sham"
R_SHAM = "sham alone"
R_NONE = "none"
RULE_MAXT = "simultaneous max-$T$, {n} contrasts"
RULE_PCT = "percentile interval on $O^m_q$"
RULE_NONE = "none"
SHAM_OWN = "the direction written"
SHAM_SEED0 = "the seed-0 normal"  # the permutation of the seed-0 logistic normal, not of the direction under test
SHAM_NONE = "--"


def _cells(s: dict, path):
    """(question, cell) pairs of one summary section, or [] when the section does not carry them."""
    try:
        return list(path(s))
    except (KeyError, TypeError, IndexError):
        return []


SHAM_RTOL = 2e-3      # the packaged shams of one block agree to this over the analyses that reuse them


def _reuses_core_sham(cell: dict, core_cell: dict):
    """True when the analysis compares against the very number the write matrix compares against; None when it
    carries no sham at all. It does not by itself say whether that number is a permutation of the direction under
    test -- an analysis that writes the seed-0 normal on the same rows reuses it legitimately."""
    a, b = cell.get("abs_sham"), (core_cell or {}).get("abs_sham")
    if a is None:
        return None
    if b is None:
        return False
    return abs(a - b) <= SHAM_RTOL * max(1e-6, abs(b))


def _random_kind(cell: dict) -> str:
    n = cell.get("random_n")
    has = cell.get("random_p95") is not None or cell.get("random_max") is not None
    if not has:
        return R_SHAM if cell.get("abs_sham") is not None else R_NONE
    if n == 18:
        return R_SMALL
    return R_FULL


ATTRIBUTES = ("view_AP", "sex_F", "age_60")


def _scan(wb: dict, section: str, path, keep=None) -> dict:
    """Blocks, rows, cells and the reference / sham the artefacts show for one analysis."""
    blocks, rows, n, refs, reuse = [], set(), 0, {}, {}
    for k, b in wb.items():
        s = b["s"].get(section)
        if not s:
            continue
        cells = _cells(s, path)
        if keep is not None:
            cells = [(q, c) for q, c in cells if keep(q, c)]
        if not cells:
            continue
        blocks.append(k)
        rows.add(s.get("n_rows"))
        core = ci.core(b)
        for q, c in cells:
            n += 1
            refs[_random_kind(c)] = refs.get(_random_kind(c), 0) + 1
            r = _reuses_core_sham(c, core.get(q, {}))
            reuse[r] = reuse.get(r, 0) + 1
    return {"blocks": sorted(blocks), "n_blocks": len(blocks), "n_rows": sorted(rows), "cells": n,
            "reference_seen": refs, "reuses_core_sham": reuse}


# ------------------------------------------------------------------ the analyses, in the order the paper meets them

def _core(s):
    return s["per_question"].items()


def _named(name):
    return lambda s: s[name]["per_question"].items()


def _ansdirt(s):
    return [(q, c) for blk in s["per_template"].values() for q, c in blk["per_question"].items()]


def _semend(s):
    return [(q, c) for ep in s["per_endpoint"].values() if "families" in ep
            for blk in ep["families"].values() for q, c in blk["per_question"].items()]


def _towerswap(s):
    return [(q, c) for cb in s["combinations"].values() if cb.get("status") == "COMPLETE"
            for q, c in cb["grade"].items()]


def _replay(s):
    return s["replay"]["grade"].items()


def _projseed(s):
    return [(q, c) for k in ("seed1", "seed2") if k in s for q, c in s[k]["per_question"].items()]


def _precision(s):
    return [(q, c) for k in ("fp32", "batch1") if k in s for q, c in s[k]["per_question"].items()]


def _altdir(s):
    return [(q, c) for f in ("dom", "pattern", "orth", "resid") if f in s for q, c in s[f]["per_question"].items()]


def _altdird(s):
    return [(q, c) for f in ("dom_disp", "pattern_disp") if f in s for q, c in s[f]["per_question"].items()]


def _tokenw(s):
    return [(q, c) for f in ("tokenw", "topq") if f in s for q, c in s[f]["per_question"].items()]


# name, summary section, cell selector, filter, declared reference, rule, sham, contrast family,
# and whether the analysis's sham is the same number the write matrix compares against (checked against the artefacts)
SPEC = [
    ("Write matrix, primary template", "core", _core, None, R_FULL, RULE_MAXT, SHAM_OWN, "30", True),
    ("Probe refits, seeds 1 and 2", None, None, None, R_FULL, RULE_MAXT, SHAM_SEED0, "30", None),
    ("Logit-scale grade", None, None, None, R_FULL, RULE_PCT, SHAM_OWN, "--", None),
    ("Unsaturated-row grade", None, None, None, R_FULL, RULE_MAXT, SHAM_OWN, "30", None),
    ("Alternative direction constructions", "altdir", _altdir, None, R_FULL, RULE_MAXT, SHAM_SEED0, "30", True),
    ("Displacement lifts", "altdird", _altdird, None, R_FULL, RULE_MAXT, SHAM_SEED0, "30", True),
    ("Extended competitor family", "extcomp", _core, None, R_FULL, RULE_MAXT, SHAM_OWN, "8 or 11", True),
    ("Token-weighted writes", "tokenw", _tokenw, None, R_FULL, RULE_MAXT, SHAM_OWN, "30", True),
    ("fp32 and batch-size-one rescoring", "precision", _precision, None, R_FULL, RULE_MAXT, SHAM_OWN, "30", False),
    ("Answer direction, primary template", "ansdir", _core, None, R_FULL, RULE_MAXT, SHAM_OWN, "30", False),
    ("Answer direction, held-out templates", "ansdirt", _ansdirt, None, None, RULE_MAXT, SHAM_OWN, "30", False),
    ("Non-clinical attributes", "attr", _core, lambda q, c: q in ATTRIBUTES, R_FULL, RULE_MAXT, SHAM_OWN, "72", False),
    ("Findings in the attribute family", "attr", _core, lambda q, c: q not in ATTRIBUTES, R_FULL, RULE_MAXT, SHAM_OWN, "72", None),
    ("Radiologist-label regrade", "valid", _core, None, R_FULL, RULE_MAXT, SHAM_OWN, "30", False),
    ("Expert-label refits", "validfit", _core, None, R_FULL, RULE_MAXT, SHAM_SEED0, "30", True),
    ("Projection seeds 1 and 2", "projseed", _projseed, None, R_FULL, RULE_MAXT, SHAM_OWN, "30", False),
    ("Crossed tower and reader", "towerswap", _towerswap, None, R_FULL, RULE_MAXT, SHAM_OWN, "30", False),
    ("Identical-tensor replay", "replay", _replay, None, R_FULL, RULE_MAXT, SHAM_OWN, "30", False),
    ("Semantic endpoints", "semend", _semend, None, R_SMALL, RULE_MAXT, SHAM_OWN, "30", False),
    ("Fine-grained object family", "fgobj", _core, None, R_FULL, RULE_MAXT, SHAM_OWN, "30", False),
]

# qualifications the reference column carries, one clause each
NOTES = {
    "Logit-scale grade": ", on the margin scale",
    "Unsaturated-row grade": ", on those rows",
    "Probe refits, seeds 1 and 2": ", the write matrix's",
    "Alternative direction constructions": ", the write matrix's",
    "Displacement lifts": ", the write matrix's",
    "Expert-label refits": ", the write matrix's",
    "Token-weighted writes": ", the sham unweighted",
    "Extended competitor family": "",
    "Semantic endpoints": ", the endpoint's own",
    "Fine-grained object family": ", the family's own",
    "Projection seeds 1 and 2": ", that projection's",
    "Crossed tower and reader": ", of the tower in use",
    "Identical-tensor replay": ", the source block's",
}

# the three analyses whose rows are a restriction or a rescaling of the write matrix rather than a module of their own
DERIVED = {
    "Probe refits, seeds 1 and 2": ("refit.json", "600 test"),
    "Logit-scale grade": ("scale.json", "600 test"),
    "Unsaturated-row grade": ("scale.json", "unsaturated test"),
}


def _derived_counts() -> dict:
    refit = json.loads((ROB / "refit.json").read_text())
    scale = json.loads((ROB / "scale.json").read_text())
    n_ref = sum(len(b["per_seed"][str(k)]["per_question"]) for b in refit["blocks"] for k in (1, 2)
                if str(k) in b.get("per_seed", {}))
    unsat = scale["item2_ceiling"]["unsaturated_full_grade"]
    return {"Probe refits, seeds 1 and 2": {"cells": n_ref, "n_blocks": refit["meta"]["n_blocks"], "rows": "600 test"},
            "Logit-scale grade": {"cells": scale["item1_margin_scale"]["all"]["n_cells"],
                                  "n_blocks": scale["item1_margin_scale"]["all"]["n_blocks"], "rows": "600 test"},
            "Unsaturated-row grade": {"cells": sum(unsat[ds]["cells"] for ds in ci.DATASETS),
                                      "n_blocks": scale["blocks_ok"],
                                      "rows": "unsaturated test"}}


def ledger(wb: dict) -> list[dict]:
    """One row per graded analysis, with every declared reference and sham checked against the artefacts."""
    derived = _derived_counts()
    out = []
    for name, sec, path, keep, ref, rule, sham, family, eq_core in SPEC:
        if sec is None:
            d = derived[name]
            out.append({"analysis": name, "rows": d["rows"], "n_blocks": d["n_blocks"], "cells": d["cells"],
                        "reference": ref + NOTES.get(name, ""), "rule": rule.format(n=family), "sham": sham,
                        "family": family, "split": None, "reuses_write_matrix_sham": None})
            continue
        got = _scan(wb, sec, path, keep)
        if not got["cells"]:
            continue
        if len(got["n_rows"]) != 1:
            raise RuntimeError(f"{name}: blocks score {got['n_rows']} rows; the ledger states one row count per analysis")
        role = "valid" if sec == "valid" else "test"
        rows = f"{got['n_rows'][0]} {role}"
        seen_ref = sorted(got["reference_seen"], key=lambda k: -got["reference_seen"][k])
        split = None
        if ref is None:                                   # the analysis applies two references; report the split
            if set(seen_ref) != {R_FULL, R_SHAM}:
                raise RuntimeError(f"{name}: expected a split reference, the artefacts show {got['reference_seen']}")
            ref = (f"{R_FULL} in {got['reference_seen'][R_FULL]} pairs; {R_SHAM} in "
                   f"{got['reference_seen'][R_SHAM]}")
            split = got["reference_seen"]
        elif seen_ref[:1] != [ref] or len(seen_ref) != 1:
            raise RuntimeError(f"{name}: declared reference {ref!r}, the artefacts show {got['reference_seen']}")
        seen_eq = got["reuses_core_sham"]
        # an analysis that reuses the write matrix's sham reuses it in every cell; one that carries its own may still
        # land within tolerance of it in a few cells whose sham is near zero, so the negative side is a loose bound
        share_true = seen_eq.get(True, 0) / max(1, got["cells"])
        if eq_core is True and share_true < 0.99:
            raise RuntimeError(f"{name}: the ledger says it reuses the write matrix's sham, the artefacts show {seen_eq}")
        if eq_core is False and share_true > 0.10:
            raise RuntimeError(f"{name}: the ledger says it carries its own sham, the artefacts show {seen_eq}")
        out.append({"analysis": name, "rows": rows, "n_blocks": got["n_blocks"], "cells": got["cells"],
                    "reference": ref + NOTES.get(name, ""), "rule": rule.format(n=family), "sham": sham,
                    "family": family, "split": split, "reuses_write_matrix_sham": eq_core})
    return out


def totals(rows: list[dict]) -> dict:
    """Headline counts of the ledger for the prose."""
    by = {r["analysis"]: r for r in rows}
    ansdirt = by["Answer direction, held-out templates"]["split"]
    seed0 = [r["analysis"] for r in rows if r["sham"] == SHAM_SEED0]
    return {"n_analyses": len(rows), "cells": sum(r["cells"] for r in rows),
            "n_seed0_sham": sum(1 for r in rows if r["sham"] == SHAM_SEED0),
            "cells_seed0_sham": sum(r["cells"] for r in rows if r["sham"] == SHAM_SEED0),
            "n_own_sham": sum(1 for r in rows if r["sham"] == SHAM_OWN),
            "n_max_t": sum(1 for r in rows if r["rule"] != RULE_PCT),
            "n_percentile": sum(1 for r in rows if r["rule"] == RULE_PCT),
            "cells_percentile": sum(r["cells"] for r in rows if r["rule"] == RULE_PCT),
            "n_full_reference": sum(1 for r in rows if r["split"] is None and r["reference"].startswith(R_FULL)),
            "ansdirt_random": ansdirt[R_FULL], "ansdirt_sham": ansdirt[R_SHAM],
            "ansdirt_cells": ansdirt[R_FULL] + ansdirt[R_SHAM],
            "semend_cells": by["Semantic endpoints"]["cells"],
            "semend_random": 18, "seed0_sham_analyses": seed0}
