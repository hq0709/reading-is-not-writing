# Reproduce the submission PDF

Use Tectonic `Tectonic 0.17.0` with binary SHA-256
`99ffcfdbf1ebf8bdda9e791942e3d06aedb12463fddc33f07de6f5211c8bf08d`. From this directory, run:

```bash
SOURCE_DATE_EPOCH=1788439147 FORCE_SOURCE_DATE=1 tectonic -X compile main.tex --keep-logs
```

The command produces `main.pdf`.
`MANIFEST.sha256` records every submitted source asset.

## Six-gate evidence provenance

The accepted evidence snapshot is the read-only `concept-flow-code` repository at
commit `20fbecde8bc9c3d9b358a65dfe50345929b3547f`. The two prospective
mechanism runs are `20260904T125710Z-a3bd883540eb-causal-ownership`
(304,800 outcomes; 5,000 bootstraps) and
`20260904T162317Z-8a55a4c2f6b6-consolidation-closure` (6,450 outcomes;
257 decision arrays reproduced exactly). The paper repository includes the
submission sources, figure-generation scripts, aggregate plotting data, and
reproducibility metadata.

## Answer-encoding evidence

The paired sensitivity experiment is
`20260904T230152Z-21dad2f72726-answer-encoding`, immutable code commit
`21dad2f727261491c372bf9ffccf2dafd7c01413`. Its
`artifacts/answer-encoding-summary.json`, `meta.json`, `per-image.csv`, and
`calibration.csv` record 198,000 intervention and 2,400 calibration outcomes.
Accepted results and review documentation are in the code repository at
`b1f58b5`, under `docs/ANSWER_ENCODING_RESULTS.md` and
`docs/reviews/qwen7b-vislast-answer-encoding.md`.
The preceding score-coordinate analysis reuses the 400-patient ownership
cohort in run `20260904T224225Z-1ab7576193a7-ownership-diagnostics`;
its primary artifact is `artifacts/ownership-diagnostics.json`.
`sections/B_answer_encoding.tex` contains the diagnostic and encoding
protocols and tables; `tables/table_encoding_mass.tex` contains the
descriptive Mass prompt comparison. The experiment archives remain the
source of full-precision results.

## Independent Mass confirmation

Run `20260905T015725Z-1c9820d7b576-mass-prompts` fixes source commit
`1c9820d7b5765c2ec220d9f4afaafd4c743e43e4` and records 177,800
confirmation plus 1,400 calibration outcomes in
`artifacts/mass-prompt-specificity-summary.json`. The accepted result and
review are in the code repository at `9bc918b414ff14af804873aba3642ac4dcbb3ff3`,
under `docs/MASS_PROMPT_SPECIFICITY_RESULTS.md` and
`docs/reviews/qwen7b-mass-prompt-specificity-results.md`.
`sections/C_mass_confirmation.tex` contains the seven-condition protocol,
calibration, paired contrasts, and predictive metrics;
`tables/table_mass_confirmation.tex` contains the coded-cell main table.
The source summary retains all full-precision estimates and all 20
simultaneous clinical comparisons. Both registered specificity flags are
false; all seven calibration cells are eligible. The accepted scientific
review receipt is
`/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T051404447778Z.json`.

## Appendix figures

Figures A1--A6 are vector PDFs generated with Matplotlib and NumPy. From this
directory, install the plotting dependencies and regenerate them with:

```bash
python -m pip install -r scripts/requirements-figures.txt
python scripts/plot_appendix_diagrams.py
python scripts/plot_appendix_quantitative.py
```

`scripts/appendix_style.py` supplies the common typography, palette, and export
settings. Each script writes publication assets to `figures/` and PNG previews
to the ignored `tmp/appendix_figures/` directory. PDF creation timestamps are
omitted for reproducible exports.

`data/accepted_results.json` contains the accepted aggregate estimates from the
six experiments. Its source locations are expressed relative to the experiment
archive. `data/ownership_summary.json` contains the full-precision six-by-six
surface from run `20260904T125710Z-a3bd883540eb-causal-ownership`. Statistical
figures read these snapshots directly; the diagram script encodes the fixed
experimental interface and evidence-flow descriptions in the manuscript.

## Direction-specificity figure provenance

`figures/fig3_direction_specificity.pdf` is generated from the read-only
`paper/scripts/gen_fig3.py` and `paper/data/accepted_results.json` in the
sibling `concept-flow-code` repository at commit
`313a258c19c65857e1eaeeac3b5c0bb6765303b1`. The right-hand panel uses the
response-surface notation `$O_{\mathrm{Effusion}}$` and the axis label
`Effusion ownership contrast (95% patient bootstrap)`; all data, point values,
random seed, coordinates, limits, and interval geometry are unchanged. To
regenerate the exact stored asset in an isolated copy, set these two labels in
`gen_fig3.py` before running it:

```python
margin_ax.set_yticks([0], [r"$O_{\mathrm{Effusion}}$"])
margin_ax.set_xlabel("Effusion ownership contrast\n(95% patient bootstrap)")
```
