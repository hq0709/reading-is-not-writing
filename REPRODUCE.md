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

## Paired behavioral opportunity

The shared 100-patient cohort is bound by
`20260905T052025Z-9bc918b414ff-paired-opportunity/artifacts/registered-pairs.json`.
Qwen run `20260905T053520Z-f0317eaf3f4e-paired-opportunity` uses source
`f0317eaf3f4e3d97a18c266eb3d84b0f1c09baec`; LLaVA run
`20260905T064036Z-1af90218a1d8-llava-opportunity` uses source
`1af90218a1d8022bce66452826fa408ef146e432`. Each contributes 200 outcomes,
with all 100 patients and the same 5,000 bootstrap indices retained.
The accepted joint report and review are in code commit `f774842`, under
`docs/PAIRED_OPPORTUNITY_ARCHITECTURE_RESULTS.md` and
`docs/reviews/llava-paired-opportunity-results.md`; the Qwen report is
`docs/QWEN_PAIRED_OPPORTUNITY_RESULTS.md`, with full-precision values in
`docs/reviews/qwen7b-paired-opportunity-results.md`. The LLaVA result receipt is
`/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T064601848008Z.json`.
The paper's `sections/D_paired_opportunity.tex` contains the complete
model-specific and direct patient-paired summaries.

## LLaVA validation readout screen

Run `20260905T093108Z-42a43207c848-llava-validation` uses immutable source
`42a43207c848acfcadecf2f3e0bf866ea71d9dc7`. It measures 4,200 calibration
outcomes on 700 validation patients; the reserved 100-patient write cohort
is untouched because no question meets joint reader and capability
qualification. Accepted reports are in code commit `bafd1d0`, under
`docs/LLAVA_VALIDATION_OPPORTUNITY_RESULTS.md` and
`docs/reviews/llava-validation-opportunity-results.md`.
The internal receipt is
`/home/qingchan/data/concept-flow/state/llava-validation-opportunity-internal-20260905T093743Z/receipt.json`.
Pinned result and synthesis receipts are
`/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T094007326743Z.json`
and `review-20260905T094932091464Z.json` in the same directory.
`sections/E_validation_readout.tex` records the prospective allocation,
screening criteria, complete reader and capability tables, and conditional
write-stage disposition.

## LLaVA text calibration and independent image/reader validation

Text run `20260905T221642Z-f2b4bafc8a80-llava-text` fixes source
`f2b4bafc8a802e3f046e82318fd8b5b9aa762c52` and contains 80 finite-suite
outcomes. The accepted report is `docs/LLAVA_TEXT_SEMANTIC_CALIBRATION_RESULTS.md`
in the code repository. Its internal and pinned result receipts are
`/home/qingchan/data/concept-flow/state/llava-text-calibration-internal-20260905T2218Z/receipt.json`
and `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T221855379998Z.json`.

Image run `20260905T232313Z-2a95dfbed888-llava-yesno` fixes source
`2a95dfbed888fce12b7481812065e5d68b712e8a`. It contains 2,800 image outcomes
and two image-free priors, with 700 index and 700 disjoint donor patients.
The accepted report is `docs/LLAVA_YESNO_IMAGE_DIAGNOSTIC_RESULTS.md`;
both reports are available at code commit
`3cd386c8fd582eb50f1a59e73ae3b48a9eaee90d`.
The authoritative full-precision summary is
`/home/qingchan/data/concept-flow/state/llava-yesno-image-recovery-20260905T234235Z/llava-yesno-image-diagnostic-summary.json`.
Validator-only source `06277a1fb8a487a54b9220373957b1f1d988e2f5` reuses the
unchanged native artifact; terminal validation source
`1da1ca3c6da95218a99304f0edb65e3ab407e0de` verifies the registered allocation,
scores, bootstrap and frozen-reader control. These steps create no model
outcomes. The internal and pinned result receipts are
`/home/qingchan/data/concept-flow/state/llava-yesno-image-internal-20260905T235215Z/receipt.json`
and `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T235348704249Z.json`.

`sections/F_image_readout.tex` gives the constructed prompts, both image
prompts, all primary contrasts, frozen-reader comparison and twenty control
AUROCs, and secondary aggregation, Brier, margin and token-mass diagnostics.
The pilot clinical AUROC, interval and control mean in the cross-cohort
table come from `questions[0]` in
`/home/qingchan/data/concept-flow/runs/20260905T093108Z-42a43207c848-llava-validation/artifacts/qualification.json`.
The two sampling frames differ; no between-cohort difference interval was
registered. Text calibration selects yes/no on the finite suite; independent
image opportunity is unestablished, wording is unresolved, and reader
selectivity does not replicate. No subsequent write experiment is reported.

## Appendix figure generation

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
