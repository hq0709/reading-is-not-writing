# Statistical definitions and label-conditioned responses

Run: bounded manuscript integration against paper baseline
`2602850ce64d9984fd80e24ce21b1e76b58b442e`. The evidence source is
`20260904T224225Z-1ab7576193a7-ownership-diagnostics`, artifact
`artifacts/ownership-diagnostics.json`, key `label_conditioned_performance`.
The independent 35-page review PDF and its source snapshot remain frozen.

Observation: the methods distinguish original target-only dose selection
from independent locked-dose replication and fixed random reference gates.
The max-T equations use centered draws with a fixed bootstrap standard
deviation, including the implemented degenerate-component treatment.
Control labels and probe scores have distinct notation. Two appendix
tables report all 36 label-conditioned cells and group counts at four
decimal places, with pointwise intervals from 2,000 patient bootstraps.
Results and Discussion retain finite-family, disease-conditionality,
direction-quality and prompt-scope interpretations.

Implementation evidence in the sibling research repository:

- `src/first_gate.py:543` and `src/cross_cell_gate.py:478`: target-only
  maximum over six controlled doses, same-dose random p95, absolute sham
  maximum over controlled doses.
- `src/qwen_causal_ownership_gate.py:276`: centered max-T with fixed
  bootstrap SD (`ddof=1`), component threshold `1e-12`, linear percentile
  interpolation and actual SD in the final subtraction.
- `src/qwen_causal_ownership_gate.py:332`: joint 400-patient resampling,
  the 30-contrast ownership family and separate 12-statistic alias family.
- `src/qwen_mass_prompt_specificity.py:227`: joint 200-patient,
  20-contrast Mass family using the same max-T implementation.
- `src/qwen_direction_specificity_gate.py:296`: direct bootstrap
  target-minus-recomputed-maximum margins and percentile bounds.
- `src/qwen_ownership_diagnostics.py:135`: joint label-conditioned
  patient bootstrap and pointwise percentile intervals.

Verification: the existing draft harness passes with 14 sections, seven
external table files, 16 citation keys, 86 labels, 212 abstract words and
all 27 original adjudicated values. No bibliography entry or citation key
was added. PDF compilation succeeds at 39 pages; main text and AI-use
statement end on page 10, References occupy pages 10--11, and Appendix A
starts on page 12. Rendered pages 1--10, 12, 15--17, 22--25 and 28 were
inspected for the affected text, equations, figures and table layout.
There are no overfull boxes, undefined references or duplicate destinations.
All 39 tables use resizebox scaling without manual font-size or column-width
adjustments. Prose-check rhythm and semicolon signals were read in context;
the procedural and comparison wording retains its scientific content.

Independent CCFA integrity review returns `PAPER_STATISTICAL_CLARITY PASS`
with no material required corrections. It verifies all 36 rows, 180
displayed numerical entries and both group counts per row against the
JSON, plus exact family sizes, seeds, pairing and statistical implementation.
It also confirms the 20 citation commands and 16 unique keys, the
bibliography and the scientific claims supported by those citations
are preserved from baseline.

Final pinned review: `PAPER_STATISTICAL_CLARITY PASS`, with no material
required corrections after reading the complete 39-page manuscript text
and exact equation sources. Receipt
`review-20260906T043708128134Z.json` records requested and actual model
`claude-fable-5-1`, medium effort, read-only true, exit zero and valid true.
The before/after checkout is clean at
`bb1205ca57c83ac48916937b6508a08f2d2e2a5e`; ARIS is pinned to
`94d8093ed21d20a790830318190095b9f5036ce8`. The prompt digest is
`0c427484b7ae6dcd4a1335e01c9d275a224b1fcdae9686473e12573df5ea25d0`.
The initial transport attempt rejected an extracted null-character glyph
before launching the reviewer. Parentheses were normalized in the review
input and exact equation sources supplied; only one model review ran.

The final reviewer suggested non-blocking textual clarifications. Three
method pointers now direct run identifiers and row selection to Appendix B
and captured modules to Appendix E. The original-cell rule explicitly
multiplies random changes by the selected dose's sign. The Appendix L
sentence about two errors already identifies the candidate A/B conditions,
which are distinct from the incumbent conditions in Table A28, and is
retained. The four local clarifications pass a separate read-only
verification after the sign-multiplication clause is closed with a comma.
The final source compiles to 39 pages, with the affected page rendered
and checked again.

Gate decision: `PAPER_STATISTICAL_CLARITY PASS`. Independent evidence,
local layout, final pinned review and the localized follow-up verification
pass with no outstanding required correction.

Review coordination: the project owner granted this paper task
`01a06afe-5c66-7a80-b922-b3ede71eab3e` exclusive use of the pinned reviewer
for this batch. The coordination lease is released after the final review
against the clean server
checkout `bb1205ca57c83ac48916937b6508a08f2d2e2a5e`; it is a collaboration
record, not a filesystem lock. The reviewer request remains
`claude-fable-5-1`, medium, read-only; the user-authorized routing policy
also accepts actual `claude-opus-5`, recorded separately from the request.

Next step: integrate the checked manuscript into the paper main branch
and deliver the final review receipt and explicit coordination release
to the project owner.
