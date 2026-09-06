# Replicated steering and prompt-conditioned competition

Run: a bounded narrative revision of paper baseline
`ffa6e66edf4202088e1974e60e7347e9b868d648`, using the supplied scientific
assessment in `ccfa-review-reports/concept-flow-iclr-review.md` at the
workspace root. The evidence package, models, experiments, table values,
and figure assets are unchanged.

Observation: abstract, introduction, Results and conclusion center two
findings. Qwen Effusion reproduces beyond random p95 and sham on independent
patients while its fixed-family clinical ownership contrast is negative.
Mass has independently reproduced clinical advantage under both show A/B
instructions, with negative is margins and larger random effects. Results
place Mass directly after the Effusion comparison. Matrix, closure, LLaVA
and paired-opportunity evidence remains in a shared boundary section and
its existing appendices. Ownership is defined as fixed-family relative
advantage, not a necessary condition for concept participation in
computation. Descriptive wording analyses, simultaneous clinical bounds,
random-control failures and the diagnostic-benefit boundary remain distinct.

Verification: independent read-only CCFA scientific and writing review
returns `PAPER_FINDINGS_NARRATIVE PASS` with no required corrections after
reading the final saved sections and the retained closure values in
Appendix A. The draft harness passes with 14 sections, 6 external table
files, 16 citation keys, 81 labels, 212 abstract words and all 27 original
adjudicated values. Tectonic produces 36 pages; main text and AI-use
statement end on page 9, References span pages 9--10, and the appendix
starts on page 11. Affected pages 1--9 and 17 were rendered and inspected.
There are no overfull boxes, undefined references or duplicate destinations.
All 37 tables retain purpose-only captions and resizebox-only scaling.
The prose checker's remaining rhythm and semicolon signals were reviewed
in context; no scientific wording was changed merely to clear a heuristic.

Gate decision: `PAPER_FINDINGS_NARRATIVE PASS`. Local scientific, writing
and layout checks pass, and the supplementary external review returns the
same verdict with no required corrections.
Receipt
`/home/qingchan/.codex/state/claude-review-concept-flow/review-20260906T031043107652Z.json`
records requested `claude-fable-5-1`, observed `claude-opus-5`, medium effort,
read-only true and exit zero, with valid false. Server checkout remains
clean at `f61bec358f2e5442d1876b937e2b7b47624d6cf8` before and after.
That invocation is not counted as fixed-model acceptance. After the
project owner's short identity probe passed, one authorized retry reused
the identical full manuscript and request. Receipt
`/home/qingchan/.codex/state/claude-review-concept-flow/review-20260906T031709653754Z.json`
again records observed `claude-opus-5` and valid false, with the same
request digest and unchanged clean server checkout. Neither full review
counts as acceptance under the original exact-identity rule; the short
probe does not substitute for manuscript review.

Authorization: on 2026-09-06 the user explicitly accepted service routing
from requested `claude-fable-5-1` to observed `claude-opus-5` for the current
Concept Flow acceptance. The request remains pinned to `claude-fable-5-1`
at medium effort with read-only execution and unchanged checkout checks.
The actual returned model must be recorded. The original receipts retain
valid false; this authorization does not alter their historical status
or supply their missing scientific verdicts. The cause of the routing
difference is undetermined.

Recovery: the adapter captured response JSON in process memory, persisted
only metadata, and raised before returning the result. It used
`--no-session-persistence`; neither task output nor the relevant server
state and session locations retain the two review bodies. The project
owner deployed the authorized routing rule at code commit
`bb1205ca57c83ac48916937b6508a08f2d2e2a5e`, with its authorization recorded
in `concept-flow-code/docs/RESEARCH_PLAN.md#reviewer-routing-policy`.
One identical-text review supplied the missing verdict. Manuscript and
local validation were retained without another writing or experimental pass.

External acceptance receipt:
`/home/qingchan/.codex/state/claude-review-concept-flow/review-20260906T034628983798Z.json`.
The request was `claude-fable-5-1`; the actual model was `claude-opus-5`.
The receipt records medium effort, read-only true, exit zero and valid true
under the user-authorized accepted-model set. Before and after checkouts
are clean at `bb1205ca57c83ac48916937b6508a08f2d2e2a5e`. The unchanged prompt
digest is `a1372d1e5a6f78a81f62aedf2f6572de2a12db26317dcd7a6c587bb9e60eb2fa`.
Transport acceptance and the returned manuscript verdict were checked
separately; the latter is `PAPER_FINDINGS_NARRATIVE PASS`.

The external reviewer found no material contradiction in the supplied
sources and accepted the two-finding narrative, ownership scope, interval
distinctions, random-reference interpretation and diagnostic-benefit
boundary. It offered two non-blocking observations: a closer table
cross-reference for the show yes/no margin of 0.2369, and a source-data
transcription check for 0.1675 appearing in two statistically distinct
interval summaries. Neither was identified as an error or required change;
the accepted evidence tables and narrative were retained. The external
verdict covers the supplied text, not page rendering, omitted appendices
or comparison with the baseline. Those checks are supported by the
independent local review and harness evidence above.

Remaining scientific question: do the two findings persist when the
clinical direction bundle is independently refitted? Current patient-level
uncertainty conditions on one direction bundle and cannot answer this
generalization question. No additional experiment was performed.
