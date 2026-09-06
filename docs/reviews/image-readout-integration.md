# LLaVA text calibration and independent image/reader manuscript integration

Run: integrate accepted text run `20260905T221642Z-f2b4bafc8a80-llava-text`
and image run `20260905T232313Z-2a95dfbed888-llava-yesno`, documented in
code commit `3cd386c8fd582eb50f1a59e73ae3b48a9eaee90d`. Immutable sources,
the validator-only artifact reuse, and experimental receipts are recorded
in `REPRODUCE.md`.

Observation: Results 4.4, introduction, discussion and conclusion connect
pilot screening to finite text calibration and independent image/reader
validation. Appendix K retains all pilot observations while limiting their
scope; Appendix L contains both prompt figures, all text-condition
summaries and paired changes, both image-wording summaries, the complete
simultaneous family, reader comparison and all twenty control AUROCs,
and secondary aggregation, margin, Brier and token-mass diagnostics.
Clinical versus control AUROC decomposition is explicitly descriptive;
the differing sampling frames and absence of a registered between-cohort
difference interval are stated. Opportunity remains unestablished,
wording unresolved, and positive pilot reader selectivity unreplicated.

Gate decision: `PAPER_IMAGE_READOUT_INTEGRATION PASS`. Independent internal
read-only review of the final saved sources found no material numerical,
protocol, narrative or presentation findings. The pinned text-only
scientific writing review also required no corrections. Receipt:
`/home/qingchan/.codex/state/claude-review-concept-flow/review-20260906T001834586719Z.json`.
Observed model is canonical `claude-fable-5-1`, medium effort;
valid/read-only are true, exit is zero, and server HEAD remains clean at
`3cd386c8fd582eb50f1a59e73ae3b48a9eaee90d` before and after review.
The reviewer transport was released before local publication.

Verification: Tectonic produces 36 pages. Main text and AI-use statement
end on page 9, References occupy page 10, and the appendix starts on its
own page 11. Appendix L occupies pages 32--36. Affected-page renders show
legible table contents, purpose-only captions and explanatory paragraphs;
prompt figure text is smaller than body text, with panel captions below.
No overfull boxes, undefined references, or duplicate destinations occur.
The existing draft harness passes with 14 sections, 6 external table files,
16 citation keys, 81 labels, 214 abstract words and all 27 original
adjudicated values. All 37 tables use concise purpose captions and all
38 tabular blocks use shrink-only `resizebox` scaling, without manual
table-font, spacing or column-width settings. Prior quantitative tables,
accepted data and figure assets are unchanged. No new experiment ran.

Next step: can a named clinical direction demonstrate selective answer
control beyond both clinical and random competitors in a condition with
independently established image-linked discrimination? The present evidence
does not resolve that contribution gap; any new experiment requires its
own prospective protocol.
