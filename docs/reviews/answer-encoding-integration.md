# Answer-encoding manuscript integration

Run: integrate accepted answer-encoding run `20260904T230152Z-21dad2f72726-answer-encoding` and retrospective ownership diagnostics `20260904T224225Z-1ab7576193a7-ownership-diagnostics` into the manuscript. Source evidence and experiment review are documented at code commit `b1f58b56a05762ec32a3a144cf0e62f97474e825`.

Observation: the abstract, introduction, protocol, Results, discussion, and conclusion incorporate prompt-conditioned clinical attribution. Table 3 reports Mass competition across three prompt packages. Appendices G and H contain score-coordinate diagnostics, capability calibration, exact prompt text, mapping-component definitions, and response-energy comparisons. The original dose-response figure and intervention summary remain in Appendix F.

Gate decision: `PAPER_ENCODING_INTEGRATION PASS`. Independent internal review and the pinned `claude-fable-5-1` medium-effort read-only review agree on the scientific interpretation and definitions. The final receipt is `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260905T013526301999Z.json`; it records the expected canonical model, valid transport, and unchanged clean server checkout `b1f58b56a05762ec32a3a144cf0e62f97474e825`. The numeric and initial story review is recorded by `review-20260905T013327770851Z.json` in the same directory.

Verification: Tectonic compiled 24 pages, with the main body ending on page 9 and the appendix starting separately on page 11. Full-page visual inspection covered the changed manuscript and appendix pages. The existing draft harness passed with 10 sections, 4 external table sources, 16 citations, 52 labels, 212 abstract words, and all 27 original adjudicated values. The LaTeX log contains no overfull boxes or unresolved references. Table sizing uses `resizebox`; prompt-panel labels sit below their text boxes. Figure assets and accepted numerical datasets were preserved.

Next step: does the descriptive Mass advantage reproduce on independent patients under a prospectively locked prompt and control design? Integrate that confirmation after the experiment task accepts its results.
