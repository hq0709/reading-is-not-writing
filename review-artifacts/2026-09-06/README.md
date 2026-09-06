# Review artifacts (2026-09-06)

This directory preserves the 35-page external-review input based on paper
commit `2602850ce64d9984fd80e24ce21b1e76b58b442e`. It is a historical review
copy, distinct from the 39-page formal manuscript at
`a9a5e6e5397a380f05dcb8115086a055c063b50a`.

- `concept-flow-review-35p.pdf`: frozen review PDF.
- `source.tar`: the tracked paper files from the historical baseline.
- `concept-flow-review-under35.tex`: the wrapper that applies the review
  copy's appendix layout and includes `main.tex`.

To reproduce the review layout with the project's Tectonic toolchain,
extract `source.tar` into a separate directory, copy the wrapper into that
directory, and compile `concept-flow-review-under35.tex` from there.
The source archive and wrapper together supply the review-copy source;
the current repository's `main.tex` is not its baseline.

The accompanying reports are preserved verbatim in
[`docs/reviews/concept-flow-iclr-review.md`](../../docs/reviews/concept-flow-iclr-review.md)
and
[`docs/reviews/concept-flow-external-review-triage.md`](../../docs/reviews/concept-flow-external-review-triage.md).
The first assesses baseline `ffa6e66edf4202088e1974e60e7347e9b868d648`;
the second concerns `2602850` and this 35-page review copy. Their historical
judgments and the original files outside this repository are unchanged.
