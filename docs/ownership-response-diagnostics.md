# Ownership response diagnostics

## Evidence

Run: `20260904T224225Z-1ab7576193a7-ownership-diagnostics` reuses the accepted 400-patient Qwen ownership cohort at dose +0.25. The source implementation is `1ab7576`; the primary artifact is `/home/qingchan/data/concept-flow/runs/20260904T224225Z-1ab7576193a7-ownership-diagnostics/artifacts/ownership-diagnostics.json`.

Observation: all six named ownership margins are negative in both probability and logit-margin coordinates, and all six matched-direction answer-score AUROC point estimates are below baseline. Effusion shifts the yes-minus-no logit margin upward in both label groups. The logit-effect matrix has a dominant first singular component.

Gate decision: the retrospective descriptive interpretation was accepted by the experiment task. Its pinned read-only reviewer receipt is `/home/qingchan/.codex/state/claude-review-concept-flow/review-20260904T225429332978Z.json`. The registered ownership and alias decisions retain their original inferential status.

Next step: does the response follow clinical meaning when the answer-code mapping is exchanged? Integrate these diagnostics with the accepted answer-encoding crossover evidence when organizing the manuscript.

## Appendix text

### Score coordinates and answer discrimination

We retrospectively analyze the stored outputs from the 400-patient Qwen ownership cohort at the fixed intervention dose of +0.25. The logit margin is the model's yes-token logit minus its no-token logit; its change is measured relative to the same patient's unperturbed output. In each score coordinate, the ownership margin is the matched direction's mean effect minus the largest mean effect among the five other clinical directions. Table~\ref{tab:response-diagnostics} compares these margins with answer discrimination. Here, AUROC ranks patients by the model's answer logit margin against the corresponding disease labels, rather than by a fitted probe score.

```latex
\begin{table}[ht]
\centering
\caption{Score-coordinate ownership and answer discrimination.}
\label{tab:response-diagnostics}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrr}
\toprule
Question & \shortstack{Probability\\ownership} & \shortstack{Logit-margin\\ownership} & \shortstack{Baseline\\AUROC} & \shortstack{Matched-direction\\AUROC} \\
\midrule
Effusion      & $-0.0640$ & $-0.4853$ & 0.6178 & 0.5921 \\
Atelectasis   & $-0.2295$ & $-1.0703$ & 0.6235 & 0.5479 \\
Pneumothorax  & $-0.1086$ & $-0.8944$ & 0.6666 & 0.6324 \\
Cardiomegaly  & $-0.1972$ & $-0.8559$ & 0.6217 & 0.5152 \\
Mass          & $-0.1528$ & $-0.7134$ & 0.6636 & 0.6354 \\
Nodule        & $-0.0831$ & $-0.4134$ & 0.6805 & 0.6355 \\
\bottomrule
\end{tabular}%
}
\end{table}
```

All six ownership margins remain negative in logit-margin coordinates as well as probability coordinates. The observed negative ownership pattern therefore persists before the sigmoid mapping to probability. Answer-score AUROC is lower under each matched direction in the point estimates, separating a larger affirmative response from better patient discrimination.

### Label-conditioned response and matrix geometry

For the Effusion direction on the Effusion question, the mean logit-margin shift is 2.1336 among 29 label-positive patients and 2.2712 among 371 label-negative patients. The positive-minus-negative difference is -0.1376, with a pointwise 95% patient-bootstrap percentile interval of [-0.4443, 0.1659] from 2,000 draws. Both label groups show a substantial upward shift, while the interval leaves their mean-shift difference unresolved. The Brier score, the mean squared difference between the model's yes probability and the binary label, rises from 0.0681 to 0.1035.

The six-by-six matrix of mean logit-margin changes has clinical directions as rows and questions as columns. Its first singular component accounts for 90.51% of the sum of squared matrix entries. Subtracting the row and column means and adding back the grand mean leaves an interaction component with 7.91% of that same raw matrix energy. These two summaries describe distinct decompositions with a common denominator: the intervention effects concentrate along a dominant matrix component, with a smaller residual after removing additive direction and question effects.

### Interpretation limits

These are retrospective descriptive analyses of the original ownership cohort. The AUROC comparisons are point estimates, and the reported interval is pointwise rather than a simultaneous bound. Matrix geometry characterizes response structure; establishing a shared causal channel requires intervention evidence beyond this decomposition.
