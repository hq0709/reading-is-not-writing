# Reproduce the submission PDF

Use Tectonic `Tectonic 0.17.0` with binary SHA-256
`99ffcfdbf1ebf8bdda9e791942e3d06aedb12463fddc33f07de6f5211c8bf08d`. From this directory, run:

```bash
SOURCE_DATE_EPOCH=1788439147 FORCE_SOURCE_DATE=1 tectonic -X compile main.tex --keep-logs
```

The resulting `main.pdf` must have SHA-256 `4696fffada2bef6214c01d9fed2eff18589e2bc210fa9d29ac97cd335dc6edae`.
`MANIFEST.sha256` records every submitted source asset.

## Figure 3 provenance

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
