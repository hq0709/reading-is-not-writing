# Reproduce the submission PDF

Use Tectonic `Tectonic 0.17.0` with binary SHA-256
`a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da`. From this directory, run:

```bash
SOURCE_DATE_EPOCH=1788439147 FORCE_SOURCE_DATE=1 tectonic -X compile main.tex --keep-logs
```

The resulting `main.pdf` must have SHA-256 `5e4a92cf0208beba925cca4910c5d605a964a38ed8694f6c64b4c985b943cbc4`.
`MANIFEST.sha256` records every submitted source asset.
