# Reproduce the accepted paper PDF

Use Tectonic `Tectonic 0.17.0` with binary SHA-256
`a98aa59ad5c1df39a6c9e56cbfc5088f2b11d6c179c0130b97998e4bd46a46da`. From this directory, run:

```bash
SOURCE_DATE_EPOCH=1788439147 FORCE_SOURCE_DATE=1 tectonic -X compile main.tex --keep-logs
```

The resulting `main.pdf` must have SHA-256 `e88c7f9a42baa4f04c7ccdbfdf9653501e4f96a81bd1c1ee238163671e5106e2`.
`MANIFEST.sha256` records every submitted source asset.
