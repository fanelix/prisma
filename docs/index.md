# RTS Forensics

A Python package, CLI and browser dashboard that distinguishes movement
evidence from RTS measurement-system artefacts. All results are exploratory;
frame-corrected products are experimental and network-relative.

Install with `pip install '.[web]'`, then run:

```sh
rts-forensics run /path/to/export.csv --config config.yaml --out results/
streamlit run app/app.py
```

The existing GitHub Pages dashboard also offers the forensic pipeline using
stlite. Computation runs in the browser; input uploads are not committed.
SciPy and matplotlib increase its initial download compared with legacy mode.

Outputs include audited observations, source-line provenance, repeat/noise
diagnostics, cycle/frame/membership tables, raw and corrected time series,
separate concern and reliability, evidence/event/field-check registers,
Markdown/HTML report, figures, ten-layer styled GeoPackage and SHA-256 manifest.

Input coordinates remain in the input grid (undefined Cartesian SRS -1) unless
an explicit transform is configured. The map uses local coordinates, without
guessing a geographic position.

The repository previously used `prismacore` to rank processed coordinates.
Its API is retained for compatibility. Use **Raw-observation forensics** in
the dashboard for the new handoff pipeline; legacy ranks are not equivalent.

Independent `reference_code/` and `golden_values.json` were not supplied with
this change. No claim is made that the new implementation reproduces their
real-data findings or final class counts. Synthetic ground-truth tests verify
rotation, translation, scale, local movement, day bias, dropouts, provenance,
segmentation and exports. See [method and configuration](method.md).

Existing real data were already tracked in the repository. This change does
not duplicate them or add generated field results. Keep future private data
outside git; the ignore rules cover new CSV exports and local results.
