# RTS forensics implementation plan and APIs

Authority: supplied PROJECT_HANDOFF(1).md, sections 3–4 and 9. The user
authorized implementation in fanelix/prisma; retain this repository name.

## Architecture

Add `rts_forensics/` alongside the existing `prismacore` compatibility API.
`parse.read_export(source, config)` retains all rows and source line numbers.
`cycles.assign_cycles(rows, config)` partitions stations before cycles.
`displacement.prism_cycles(rows, config)` preserves repeat provenance, uses
circular Hz medians, raw polar components and segmented delivered coordinates.
`noise.estimate(series, rows)` supplies repeat and rolling-residual noise.
`rates.blocks24(series)` and `rates.slope(series, column, days)` use elapsed time.
`frame.fit_frames(series, config)` returns parameter/uncertainty tables and
experimental corrections. No named prism or event date is embedded in code.
`investigations.investigate(series, frames, summary, config)` records evidence
and limitations for sampling, bias, spatial grouping, references and missingness.
`classify.summarize(series, rows, config)` separates concern from reliability.
`report.write_results(result, directory)` writes CSV, plots, Markdown/HTML,
configuration, GeoPackage, and SHA-256 manifest.
`gis.write_gpkg(result, path)` keeps the undefined Cartesian input grid by
default; a configured transform rotates vectors as well as points.
`pipeline.run(source, config)` coordinates these functions without pickles.
CLI: `rts-forensics run data.csv --config config.yaml --out results/`.
The six forensic dashboard pages reuse the same pipeline in Streamlit/stlite.

## Ordered work

1. Write synthetic recovery, circular-angle, provenance, segmentation,
   sampling, gap/rate, GIS and export tests. Run them before implementation.
2. Implement parser, validated YAML/JSON configuration, station/cycle assignment,
   baseline/repeat aggregation, diagnostic columns, noise and anchored rates.
3. Implement Huber joint frame fit with rank checks and standard errors; retain
   raw values and expose frame membership and exclusions. Reliability grade A
   cannot be inferred without an owner-supplied grading policy: explicit frame
   membership is permitted, automatic fits are marked provisional.
4. Implement evidence tables and exploratory screening using only handoff
   constants. Unspecified physical/statistical decision thresholds remain
   unset; tests can compute statistics without claiming a classification.
5. Implement report, ten-layer GIS export, CLI and six-page browser UI. Keep
   legacy API intact and accessible as a separately labelled legacy mode.
6. Add packaging, lint/test/docs CI, release workflow and method/config docs.
7. Run full suite, synthetic CLI/export/UI checks and current HLO input.
   `golden_values.json` is committed at `tests/fixtures/` and checked by
   `tests/test_golden_hlo.py` with explicit tolerances. `reference_code/` was
   not supplied, so final class counts are not claimed (strict xfail).
8. Review, commit and push the feature branch, and open a pull request.

## Verification limits

The repository already tracks a real export and results. Do not copy them into
new fixtures or generated tracked outputs; the golden test reads the export in
place from `data/`. Keep raw source bytes unchanged.
Existing Candrian golden tests refer to a file absent from this checkout;
skip only when their external fixture is absent, with an explicit reason.
No alarm or slope-safety assertion. TARP is applied only if fully supplied.
No invented screening, reliability or clustering thresholds.
