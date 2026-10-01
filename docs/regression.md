# Regression

## Golden values

`tests/fixtures/golden_values.json` is the machine-readable regression fixture
produced from a real monthly export. It contains the input fingerprint (file
MD5, row and prism counts, station cycle counts, period), repeatability, the
station E rotation history and processing change, the G1 cluster results,
`HLO-R7`, final class counts, tangential-only candidates, daytime-bias prisms
and prisms lost in the final 48 hours.

The fixture is committed because the site confirmed the export uses a **shifted
dummy grid**; the data owner restores the real grid separately in their own
environment. `docs/project_handoff.md` is included for the same reason.

Rules:

- Never derive a fake fixture from the rounded table in the handoff document.
  Rounded reference values are descriptions, not tolerances.
- The raw monthly export is not part of this repository yet. Drop it at
  `tests/fixtures/HLO_Sept_2026.csv` or set `RTS_HLO_CSV` to run the
  regression; section 9 of the design (site-grid transform) stays in the
  owner's local workflow.
- A different export month, station configuration or preprocessing path is a
  new regression fixture, not a tolerance adjustment.

## Public CI uses synthetic fixtures only

The public test suite generates its own data with the independent forward model
in `tests/synthetic.py` (no pipeline code shared), with explicit injected
effects: rotation step and drift, station translation, scale error, local prism
motion, daytime LOS bias and dropouts. Tests assert recovery of the injected
values with tolerances documented as fixture properties.

Whenever the raw export is absent, the real-data regression must **skip with an
explicit message**, never silently pass and never substitute synthetic data for
the real input.

## Running the real-data regression locally

1. Place the raw export where the regression test expects it (it is not
   committed):

   ```text
   tests/fixtures/HLO_Sept_2026.csv
   ```

   or point the environment at it:

   ```bash
   export RTS_HLO_CSV=/path/to/HLO_Sept_2026.csv
   ```

2. Run the regression only:

   ```bash
   python -m pytest tests/test_golden_regression.py -q
   ```

3. To run the whole suite with the export present, use the normal command. The
   regression test stays skipped with a clear reason when the export path is
   not configured:

   ```bash
   python -m pytest tests -q
   ```

4. Verify the run manifest as part of the comparison: the manifest records the
   source hash, config hash, package/runtime versions, deterministic seeds,
   algorithms and settings, schema versions and output hashes (excluding its own
   hash). Reproducing the same source and configuration must reproduce the same
   hashes.

## Tolerances and comparison rules

- Expected values, class counts and absolute/relative tolerances come from the
  fixture definitions. They are not package defaults and are not inferred here.
- Class counts are compared as counts, but a class label is never treated as a
  safety statement.
- Sign conventions are part of the assertion: `dD` positive away, `ver_raw`
  positive up, `tan_raw` positive clockwise from above, station rise against
  raw common vertical slope −1.
- The site-grid transform is an export concern only; original grid data is
  preserved and only a configured transform is applied.
