# Regression

## Golden values

`golden_values.json` is the machine-readable regression fixture produced from a
real monthly export. It contains the input fingerprint (file MD5, row and prism
counts, station cycle counts, period), repeatability, the station E rotation
history and processing change, the G1 cluster results, `HLO-R7`, final class
counts, tangential-only candidates, daytime-bias prisms and prisms lost in the
final 48 hours.

The fixture and the raw export are **private**:

- `golden_values.json` is supplied in the confidential bundle. In this
  repository it is available only locally (for example under `.reference/`);
  `golden_values.json`, `reference_code/*.json` and `tests/fixtures/private/`
  are excluded from version control.
- The raw monthly export must never be committed. It contains site locations
  and identifiers, and the export in the reference bundle is a shifted dummy
  grid whose owner converts it separately.
- Never derive a fake fixture from the rounded table in the handoff document.
  Rounded reference values are descriptions, not tolerances.

## Public CI uses synthetic fixtures only

The public test suite generates its own data with the independent forward model
in `tests/synthetic.py` (no pipeline code shared), with explicit injected
effects: rotation step and drift, station translation, scale error, local
prism motion, daytime LOS bias and dropouts. Tests assert recovery of the
injected values with tolerances documented as fixture properties.

Whenever the private files are absent, the real-data regression must **skip
with an explicit message**, never silently pass and never substitute synthetic
data for the real input.

## Running the real-data regression locally

1. Place the private files where the local-only regression test expects them,
   without adding them to git:

   ```text
   tests/fixtures/private/HLO_Sept_2026.csv     # raw export, never committed
   tests/fixtures/private/golden_values.json    # machine-readable fixture
   ```

   `tests/fixtures/private/` is listed in `.gitignore` for this purpose.

2. Run the regression only:

   ```bash
   python -m pytest tests/test_golden_regression.py -q
   ```

3. To run the whole suite while the private data are available, use the normal
   command. The regression test stays skipped with a clear reason when the
   private path is not configured:

   ```bash
   python -m pytest tests -q
   ```

4. Verify the run manifest as part of the comparison: the manifest records the
   source hash, config hash, package/runtime versions, deterministic seeds,
   algorithms and settings, schema versions and output hashes (excluding its own
   hash). Reproducing the same source and configuration must reproduce the same
   hashes.

The comparison is only meaningful on the exact input fingerprint recorded in
the fixture (file hash and row/prism counts). A different export month, a
different station configuration or a different preprocessing path is a new
regression fixture, not a tolerance adjustment.

## Tolerances and comparison rules

- Expected values, class counts and absolute/relative tolerances come from the
  fixture definitions supplied with the private data. They are not package
  defaults and are not inferred here.
- Class counts are compared as counts, but a class label is never treated as a
  safety statement.
- Sign conventions are part of the assertion: `dD` positive away, `ver_raw`
  positive up, `tan_raw` positive clockwise from above, station rise against
  raw common vertical slope −1.
- A real-data regression run happens locally or in a separately authorized
  private environment. The public GitHub Actions workflows never receive the
  private export, the golden fixture or any credential for them.
