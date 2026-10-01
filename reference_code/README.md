# Reference code

The `rts-forensics` algorithms were derived from the research scripts listed in
section 7 of `docs/project_handoff.md` (`s1_*.py`, `s2_*.py`, `s3_final.py`,
`s4_qgis_*.py`, `run_all.sh`). The scripts themselves are not in this repository
yet; drop the bundle into this directory locally when you want them available
for review.

`tests/fixtures/golden_values.json` is committed (the site confirmed the export
uses a shifted dummy grid, and the data owner restores the real grid
separately).

## How production code treats them

Production modules never import these scripts, never execute them and never
load their pickles:

- `src/rts_forensics/` imports only the package itself and its declared
  dependencies (pandas, numpy, scipy, PyYAML, plus optional jinja2/matplotlib
  and streamlit for the adapters).
- The algorithms are re-implemented as tested, configurable functions. Every
  value the scripts hard-code is either a documented supplied setting in
  `config.yaml` or `not_configured`.
- No module reads `reference_code/` at run time, and no output depends on those
  files being present.
- `golden_values.json` is used only by the regression test, which skips with an
  explicit reason when the raw export is absent. It is never bundled in a
  release.

See [Regression](../docs/regression.md) for the local real-data workflow and
[tests/fixtures/golden_values.json](../tests/fixtures/golden_values.json) for
the fixture itself.
