# Reference code (not committed)

This directory is a **placeholder**. The research scripts that the
`rts-forensics` algorithms were derived from are **not committed** to this
repository.

## Why the scripts are not in git

The scripts are site-specific and confidential. They contain hard-coded prism
names, station labels, dates, site identifiers and private file paths, and they
exchange intermediate results as pickles between steps. Publishing them would
expose site information and publish code that must never run in production.

`reference_code/*.py`, `reference_code/*.json` and bundles such as
`reference_code/*.zip` are listed in `.gitignore` so an accidental copy stays
out of version control.

## Where to place them locally

If you have the confidential bundle, unpack it into this directory on your own
machine:

```text
reference_code/
  s1_parse.py
  s1_cycles.py
  s1_core.py
  s1_summary.py
  s1_extra.py
  s1_docs.py
  s1_figs.py
  s2_frame.py
  s2_g1.py
  s2_refs.py
  s2_missing.py
  s2_hourmatch.py
  s2_register.py
  s2_figs.py
  s3_final.py
  s4_qgis_build.py
  s4_qgis_styles.py
  s4_qgis_preview.py
  run_all.sh
  golden_values.json        # private regression fixture, never committed
```

Keep the local copies excluded from git. They are reading material for
implementers and reviewers, not part of the package.

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
- The private `golden_values.json` is used only by the local regression, which
  skips with an explicit reason when the private data are absent. It is never a
  package resource and never bundled in a release.

See [Regression](../docs/regression.md) for the private-data policy and the
local real-data workflow.
