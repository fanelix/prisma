# Contributing

Thanks for helping improve `rts-forensics`. The project analyses monitoring data
that may feed engineering decisions, so correctness, auditability and honest
uncertainty matter more than features.

## Development setup

```bash
git clone <repository-url>
cd prisma            # repository root
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[dev,report]"      # report export: jinja2, matplotlib
pip install -e ".[dev,dashboard]"   # local dashboard: streamlit, matplotlib
```

The analytical core must stay usable without Streamlit and without matplotlib;
the dashboard and report extras are adapters.

## Running the checks

```bash
ruff check src tests
python -m pytest tests -q
mkdocs build --strict
```

- **Ruff** is the linter and import sorter (`line-length = 100`, rules
  `E, F, W, I, UP, B`). Keep the diff lint-clean.
- **pytest** runs the full suite from the repository root. Synthetic fixtures
  live in `tests/synthetic.py` and `tests/fixtures/synthetic/`.
- **mkdocs** builds the documentation with strict warnings. Add new pages to the
  `nav` in `mkdocs.yml`.

The dashboard contract test imports `rts_forensics.dashboard` in an environment
without Streamlit; never add a top-level `import streamlit` outside
`dashboard/app.py` and never add one at module level in `dashboard/views.py`.

## Honesty rules (non-negotiable)

1. **Never invent a threshold.** If the research bundle or the user does not
   define a decision rule, it is `not_configured`. Modules return an explicit
   status and reason; they do not guess, and they do not fall back to a hidden
   default. New supplied values must carry their provenance in `config.yaml`
   and in the docs.
2. **Never state that a slope is safe or unsafe.** The package produces
   exploratory evidence and separate reliability grades. It does not produce
   factor-of-safety statements or alarm states without a user-supplied TARP.
3. **Raw data are never modified, and never replaced.** Every derived value
   links to its source lines. Frame-corrected values appear beside raw values
   in every table, plot, report and GIS product — never instead of them.
4. **Flags are retained, not deleted.** Spikes, outliers and inconsistent rows
   are marked and still contribute; they remain visible through the pipeline.
5. **Coordinates stay in the input grid** unless a transform/CRS is explicitly
   configured. No CRS is inferred, and unobserved periods are never described
   as stable.
6. **Movement concern and reliability stay separate fields.** Do not collapse
   them into one grade or one colour.

## Private data

- Never commit a raw site export, `golden_values.json`, generated site results,
  credentials or the research scripts. They are site-specific and confidential.
- `reference_code/` explains where the research scripts go locally and why
  production modules never import or execute them or load their pickles.
- `tests/fixtures/private/` is the local-only place for real-data regression
  fixtures and is gitignored.
- Public CI must keep passing with synthetic fixtures only.

See [Regression](docs/regression.md) for the real-data workflow and
[Limitations](docs/limitations.md) for the statements every report and
dashboard page must keep.

## Changes to algorithms or configuration

- Add tests first, with the expected numeric behaviour and tolerances stated as
  fixture properties.
- Sign conventions are part of the contract: `dD` positive away from the
  instrument, `ver_raw` positive up, `tan_raw` positive clockwise from above.
- If a change alters a numeric result on the real-data regression, update the
  private fixture through the documented process; never loosen a tolerance to
  hide a change.
- Keep the public API and the data contracts in `rts_forensics.models` in sync
  with the docs (`docs/api.md`, `docs/configuration.md`).

## Documentation

- Docs are English and live in `docs/`.
- Document supplied values with their provenance and keep `not_configured`
  rules explicit.
- Do not add emojis.
