# API

The package is organised around stable data contracts in
`rts_forensics.models`: raw observations with provenance, typed result
dataclasses and registered output tables (`TABLE_SPECS`). Functions return new
objects; raw inputs are never mutated.

Two namespaces with similar sounds are deliberately distinct:

- `rts_forensics.tests` — analytical hypothesis tests exposed as a package
  (group, sampling, reference, missingness and event tests);
- `tests/` — pytest software tests for this repository, including synthetic
  data generators and the dashboard contract test.

## Quick start: full pipeline

```python
from rts_forensics.config import load_config, validate_config
from rts_forensics.pipeline import export_run, run_analysis

config = load_config("config.yaml")
audit = validate_config(config)
run = run_analysis("HLO_Sept_2026.csv", config=config)
manifest = export_run(run, "results/")
print(run.classification.statuses)      # per-family statuses, including not_configured
print(manifest.outputs)                 # written files with SHA-256
```

`run_analysis` parses the upload once, preserves the original bytes and source
line numbers, runs every configured stage, and returns a `RunResult`. Stages
whose decision rules are not configured come back with an explicit status and
reason rather than guessed values. `export_run` writes the same outputs the CLI
writes and returns the output manifest.

## CLI

```bash
rts-forensics run data.csv --config config.yaml --out results/
```

The CLI is a thin wrapper over `run_analysis` and `export_run`; there is no
separate command-line algorithm. The same exporters are used by the dashboard's
Downloads page.

## `rts_forensics.config`

| Object | Purpose |
|---|---|
| `load_config(path=None, overrides=None) -> AnalysisConfig` | load YAML (partial allowed, unknown top-level keys rejected), deep-merge overrides |
| `validate_config(config) -> ConfigAudit` | dotted-path audit of configured / not-configured decisions plus warnings |
| `AnalysisConfig` | typed sections: `input`, `cycles`, `stations`, `baseline`, `noise`, `rates`, `screening`, `reliability`, `frame`, `forensics`, `tarp`, `gis`, `provenance`; `to_dict()`, `from_dict()`, `hash()` |
| `ConfigAudit` | `configured`, `not_configured`, `warnings`, `summary()`, `requires(path)` |
| `default_config()` | versioned defaults |

## `rts_forensics.models`

Canonical schemas (`OBSERVATION_COLUMNS`, `PRISM_CYCLE_COLUMNS`,
`BASELINE_COLUMNS`, `DISPLACEMENT_COLUMNS`, `RATE_COLUMNS`,
`EVIDENCE_COLUMNS`, `INVESTIGATION_COLUMNS`, `CONCERN_COLUMNS`,
`RELIABILITY_COLUMNS`, `TARP_COLUMNS`, `SOURCE_LINK_COLUMNS`), the status
vocabulary (`NOT_CONFIGURED`, `STATUS_OK`, ...), the table registry
`TABLE_SPECS` / `table_dictionary()` and the result dataclasses
(`ParsedExport`, `CycleResult`, `DisplacementResult`, `NoiseResult`,
`FrameResult`, `ForensicResult`, `ClassificationResult`, `RunResult`).

## `rts_forensics.provenance`

| Function | Purpose |
|---|---|
| `sha256_bytes(data)` / `sha256_file(path)` | content hashes |
| `make_observation_id(source_sha256, src_line)` | immutable observation ID |
| `write_raw_sources(sources, raw_dir)` | copy original bytes, return hashes |
| `build_source_links(table_name, rows)` | normalized provenance links |
| `runtime_audit(seed=0)` | package/runtime versions and deterministic seed |
| `hash_outputs(paths, root=None)` | output SHA-256 map |
| `build_manifest(...)` / `write_manifest(manifest, path)` | run manifest (excludes its own hash) |

## `rts_forensics.io`

| Function | Purpose |
|---|---|
| `detect_format(source) -> FormatSpec` | detect delimiter, encoding, time format and schema |
| `read_geomos(source, *, config) -> ParsedExport` | parse once, retain original bytes, line numbers, diagnostics and source links |
| `parse_dms(value) -> float` | explicit DMS parsing (degree/minute/second, sign) |

```python
from rts_forensics.config import load_config
from rts_forensics.io import read_geomos

parsed = read_geomos("data.csv", config=load_config("config.yaml"))
print(parsed.sources[0].sha256, len(parsed.observations))
```

## `rts_forensics.cycles`

| Function | Purpose |
|---|---|
| `assign_stations(observations, *, config) -> StationAssignment` | station clusters (auto-detected or configured anchors), assignment diagnostics |
| `build_cycles(observations, *, config) -> CycleResult` | elapsed-time cycles, cycle table, processing segments, candidate events |
| `detect_processing_changes(cycles, *, config) -> ChangeResult` | station-coordinate/orientation changes, reported without fixed dates |

## `rts_forensics.displacement`

| Function | Purpose |
|---|---|
| `aggregate_prism_cycles(observations, *, config) -> DataFrame` | median repeat aggregation, circular Hz |
| `compute_displacements(prism_cycles, *, segments, config) -> DisplacementResult` | `dD`, `rad_raw`, `ver_raw`, `tan_raw`, segmented `dE/dN/dZ`, baselines, source links |

## `rts_forensics.noise`

| Function | Purpose |
|---|---|
| `pooled_repeatability(observations, *, cycles) -> DataFrame` | pooled within-cycle repeat SD |
| `estimate_noise(series, *, config) -> NoiseResult` | robust MAD residual noise |
| `flag_spikes(series, *, noise, config) -> DataFrame` | retained spike flags (never deleted) |

## `rts_forensics.rates`

| Function | Purpose |
|---|---|
| `backward_blocks(series, *, duration, anchor) -> DataFrame` | 24-hour blocks counted back from the final record |
| `theil_sen_rate(series, *, metric, window, config) -> RateEstimate` | Theil–Sen on block medians with CI; 72-hour window uses cycles and states its autocorrelation limitation |
| `compute_rates(displacements, *, config) -> DataFrame` | all configured metrics and windows |

## `rts_forensics.frame`

| Function | Purpose |
|---|---|
| `select_frame_set(summary, *, exclusions, config) -> FrameSelection` | reliability and minimum-cycle rules, published membership/exclusions |
| `fit_cycle_frame(prism_cycle, *, frame_set, config) -> CycleFrameFit` | weighted Huber IRLS eight-term fit with rank/conditioning diagnostics |
| `fit_frames(series, *, frame_set, config) -> FrameResult` | per-cycle fits and the corrected series |
| `apply_frame(series, frames) -> DataFrame` | raw columns kept beside `los_fc_mm`, `tan_fc_mm`, `ver_fc_mm` |

A failed or rank-deficient fit returns a status and missing corrected values,
never a zero correction.

## `rts_forensics.tests` (analytical tests)

| Module | Public API |
|---|---|
| `groups` | `candidate_clusters`, `spatial_placebo`, `temporal_split`, `change_point`, `pairwise_los`, `movement_vectors` |
| `sampling` | `day_night_comparison`, `hour_matched_change`, `sampling_composition`, `daytime_bias` |
| `references` | `reference_inference`, `resection_validation` |
| `missingness` | `coverage`, `lost_vs_retained`, `pre_loss_trends`, `observability_by_hour` |
| `events` | `balanced_step_test`, `common_mode_steps` |

All take named inputs such as `series`, `cycles`, `coordinates`, `station_series`
and `config`. No test accepts a hard-coded site prism list or event date.
> These modules are exposed by the package contract; they are imported by
> `pipeline` only, so the root `tests/` suite and the dashboard stay independent
> of any single test family.

## `rts_forensics.classify`

| Function | Purpose |
|---|---|
| `screen_movement(summary, *, config) -> DataFrame` | exploratory movement concern (no TARP meaning) |
| `grade_reliability(summary, *, config) -> DataFrame` | reliability grades, separate from concern |
| `apply_tarp(series, *, tarp) -> DataFrame` | apply a user TARP only when supplied |
| `classify(summary, evidence, *, config) -> ClassificationResult` | concern, reliability, flags and TARP as separate fields |

## `rts_forensics.report`

| Function | Purpose |
|---|---|
| `write_tables(run, out_dir)` | CSV tables plus the generated data dictionary |
| `plot_results(run, out_dir)` | figures with raw and corrected series side by side |
| `render_report(run, out_dir) -> ReportPaths` | `report.md` and `report.html` including the limitations |
| `write_manifest(run, out_dir) -> Path` | SHA-256 audit manifest |

## `rts_forensics.gis`

| Function | Purpose |
|---|---|
| `write_geopackage(run, path, *, config) -> GISExport` | ten layers, undefined Cartesian SRS by default, QML styles in `layer_styles` |
| `transform_grid(points, vectors, uncertainties, *, transform) -> GISData` | export-only transform rotating vector and covariance components with the geometry |
| `write_qml(...)` | write layer styles without modifying the source grid |

## `rts_forensics.pipeline`

| Function | Purpose |
|---|---|
| `run_analysis(source, *, config) -> RunResult` | deterministic orchestration shared by CLI and dashboard; parses once and preserves provenance |
| `export_run(run, out_dir) -> OutputManifest` | write all outputs and the manifest |

## `rts_forensics.dashboard`

| Object | Purpose |
|---|---|
| `dashboard.PAGES` | ordered six-page registry (`key`, `title`) |
| `dashboard.PAGE_KEYS` | the six stable keys |
| `dashboard.views.PAGE_RENDERERS` | page key to renderer function; Streamlit is imported lazily inside the renderers |
| `dashboard.app:main` | Streamlit entry point (`streamlit run src/rts_forensics/dashboard/app.py`) |

The dashboard is not imported by the core: `import rts_forensics` works without
Streamlit, and the dashboard's Downloads page calls the same `report` and `gis`
exporters as the CLI.
