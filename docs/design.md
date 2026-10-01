# rts-forensics — repository and API proposal

Status: design proposal; implementation and publication have not started.
Source: the supplied `PROJECT_HANDOFF.md`, especially sections 3, 4, 8 and 9.

## Purpose and acceptance criteria

Build a Python package, CLI and Streamlit dashboard for auditable analysis of monthly RTS exports. Separate movement concern from measurement reliability. Preserve original input bytes, observations, repeat measurements and flags. Retain raw and experimental frame-corrected results side by side in tables, plots, reports and GIS products. Apply no TARP unless supplied. Never infer a CRS, identify a slope as safe/unsafe, or describe unobserved periods as stable.

Only `PROJECT_HANDOFF.md` is currently available. `golden_values.json` and `reference_code/` are needed for faithful refactoring and exact regression validation. Reference values in section 5 are rounded descriptions and must not be substituted for the missing machine-readable fixture or its tolerances.

## Architecture choice

Recommended: a pandas/numpy/scipy analytical core with typed configuration/result dataclasses, a thin CLI, and a thin Streamlit dashboard. The core never imports Streamlit. Both interfaces invoke the same pipeline, so there is one set of algorithms and one audit trail.

Alternative: FastAPI and a separate JavaScript frontend. This adds deployment and interface maintenance without improving the numerical MVP. Keeping the research scripts as wrappers would be faster initially, but retains hidden configuration, pickle dependencies and hard-coded lists; it does not meet the requested specification.

## Proposed repository tree

```text
rts-forensics/
  pyproject.toml
  README.md
  LICENSE
  CONTRIBUTING.md
  config.yaml
  .gitignore
  .github/workflows/ci.yml
  .github/workflows/release.yml
  src/rts_forensics/
    __init__.py
    config.py
    models.py
    provenance.py
    geometry.py
    io.py
    cycles.py
    displacement.py
    noise.py
    rates.py
    frame.py
    classify.py
    pipeline.py
    cli.py
    report.py
    gis.py
    tests/
      __init__.py
      groups.py
      sampling.py
      references.py
      missingness.py
      events.py
    dashboard/
      app.py
      views.py
    templates/
      report.html.j2
      qgis/
  tests/
    conftest.py
    synthetic.py
    test_io.py
    test_cycles.py
    test_displacement.py
    test_noise.py
    test_rates.py
    test_frame.py
    test_forensic_tests.py
    test_classify.py
    test_report.py
    test_gis.py
    test_cli.py
    test_dashboard_contract.py
    test_golden_regression.py
    fixtures/synthetic/
  docs/
    design.md
    method.md
    pitfalls.md
    limitations.md
    configuration.md
    api.md
    regression.md
  reference_code/
    README.md
```

`rts_forensics.tests` contains analytical tests of hypotheses; the root `tests/` contains pytest software tests. The two namespaces are deliberately distinct. Supplied research scripts may be retained after inspection for private data or embedded confidential information; production modules will not execute them or load their pickles. Raw exports, private fixtures, generated site results, credentials and unreviewed source bundles are excluded from version control. A project license must be chosen explicitly before public publication.

> Implementation note: the site confirmed that the supplied export uses a shifted dummy grid (the data owner restores the real grid separately), so `golden_values.json` and the handoff document are included in this repository. The research scripts are still expected to be dropped into `reference_code/` locally; production modules never import them.

## Shared data contracts

All tables use stable identifiers and explicit units. Distances used for geometry are metres; displacements and translations are millimetres; angles used internally are radians; exported rotation is arcseconds; rates are mm/day. Scale in ppm is recorded explicitly. Every table declares its grain, units and key columns in a generated dictionary.

Types:

- `ParsedExport`: canonical observations, original column mapping, parser diagnostics, original file SHA-256 and provenance index.
- `CycleResult`: observations with station/cycle IDs, cycle table, station-assignment diagnostics, processing segments and candidate frame events.
- `DisplacementResult`: prism-cycle rows, segment baselines and source links.
- `NoiseResult`: pooled repeatability, robust residual noise and flags.
- `FrameResult`: coefficients, standard errors, rank/conditioning diagnostics, residuals, frame-member decisions and correction status.
- `ForensicResult`: evidence tables and investigation records; unavailable/disabled tests have an explicit reason.
- `ClassificationResult`: movement concern, reliability, measurement flags and optional TARP state as separate fields.
- `RunResult`: the above results, registered output tables, effective configuration, execution audit and manifest.

Each source observation has `(source_sha256, src_line)` and an immutable observation ID. Physical source line numbers are retained even with CRLF, quoted delimiters or multiline CSV fields. Aggregations retain all contributing observation IDs in a normalized `source_links` table; a displacement also links to its baseline observations. Frame-corrected values link to the fitted cycle and the frame-set observations. Inferential outputs link to the specific rows/blocks used, not merely to the entire input file.

## Proposed public APIs

Signatures below describe contracts, not implemented functions. DataFrame inputs are validated against the documented schemas. Functions return new objects rather than mutating supplied raw data.

| Module | Public API | Responsibility |
|---|---|---|
| `config` | `load_config(path) -> AnalysisConfig`; `validate_config(config) -> ConfigAudit` | Validate units, windows, settings and threshold provenance; distinguish numerical settings from screening and TARP. |
| `io` | `detect_format(source) -> FormatSpec`; `read_geomos(source, *, config) -> ParsedExport`; `parse_dms(value) -> float` | Detect delimiter/encoding/schema, explicitly parse site-local times and DMS, preserve raw bytes and row provenance. |
| `cycles` | `assign_stations(observations, *, config) -> StationAssignment`; `build_cycles(observations, *, config) -> CycleResult`; `detect_processing_changes(cycles, *, config) -> ChangeResult` | Assign stations across resection shifts; group cycles using actual elapsed times; flag processing changes without fixed dates. |
| `displacement` | `aggregate_prism_cycles(observations, *, config) -> DataFrame`; `compute_displacements(prism_cycles, *, segments, config) -> DisplacementResult` | Keep repeats, circularly aggregate Hz, compute raw polar displacement and independently segmented delivered coordinates. |
| `noise` | `pooled_repeatability(observations, *, cycles) -> DataFrame`; `estimate_noise(series, *, config) -> NoiseResult`; `flag_spikes(series, *, noise, config) -> DataFrame` | Pooled repeat SD, MAD residual noise, retained spike flags. |
| `rates` | `backward_blocks(series, *, duration, anchor) -> DataFrame`; `theil_sen_rate(series, *, metric, window, config) -> RateEstimate` | Anchor full 24-hour blocks at the final observation; use elapsed days, report CI and partial-block handling. |
| `frame` | `select_frame_set(summary, *, exclusions, config) -> FrameSelection`; `fit_cycle_frame(prism_cycle, *, frame_set, config) -> CycleFrameFit`; `fit_frames(series, *, frame_set, config) -> FrameResult`; `apply_frame(series, frames) -> DataFrame` | Weighted Huber IRLS rotation/translation/scale/vertical reconstruction with explicit rank and uncertainty diagnostics. |
| `tests.groups` | `candidate_clusters(series, *, coordinates, config) -> ClusterResult`; `spatial_placebo(...)`; `temporal_split(...)`; `change_point(...)`; `pairwise_los(...)`; `movement_vectors(...)` | Find spatial candidates without a fixed G1 list; investigate robustness and vector consistency. |
| `tests.sampling` | `day_night_comparison(...)`; `hour_matched_change(...)`; `sampling_composition(...)`; `daytime_bias(...)` | Compare matching observation hours and group membership; expose diurnal and changing-composition effects. |
| `tests.references` | `reference_inference(...)`; `resection_validation(...)` | Cycle-detrended Spearman tests with Bonferroni correction and expected station-rise/raw-vertical slope −1. |
| `tests.missingness` | `coverage(...)`; `lost_vs_retained(...)`; `pre_loss_trends(...)`; `observability_by_hour(...)` | Identify final-window dropouts, coverage/gaps and missingness patterns. |
| `tests.events` | `balanced_step_test(...)`; `common_mode_steps(...)` | Compare frame changes in time-of-day-balanced windows; record candidate events and competing explanations. |
| `classify` | `screen_movement(summary, *, config) -> DataFrame`; `grade_reliability(summary, *, config) -> DataFrame`; `apply_tarp(series, *, tarp) -> DataFrame`; `classify(summary, evidence, *, config) -> ClassificationResult` | Exploratory evidence screening, separately configured reliability grades and separately applied user TARP. |
| `report` | `write_tables(run, out_dir)`; `plot_results(run, out_dir)`; `render_report(run, out_dir) -> ReportPaths`; `write_manifest(run, out_dir) -> Path` | CSV tables, figures, Markdown/HTML and SHA-256 audit manifest. |
| `gis` | `write_geopackage(run, path, *, config) -> GISExport`; `transform_grid(points, vectors, uncertainties, *, transform) -> GISData`; `write_qml(...)` | Ten layers, undefined Cartesian SRS by default, paired vectors, styles and correctly transformed vector/covariance components. |
| `pipeline` | `run_analysis(source, *, config) -> RunResult`; `export_run(run, out_dir) -> OutputManifest` | Deterministic orchestration, shared by CLI and dashboard. |

All forensic-test APIs use named inputs such as `series`, `cycles`, `coordinates`, `station_series` and `config`; final signatures will be fixed after inspecting their reference algorithms. No test accepts a hard-coded site prism list or event date.

## Numerical and geotechnical behavior

1. Treat D, Hz and V as observations, and delivered target/station coordinates as processing products. Use a normalized local azimuth obtained from geometry/orientation diagnostics rather than interpreting Hz as a grid azimuth.
2. Assign stations using configured spatial association rules or explicit station anchors. Exact station-coordinate equality cannot be used because resection changes coordinates. Never choose a hidden clustering tolerance.
3. Keep repeated observations. Aggregate Hz circularly, including wrap-around at 0/360 degrees. Do not compare circular angles by ordinary subtraction.
4. Create baselines from the documented 48-hour window. Baseline anchor, late-starting-prism handling, minimum completeness and end-change estimator must be recovered from the reference code or explicitly configured.
5. Delivered coordinate changes have a separate baseline for every processing segment. A rate, net change, plot line or block statistic may never bridge these segments. Raw polar series remain available across the event when supported by observations.
6. Fit the eight documented frame terms: rotation, east/north translation, scale, height, vertical index and two tilts. Check identifiability and geometry before fitting. A rank-deficient or failed fit produces a status and missing corrected values, never fabricated zero correction. Any geometric conditioning decision needs an explicit rule.
7. Use the documented weights (Hz 0.8 arcsec, LOS 0.8 mm, vertical 3 mm) only as attributed research settings. Huber tuning, convergence rules, covariance convention and frame exclusion logic must come from the scripts or user configuration.
8. Keep raw and corrected LOS, vertical and tangential columns adjacent. Frame correction is marked experimental and relative to the chosen network. Publish frame membership and exclusions. Leave-one-out/reference-under-test exclusions prevent a tested point from proving its own stability.
9. Preserve raw tangential signals as well as corrected ones. Tangential-only observations require independent consistency evidence before any movement label.
10. Use 24-hour block medians counted backwards from the final record for the specified 30-day and 7-day rates. The 72-hour estimate uses all observed cycles and visibly notes its autocorrelation limitation. Retain block counts and start/end bounds. Do not silently include biased partial blocks.
11. Group statistics record member IDs and actual contributors for every epoch; constant-membership and hour-matched comparisons are separate products. A group shrinking to one prism must be apparent.
12. Apply a configured transform only on export; preserve original grid data. Rotate vectors and transform their covariance/uncertainty geometry together with point coordinates. Without external CRS metadata use undefined Cartesian SRS (srs_id −1), not EPSG:4326.

## Configuration and threshold policy

`config.yaml` is versioned and its effective values are written to every run. Site-specific prism exclusions, station definitions and known processing events are configuration inputs, never package constants. Configurations may be provided via the CLI or dashboard and must produce identical results.

Numbers explicitly stated in the handoff can be offered as attributed exploratory research settings:

| Setting | Supplied value | Qualification |
|---|---|---|
| Baseline window | 48 h | Anchor/completeness semantics still require the reference algorithm. |
| Final no-data window | 48 h | Unobserved means unavailable, not stable. |
| Cycle gaps | E 30 min; W 15 min | Per-station configuration after station assignment; do not assume other stations match E or W. |
| Noise rolling window | 24 h, centered | Edge handling and minimum observations must be specified. |
| Robust SD scale | 1.4826 × MAD | Statistical conversion factor, not an alarm. |
| Rate blocks/windows | 24 h; 30 d, 7 d, 72 h | 72 h uses cycles and its CI ignores autocorrelation. |
| Rate confidence | 95% | Same block medians as the rate. |
| Exploratory detection | net ≥ max(3σ, floor) plus same-sign CI excluding zero | LOS floor 2 mm; vertical floor 5 mm. No TARP meaning. |
| Insufficient observations | fewer than 20 prism-cycles | Exploratory screen. |
| Spike flag | absolute robust z > 6 | Flag retained observations. |
| Frame minimum cycles | 100 | Also reliability A and configured exclusions. |
| Frame residual scales | 0.8 arcsec; 0.8 mm; 3 mm | Research weighting assumptions, not thresholds for safety. |
| Spatial placebo neighborhood | 7 nearest prisms | Full comparison/membership semantics require the script. |

No unsupported value is assigned to:

- station association tolerance;
- station-coordinate/orientation jump and common-mode detection thresholds;
- Huber cutoff, convergence settings and conditioning limits;
- reliability A–D boundaries and weaker-evidence/possible criteria;
- spatial cluster size, radius, coherence and significance decisions;
- daytime bias, reference inference and missingness decision significance;
- day/night hour definitions, sampling sufficiency and balanced-window completeness;
- TARP thresholds, metric conventions, averaging, persistence and missing-data policy;
- regression absolute/relative tolerances.

Unprovided rules have a machine-readable `not_configured` status; no automatic grade or classification is invented. An audit-only run can still parse, preserve provenance, show diagnostics and compute metrics supported by explicit settings. A requested full run reports the missing required configuration rather than silently completing an incomplete analysis. The exact research labels from section 5 are not a classifier trained on prism names.

## Outputs and dashboard

The CLI is exactly:

```bash
rts-forensics run data.csv --config config.yaml --out results/
```

Outputs include the section-6 tables, canonical observations with provenance, prism-cycle series, baseline table, repeatability/noise, processing segments, frame membership/diagnostics, source links, hypothesis evidence, CSV dictionaries, figures, `report.md`, `report.html`, effective config, styled GeoPackage and SHA-256 manifest. The manifest contains source/config hashes, package/runtime versions, deterministic random seeds, algorithms/settings, schema versions and output hashes. It excludes its own hash to avoid a recursive digest.

Proposed ten GIS layers:

1. prism_summary
2. station_cycles
3. frame_cycles
4. timeseries_24h (temporal)
5. event_register
6. investigation_register
7. field_checks (QField form)
8. sight_lines
9. candidate_zones
10. movement_vectors

Nonspatial tables use GeoPackage attributes registration. Movement-vector features have an explicit raw/frame-corrected/artefact role, method, contributing interval and uncertainty fields. Candidate-zone construction will follow an explicit configured spatial rule. QML is stored in default `layer_styles`, with geometry-generator arrows scaled by `@hlo_vec_scale`. Exact compatibility with the source bundle's ten layers will be checked after the GIS scripts are supplied.

Streamlit exposes the six requested pages: Upload & audit; Station frame; Prism explorer; Map; Events & investigations; Downloads. Movement concern and reliability are separate controls and fields. Raw versus corrected series, day/night composition, correction availability, frame uncertainty and source lines remain inspectable. The map is drawn in the supplied Cartesian grid without a geographic basemap unless a real transform/CRS is configured. Downloaded results are generated by the same exporter as the CLI.

Every report and dashboard displays the section-8 limitations. No slope stability/FoS from RTS alone; no alarm state without a TARP; no absolute movement without external control; no causal claims without external drivers; no confirmed resection-reference identities from correlations; no stability claim over gaps.

## Test-first implementation sequence

Each module follows write test → observe expected failure → implement → verify → refactor. Tests use synthetic IDs and dates unrelated to the site. Synthetic signal amplitudes and numerical assertion tolerances are test-fixture properties and are documented separately from configurable operational decisions.

1. **Fixtures and parser:** independent forward model for polar observations; cp1252 degree signs, quoted DMS, CRLF, source lines, schema detection, malformed rows and original-byte digest. Inject rotation step/drift, translation, scale, local motion, diurnal bias and dropouts. Keep the fixture generator independent of production frame/displacement routines so sign mistakes are detectable.
2. **Stations and cycles:** multiple stations, resection coordinate changes, interleaving, out-of-order rows, repeated observations and gap-boundary cases. Cycle decisions use elapsed minutes.
3. **Displacement:** circular aggregation, polar signs/units, repeated-source lineage, baseline uncertainty/completeness and delivered-coordinate segmentation. Include a processing jump that must never enter a bridged rate.
4. **Noise and rates:** pooled within-cycle SD, MAD noise, retained spikes, millimetre quantization, backwards-block anchoring, partial-day/diurnal bias, irregular timestamps and explicit CI/insufficient-data behavior.
5. **Frame:** recover known eight-term effects with geometry sufficient to identify them; test outlier robustness, independent local movement, excluded references, sparse/rank-deficient geometry, correction availability and raw immutability. Exact recovery tolerances require agreement/reference provenance.
6. **Forensics:** unknown cluster IDs, spatial placebo, fixed membership, day/night/hour matching, temporal split, change point, paired LOS, vector signs, multiple-testing correction, resection slope, biased targets and dropouts.
7. **Classification:** boundary tests for supplied screening settings, separate concern/reliability, unsupported rules marked unconfigured, no alarm without TARP and supplied persistence/missingness behavior.
8. **Exports/UI:** CLI end-to-end run, raw/corrected paired columns/plots, source links, manifest verification, undefined SRS, ten-layer schema, QML/default styles, rotated vectors/covariances and the six dashboard page contracts.
9. **Real-data regression:** locally supplied raw export plus actual `golden_values.json`; run when the private data path is configured and otherwise clearly skip. Validate metrics and class counts only through explicit fixture definitions/tolerances. Do not commit private input or derive a fake fixture from the rounded handoff table.

CI runs ruff, pytest and a documentation build; build/package checks run against the supported Python versions. Release publishes an explicitly tagged build using configured trusted publishing. Real-data regression runs locally or in a separately authorized private environment; public CI uses synthetic fixtures only.

## Inputs needed before faithful implementation

1. Upload the original bundle containing `golden_values.json` and `reference_code/` (a ZIP is sufficient). The private monthly export is optional for implementation and required later to run the real-data regression.
2. Resolve unspecified decision rules through the supplied scripts or user settings. Reuse documented research screening values with attribution; never introduce TARP or other unknown decisions silently. If the bundle does not define a needed value, ask for that value before implementing the decision.
3. For GitHub publication, the authenticated account is `fanelix`; `fanelix/rts-forensics` was not accessible/existing at inspection. Current GitHub connector tools do not expose repository creation, so publication requires an available authorized creation route or an empty repository supplied by the user. Source implementation can proceed locally once the essential inputs are present.
4. Choose an open-source license before public release; do not publish the raw site export or confidential bundle content.
