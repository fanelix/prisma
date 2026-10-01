# Configuration

All operational decisions live in `config.yaml`. Every run writes its effective
configuration and a configuration audit into the output directory, so a result
can be reproduced from the exact settings that produced it. The CLI and the
dashboard accept the same file and produce identical results.

## Policy: nothing is invented

The research bundle defines some numbers and deliberately leaves others open.

- Values explicitly stated in the handoff and the research scripts are included
  as **supplied research settings**. They are exploratory and are labelled with
  their provenance in `config.yaml`. A supplied value is never a safety
  threshold and never a TARP.
- Values the bundle does not define are `not_configured`. A module that needs
  one returns an explicit status and reason instead of guessing. Unavailable
  tests remain visible in the investigation/status tables.
- Site-specific entries (station labels, processing breaks, prism exclusions,
  a TARP, a grid transform) are **configuration inputs**, never package
  constants.
- An audit-only run can still parse, preserve provenance, show diagnostics and
  compute everything supported by explicit settings. A requested full analysis
  reports the missing required configuration instead of silently completing an
  incomplete analysis.

Use the literal string `not_configured` (or `null` where the schema allows it)
for an undecided value. `rts_forensics.config.validate_config` walks the
configuration and reports every dotted path as configured or not configured; the
third state in the audit is a list of warnings for rules that will be reported
as unavailable at run time.

## Loading, precedence and audit

```python
from rts_forensics.config import load_config, validate_config

config = load_config("config.yaml")          # partial file, defaults fill gaps
audit = validate_config(config)
print(audit.summary())
```

- A partial file is allowed; absent keys keep their documented defaults.
- Unknown top-level keys are rejected, to catch typos.
- Programmatic overrides are deep-merged over the file.
- `config.hash()` is SHA-256 of the canonical effective configuration.
- The CLI writes `effective_config.yaml` and the configuration audit with the
  results; the dashboard exposes the same information on the audit page.
- A missing file raises an error; the dashboard shows it instead of running
  with silent defaults.

Units used throughout: geometry metres, displacement/translation millimetres,
internal angles radians, exported rotation arcseconds, rates mm/day, scale ppm,
times site-local naive.

## `provenance`

| Key | Value | Provenance | Meaning |
|---|---|---|---|
| `preserve_raw_bytes` | `true` | supplied policy | copy the original upload bytes under `<out>/raw/` |
| `store_observation_ids` | `true` | supplied policy | keep immutable `(source_sha256, src_line)` identifiers |

## `input`

| Key | Value | Provenance | Meaning |
|---|---|---|---|
| `delimiter` | `null` | auto-detect | GeoMoS uses `;` |
| `encodings` | `[cp1252, utf-8, iso-8859-1]` | supplied | tried in order |
| `time_formats` | day-first plus ISO variants | supplied | day-first first, minute resolution |
| `timezone` | `site-local` | supplied | timezone is unstated in the export; never converted |
| `column_map` | `{}` | auto-detect | canonical name to source column; set for other formats |

## `cycles`

| Key | Value | Provenance | Meaning |
|---|---|---|---|
| `gap_minutes` | `{E: 30, W: 15}` | supplied site schedule | per-station cycle gap; other stations must be configured explicitly |
| `default_gap_minutes` | `30` | supplied fallback | used when no station-specific value exists |
| `processing_break` | `"2026-09-21 18:00"` | supplied known event | auto-detection also runs; both are reported. As-delivered series are never bridged across it |

## `stations`

| Key | Value | Provenance | Meaning |
|---|---|---|---|
| `association_tolerance_m` | `not_configured` | not supplied | no hidden clustering tolerance; stations are auto-detected from the largest gap in station easting |
| `anchors` | E ≈ 176064, W ≈ 175087 | supplied site configuration | explicit labels for the detected clusters; exact coordinate equality is unusable across resections |

## `baseline`

| Key | Value | Provenance | Meaning |
|---|---|---|---|
| `window_hours` | `48` | supplied | baseline window (anchor semantics from the reference algorithm) |
| `min_observations` | `6` | supplied fallback | minimum baseline observations before the first-rows fallback |
| `anchor` | `first_observations_of_segment` | supplied semantics | baselines restart at each processing segment |
| `end_change_estimator` | `median_final_48h` | supplied semantics | net change is a median of the final window |

## `noise`

| Key | Value | Provenance | Meaning |
|---|---|---|---|
| `rolling_window_hours` | `24` | supplied | centred rolling median for residual noise |
| `centered` | `true` | supplied | centred window |
| `min_observations` | `3` | supplied | minimum points for the rolling median |
| `robust_scale` | `1.4826` | statistical conversion | MAD to SD factor, not an alarm |
| `spike_z` | `6.0` | supplied | exploratory spike flag; observations are retained, never deleted |

## `rates`

| Key | Value | Provenance | Meaning |
|---|---|---|---|
| `block_hours` | `24` | supplied | block length |
| `windows_days` | `[30, 7]` | supplied | rate windows |
| `windows_hours` | `[72]` | supplied | uses all cycles; its CI ignores autocorrelation (stated with the result) |
| `confidence` | `0.95` | supplied | CI from the same block medians |
| `block_anchor` | `final_record` | supplied | blocks counted backwards from the final observation; calendar days bias diurnal signals |

## `screening`

Exploratory evidence screening only; no TARP meaning.

| Key | Value | Provenance |
|---|---|---|
| `sigma_multiplier` | `3.0` | supplied |
| `los_floor_mm` | `2.0` | supplied |
| `vertical_floor_mm` | `5.0` | supplied |
| `possible_sigma_multiplier` | `2.0` | supplied weaker-evidence rule |
| `possible_los_floor_mm` | `2.0` | supplied weaker-evidence rule |
| `possible_vertical_floor_mm` | `4.0` | supplied weaker-evidence rule |
| `min_prism_cycles` | `20` | supplied; below this the result is *insufficient data* |
| `require_same_sign_ci` | `true` | supplied; the rate CI must exclude zero with the same sign as the net change |

## `reliability`

Reliability is graded separately from movement concern. The supplied research
boundaries are exploratory; they are not alarms.

| Key | Value | Provenance |
|---|---|---|
| `boundaries.insufficient_below_cycles` | `20` | supplied |
| `boundaries.coverage_weak_pct` | `50` | supplied |
| `boundaries.coverage_moderate_pct` | `80` | supplied |
| `boundaries.max_gap_hours` | `72` | supplied |
| `boundaries.noisy_los_mm` | `2.0` | supplied |
| `boundaries.noisy_vertical_mm` | `6.0` | supplied |
| `boundaries.spike_flags_moderate` | `5` | supplied |
| `frame_min_cycles` | `100` | supplied requirement for frame membership |

## `frame`

| Key | Value | Provenance | Meaning |
|---|---|---|---|
| `weights.hz_arcsec` | `0.8` | supplied research weighting | observation weight, not a safety threshold |
| `weights.los_mm` | `0.8` | supplied research weighting | |
| `weights.vertical_mm` | `3.0` | supplied research weighting | |
| `huber_cutoff` | `1.5` | supplied | Huber IRLS cutoff |
| `convergence.max_iterations` | `30` | supplied | IRLS iterations |
| `convergence.scale_floor` | `1.0` | supplied | robust-scale floor |
| `conditioning_limit` | `not_configured` | not supplied | rank and pseudo-inverse diagnostics are reported either way |
| `min_prism_cycles_per_fit` | `20` | supplied | minimum prisms in a cycle fit |
| `frame_set_reliability` | `A` | supplied | frame members must be reliability A |
| `exclusions` | `[]` | user input | prism IDs never used as frame references |
| `site_exclusions` | `[]` | site configuration | suspected movers and biased prisms |

A rank-deficient, insufficient or failed fit returns an explicit status and
missing corrected values; it never fabricates a zero correction.

## `forensics`

| Key | Value | Provenance | Meaning |
|---|---|---|---|
| `placebo_neighbors` | `7` | supplied | nearest-neighbour placebo groups |
| `hour_class_hours` | `4` | supplied | hour classes for matched comparisons |
| `day_bands.night` / `.day` | `[20, 0, 4]` / `[8, 12, 16]` | supplied | time-of-day bands |
| `cluster.radius_m` | `not_configured` | not supplied | candidate clustering must not invent a radius |
| `cluster.min_members` | `not_configured` | not supplied | |
| `cluster.coherence` | `not_configured` | not supplied | |
| `references.alpha` | `0.05` | supplied | significance level |
| `references.bonferroni` | `true` | supplied | multiple-testing correction |
| `references.expected_rise_vertical_slope` | `-1.0` | supplied | resection validation expectation |
| `missingness.final_window_hours` | `48` | supplied | unobserved is not stable |
| `missingness.permutation_n` | `5000` | supplied | permutation test size |
| `missingness.seed` | `1` | supplied | deterministic seed |
| `missingness.decision_significance` | `not_configured` | not supplied | missingness decisions stay unavailable |
| `events.balanced_windows` | `[[16, 20, 0, 4], [20, 0, 4, 8]]` | supplied | balanced step-test windows |
| `events.step_significance` | `not_configured` | not supplied | step results report evidence, no invented decision rule |

## `tarp`

`tarp: not_configured`. A TARP is applied only when the user supplies one with
explicit thresholds, metric conventions, averaging, persistence and
missing-data policy. While it is absent, no alarm state is produced, the
dashboard shows the exploratory notice, and no slope is declared safe or
unsafe. Supplying a TARP never changes raw values or the exploratory evidence;
it adds a separate, attributable state.

## `gis`

| Key | Value | Provenance | Meaning |
|---|---|---|---|
| `srs_id` | `-1` | supplied policy | undefined Cartesian; never guess an EPSG code |
| `srs_wkt` | `null` | not supplied | a real CRS must be supplied explicitly |
| `transform` | `not_configured` | not supplied | applied on export only; original grid data preserved |
| `vector_scale` | `2500` | supplied default | `@hlo_vec_scale` arrow scaling |
| `candidate_zone_rule` | `not_configured` | not supplied | candidate zones follow an explicit configured spatial rule |

If a transform or CRS is configured, vectors and their covariance/uncertainty
geometry are rotated together with the point coordinates. Without one, the map
stays in the input grid with no geographic basemap.
