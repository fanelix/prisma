# Method

`rts-forensics` turns a monthly Robotic Total Station (RTS) prism export into an
audit trail that separates **movement concern** from **measurement reliability**.
It works from the raw observations (horizontal direction, zenith angle, slope
distance) and reconstructs the instrument frame from the network itself, because
the delivered coordinates are a processing product that can contain a whole
instrument rotation.

This page describes the pipeline stages, units, sign conventions and equations.
Threshold provenance and the `not_configured` policy are in
[Configuration](configuration.md); the failure modes each stage guards against
are in [Pitfalls](pitfalls.md); what the data cannot establish is in
[Limitations](limitations.md).

## Honesty policy in one paragraph

Raw observations are never modified. Every derived value links back to source
lines. No threshold is invented: settings the research bundle defines are used
with attribution and settings it does not define are reported as
`not_configured`. Frame-corrected values are always shown beside raw values,
never instead of them. Flags are retained, not deleted. No slope is declared
safe or unsafe, and no alarm state is produced without a site TARP.

## Units and sign conventions

| Quantity | Unit | Sign / convention |
|---|---|---|
| Geometry distances (`D`, target and station coordinates) | metre | as exported; coordinates may be a shifted site grid |
| Displacements, translations | millimetre | see individual products below |
| `dD` / raw radial displacement | millimetre | **positive = away from the instrument** |
| `ver_raw` (Δ of D·cos V) | millimetre | **positive = up**; independent of station-height estimates |
| `tan_raw` (D·sin V·ΔHz) | millimetre | **positive = clockwise from above**; contains instrument rotation |
| `rad_raw` (Δ of D·sin V) | millimetre | radial component in the horizontal plane |
| Delivered `dE`, `dN`, `dZ` | millimetre | change from the **segment** baseline; never bridged across processing changes |
| Internal angles | radian | |
| Observed/exported angles | degree, or arcsecond where explicitly labelled | Hz is a circle reading, **not** a grid azimuth |
| Frame rotation | arcsecond | |
| Scale | ppm | |
| Rates | mm/day | |
| Time | site-local naive | timezone unstated in the export; never converted implicitly |

## Stage 1 — parse once, keep provenance

1. The GeoMoS-style CSV is detected (delimiter, encoding, time format) and
   parsed once. The documented export is `;`-separated, cp1252, CRLF, with DMS
   angle strings and `dd/mm/yyyy HH:MM` day-first timestamps.
2. Every observation receives an immutable ID `obs_id = sha256:src_line` and
   keeps `(source_sha256, src_line)`. Physical source line numbers survive CRLF,
   quoted delimiters and multiline fields.
3. The original bytes are hashed and optionally copied under `<out>/raw/`. The
   upload itself is never modified.
4. DMS values are parsed with an explicit regex; malformed rows produce parser
   diagnostics instead of silent drops.
5. Aggregations write a normalized `source_links` table so a displacement,
   rate, frame fit or classification row can name the observations it used.

## Stage 2 — stations and cycles

1. Observations are split by station. Stations are auto-detected from the
   largest gap in station easting, or matched to explicit configured anchors.
   Exact station-coordinate equality cannot be used because resections change
   the coordinates.
2. A new cycle starts when the elapsed gap to the previous observation at the
   same station exceeds the configured threshold. Cycles use elapsed minutes,
   never row indices.
3. Consistency diagnostics compare raw and delivered values within a cycle:
   - `hz_deg − coordinate azimuth` is the orientation offset (constant within a
     cycle when the coordinates are a polar computation);
   - `D` versus coordinate distance gives the scale;
   - `V` versus coordinate zenith gives curvature/refraction.
4. A cycle table records times, observation counts, median station coordinates,
   orientation offset and its spread, and scale.
5. Processing changes (station-coordinate changes and orientation jumps) are
   detected and also accepted from configuration. The delivered-coordinate
   series is split into processing segments (`E-S1`, `E-S2`, `W1` in the
   reference export), and as-delivered values are never bridged across a
   segment boundary.

## Stage 3 — displacement products

Repeats are real repeat measurements, so they are retained: repeated
observations are aggregated per prism and cycle (circular median of Hz, median
of the other quantities), never treated as duplicates.

With the baseline defined per prism and per processing segment (the first
configured baseline window of the segment), the products below are all in
millimetres:

- **Primary, frame-independent metric:**
  `dD = (D − D0) · 1000`, positive away from the instrument.
- **Vertical, independent of station-height estimates, sensitive to refraction:**
  `ver_raw = (D·cos V − D0·cos V0) · 1000`, positive up.
- **Horizontal radial:** `rad_raw = (D·sin V − D0·sin V0) · 1000`.
- **Tangential:** `tan_raw = D0·sin V0·ΔHz` with ΔHz converted to radians;
  positive clockwise from above. It contains the instrument rotation and is the
  noisiest direction.
- **Delivered coordinates:** `dE`, `dN`, `dZ` from the **segment** baseline.

Every row keeps its contributing observation IDs and its baseline observation
IDs. Raw polar series remain available across a processing change whenever the
observations support them; only the delivered-coordinate series is segmented.

## Stage 4 — noise

- **Pooled within-cycle repeatability** uses the repeated observations:
  `sqrt( Σ(x − x̄_group)² / (N − k) )` for distance (mm), Hz (arcsec), V
  (arcsec) and coordinates (mm), reported for all repeats and for the
  same-minute and `≥ 10 min` subsets.
- **Robust residual noise** per prism and metric is
  `1.4826 · MAD(residuals from a centred rolling median)`. The rolling window,
  minimum observations and robust scale are configuration values.
- **Spike flags:** `|robust z| > 6` against the rolling median. Spikes are
  flagged and retained, never deleted; the observation still contributes to
  every product.

## Stage 5 — rates

- Rates use **24-hour block medians counted backwards from the final record**,
  not calendar days. Partial days bias daily values because daytime LOS and
  vertical readings differ systematically from night readings.
- Theil–Sen slopes are computed on the block medians, with the confidence
  interval from the same block medians. An irregular row count cannot create a
  velocity: elapsed time is always used, never a row index.
- Distance has 1 mm resolution, so Theil–Sen on raw cycles produces many zero
  slopes; the block medians are mandatory for this reason.
- The configured windows are 30 days, 7 days and 72 hours. The 72-hour estimate
  uses all cycles rather than blocks and its confidence interval ignores
  autocorrelation; that limitation is reported with the result.
- Block counts, start/end bounds and partial-block handling are retained.

## Stage 6 — coverage, screening, reliability and flags

- Coverage per prism: observation count, first/last time, longest gap,
  scheduled cycles, coverage percentage, gaps above the configured threshold
  and observations in the final configured window. Unobserved means
  unavailable, never stable.
- **Exploratory movement screening** combines a net-change magnitude test with
  a rate test: `|net| ≥ max(σ multiplier · σ, floor)` **and** the 30-day rate
  confidence interval excludes zero with the same sign as the net change.
  Weaker evidence is labelled *possible*; fewer than the configured minimum
  prism-cycles is *insufficient data*; otherwise the result is
  *"no credible movement detected within available observations and
  sensitivity"*. These labels are exploratory research screening, not alarms
  and not TARP states.
- **Reliability is graded separately** from movement concern, using coverage,
  gaps, noise and spike counts against the configured boundaries. A prism can
  be reliable but stable, or unreliable with a large apparent change; the two
  fields are never merged.
- The reference export's supplied screening values (LOS floor 2 mm, vertical
  floor 5 mm, 3σ, minimum 20 prism-cycles, spike `|z| > 6`) and reliability
  boundaries live in `config.yaml` with their provenance comments.

## Stage 7 — frame reconstruction

The core algorithm fits an eight-term frame per station cycle with weighted
Huber IRLS over a frame set of prisms (reliability A, at least the configured
minimum number of cycles, excluding suspected movers, biased prisms and any
reference under test). With azimuth `a`, horizontal distance `HD`, slope
distance `D` and zenith `V`:

```text
Hz :  dHz_i [arcsec] = w_c + 206.265 · (−tE·cos a_i + tN·sin a_i) / HD_i
LOS:  dD_i  [mm]     = −(tE·sin a_i + tN·cos a_i)·sin V_i + s_c · D_i[km]
Ver:  ver_i [mm]     = h_c + D_i[km] · (iota_c + ts_c·sin a_i + tc_c·cos a_i)
```

- `w_c` rotation, `tE`/`tN` translation, `s_c` scale; `h_c` vertical offset,
  `iota_c` vertical index, `ts_c`/`tc_c` tilt terms.
- Weights are the supplied research assumptions σ_Hz 0.8″,
  σ_D 0.8 mm, σ_V 3 mm. They are weighting assumptions, not safety thresholds.
- Rank and conditioning are reported. A rank-deficient, insufficient or failed
  fit gets an explicit status and missing corrected values; it never fabricates
  a zero correction. Geometry is checked before fitting.
- The corrected series subtract the fitted prediction from the observation:

```text
tan_fc = (dHz − pred_hz) · HD / 206.265
los_fc = dD − ( −(sin a·tE + cos a·tN)·sin V + s·D[km] )
ver_fc = ver_raw − ( h + D[km]·(iota + ts·sin a + tc·cos a) )
```

- **Frame correction is experimental and relative to the chosen network.** The
  frame set is published with the exclusions, so a tested point never proves
  its own stability. Raw and corrected columns travel together in every table,
  plot, report and GIS product.

## Stage 8 — forensic tests

The test families investigate competing explanations rather than issuing
labels:

- **Groups:** candidate spatial clusters from coherent frame residuals (no
  hard-coded prism list); spatial placebo on every other nearest-neighbour
  group; night-only versus day-only net change; hour-matched net change;
  temporal split; change point on block medians; period rates; pairwise LOS
  differences; a 3D vector from frame-corrected LOS, tangential and vertical
  components.
- **Sampling:** day/night comparison, hour-matched change, sampling composition
  and daytime-bias tests (day-minus-night LOS, weekly stability, angle shifts,
  night-only trends, same-geometry control). Changing observation hours can
  create apparent vertical trends; like must be compared with like.
- **References:** cycle-detrended Spearman correlations of each prism's raw
  vertical against station height and Hz against orientation, with Bonferroni
  correction; the resection-validation test expects a real station rise to fall
  in raw common vertical with slope −1. Correlation suggests a candidate, it
  does not identify a reference.
- **Missingness:** coverage, lost versus retained prisms, pre-loss trend and
  observability by hour. A prism missing at night is not a prism that moved.
- **Events:** time-of-day-balanced step tests and common-mode steps. Every
  candidate event records its evidence, competing explanations, confidence and
  follow-up status.

Tests whose decision rule is not configured return `not_configured` with a
reason; they are visible in the investigation register, not silently skipped.

## Stage 9 — synthesis and classification

The final result keeps separate fields for movement concern, measurement
reliability, retained flags and optional TARP state. Generic labels come from
the exploratory screen; site-specific overrides (known clusters, references,
biased prisms) are configuration inputs, never package constants. Applying a
TARP is a separate step that runs only when a user-supplied TARP exists;
otherwise the state stays `not_configured` and no alarm is produced.

## Stage 10 — exports

The CLI and the dashboard call the same pipeline and the same exporters. Outputs
include the canonical observations with provenance, prism-cycle series,
baselines, repeatability/noise, rates, processing segments, frame
membership/diagnostics, source links, hypothesis evidence, CSV dictionaries,
figures, `report.md`/`report.html`, the effective configuration and a styled
GeoPackage with ten layers. The GIS default is an undefined Cartesian SRS
(`srs_id −1`); original grid coordinates are preserved and only a configured
transform is applied on export, rotating vector and covariance components with
the geometry. Every export is covered by a SHA-256 manifest that excludes its
own hash.

## Why as-delivered coordinates are insufficient

The delivered easting/northing/elevation are processing products, not raw
observations. In the reference export:

- an uncorrected instrument rotation reached 12.9″ later in the month, a
  network-median 3D artefact of 26 mm, and 54 mm at the backsight;
- station coordinates and orientation were re-estimated mid-record, so a single
  coordinate series would silently bridge two different processing regimes;
- a "reference" or backsight prism can itself drift, so stability cannot be
  assigned by name;
- daytime LOS readings differ systematically from night readings, so calendar
  day statistics mix two different measurement populations.

For those reasons the analysis always starts from `D`, `Hz` and `V`, keeps the
delivered coordinates as a separate segmented product, and reconstructs the
frame from the network with published membership.
