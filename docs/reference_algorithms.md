# Reference algorithms (condensed from the handoff and `reference_code/`)

This file is a working aid for implementation. `reference_code/` is not committed
(site-specific). Every decision that is *not* defined here stays `not_configured`.

## 1. Parser (s1_parse.py)

- CSV, `;`, cp1252, CRLF, 18 columns. Header line 1; data `src_line` starts at 2.
- DMS regex: `^\s*(-?\d+)\D+(\d+)\D+([\d.]+)\D*$` → `d + m/60 + s/3600`.
- Time format `%d/%m/%Y %H:%M`, day-first, minute resolution; timezone unstated
  (treat as site local).
- Numeric fields stripped then `to_numeric`.
- Source columns → canonical observation names:
  `pid→point_id, time_str→time, hz_str→hz_deg, v_str→v_deg, D→d_m, E→te_m,
  N→tn_m, Z→tz_m, stE→se_m, stN→sn_m, stH→sh_m, nullm→null_meas_m, HD→horz_dist_m,
  ppm→ppm, temp→temp_c`.
- Keep original bytes (SHA-256) and physical line numbers even with quoted
  multiline fields.

## 2. Stations and cycles (s1_cycles.py)

- Station split by station easting. Reference used a fixed threshold
  (`stE < 175500 → W`); the app must auto-detect: sort unique station eastings
  and split at the largest gap (configurable explicit anchors override).
  Site config labels the clusters `E` (≈176064) and `W` (≈175087).
- Cycle: within a station, a new cycle starts when the gap (minutes) to the
  previous observation exceeds the configured threshold (E 30, W 15).
  `cycle_id = station + zero-padded cycle number` (e.g. `E007`).
- Consistency diagnostics per observation:
  `dE=E-stE`, `dN=N-stN`, `dZ=Z-stH`; `hd_c=hypot(dE,dN)`; `sd_c=hypot(hd_c,dZ)`;
  `az=degrees(atan2(dE,dN))%360`; `zen=degrees(atan2(hd_c,dZ))`;
  `dHz=((hz_deg-az+180)%360-180)*3600` (arcsec);
  `dV=(v_deg-zen)*3600`; `dSD=(D-sd_c)*1000`; `dHDrep=(HD-hd_c)*1000`;
  `scale_ppm=(sd_c/D-1)*1e6`.
- Cycle table: t0,t1,n,npid, median station coords, n distinct heights,
  median `dHz` (orientation), `dHz` sd, median scale, median `dV`, duration.

## 3. Core products (s1_core.py)

Group prefix: `grp = regexp_replace(pid, r'[-_]?\d+[A-Za-z]?$', '')`.

Segments: station W → `W1`; station E split at the configured processing break
(handoff: 2026-09-21 18:00, first new cycle 20:03) into `E-S1` / `E-S2`.
**As-delivered coordinates are never bridged across a segment.**

Prism-cycle aggregation (repeats retained):
- circular median of Hz (`circ_med`: `(ref + median((a-ref+180)%360-180))%360`),
  median of V, D, E, N, Z, HD, station coords, null measurement;
  counts, `src_lines`.

Baseline per prism:
- first `baseline.window_hours` (48 h) of the segment;
- if fewer than `baseline.min_observations` (6), fall back to the first 6 rows;
- `hz0` circular median, `v0`, `D0` medians.

Displacement products (mm):
- `dD   = (D - D0)*1000`                       (+ = away from instrument)
- `dHz_as = ((hz-hz0+180)%360-180)*3600`       (arcsec)
- `tan_raw = D0*sin(v0)*radians(dHz_as/3600)*1000`  (+ = clockwise from above)
- `rad_raw = (D*sin(v) - D0*sin(v0))*1000`
- `ver_raw = (D*cos(v) - D0*cos(v0))*1000`     (+ = up; independent of station height)
- As-delivered per segment: `dE,dN,dZ` in mm from the segment baseline.
- `az_deg` from median (E,N) and median station; `dRad_rep=dE sin+dN cos`,
  `dTan_rep=dE cos-dN sin`.
- Experimental: `tan_bsref = D0*sin(v0)*radians((dHz_as - bs_dHz_as)/3600)*1000`
  for station E, using the configured backsight prism (`BS_HL_5`) — labelled
  experimental, never a replacement.
- Every row keeps its observation IDs and baseline observation IDs.

Noise:
- Pooled within-cycle repeat SD: `sqrt(sum((x - mean_group)^2)/(N - k))`
  over repeated groups, for D (mm), Hz (arcsec), V (arcsec), E/N/Z (mm).
  Report `all`, `same_minute` (span ≤ 1 min) and `>=10min` variants.
- Per-prism robust SD: residual from a **24 h centred rolling median**
  (`min_periods=3`), then `1.4826*MAD(residual)`; requires ≥ 5 residual points.
- Spike flag: `|residual / max(robust_sd, floor)| > 6`, floors
  dD 0.5, ver_raw 1.0, tan_raw 1.0, dZ 1.0 mm. Flags are retained, never deleted.

Rates:
- 24 h blocks **counted backwards from the final record**:
  `day = T_END - (floor((T_END - t)/1day) + 0.5) days`.
- Theil–Sen (`scipy.stats.theilslopes`, α 0.95) on block medians (point and CI
  from the same block medians). Windows: 30 d (all), 7 d, 72 h.
  - 72 h uses all cycles (not blocks); its CI ignores autocorrelation — state it.
- Metrics: dD, ver_raw, tan_raw, tan_bsref; last-segment dE/dN/dZ; early window
  = record start → 7 d before end (for change screening).
- Distance quantisation: 1 mm on D → block medians are mandatory.

## 4. Summary, screening, reliability (s1_summary.py)

Coverage per prism: n_cycles, first/last time, longest gap (h), scheduled cycles
(AE prisms: only cycles in which any AE was measured; otherwise all station
cycles), coverage %, gaps > 24 h, observations in the final 48 h,
`HD - Null` median over the first 48 h.

Net change: final 48 h median minus first 48 h median (own last 48 h if the
prism is stale; window label marks this).

Exploratory LOS/vertical frame check on net values (IRLS Huber k=1.345, 20
iterations, scale 1.4826·MAD or floor):
- LOS: `A = [-sin(az)sinV, -cos(az)sinV, -cosV, D/1000]`
- vertical: `Av = [1, D/1000, D/1000·sin az, D/1000·cos az]`
- These residuals are shown beside raw values, never instead.

Screening (exploratory):
- `crit = |net| ≥ max(3σ, floor) AND 30-d rate CI excludes 0 AND sign(rate) == sign(net)`
  with floors LOS 2 mm, vertical 5 mm.
- `possible = |net| ≥ max(2σ, 2 mm)` (LOS) / `max(2σ, 4 mm)` (vertical) and CI excludes 0.
- `n_cycles < 20` → insufficient data.
- Otherwise: "no credible movement detected within available observations and sensitivity".

Reliability (separate from movement; independent of screening):
- flags: `coverage<50%`, `coverage50-80%`, `gap>72h`, `no obs last48h`,
  `noisy LOS` (σ_dD > 2 mm), `noisy vertical` (σ_ver_raw > 6 mm),
  `≥5 spike flags`, `AE evening-only schedule`, `W night-only station`.
- grade: `<20 cycles` → D (insufficient); if any hard flag
  (coverage<50%, gap>72h, no obs last 48 h, noisy) → C (weak);
  else if coverage50-80% or any spikes → B (moderate); else A (good).
  These boundaries are supplied research rules; they live in config.

## 5. Frame reconstruction (s2_frame.py)

Per cycle (≥ 20 prisms in the frame set) weighted robust joint fit:
- `a = radians(az_deg)`, `HD = D·sin(radians(v))`, `Dk = D/1000`, `sinV = sin(radians(v))`.
- Horizontal design (2n×4), weighted by σ_Hz = 0.8″ and σ_D = 0.8 mm:
  - Hz rows: `[1, 206.265·(-cos a)/HD, 206.265·(sin a)/HD, 0]`, y = `dHz_as`
  - LOS rows: `[0, -sin a·sinV, -cos a·sinV, Dk]`, y = `dD`
  - unknowns `[w (rotation, arcsec), tE (mm), tN (mm), s (mm/km)]`
- Vertical design (n×4), weight σ_V = 3 mm:
  - `[1, Dk, Dk·sin a, Dk·cos a]`, y = `ver_raw`
  - unknowns `[h (mm), iota (mm/km), tilt_s, tilt_c]`
- Huber IRLS: `k = 1.5`, 30 iterations; `s = 1.4826·MAD(r)` or 1.0;
  `w = clip(k·s/max(|r|, 1e-12), 0, 1)`; covariance `s²·pinv(AwᵀAw)`,
  standard errors `sqrt(diag)`; report rank and RMS
  (`1.4826·MAD(residual)·σ`). Rank-deficient or failed → explicit status,
  never zero correction.
- Frame set: configured exclusions, reliability A, ≥ `frame.min_cycles` (100).
  Deterministic; membership and exclusions are published.
- Frame-corrected series (always beside raw):
  `pred_hz = w + 206.265·(-cos a·tE + sin a·tN)/HD`;
  `tan_fc = (dHz_as - pred_hz)·HD/206.265`;
  `los_fc = dD - (-(sin a·tE + cos a·tN)·sinV + s·Dk)`;
  `ver_fc = ver_raw - (h + Dk·(iota + tilt_s·sin a + tilt_c·cos a))`.

## 6. Forensic tests (s2_*.py)

All thresholds are config; where the reference has no numeric threshold the
result carries `not_configured`:

- **Groups**: candidate clusters from spatially coherent frame-residual
  neighbourhoods (no hard-coded list); spatial placebo = the same statistic on
  every other 7-nearest-prism group; night-only vs day-only net change;
  hour-matched net change (4-h classes, baseline first 72 h of each class vs
  final 72 h of the same class, median across classes); temporal split;
  hinge/three-segment change point on block medians; period rates; pairwise LOS
  differences; 3-D vector from frame-corrected LOS/tangential/vertical
  (`rad_h = (los - ver·cosV)/sinV`, `dE = rad_h·sin a + tan·cos a`,
  `dN = rad_h·cos a - tan·sin a`).
- **Sampling**: day/night comparison, hour-matched change, sampling composition,
  daytime bias (weekly day-minus-night median; angle shifts; night-only trends;
  same-geometry control).
- **References**: per-cycle-detrended (subtract cycle median) Spearman
  correlations of prism vertical vs station height and Hz vs orientation;
  Bonferroni correction (`p·n < 0.05`); expected station-rise/raw-vertical
  slope −1 for resection validation.
- **Missingness**: coverage, lost vs retained (Mann–Whitney; permutation test of
  distance to a candidate cluster; report n and seed), pre-loss trend rank,
  observability by hour.
- **Events**: balanced step test comparing before/after windows with the same
  time-of-day composition; common-mode steps; every candidate records competing
  explanations.

## 7. Classification (s1_summary.py + s3_final.py)

- Movement concern and reliability are separate output fields.
- Generic final class comes from the exploratory screen, not from a fixed site
  list. Site-specific overrides (known clusters, references, biased prisms) are
  configuration inputs; without them the label stays the generic screening text.
- TARP: only if supplied; otherwise state `not_configured` and produce no alarm.
- Never say a slope is safe/unsafe or that an unobserved period is stable.

## 8. GIS (s4_qgis_*.py)

Ten layers, undefined Cartesian SRS (`srs_id = -1`), original grid preserved;
only a user-configured transform is applied on export, rotating vector and
covariance components with the geometry. Nonspatial tables are registered as
GeoPackage attributes. Default styles live in `layer_styles`; arrows are
geometry generators scaled by a project variable (`@hlo_vec_scale`).
Proposed layer names (plan): `prism_summary`, `station_cycles`, `frame_cycles`,
`timeseries_24h`, `event_register`, `investigation_register`, `field_checks`,
`sight_lines`, `candidate_zones`, `movement_vectors`.
