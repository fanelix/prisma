# PROJECT HANDOFF — RTS Prism Forensic Analyser (HLO September 2026)

**How to use this file:** give it to ChatGPT together with the code in this bundle and say:
*"Build the GitHub app described in section 9, reusing the algorithms in sections 3–4 and the scripts in `reference_code/`. Use `golden_values.json` as regression tests."*

Everything below comes from code that was actually run on the real export. Numbers in section 5 are reference values the app must reproduce on the same input.

---

## 1. What the project does (one paragraph)

It takes a raw Robotic Total Station (RTS) prism export (Leica GeoMoS-style CSV), and separates **real ground movement** from **measurement-system artefacts**:

- instrument rotation;
- resection errors;
- atmospheric and refraction effects;
- sampling bias;
- lost targets.

For each prism it classifies movement concern and data reliability *separately*. It then produces a report, tables, figures and a styled QGIS GeoPackage.

The key insight: as-delivered coordinates can show centimetre-level "movement" that is entirely instrument rotation. The analysis therefore works from the **raw observations** (Hz, V, slope distance) and reconstructs the reference frame itself.

---

## 2. Input data specification

### File format
- CSV, `;` separator, encoding cp1252, CRLF line endings, 18 columns, one row per observation.
- Timestamp `dd/mm/yyyy HH:MM`, day-first, minute resolution. The timezone is not stated; treat times as site local.
- Angles are DMS strings, e.g. `" 148° 36' 38.36901"` (embedded quotes; degree sign in cp1252).

| Column | Meaning | Notes |
|---|---|---|
| Point ID | Prism ID | ID prefixes give the group (HLO-, SP_, STG4N-, DAM5_, AE, HLO-R, HLO-W, BS_HL_5) |
| Time | Timestamp | See above |
| Hz [dms] | Raw horizontal circle reading | NOT grid azimuth. Orientation offset per station (E: +29.9768°) |
| V [dms] | Zenith angle | 90° = horizontal |
| D [m] | Raw slope distance | 1 mm resolution |
| PPM Type, PPM, Pressure, Av Temp, Add Const | Atmospheric metadata | Constant defaults in this export (0 ppm, 1013.25 mbar, 11.1012 °C, 0). Treat them as "no met data" |
| Target Easting/Northing/Elevation | Processed coordinates (m) | Derived, not raw. Polar computation with height reduction (~ −h/R) and curvature/refraction k = 0.125 |
| Station Easting/Northing/Height | Station coordinates used in that cycle | May change cycle to cycle when resection is on |
| Null Measurement [m] | Constant per-prism reference horizontal distance | Epoch unknown; do not use as baseline |
| Horz Distance [m] | Horizontal distance consistent with coordinates | Derived |

### Important properties found in this export
- **Two stations** in one file, identified by Station Easting (E ≈ 176064, W ≈ 175087). No prism is observed from both.
- **Station E:** 178 cycles every 4 h (00/04/08/12/16/20), each about 37 min.
  - Station coordinates are fixed until 21 Sep 16:41. After that, station and orientation are re-estimated only in the 20/00/04 h cycles.
  - HLO prisms are measured 2–3 times per cycle. These are not duplicates; keep them and take the median.
- **Station W:** 462 hourly cycles, 17:00–07:59 only. AE prisms are measured 17:00–19:00 only. Station coordinates are re-estimated almost every cycle.
- **Coordinates are a shifted dummy grid.** The owner converts them to the real grid separately. Never infer CRS or location from them.

---

## 3. Processing pipeline (stages and algorithms)

### Stage 1 — Audit and core EDA (`s1_*.py`)

1. **Parse once, keep provenance.**
   - `src_line` = 1-based line number in the source file.
   - Decode DMS with a regex. Parse the timestamp with an explicit day-first format.
2. **Station assignment and cycles.**
   - Split by Station Easting.
   - A new cycle starts when the gap to the previous observation at the same station is more than 30 min (E) or 15 min (W).
3. **Consistency checks.**
   - Hz − coordinate azimuth gives the orientation offset; it is constant within a cycle when coordinates are a polar computation.
   - D vs coordinate distance gives the scale.
   - V vs coordinate zenith gives curvature/refraction k.
4. **Prism-cycle series:** median of repeats (circular median for Hz).
5. **Displacement products (all in mm), baseline = median of the first 48 h:**
   - `dD` = LOS range change (+ = away from instrument). **Primary, frame-independent metric.**
   - `ver_raw` = Δ(D·cos V) (+ = up). Independent of station-height estimates, but sensitive to refraction.
   - `rad_raw` = Δ(D·sin V).
   - `tan_raw` = D·sin V·ΔHz (+ = clockwise from above). Contains instrument rotation.
   - `dE, dN, dZ` = as-delivered coordinate change. **Segmented** at processing changes (E-S1 / E-S2 / W1) and never bridged.
6. **Noise:**
   - Pooled within-cycle repeat SD.
   - Per-prism robust SD = 1.4826 × MAD of residuals from a 24-h centred rolling median.
7. **Rates:**
   - Theil–Sen slope on **24-h block medians**, with blocks counted back from the last record (calendar days create partial-day diurnal bias).
   - 95% CI from the same block medians. Windows: 30 d, 7 d, 72 h (72 h uses all cycles; its CI ignores autocorrelation).
8. **Coverage:** observations, coverage %, longest gap, observations in the final 48 h.
9. **Screening** (exploratory, not alarms):
   - *detected*: |net| ≥ max(3σ, 2 mm LOS / 5 mm vertical) **and** the 30-d rate CI excludes 0 with the same sign.
   - *possible*: weaker evidence.
   - otherwise: *"no credible movement detected within available observations and sensitivity"*.
   - fewer than 20 prism-cycles: insufficient data.
   - Reliability is graded A–D separately (coverage, gaps, noise, spikes).
10. **Spikes:** |robust z| > 6 against the rolling median. Flag them, never delete.

### Stage 2 — Forensic tests (`s2_*.py`)

**A. Frame reconstruction (core algorithm, `s2_frame.py`).** For each cycle c, run a robust (Huber IRLS) joint fit over a "frame set" of prisms. The frame set is reliability A, at least 100 cycles, and excludes suspected movers, biased prisms and the reference points under test.

```
Hz : dHz_i [arcsec] = w_c + 206.265·(−tE·cos a_i + tN·sin a_i)/HD_i[m]   (rotation + translation)
LOS: dD_i  [mm]     = −(tE·sin a_i + tN·cos a_i)·sin V_i + s_c·D_i[km]     (translation + scale)
Ver: ver_i [mm]     = h_c + D_i[km]·(iota_c + ts_c·sin a_i + tc_c·cos a_i) (height + vertical index + tilt)
```

Weights: σ_Hz 0.8″, σ_D 0.8 mm, σ_V 3 mm.

- Output per cycle: rotation, translation, scale, vertical terms, standard errors and RMS.
- Frame-corrected series = observation minus the fitted frame prediction: `los_fc`, `tan_fc`, `ver_fc`.
- These are EXPERIMENTAL. Always show them beside the raw values.

**Other tests:**
- **Step test:** compare the frame before and after an event in windows balanced by time of day.
- **Resection validation:** a real station rise ⇒ raw common vertical falls by the same amount (expected slope −1). Compare raw-fitted translation with exported station coordinates.
- **Reference inference:** Spearman correlation of cycle-detrended raw vertical / Hz of every prism against station height / orientation, with Bonferroni correction. Strong negative ρ for height suggests that point is a resection reference.
- **Group tests (G1):**
  - spatial placebo (median of every other 7-nearest-prism group);
  - night-only vs day-only net change;
  - hour-matched net change (per 4-h class, then median);
  - temporal split (two halves);
  - hinge/three-segment change point on block medians;
  - period rates;
  - pairwise LOS differences;
  - 3D vector from frame-corrected LOS, tangential and vertical.
- **Missingness:**
  - lost vs retained prisms (Mann–Whitney, permutation of distance to the moving zone);
  - pre-loss trend rank;
  - night/day observation success over time.
- **Sampling-composition test:** net change recomputed within hour classes.
- **Daytime-bias test:** day-minus-night LOS per prism, weekly stability, angle shifts, night-only trends, same-geometry control group.

### Stage 3 — Synthesis (`s3_final.py`)
- Final class per prism, measurement flags, final event register (16 events), independent verification and a SHA-256 manifest.

### Stage 4 — GIS export (`s4_qgis_*.py`)
- GeoPackage with 10 layers and QGIS styles stored as defaults in `layer_styles`.
- Geometry-generator arrows scaled by `@hlo_vec_scale`, a QField-ready form, and a temporal layer. Undefined Cartesian SRS (srs_id −1).
- A site-grid transform helper rotates the vector components as well as the geometry.

---

## 4. Pitfalls a naive implementation gets wrong (must be handled)

1. **Trusting as-delivered coordinates.** Here they contain an uncorrected instrument rotation of up to 12.9″ (26 mm network median, 54 mm at the backsight). Always work from raw D/Hz/V and fit the frame.
2. **Bridging across a processing change.** Station re-estimation started mid-month. Segment as-delivered series at that point.
3. **Assuming a "reference" or backsight is stable.** BS_HL_5 drifted about 10 mm against the network. HLO-R7 (named like a reference) is subsiding.
4. **Calendar-day medians.** Partial days bias daily values, because daytime LOS reads about 2 ppm shorter and vertical shows a diurnal swing of about 4 mm. Use 24-h blocks counted back from the last record, or hour-matched comparisons.
5. **Changing sampling hours.** Prisms that lose night observations show fake vertical trends of 5–10 mm. Always compare like hours.
6. **Group medians with missing members.** On 28–29 Sep the HLO-R group median was HLO-R7 alone.
7. **Treating repeats as duplicates.** They are real repeat measurements and give the best repeatability estimate.
8. **Velocity from row index.** Always use elapsed time.
9. **Distance quantisation.** D has 1 mm resolution, so Theil–Sen on raw cycles gives many zero slopes. Use block medians.
10. **Tangential component.** It is the weakest; |frame-corrected tangential| has a network 90th percentile of 5.6 mm. Do not call tangential-only signals movement without consistency tests.
11. **Removing a common mode automatically.** It may hide real movement. Show raw and corrected side by side.
12. **Thresholds.** No TARP was supplied. Never invent alarm thresholds; screening labels stay labelled as exploratory.

---

## 5. Key results (reference values; also in `golden_values.json`)

| Finding | Value | Confidence |
|---|---|---|
| Station E rotation step, 7 Sep 04:34–08:04 | −4.41″ (se ≈ 0.15″); translation < 0.1 mm; no LOS change | High (instrument, not ground) |
| Station E rotation drift | −0.07″/day (7–21 Sep) → −0.67″/day (21 Sep–1 Oct); −12.84″ at the end; no diurnal cycle | High |
| As-delivered E-S1 artefact | Network-median 3D 26 mm on 21 Sep; BS_HL_5 dE +48.5 mm | High |
| Station E resection after 21 Sep 20:03 | Exported ΔN −4…−10 mm, ΔH −4…+17 mm not supported by raw data (slope +0.09 vs −1 expected); orientation offset −2.07″ | High |
| Cycle E158 (28 Sep 00:03) | Station height 61.602 m (+15/+19 mm outlier); 60 of 123 prisms flagged | High (verified on raw rows 35715/35906/36018) |
| **G1 cluster** (HLO-279, 128, 189, 289, 320, 123, 231) | Frame-corrected LOS −2.1 to −5.9 mm; horizontal 3.4–8.0 mm toward az 27–82°; onset about 2–3 Sep; rates −0.16 → −0.05 mm/day; placebo 0/139 | Moderate–high (credible movement) |
| **HLO-R7** (station W) | Raw vertical −12.7 mm, LOS +4 mm since about 12 Sep; only prism correlated with W station height (ρ −0.32) | Moderate–high |
| Station W frame | Station height +2.1 mm/30 d (+4.5 mm at end) → false uplift on W prisms | Moderate |
| Night observability | HLO-320 no night data after 25 Sep; HLO-123 and HLO-290 failing at night | High (pattern) |
| Daytime LOS bias | 10 SP prisms read 3–48 mm short in 08–12 h cycles; constant week to week | High (artefact) |
| Lost prisms | 12 with no data in the final 48 h; cluster near G1 (perm p = 0.012) | Low–moderate |
| Vertical reliability | Not robust below about 5 mm (28 prisms shift ≥ 3 mm under hour-matching) | High |
| Tangential-only candidates | DAM5_3, HLO-112, HLO-274, STG4N-10 (unverified) | Low |
| Repeatability | D 0.30 mm, Hz 0.75″, V 0.79″; cycle σ LOS 0.74 mm, vertical 2.8 mm | — |

**Final class counts (208 prisms):**

| Class | Count |
|---|---|
| No credible movement detected | 167 |
| Daytime artefact | 10 |
| Insufficient data | 9 |
| G1 credible | 7 |
| Unresolved | 5 |
| Ambiguous vertical | 4 |
| Sampling artefact | 2 |
| Reference points | 2 |
| HLO-R7 credible | 1 |
| Possible (HLO-107) | 1 |

---

## 6. Output data model

| Output | Grain | Key fields |
|---|---|---|
| `prism_summary` | 1 row per prism | pid, station, group, final_class, reliability, flags, coverage, net LOS raw/fc/hour-matched, vertical raw/fc/hour-matched, rates + CI, sigmas, night success |
| `cycle_table` | 1 row per station cycle | times, counts, station coords, orientation, scale |
| `frame_cycles` (E) | 1 row per cycle | rotation, translation, scale, vertical terms, SEs, applied-minus-fitted orientation, exported station shifts |
| `timeseries_24h` | prism × 24-h block | block start/end, LOS raw/fc, vertical raw/fc, tangential fc, as-delivered dE/dN/dZ, n, n_night |
| `event_register` | 1 row per event | id, class, prisms, start, end, effect, evidence, alternatives, confidence, follow-up, status |
| `investigation_register` | 1 row per test | hypothesis, expected signature, test, result, competing explanation, limitation, implication |
| `field_checks` | 1 row per check | question, supports-if / alternative-if, status, outcome, notes, photo |
| GeoPackage | 10 layers | as above + sight lines, zones, vectors, artefact vectors |

---

## 7. Reference code in this bundle (`reference_code/`)

| Script | Role |
|---|---|
| `s1_parse.py` | Parse CSV with provenance |
| `s1_cycles.py` | Stations, cycles, consistency diagnostics |
| `s1_core.py` | Prism-cycle series, displacement products, noise, rates, spikes |
| `s1_summary.py` | Coverage, net change, frame fits, screening, reliability |
| `s1_extra.py` | Time-of-day bias screen, step screen, verification |
| `s1_docs.py`, `s1_figs.py` | Data dictionary, QA table, event register, figures |
| `s2_frame.py` | **Per-cycle frame reconstruction (core)** |
| `s2_g1.py` | Group tests (placebo, day/night, change point, vectors, pairs) |
| `s2_refs.py` | Reference inference, station W leakage |
| `s2_missing.py` | Missingness, coverage decline, SP daytime bias |
| `s2_hourmatch.py` | Sampling-composition test |
| `s2_register.py`, `s2_figs.py` | Investigation register, figures |
| `s3_final.py` | Final classes, event register, verification, manifest |
| `s4_qgis_build.py`, `s4_qgis_styles.py`, `s4_qgis_preview.py` | GeoPackage, QML styles, preview |
| `run_all.sh` | Runs Stages 1–3 in about 70 s |

The scripts are linear research code: hard-coded names (G1 list, SP list, dates) and pickles between steps. The app must turn them into a configurable library (section 9).

Environment used: Python 3.11, pandas 3.0, numpy 2.4, scipy 1.17, matplotlib 3.10, geopandas 1.2, pyogrio 0.13 (GDAL 3.12), lxml.

---

## 8. What the data cannot establish (keep in the app's UI text)
- No slope stability or factor of safety from RTS data alone.
- No alarm status without a site TARP.
- No absolute movement without external control: the frame is defined by the network.
- No causes without rainfall, blasting, stacking or geology data.
- Resection references are not identifiable from the export.
- Unobserved periods are not evidence of stability.

---

## 9. App to build (specification for ChatGPT)

### Goal
An open-source GitHub repository, **`rts-forensics`**: a Python package + CLI + web dashboard. A user uploads a monthly RTS export, and the app runs the pipeline and shows credible movement vs measurement artefacts. The results must be reproducible and auditable.

### Must-haves (MVP)
1. **Package `rts_forensics/` with modules:**
   - `io` (GeoMoS CSV parser, provenance, format auto-detect);
   - `cycles`;
   - `displacement` (raw LOS / vertical / tangential, segmented coordinates);
   - `noise`;
   - `rates` (Theil–Sen on 24-h blocks);
   - `frame` (per-cycle robust frame fit);
   - `tests` (placebo, day/night, hour-matched, split, change point, reference inference, missingness, daytime bias);
   - `classify`;
   - `report` (Markdown/HTML);
   - `gis` (GeoPackage + QML).
2. **Config file `config.yaml`** instead of hard-coded values:
   - baseline/end windows, cycle gap thresholds, frame-set rules, screening floors;
   - known processing-change dates (auto-detect too);
   - optional TARP block (thresholds, averaging period, persistence), applied only if provided;
   - optional site-grid transform.
3. **Auto-detection** instead of hard-coding:
   - stations;
   - processing changes (station-coordinate changes, orientation jumps);
   - common-mode steps;
   - candidate clusters (spatially coherent frame-residual groups) instead of a fixed G1 list;
   - daytime-biased prisms;
   - lost prisms.
4. **CLI:** `rts-forensics run data.csv --config config.yaml --out results/`. It writes CSV tables, figures, `report.md`/`report.html`, the GeoPackage and a manifest with SHA-256.
5. **Web dashboard** (Streamlit is simplest; FastAPI + a JS front end is an option) with these pages:
   1. Upload & audit.
   2. Station frame (rotation / translation / station-coordinate plots).
   3. Prism explorer (time series raw vs frame-corrected, day/night split).
   4. Map (prisms by class, arrows with uncertainty circles, artefact-vector toggle, sight-line issues).
   5. Events & investigations.
   6. Downloads.

   Always show movement concern and reliability separately. Always label outputs as exploratory unless a TARP is configured.
6. **Tests (pytest):** a small synthetic dataset with injected rotation step, translation, scale, local movement, daytime bias and dropouts. The pipeline must recover each one. Plus a regression test against `golden_values.json` when the real file is available locally (never commit private data).
7. **GitHub Actions CI:** lint (ruff), tests, build docs; release workflow publishes the package.
8. **Docs:** README with method summary (sections 3–4), pitfalls, limitations (section 8), quick start, config reference.

### Nice-to-haves
- Multi-month comparison (October vs September rates for G1, R7 and the tangential candidates).
- DataPlotly-style charts in the dashboard.
- QField form round-trip: import completed field checks back into the event register.
- Plugin interface for other export formats (Trimble, Topcon).
- Optional external drivers (rainfall, blasting) with lag analysis, only when data are supplied.

### Non-negotiable rules for the app
- Raw data are never modified; every derived value is traceable to source line numbers.
- Never invent thresholds. Never state that a slope is safe or unsafe.
- Frame-corrected values are shown beside raw values, never instead of them.
- Flags are shown, not deleted.
- Coordinates stay in the input grid unless the user configures a transform.

---

## 10. Open questions (answers would improve the app's defaults)
1. Which points are the resection/orientation references at E (after 21 Sep) and W? Is HLO-R7 one of them?
2. What happened at station E on 7 Sep between 04:34 and 08:04?
3. Was the 21 Sep 20:03 change a deliberate software setting?
4. Timestamp timezone; Null Measurement epoch.
5. Site TARP, rainfall, blasting and stacking logs for later versions.

## 11. Suggested prompt for ChatGPT

> You are a senior Python engineer and geotechnical-monitoring software developer. Using PROJECT_HANDOFF.md, golden_values.json and the scripts in reference_code/, create the GitHub repository `rts-forensics` described in section 9.
> 1. Propose the repository tree and module APIs first.
> 2. Then implement module by module, refactoring the research scripts into tested, configurable functions (no hard-coded prism lists or dates).
> 3. Write the synthetic-data tests first.
> 4. Follow every pitfall in section 4 and every rule in section 9.
> 5. Keep raw vs frame-corrected outputs side by side.
>
> Ask me before inventing any threshold.
