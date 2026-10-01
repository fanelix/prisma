# Verification and implementation decisions

Assessment, 1 October 2026 (after the golden fixture was added): **133 passed,
6 skipped, 2 xfailed** with numeric runtime warnings treated as errors. Ruff
lint and format, strict documentation build, wheel/sdist build (no field data
or golden JSON in the sdist) and the CLI on the HLO export passed.

The six skips are five legacy Candrian checks and the legacy sqlite-free check,
all needing `data/Candrian_Sep_w1_w2_2026.csv`, which is no longer in `data/`.
The sdist content check also skips when no build precedes pytest, as in CI.
The two expected failures are the golden values listed below as not reproduced.

## Golden regression on the HLO export

`tests/test_golden_hlo.py` checks the owner's `golden_values.json` against
`data/HLO Sept 2026.csv` (MD5 `99b7183d…` matches `input.md5`). Station IDs are
mapped from station easting (S1 = W, S2 = E). Golden names "E-S1" refer to the
first station E processing segment.

| Golden value | Golden | Pipeline | Tolerance |
|---|---|---|---|
| Rows / prisms / E,W prisms / E,W cycles / period | 39822 / 208 / 188,20 / 178,462 | identical | exact |
| Pooled repeat SD D / Hz / V | 0.30 mm / 0.75″ / 0.79″ | 0.304 / 0.755 / 0.794 | 0.01 |
| Median cycle σ LOS / vertical | 0.74 / 2.8 mm | 0.741 / 2.759 | 0.02 / 0.1 |
| E rotation at end | −12.84″ | −12.837″ | 0.05″ |
| E step, 7 Sep 04:34–08:04 (24 h windows) | −4.41″ | −4.42″ | 0.15″ |
| E drift 7–21 Sep / after 21 Sep | −0.07 / −0.67″/day | −0.069 / −0.671 | 0.02 |
| Processing change (auto-detected) | 21 Sep 20:03 | 21 Sep 20:03 | exact |
| Cycle E158 station height | 61.602 m | 61.602 m | 0.5 mm |
| Applied-minus-fitted orientation after change | −2.07″ | −2.17″ | 0.15″ |
| BS_HL_5 median dE over E-S1 | 48.5 mm | 48.5 mm | 0.5 mm |
| Network median as-delivered 3D, end of E-S1 | 26 mm | 26.9 mm (cycles 26.0–27.0) | 1.0 mm |
| G1 frame-corrected net LOS (7 targets) | −2.08 … −5.90 mm | max deviation 0.21 mm | 0.3 mm |
| G1 raw net LOS (7 targets) | −3.0 … −6.5 mm | identical | 0.5 mm |
| G1 group median | −3.16 mm | −3.03 mm | 0.3 mm |
| G1 period rates | −0.158 / −0.109 / −0.106 / −0.050 | −0.150 / −0.120 / −0.123 / −0.050 | 0.025 mm/day |
| G1 placebo: groups as extreme / minimum | 0 / −0.61 mm | 0 of 154 / −0.64 mm | 0.1 mm |
| HLO-R7 raw vertical / LOS net | −12.68 / 4.0 mm | −12.681 / 4.0 | 0.05 / 0.5 |
| HLO-R7 vertical trend per 30 d | −14.89 mm | −15.43 (−14.91 with partial block) | 0.6 / 0.05 |
| HLO-R7 only negative height-correlated W target | yes | yes (ρ −0.20, p_Bonf 0.0004) | — |
| Lost in final 48 h | 12 targets | same 12 | exact set |
| SP daytime-biased targets | 10 targets | the 10 most negative SP biases | exact set |
| Tangential-only candidates | 4 targets | all above network p90 of 5.7 mm | — |
| Insufficient data | 9 | 9 | exact |

Not reproduced (strict xfail, reasons in the test):

- **HLO-R7 ρ = −0.317.** The cycle-detrended definition gives −0.20 to −0.23,
  depending on which cycle median is removed; the research script is not
  available. The qualitative finding (HLO-R7 is the only target with a
  significant negative correlation) is reproduced.
- **Final class counts.** They come from stage-3 synthesis with named clusters
  and artefact classes; the pipeline reports exploratory screening labels.

The HLO-R7 trend gap is the initial partial 24-hour block: the pipeline drops
it, as documented (handoff pitfall 4); the golden value includes it.

## Defects found and fixed during the assessment

- Reference inference used time-differenced series, so slow co-variation
  such as HLO-R7's subsidence was invisible (ρ −0.03). It now follows the
  handoff's cycle-detrended definition; the differenced form is kept as a
  labelled robustness check.
- Pooled repeat SD (handoff stage 1) was missing; it is now the
  `repeatability` table.
- Targets spanning less than the baseline plus end windows reported a net
  change of exactly 0 mm, from the same observations (e.g. HLO-R8). That net
  change is now unavailable and flagged.
- At station W all targets lie in a 46° azimuth fan, and the four-parameter
  vertical frame is ill-conditioned (fitted height SE ~26 mm against ~3 mm raw
  noise; frame-corrected AE verticals up to 60 mm). Targets whose fit SE
  exceeds their raw noise are now flagged and a run warning is issued; values
  are kept.
- The pipeline took about 6 minutes on the HLO export, mostly pandas overhead
  in 24-hour blocking. It now takes about 80 s with bit-identical outputs.
- For 46 of 208 targets the robust LOS σ was 0.0 mm, because the MAD of
  1 mm-quantised residuals collapses, so every nonzero residual was flagged as
  a spike (2,821 of 33,618 prism-cycles). With the owner-approved floor of
  resolution/√12 = 0.29 mm, 285 prism-cycles (0.85%) are spikes. Screening,
  frames and 24-hour series are unchanged; the median σ stays 0.741 mm. With a
  measurable LOS σ, the frame-precision flag now also marks 19 station W
  targets whose LOS fit SE exceeds raw noise.

## Open decisions for the owner

- **Station W vertical frame.** A reduced vertical model (height and index
  only) or explicit W frame membership would avoid the ill-conditioning; the
  research script's W model is unknown.
- **Bias floor.** Without `detection.bias_floor_mm`, 181 of 208 targets carry
  a "day/night bias candidate" flag. The golden SP set suggests a floor of a
  few millimetres.
- **"Possible" screening.** 83 targets are "possible (exploratory)": either
  criterion alone triggers it. The handoff only says "weaker evidence".
- **Golden tolerances** in `tests/test_golden_hlo.py` are proposals; confirm
  or tighten them.

## Independent review and fixes

A read-only independent reviewer checked the complete implementation. Its
important findings were fixed: anchored missing-correction windows, TARP
missingness/gap persistence, mixed-regime repeat coordinates, vertical-mover
exclusion, exported-orientation reference inference, configuration validation,
effective configured bias floors, and truthful LOS-SE labelling on the map.
Tests reproduce the affected behaviours. Additional coverage verifies vertical
concern visibility, prefix groups, sampling success, pre-loss ranks, angle bias,
same-geometry controls, URL encoding for space-containing repository filenames,
and distribution exclusion of field data.

## Decisions made within the authorized handoff

- Retain `fanelix/prisma`, its legacy API and legacy dashboard mode; add the
  separately packaged `rts-forensics` raw pipeline. This preserves compatibility
  but the two methods have different meanings and must not be compared as if
  they were identical.
- A–D boundaries were not supplied. Grade them only with an owner policy;
  otherwise report ungraded reliability and a provisional network frame. That
  frame can absorb common ground motion and requires external control.
- Unspecified physical step, bias and cluster thresholds remain null. Statistical
  day/night effects are descriptive; without a supplied physical bias floor,
  they do not automatically exclude targets from a provisional frame. The
  consequence is that automatic physical event/cluster decisions need site input.
- TARP additionally requires an owner-supplied `max_gap_hours`. This prevents a
  sparse record from defining its own permissible gap; existing incomplete TARP
  mappings must be completed before use. No default alarm values are provided.
- Uncertainty circles are explicitly median LOS fit-SE diagnostics, not
  uncertainty in a horizontal endpoint vector or total survey error.
- Missing independent research assets prevent a full real-data regression
  claim. No narrative class counts or thresholds were fabricated to fill the gap.

Actual QGIS/QField rendering and a live stlite browser execution have not been
verified in those applications. Default QML and browser assets are supplied;
the local Streamlit test harness and browser-compatible GIS backend are checked.
