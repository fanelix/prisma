# Method and configuration

## What is preserved

Parse cp1252 semicolon CSV and explicit `%d/%m/%Y %H:%M` timestamps once.
Retain every row, source filename, 1-based physical source line and input hash.
Malformed rows stay in the audit; they do not enter calculations. Repeat
observations are kept and pooled to estimate repeat SD. Source references
on each prism-cycle median identify all supporting raw rows. Overlapping
input files are not silently deduplicated: upload each observation once.

Station assignment uses connected overlapping station-easting intervals per
prism, without a guessed distance threshold. It assumes each target is observed
at one station, as in this handoff. Shared targets, identical station eastings,
or disjoint per-prism station regimes need owner-supplied `station_by_pid`.
Review the audit and station table before accepting automatic partitions.

Cycles are split independently at each station using configured observation
gaps. The sample config uses the supplied 15-minute W and 30-minute E gaps.
No fixed station coordinate, prism list or event date is embedded in code.

## Products and pitfalls

Baseline: first 48 hours per prism/station (configurable). Do not use Null
Measurement as baseline; its epoch is unknown. Raw LOS = change in slope
distance (+ away); raw vertical = change in D cos V (+ up); raw radial =
change in D sin V; raw tangential = D sin V times circular Hz change (+ clockwise).
Hz is a circle reading, not a grid bearing; estimate orientation from the
exported station/target vector as a consistency diagnostic. Apparent scale and
refraction are diagnostics, confounded by export height reduction.

Delivered coordinates are segmented at configured processing changes and at
the first departure from a station initially fixed throughout the baseline.
A station varying from the beginning remains one processing regime. This is
a conservative regime heuristic: review it and supply change dates for other
software changes. Coordinates alone cannot reliably distinguish every change.
Orientation differences are reported; step events require an explicit
`detection.orientation_step_arcsec`. No unprovided step threshold is assumed.

Noise uses 1.4826 MAD of residuals from a centered 24-hour rolling median.
LOS is quantised to the range resolution (`screening.range_resolution_mm`,
1 mm in the handoff), so its MAD can collapse to 0; LOS σ is therefore floored
at the uniform quantisation SD, resolution/√12 (0.29 mm). The unfloored MAD is
kept as `sigma_los_raw_mad`. Vertical and tangential σ are not floored: the
range quantum is scaled by cos V in vertical and does not enter tangential.
Spikes above the supplied robust-z 6 are retained as flags. Rates use Theil–Sen
on 24-hour block medians counted backward from the final record, using median
observation timestamps and excluding the initial partial block. Windows are
30 days, 7 days, and 72 hours (the latter uses cycles, with CI ignoring serial
autocorrelation). These rates are not calendar-day medians or row-index slopes.

Net changes are end-window minus baseline-window medians. Hour-matched nets
compare the same observed hours in those windows. Night/day diagnostics use
paired same-date medians; descriptive 95% normal CIs do not establish cause.
The window for a lost target ends at its last record, not at the file endpoint;
the missing tail is reported separately and never evidence of stability.

## Frame reconstruction

Huber IRLS jointly fits horizontal rotation, E/N translation, range scale,
vertical height/index and two tilt terms using the handoff's equations and
sigmas (0.8 arcsec Hz, 0.8 mm D, 3 mm vertical). Huber tuning 1.345 is the
standard numerical estimator constant, not a movement or alarm threshold.
Fits with insufficient rank are unavailable; raw series remain accessible.
Report parameter SEs, geometry condition, RMS, membership and exclusions.
Nominal fit uncertainty excludes model bias, external-control uncertainty and
unmodelled atmosphere. Ill-conditioned fits require manual review.

The vertical model is chosen per station from a ladder, richest first:
`full` (height, index, tilt sin/cos: the handoff model), `merged` (height,
index plus the tilt along the members' mean azimuth, tilt across it),
`height_index`, and `height`. Across a narrow fan the index and the along-fan
tilt both scale with distance and cannot be separated, and corrections for
targets outside the fan are extrapolated. The automatic rule
(`frame.vertical_model: auto`) uses the richest model for which every target's
a-priori correction SE, from the members' geometry and the supplied 3 mm
vertical sigma alone, has a median over cycles no greater than that sigma.
In words: no target gets a correction less precise than one raw vertical
observation. No new constant is introduced, and the choice depends on
geometry, not on the fitted values. A model name, or a `{station: model}`
mapping, overrides the rule. `frame_cycles.csv` records the model, how it was
chosen, the fan axis and the full model's worst target SE. With a reduced
model, `height_mm` is the intercept at the instrument; it can stay poorly
determined (HLO W: SE ~6 mm) while target corrections are within sigma, so
resection validation against exported height is weak at such a station.

Frame values are experimental. They may remove common ground movement;
raw/FC must be examined together. Reference names do not establish stability.
Explicit `frame.include`, `frame.exclude` and `frame.references_under_test`
control membership. Automatic selection requires the supplied 100 cycles,
excludes raw detected movers and, when a physical bias floor is supplied,
statistically persistent day/night candidates above that floor,
and excludes detected raw vertical movement, using grade A **only if an owner-defined grading policy exists**. With no
policy it is explicitly provisional, so the frame set differs from the
research frame set; frame-corrected golden values agree within 0.3 mm.
A target whose median frame fit SE exceeds its raw noise is flagged: at a
station whose targets span a narrow azimuth range (HLO station W) the vertical
frame is ill-conditioned and `ver_fc` is less precise than `ver_raw`.

## Decisions that require site input

Exploratory detection uses only the handoff's floors: 20 cycles, |net| at least
max(3 sigma, 2 mm LOS / 5 mm vertical) and a same-sign 30-day slope CI excluding
zero. Raw and corrected concern are reported separately; corrected concern
is labelled experimental. Tangential-only effects are evidence to investigate,
not movement classifications. No slope safety or alarm claims are made.

`reliability: null` leaves A–D ungraded. To supply a policy use A/B/C mappings,
each with `coverage_min`, `max_gap_hours`, `max_los_noise_mm`, and
`max_spike_fraction`; D comprises remaining cases. Choose boundaries with the
site owner. Physical `bias_floor_mm`, `orientation_step_arcsec` and
`cluster_radius_m` are unset until supplied; spatial seven-neighbour placebo
effects are computed descriptively. Candidate clusters and zones require an
explicit radius. Missing group members and overlapping placebo groups limit
interpretation; source series should be inspected.

TARP is disabled by default. An optional block must provide all of:
`metric` (los_raw/ver_raw/los_fc/ver_fc, mm), `threshold`, `averaging_hours`,
`persistence_hours`, owner-supplied `max_gap_hours`, `direction` (above/below/absolute), and `label`. This
version supports one fully specified displacement condition. It checks the
condition at the last observed record and resets persistence across gaps.
Both averaging and persistence reset across missing metrics or configured
gaps; an unavailable final metric remains unavailable. It is an observation-condition
evaluation, not a live monitoring alarm system.

`site_transform` accepts `rotation_deg`, `offset_e`, `offset_n` in metres,
and optional `srs_id` plus explicit `srs_wkt`. Positive rotation is mathematical
counterclockwise in E/N axes. Both points and vector components rotate. SRS is
never inferred. QGIS default styles contain vector geometry generators using
`@hlo_vec_scale`; optional map circles are a median LOS fit-SE diagnostic,
not horizontal vector uncertainty or total survey error.

## Evidence and remaining interpretation limits

Investigations include balanced step windows (when a step limit is supplied),
raw-fit/exported station shift comparisons, differenced Spearman reference
inference with Bonferroni p-values, same-date and weekly bias, hour matching,
night/day trends, temporal splits, hinge/three-segment fit comparisons,
seven-neighbour placebos, candidate pairwise LOS, missing-target distance
Mann–Whitney and seeded permutation evidence. Descriptive optimizations and
overlapping placebo counts are not significance tests. Spatial clustering is
configurable and remains exploratory. Independent verification must use
external controls and observations; no automated test establishes slope safety.

No rainfall, blasting, stacking or geology is inferred when absent. No absolute
movement without control; no identifiable resection references from export
alone; no factor of safety; no alarm without a site TARP; no evidence of
stability during unobserved intervals. Vertical below about 5 mm is sensitive
to atmosphere and sampling. Group-median composition changes must be reviewed.

## Golden regression

`tests/test_golden_hlo.py` runs the pipeline on `data/HLO Sept 2026.csv` with
`config.yaml` and compares it with `tests/fixtures/golden_values.json`, the
owner's file committed byte for byte. The test is skipped with a reason if
either file is absent or the export's MD5 differs from `input.md5`. Set
`RTS_GOLDEN_DATA` and `RTS_GOLDEN_VALUES` to check another copy.

The golden file holds values only. Tolerances sit in one `TOLERANCES` table in
the test, each with its reason (golden rounding, 1 mm range quantisation,
provisional frame set). They are regression tolerances for owner review, not
site, movement or alarm thresholds. Quantities without a single pipeline
output (period rates, placebo groups, the 7 Sep step) are derived in the test
from pipeline tables, and each derivation is stated there. Golden values the
implementation does not reproduce are `xfail(strict=True)` with the reason, so
that a later fix fails the test until the marker is removed.

Reference inference follows the handoff's cycle-detrended definition: each
target's raw vertical (or Hz) minus the leave-one-out cycle median of the
station's other targets, correlated with exported station height (or
orientation). A time-differenced variant is kept as a robustness check; it
cannot detect slow co-variation such as subsidence.
