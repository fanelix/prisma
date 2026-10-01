# Pitfalls

The twelve failure modes a naive implementation gets wrong, and the
countermeasure in `rts-forensics`. They come from the research bundle's own
findings on a real monthly export; the numbers below are observations from that
export, not thresholds the app applies automatically.

## 1. Trusting as-delivered coordinates

**What goes wrong.** The delivered coordinates contain an uncorrected
instrument rotation of up to 12.9″ — a network-median 3D artefact of 26 mm and
54 mm at the backsight. Reporting coordinate change directly would present an
instrument error as ground movement.

**Countermeasure.** The analysis works from the raw observations (Hz, V, slope
distance) and reconstructs the reference frame itself. Delivered coordinates
are kept as a separate, segmented product and every raw/corrected pair is shown
beside its raw counterpart.

## 2. Bridging across a processing change

**What goes wrong.** Station re-estimation started mid-month. A rate, net
change, plot line or block statistic that spans the change would blend two
different processing regimes and can invent centimetres of movement.

**Countermeasure.** As-delivered coordinate series are split into processing
segments at configured and auto-detected changes (`E-S1`, `E-S2`, `W1` in the
reference export) and are never bridged across a boundary. Raw polar series
remain available across the change when observations support them. Baselines
are computed per prism per segment.

## 3. Assuming a "reference" or backsight is stable

**What goes wrong.** `BS_HL_5` drifted about 10 mm against the network, and
`HLO-R7` — a name that looks like a reference — is subsiding. A named reference
can itself move, and a tested point must not prove its own stability.

**Countermeasure.** Frame membership is explicit and published; leave-one-out
and reference-under-test exclusions are applied. Reference inference reports
correlations and their limitations and never identifies a reference from the
export alone. Known movers and biased prisms can be excluded through
configuration.

## 4. Calendar-day medians

**What goes wrong.** Partial days bias daily values: daytime LOS reads about
2 ppm shorter and vertical shows a diurnal swing of about 4 mm. Calendar-day
averages mix different hours of day and different numbers of observations.

**Countermeasure.** Rates use 24-hour block medians counted backwards from the
final record, anchored at the final observation, and block counts and bounds
are retained. Hour-matched and day/night comparisons are separate products.

## 5. Changing sampling hours

**What goes wrong.** Prisms that lose their night observations show fake
vertical trends of 5–10 mm, because only the biased daytime population remains.

**Countermeasure.** The app compares like hours: day/night bands, hour-matched
change tests and sampling-composition tests are part of the forensic stage, and
coverage is reported by hour and in the final window. Missing-at-night is
treated as missing data, not as stability.

## 6. Group medians with missing members

**What goes wrong.** On 28–29 September the `HLO-R` group median was `HLO-R7`
alone. A shrinking group silently changes the statistic's meaning.

**Countermeasure.** Group statistics record member IDs and actual contributors
for every epoch; a constant-membership product is separate from the
hour-matched product, and the number of members is explicit, so a group
shrinking to one prism is visible.

## 7. Treating repeats as duplicates

**What goes wrong.** HLO prisms are measured two or three times per cycle.
Dropping the extra rows as "duplicates" destroys the only direct estimate of
repeatability and changes the series.

**Countermeasure.** Repeats are retained. Aggregation uses a circular median
for Hz (with wrap-around at 0/360 degrees, never ordinary subtraction) and
medians for the remaining quantities, and the pooled within-cycle repeatability
is a first-class output.

## 8. Velocity from row index

**What goes wrong.** Using the row number as a time axis makes velocity depend
on sampling density and on gaps, and breaks entirely when a station has a
different schedule.

**Countermeasure.** Every rate, gap and cycle decision uses elapsed time with
explicit site-local timestamps; rates use elapsed days against block medians.

## 9. Distance quantisation

**What goes wrong.** `D` has 1 mm resolution, so Theil–Sen on raw cycles gives
many zero slopes and an unstable estimate.

**Countermeasure.** Repeats are pooled per cycle and rates are computed on
24-hour block medians, which absorbs quantisation. Theil–Sen remains robust to
the remaining outliers.

## 10. Tangential component

**What goes wrong.** The tangential component is the weakest: this export's
frame-corrected tangential reaches a network 90th percentile of 5.6 mm, and it
also contains instrument rotation.

**Countermeasure.** Raw tangential signals are preserved alongside corrected
ones, and a tangential-only signal requires independent consistency evidence
before any movement label. The frame fit estimates the rotation explicitly
rather than absorbing it into a prism.

## 11. Removing a common mode automatically

**What goes wrong.** Subtracting a network median or common mode may remove
real movement that affects every prism, and there is no way to see that from a
corrected series alone.

**Countermeasure.** Raw and frame-corrected values are always shown side by
side in tables, plots, reports, the dashboard and the GIS products. Frame
correction is marked experimental and relative to the chosen network; raw
values are never replaced.

## 12. Inventing thresholds

**What goes wrong.** No site TARP was supplied. Inventing an alarm threshold
(or reading the exploratory screening as an alarm) would turn a research label
into a safety claim the data cannot support.

**Countermeasure.** Screening labels stay labelled as exploratory. TARP exists
only if the user supplies one; otherwise its state is `not_configured` and no
alarm is produced. Every value the handoff does not define stays
`not_configured` in configuration, and the effective configuration is written
into every run for audit.
