# Limitations

These statements are part of the product, not fine print: every report and
every dashboard page displays them. They describe what an RTS prism monitoring
export cannot establish, regardless of how the pipeline is configured.

- **No slope stability or factor of safety from RTS data alone.** RTS
  displacement observations describe how targets moved relative to an
  instrument network. Stability analysis requires geotechnical models, material
  properties and other monitoring, none of which are in this data set.
- **No alarm status without a site TARP.** A Trigger Action Response Plan is a
  site decision with agreed thresholds, averaging periods, persistence rules
  and missing-data policy. `rts-forensics` applies a TARP only when one is
  supplied in configuration; otherwise the TARP state is `not_configured` and
  no alarm is produced.
- **No absolute movement without external control.** The frame is defined by
  the network being monitored. Frame-corrected values are relative to the
  chosen frame set; if every prism in the network moved together, the
  correction would remove it. Absolute motion requires external control that
  this export does not contain.
- **No causes without rainfall, blasting, stacking or geology data.** The
  forensic tests can show that a pattern is unlikely to be a sampling or
  instrument artefact, but causal attribution requires external driver data
  that must be supplied separately and analysed with explicit lags.
- **Resection references are not identifiable from the export.** Correlation
  tests can suggest which prisms track station height or orientation, but the
  export does not state which points are the resection references, and a
  correlation has competing explanations.
- **Unobserved periods are not evidence of stability.** Gaps, dropouts and
  prisms lost in the final window mean the metric is unavailable for that
  period. Coverage and gap statistics are reported so missingness is visible;
  it is never described as stable.

The exploratory screening labels produced by the pipeline
(*no credible movement detected*, *possible*, *insufficient data*) and the
separate reliability grades comment on the **measurement** and on the
**evidence in this export**. They are not statements that a slope is safe or
unsafe, and they do not replace a site TARP or engineering judgement.
