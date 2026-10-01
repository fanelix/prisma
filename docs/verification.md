# Verification and implementation decisions

Local verification, 1 October 2026: **113 passed, 7 skipped** with numeric
runtime warnings treated as errors. Lint, formatting, strict documentation
build, wheel/sdist build and installed-wheel CLI smoke test passed.
Six dashboard pages, both map toggles, and pure-Python GeoPackage integrity,
ten layer records and ten default styles were checked.

The seven skips are five legacy Candrian checks whose external CSV is absent,
the optional HLO independent-golden regression whose JSON is absent, and an
existing optional GDAL interoperability check. No failed test is hidden by
these skips; each retains its verification when its fixture/runtime exists.

The available HLO CSV contained 39,822 rows, all retained without parse errors,
208 targets, 462 western cycles and 178 eastern cycles. An end-to-end run
produced eastern fitted final rotation -12.836964 arcsec and HLO-R7 raw vertical
net -12.680667 mm, consistent with the narrative handoff's rounded values.
This is a sanity check, not an independent golden regression. Research
reference scripts and golden JSON were not supplied.

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
