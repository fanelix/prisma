"""Pure coordinator; input bytes are read once and never modified."""

import pandas as pd

from .classify import summarize
from .config import load_config
from .cycles import assign_cycles, cycle_table
from .displacement import prism_cycles
from .frame import fit_frames
from .investigations import investigate
from .noise import estimate, pooled_repeatability
from .parse import read_export
from .rates import blocks24


def run(source, config=None):
    config = load_config(overrides=config) if config else load_config()
    inputs = []
    observations = []
    sources = source if isinstance(source, list) else [source]
    for item in sources:
        rows, info = read_export(item, config)
        observations.append(rows)
        inputs.append(info)
    rows = assign_cycles(pd.concat(observations, ignore_index=True), config)
    if (~rows.parse_error).sum() == 0:
        raise ValueError("No valid observations; all source rows failed audit")
    series = prism_cycles(rows, config)
    series, noise = estimate(series, rows, config)
    for column in ["los_fc", "tan_fc", "ver_fc"]:
        series[column] = float("nan")
    initial = summarize(series, noise, config)
    series, frames, members = fit_frames(series, initial, config)
    summary = summarize(series, noise, config)
    investigations, events = investigate(series, frames, summary, config)
    warnings = [
        "All outputs are exploratory; no slope safety or factor-of-safety inference.",
        "Frame corrections are experimental and network-relative; common ground movement may be removed.",
        "Raw and corrected values must be interpreted together; unobserved periods are not stability evidence.",
        "No causes can be established without rainfall, blasting, stacking or geological context.",
        "Resection references cannot be identified from this export alone.",
        "Vertical sensitivity below about 5 mm and tangential-only signals require independent checks.",
        "72-hour rate confidence intervals ignore serial autocorrelation.",
    ]
    if config["reliability"] is None:
        warnings.append(
            "Reliability grading is unconfigured; automatic frame selection is provisional, not reliability A."
        )
    if config["tarp"] is None:
        warnings.append("No TARP configured; no alarm status is assigned.")
    if config["detection"]["bias_floor_mm"] is None:
        warnings.append(
            "No physical bias floor supplied: day/night effects are descriptive candidates and do not automatically exclude targets from the provisional frame. Review frame membership."
        )
    flags = summary["flags"].fillna("")
    for text, message in [
        (
            "frame-corrected vertical less precise",
            "Frame-corrected vertical is less precise than raw for {n} target(s) (fit SE exceeds raw noise; "
            "typically a narrow-azimuth station geometry). Interpret ver_fc for those targets with raw values.",
        ),
        (
            "frame-corrected LOS less precise",
            "Frame-corrected LOS is less precise than raw for {n} target(s); compare raw and corrected series.",
        ),
        (
            "windows overlap",
            "{n} target(s) span less than the baseline plus end windows; their net change is unavailable.",
        ),
    ]:
        n = int(flags.str.contains(text, regex=False).sum())
        if n:
            warnings.append(message.format(n=n))
    if any(i["invalid_rows"] for i in inputs):
        warnings.append("Invalid rows are retained in observations/audit and excluded from calculations.")
    met = rows[["ppm", "pressure", "temperature", "add_const"]]
    if all(met[c].nunique(dropna=True) <= 1 for c in met):
        warnings.append(
            "Atmospheric metadata are constant or missing; treat as no measured meteorological data."
        )
    field_checks = pd.DataFrame(
        [
            {
                "check_id": f"F{i + 1:03}",
                "question": q,
                "supports_if": "independent evidence agrees with raw observation signature",
                "alternative_if": "evidence suggests instrument, atmosphere or sampling effects",
                "status": "open",
                "outcome": "",
                "notes": "",
                "photo": "",
            }
            for i, q in enumerate(
                [
                    "Check station/resection logs around frame changes",
                    "Independently survey movement candidates and external controls",
                    "Inspect obscured/lost targets and sight lines",
                    "Confirm site grid transform and local timestamp convention",
                ]
            )
        ]
    )
    return {
        "observations": rows,
        "series": series,
        "prism_summary": summary,
        "cycle_table": cycle_table(rows),
        "frame_cycles": frames,
        "frame_members": members,
        "noise": noise,
        "repeatability": pooled_repeatability(rows),
        "timeseries_24h": blocks24(series),
        "event_register": events,
        "investigation_register": investigations,
        "field_checks": field_checks,
        "inputs": inputs,
        "config": config,
        "warnings": warnings,
    }
