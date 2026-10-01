"""
models.py — typed contracts and documented table schemas for rts-forensics.

Everything crossing a module boundary is described here: canonical column
names, units, grain and keys. Modules must not invent column names; output
tables registered in :data:`TABLE_SPECS` are what the CSV dictionary, the
report and the GeoPackage export read.

Honesty rules encoded in these types:
  * a computation that cannot be performed carries ``status="not_configured"``
    (or another explicit status) and a human-readable ``reason`` — never a
    fabricated numeric value;
  * raw and frame-corrected columns live side by side, never overwriting each
    other;
  * provenance is explicit: every observation has ``(source_sha256, src_line)``
    and every aggregation links back through ``source_links``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
#  Status vocabulary
# ---------------------------------------------------------------------------
NOT_CONFIGURED = "not_configured"
STATUS_OK = "ok"
STATUS_DISABLED = "disabled"
STATUS_UNAVAILABLE = "unavailable"
STATUS_INSUFFICIENT = "insufficient_data"
STATUS_RANK_DEFICIENT = "rank_deficient"
STATUS_FAILED = "failed"

# ---------------------------------------------------------------------------
#  Units
# ---------------------------------------------------------------------------
UNITS: dict[str, str] = {
    "geometry_distance": "m",
    "displacement": "mm",
    "translation": "mm",
    "angle_internal": "rad",
    "angle_observation": "deg",
    "rotation_export": "arcsec",
    "rate": "mm/day",
    "scale": "ppm",
    "time": "site-local naive",
}

# ---------------------------------------------------------------------------
#  Canonical observation schema
# ---------------------------------------------------------------------------
OBSERVATION_COLUMNS: dict[str, str] = {
    "obs_id": "str — immutable, f'{source_sha256[:16]}:{src_line}'",
    "source_sha256": "str — SHA-256 of the original file bytes",
    "src_line": "int — physical line number in the source file (1-based)",
    "source_name": "str — original file name",
    "point_id": "str — prism/target identifier as exported",
    "time": "datetime64[ns] — site-local, naive; never converted implicitly",
    "hz_deg": "float — horizontal direction, degrees [0, 360)",
    "v_deg": "float — vertical angle, degrees, as exported",
    "d_m": "float — slope distance, metres",
    "te_m": "float — delivered target easting, metres (processing product)",
    "tn_m": "float — delivered target northing, metres (processing product)",
    "tz_m": "float — delivered target elevation, metres (processing product)",
    "se_m": "float — station easting, metres",
    "sn_m": "float — station northing, metres",
    "sh_m": "float — station height, metres",
    "temp_c": "float — air temperature, degrees Celsius (optional)",
    "ppm": "float — EDM scale correction, ppm (optional)",
    "horz_dist_m": "float — instrument-computed horizontal distance, m (optional)",
    "null_meas_m": "float — null measurement, m (optional)",
}

# ---------------------------------------------------------------------------
#  Prism-cycle schema (repeats retained, Hz aggregated circularly)
# ---------------------------------------------------------------------------
PRISM_CYCLE_COLUMNS: dict[str, str] = {
    "point_id": "str",
    "station_id": "str — assigned station",
    "cycle_id": "int — elapsed-time cycle within the station",
    "cycle_start": "datetime64[ns]",
    "cycle_end": "datetime64[ns]",
    "n_obs": "int — repeats retained in this cycle",
    "hz_deg": "float — circular mean Hz [0, 360)",
    "v_deg": "float — mean vertical angle",
    "d_m": "float — mean slope distance",
    "te_m": "float — mean delivered easting",
    "tn_m": "float — mean delivered northing",
    "tz_m": "float — mean delivered elevation",
    "se_m": "float — station easting",
    "sn_m": "float — station northing",
    "sh_m": "float — station height",
    "sd_hz_deg": "float — within-cycle repeat SD of Hz",
    "sd_v_deg": "float — within-cycle repeat SD of V",
    "sd_d_m": "float — within-cycle repeat SD of D",
    "obs_ids": "object — tuple of contributing observation IDs",
}

# ---------------------------------------------------------------------------
#  Baseline schema (one baseline per point per processing segment)
# ---------------------------------------------------------------------------
BASELINE_COLUMNS: dict[str, str] = {
    "point_id": "str",
    "segment_id": "str — processing segment this baseline belongs to",
    "start": "datetime64[ns] — first observation in the baseline window",
    "end": "datetime64[ns] — last observation in the baseline window",
    "n_obs": "int",
    "complete": "bool — whether the configured window/completeness rule was met",
    "te_m": "float — baseline easting",
    "tn_m": "float",
    "tz_m": "float",
    "obs_ids": "object — tuple of contributing observation IDs",
    "status": "str — ok / insufficient_data / not_configured",
}

# ---------------------------------------------------------------------------
#  Displacement schema (raw polar + delivered, segmented)
# ---------------------------------------------------------------------------
DISPLACEMENT_COLUMNS: dict[str, str] = {
    "point_id": "str",
    "segment_id": "str",
    "station_id": "str",
    "cycle_id": "int",
    "ts": "datetime64[ns]",
    "d_rad_mm": "float — raw radial displacement from baseline, mm",
    "d_tan_mm": "float — raw tangential displacement from baseline, mm",
    "d_vert_mm": "float — raw vertical displacement from baseline, mm",
    "d_east_mm": "float — delivered coordinate change, segment baseline, mm",
    "d_north_mm": "float — delivered coordinate change, mm",
    "d_up_mm": "float — delivered elevation change, mm",
    "baseline_status": "str — ok / insufficient_data / not_configured",
    "obs_ids": "object — contributing prism-cycle observation IDs",
    "baseline_obs_ids": "object — baseline observation IDs",
}

# ---------------------------------------------------------------------------
#  Source links (normalized provenance)
# ---------------------------------------------------------------------------
SOURCE_LINK_COLUMNS: dict[str, str] = {
    "table_name": "str — output table that used the observation",
    "row_key": "str — stable key of the consuming row (e.g. point#segment@cycle)",
    "role": "str — observation / baseline / frame / exclusion",
    "obs_id": "str",
    "source_sha256": "str",
    "src_line": "int",
}

# ---------------------------------------------------------------------------
#  Rates
# ---------------------------------------------------------------------------
RATE_COLUMNS: dict[str, str] = {
    "point_id": "str",
    "segment_id": "str",
    "metric": "str — d_rad / d_vert / d_tan / d_east / d_north / d_up",
    "window": "str — 30d / 7d / 72h",
    "n_blocks": "int — complete 24 h blocks used (72 h window: cycles)",
    "n_obs": "int",
    "t_start": "datetime64[ns]",
    "t_end": "datetime64[ns]",
    "rate_mm_day": "float",
    "ci_low": "float",
    "ci_high": "float",
    "method": "str — theil_sen / blocked_median",
    "partial_block": "bool — whether an excluded partial block influenced the result",
    "status": "str",
    "reason": "str — populated when status != ok",
}

# ---------------------------------------------------------------------------
#  Forensics
# ---------------------------------------------------------------------------
EVIDENCE_COLUMNS: dict[str, str] = {
    "test": "str — test family (groups / sampling / references / missingness / events)",
    "hypothesis": "str",
    "metric": "str",
    "statistic": "float",
    "p_value": "float",
    "alpha": "float",
    "members": "object — identifiers of the compared rows/blocks",
    "status": "str",
    "reason": "str",
}

INVESTIGATION_COLUMNS: dict[str, str] = {
    "investigation_id": "str",
    "opened": "datetime64[ns]",
    "topic": "str",
    "candidate_explanation": "str",
    "competing_explanation": "str",
    "evidence": "str — links to evidence table rows",
    "status": "str — open / not_configured / closed",
}

# ---------------------------------------------------------------------------
#  Classification
# ---------------------------------------------------------------------------
CONCERN_COLUMNS: dict[str, str] = {
    "point_id": "str",
    "segment_id": "str",
    "metric": "str",
    "net_mm": "float",
    "sigma_mm": "float",
    "floor_mm": "float",
    "ci_low": "float",
    "ci_high": "float",
    "exceeds": "bool",
    "status": "str",
    "reason": "str",
}

RELIABILITY_COLUMNS: dict[str, str] = {
    "point_id": "str",
    "segment_id": "str",
    "n_prism_cycles": "int",
    "frame_member": "bool",
    "noise_mm": "float",
    "coverage": "float",
    "grade": "str — not_configured until boundaries are supplied",
    "status": "str",
    "reason": "str",
}

TARP_COLUMNS: dict[str, str] = {
    "point_id": "str",
    "segment_id": "str",
    "metric": "str",
    "value_mm": "float",
    "tarp_mm": "float",
    "state": "str",
    "status": "str — not_configured / applied",
    "reason": "str",
}

# ---------------------------------------------------------------------------
#  Table dictionary registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TableSpec:
    name: str
    grain: str
    keys: tuple[str, ...]
    units: dict[str, str]
    description: str
    columns: tuple[str, ...] = ()


TABLE_SPECS: dict[str, TableSpec] = {
    "observations": TableSpec(
        "observations", "one row per source observation",
        ("obs_id",), {"distance": "m", "angle": "deg", "time": "site-local naive"},
        "Canonical observations with immuteful provenance IDs", tuple(OBSERVATION_COLUMNS)),
    "prism_cycles": TableSpec(
        "prism_cycles", "one row per prism per cycle",
        ("point_id", "station_id", "cycle_id"),
        {"distance": "m", "angle": "deg"}, "Repeats retained, Hz circularly aggregated",
        tuple(PRISM_CYCLE_COLUMNS)),
    "baselines": TableSpec(
        "baselines", "one row per prism per processing segment",
        ("point_id", "segment_id"), {"distance": "m"}, "Segment baseline definitions",
        tuple(BASELINE_COLUMNS)),
    "displacements": TableSpec(
        "displacements", "one row per prism per cycle",
        ("point_id", "segment_id", "cycle_id"),
        {"displacement": "mm"}, "Raw polar and delivered displacements, segmented",
        tuple(DISPLACEMENT_COLUMNS)),
    "rates": TableSpec(
        "rates", "one row per prism per metric per window",
        ("point_id", "segment_id", "metric", "window"),
        {"rate": "mm/day"}, "Backwards 24 h block rates with CI",
        tuple(RATE_COLUMNS)),
    "evidence": TableSpec(
        "evidence", "one row per hypothesis test",
        ("test", "hypothesis", "metric"), {}, "Forensic evidence records",
        tuple(EVIDENCE_COLUMNS)),
    "investigations": TableSpec(
        "investigations", "one row per investigation",
        ("investigation_id",), {}, "Investigation register",
        tuple(INVESTIGATION_COLUMNS)),
    "concern": TableSpec(
        "concern", "one row per prism per metric per segment",
        ("point_id", "segment_id", "metric"), {"displacement": "mm"},
        "Movement-concern screening (no TARP meaning)", tuple(CONCERN_COLUMNS)),
    "reliability": TableSpec(
        "reliability", "one row per prism per segment",
        ("point_id", "segment_id"), {}, "Measurement reliability (separate from concern)",
        tuple(RELIABILITY_COLUMNS)),
    "tarp": TableSpec(
        "tarp", "one row per prism per metric per segment",
        ("point_id", "segment_id", "metric"), {"displacement": "mm"},
        "User-supplied TARP state; not_configured when absent", tuple(TARP_COLUMNS)),
    "source_links": TableSpec(
        "source_links", "one row per (output row, contributing observation)",
        ("table_name", "row_key", "obs_id"), {}, "Normalized provenance links",
        tuple(SOURCE_LINK_COLUMNS)),
}


def empty_frame(schema: dict[str, str] | tuple[str, ...]) -> pd.DataFrame:
    """An empty DataFrame with exactly the documented columns."""
    cols = list(schema)
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def table_dictionary() -> pd.DataFrame:
    """Machine-readable dictionary of every registered output table."""
    rows = []
    for spec in TABLE_SPECS.values():
        for col in spec.columns:
            rows.append({
                "table": spec.name,
                "grain": spec.grain,
                "key_columns": ", ".join(spec.keys),
                "column": col,
                "units": spec.units.get(col, spec.units.get("distance", "")),
                "description": spec.description,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FormatSpec:
    delimiter: str
    encoding: str
    time_format: str
    column_map: dict[str, str] = field(default_factory=dict)
    has_header: bool = True


@dataclass
class SourceRecord:
    name: str
    sha256: str
    n_bytes: int
    n_rows: int = 0
    raw_path: str | None = None


@dataclass
class ParsedExport:
    observations: pd.DataFrame
    format: FormatSpec
    sources: list[SourceRecord] = field(default_factory=list)
    diagnostics: pd.DataFrame = field(default_factory=lambda: empty_frame(
        {"line": "int", "severity": "str", "message": "str"}))
    source_links: pd.DataFrame = field(default_factory=lambda: empty_frame(SOURCE_LINK_COLUMNS))
    raw_bytes: dict[str, bytes] = field(default_factory=dict, repr=False)


@dataclass
class StationAssignment:
    observations: pd.DataFrame
    stations: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: str = NOT_CONFIGURED
    reason: str | None = "station association tolerance not configured"


@dataclass
class ChangeResult:
    changes: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: str = NOT_CONFIGURED
    reason: str | None = "processing-change thresholds not configured"


@dataclass
class CycleResult:
    observations: pd.DataFrame
    cycles: pd.DataFrame = field(default_factory=pd.DataFrame)
    assignment: StationAssignment | None = None
    segments: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidate_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    changes: ChangeResult | None = None
    status: str = STATUS_OK
    reason: str | None = None


@dataclass
class DisplacementResult:
    prism_cycles: pd.DataFrame
    displacements: pd.DataFrame = field(default_factory=lambda: empty_frame(DISPLACEMENT_COLUMNS))
    baselines: pd.DataFrame = field(default_factory=lambda: empty_frame(BASELINE_COLUMNS))
    source_links: pd.DataFrame = field(default_factory=lambda: empty_frame(SOURCE_LINK_COLUMNS))
    status: str = STATUS_OK
    reason: str | None = None


@dataclass
class NoiseResult:
    pooled: pd.DataFrame = field(default_factory=pd.DataFrame)
    series: pd.DataFrame = field(default_factory=pd.DataFrame)
    flags: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: str = STATUS_OK
    reason: str | None = None


@dataclass
class FrameSelection:
    members: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: str = NOT_CONFIGURED
    reason: str | None = "frame exclusion logic requires the reference bundle"


@dataclass
class CycleFrameFit:
    cycle_id: int | None = None
    coefficients: dict[str, float] = field(default_factory=dict)
    std_errors: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    condition_number: float | None = None
    residuals: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: str = NOT_CONFIGURED
    reason: str | None = "Huber tuning and convergence rules require the reference bundle"


@dataclass
class FrameResult:
    coefficients: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    residuals: pd.DataFrame = field(default_factory=pd.DataFrame)
    membership: FrameSelection | None = None
    corrected: pd.DataFrame = field(default_factory=pd.DataFrame)
    correction_status: str = NOT_CONFIGURED
    experimental: bool = True
    status: str = NOT_CONFIGURED
    reason: str | None = "frame correction not configured"


@dataclass
class ForensicResult:
    evidence: dict[str, pd.DataFrame] = field(default_factory=dict)
    investigations: pd.DataFrame = field(default_factory=lambda: empty_frame(INVESTIGATION_COLUMNS))
    statuses: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ClassificationResult:
    concern: pd.DataFrame = field(default_factory=lambda: empty_frame(CONCERN_COLUMNS))
    reliability: pd.DataFrame = field(default_factory=lambda: empty_frame(RELIABILITY_COLUMNS))
    flags: pd.DataFrame = field(default_factory=pd.DataFrame)
    tarp: pd.DataFrame = field(default_factory=lambda: empty_frame(TARP_COLUMNS))
    statuses: dict[str, str] = field(default_factory=dict)


@dataclass
class RunResult:
    observations: pd.DataFrame
    parsed: ParsedExport
    cycles: CycleResult
    displacements: DisplacementResult
    noise: NoiseResult
    frames: FrameResult
    forensics: ForensicResult
    classification: ClassificationResult
    rates: pd.DataFrame = field(default_factory=lambda: empty_frame(RATE_COLUMNS))
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    config: Any = None
    audit: Any = None
    manifest: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)

    def table(self, name: str) -> pd.DataFrame:
        if name in self.tables:
            return self.tables[name]
        raise KeyError(f"table not registered in this run: {name}")
