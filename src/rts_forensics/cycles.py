"""
cycles.py — station assignment, elapsed-time cycles and change diagnostics.

Three layers, in pipeline order:

* :func:`assign_stations` assigns every observation to a station. Explicit
  ``config.stations.anchors`` label the site clusters by nearest
  ``easting_ref``; otherwise stations are auto-detected by splitting the
  sorted unique station eastings at the largest gap and labelling ``ST1``,
  ``ST2``, ... by descending easting. ``config.stations.association_tolerance_m``
  is not configured and is never applied as a hidden tolerance.
* :func:`build_cycles` derives the ALGORITHMS.md section 2 consistency
  diagnostics, starts a new cycle when the elapsed gap to the previous
  observation at the same station exceeds the configured per-station
  threshold, and produces the cycle table, processing segments, measured
  processing-change deltas and an explicit (empty) candidate-event table.
* :func:`detect_processing_changes` reports every station transition with its
  measured east/north/height/orientation deltas. The classification thresholds
  are not configured, so the status stays ``not_configured`` and no jump is
  ever classified.

Input frames are never mutated: every function returns a copy with added
columns. Cycle identifiers are ``f"{label}{n:03d}"`` with ``n`` starting at 0
per station.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig, is_not_configured
from .geometry import (
    angular_difference_deg,
    azimuth_deg,
    circular_mean_deg,
    circular_std_deg,
)
from .models import (
    NOT_CONFIGURED,
    STATUS_OK,
    ChangeResult,
    CycleResult,
    StationAssignment,
)

#: Grain of the cycle table: one row per station per elapsed-time cycle.
CYCLE_COLUMNS: tuple[str, ...] = (
    "cycle_id", "station_id", "t0", "t1", "n", "npid",
    "se_m", "sn_m", "sh_m", "n_distinct_heights",
    "orientation_deg", "orientation_sd_arcsec",
    "scale_ppm", "dv_arcsec", "dur_min",
)

#: Per-observation diagnostics appended to the canonical observations.
OBSERVATION_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "dE", "dN", "dZ", "hd_c", "sd_c", "az_deg", "zen_deg",
    "dHz_arcsec", "dV_arcsec", "dSD_mm", "dHDrep_mm", "scale_ppm",
)

#: One row per prism per processing segment.
SEGMENT_COLUMNS: tuple[str, ...] = (
    "point_id", "segment_id", "station_id", "t_start", "t_end", "n_obs",
)

#: Measured per-transition station deltas (never a classification).
CHANGE_COLUMNS: tuple[str, ...] = (
    "t", "station_id", "d_east_mm", "d_north_mm",
    "d_height_mm", "d_orientation_arcsec",
)

#: Placeholder table; candidate frame events need configured thresholds.
CANDIDATE_EVENT_COLUMNS: tuple[str, ...] = (
    "event_id", "kind", "start", "end", "note",
)

STATION_COLUMNS: tuple[str, ...] = (
    "label", "easting_ref", "n_obs", "median_se_m", "median_sn_m", "median_sh_m",
)

_STATION_DIAGNOSTIC_COLUMNS: tuple[str, ...] = ("method", "station_id", "message")

_NOT_CONFIGURED_REASON = (
    "processing-change classification thresholds are not configured; measured "
    "per-transition station deltas are reported without a classification"
)


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


# ---------------------------------------------------------------------------
#  A. Station assignment
# ---------------------------------------------------------------------------
def assign_stations(observations: pd.DataFrame, *,
                    config: AnalysisConfig) -> StationAssignment:
    """Assign every observation to a station without inventing a tolerance.

    With configured ``config.stations.anchors`` each observation is assigned to
    the anchor whose ``easting_ref`` is closest in absolute difference and the
    label comes from the anchor. Without anchors the sorted unique station
    eastings are split once at the largest gap; the higher cluster is ``ST1``
    and the lower is ``ST2`` (labels are stated to be automatic in the
    diagnostics). The returned observations are a copy with ``station_id``
    added; the input frame is never mutated.
    """
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame")
    missing = {"point_id", "se_m"} - set(observations.columns)
    if missing:
        raise ValueError(f"observations missing required columns: {sorted(missing)}")

    work = observations.copy()
    anchors = [anchor for anchor in (config.stations.anchors or []) if anchor]
    if anchors:
        labels, stations, diagnostics = _assign_by_anchors(work, anchors)
    else:
        labels, stations, diagnostics = _assign_by_easting_gap(work)
    work["station_id"] = labels
    return StationAssignment(
        observations=work,
        stations=stations,
        diagnostics=diagnostics,
        status=STATUS_OK,
        reason=None,
    )


def _anchor_field(anchor: Any, key: str, default: Any = None) -> Any:
    if isinstance(anchor, dict):
        return anchor.get(key, default)
    return getattr(anchor, key, default)


def _assign_by_anchors(observations: pd.DataFrame, anchors: list[Any]):
    labels = [str(_anchor_field(anchor, "label", f"ST{index + 1}"))
              for index, anchor in enumerate(anchors)]
    refs = np.asarray([float(_anchor_field(anchor, "easting_ref"))
                       for anchor in anchors], dtype=float)
    easting = pd.to_numeric(observations["se_m"], errors="coerce").to_numpy(dtype=float)
    assigned = pd.Series(pd.NA, index=observations.index, dtype="object")
    valid = np.isfinite(easting)
    if valid.any():
        distances = np.abs(easting[valid, None] - refs[None, :])
        nearest = np.argmin(distances, axis=1)
        assigned.loc[valid] = [labels[index] for index in nearest]

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for label, ref in zip(labels, refs, strict=True):
        mask = assigned == label
        subset = observations.loc[mask]
        rows.append({
            "label": label,
            "easting_ref": float(ref),
            "n_obs": int(mask.sum()),
            "median_se_m": _median(subset, "se_m"),
            "median_sn_m": _median(subset, "sn_m"),
            "median_sh_m": _median(subset, "sh_m"),
        })
        diagnostics.append({
            "method": "anchors",
            "station_id": label,
            "message": (
                f"assigned by nearest configured easting_ref={ref:g}; "
                "association_tolerance_m is not configured and was not used as a gate"),
        })
    return (assigned,
            pd.DataFrame(rows, columns=list(STATION_COLUMNS)),
            pd.DataFrame(diagnostics, columns=list(_STATION_DIAGNOSTIC_COLUMNS)))


def _assign_by_easting_gap(observations: pd.DataFrame):
    easting = pd.to_numeric(observations["se_m"], errors="coerce")
    finite = easting.dropna()
    if finite.empty:
        assigned = pd.Series(pd.NA, index=observations.index, dtype="object")
        diagnostics = pd.DataFrame([{
            "method": "auto_easting_gap",
            "station_id": pd.NA,
            "message": "no finite station eastings; automatic labels unavailable",
        }], columns=list(_STATION_DIAGNOSTIC_COLUMNS))
        return assigned, _empty(STATION_COLUMNS), diagnostics

    unique = np.sort(finite.unique())
    if unique.size == 1:
        threshold = float(unique[0])
    else:
        gaps = np.diff(unique)
        split = int(np.argmax(gaps))
        threshold = float(unique[split] + unique[split + 1]) / 2.0

    assigned = pd.Series(pd.NA, index=observations.index, dtype="object")
    valid = easting.notna()
    assigned.loc[valid] = [
        "ST1" if float(value) >= threshold else "ST2"
        for value in easting.loc[valid]
    ]

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for label in ("ST1", "ST2"):
        mask = assigned == label
        if not mask.any():
            continue
        subset = observations.loc[mask]
        ref = float(pd.to_numeric(subset["se_m"], errors="coerce").median())
        rows.append({
            "label": label,
            "easting_ref": ref,
            "n_obs": int(mask.sum()),
            "median_se_m": _median(subset, "se_m"),
            "median_sn_m": _median(subset, "sn_m"),
            "median_sh_m": _median(subset, "sh_m"),
        })
        diagnostics.append({
            "method": "auto_easting_gap",
            "station_id": label,
            "message": (
                "station labels are automatic: split at the largest gap in the "
                "unique station eastings; no anchors configured"),
        })
    return (assigned,
            pd.DataFrame(rows, columns=list(STATION_COLUMNS)),
            pd.DataFrame(diagnostics, columns=list(_STATION_DIAGNOSTIC_COLUMNS)))


def _median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.median())


# ---------------------------------------------------------------------------
#  B. Cycles, diagnostics and segments
# ---------------------------------------------------------------------------
def processing_break(config: AnalysisConfig) -> pd.Timestamp | None:
    """Configured processing break as a timestamp, or ``None`` when absent."""
    value = config.cycles.processing_break
    if is_not_configured(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cycles.processing_break is not a valid timestamp: {value!r}") from exc


def segment_label(station_id: Any, time: Any,
                  break_time: pd.Timestamp | None) -> Any:
    """Segment identifier for one observation.

    ``{label}1`` without a configured break; otherwise ``{label}-S1`` strictly
    before the break and ``{label}-S2`` from the break time on. Missing station
    or time yields ``pd.NA`` (no segment is invented).
    """
    if station_id is None or pd.isna(station_id):
        return pd.NA
    label = str(station_id)
    if break_time is None:
        return f"{label}1"
    timestamp = pd.Timestamp(time)
    if pd.isna(timestamp):
        return pd.NA
    return f"{label}-S1" if timestamp < break_time else f"{label}-S2"


def build_cycles(observations: pd.DataFrame, *,
                 config: AnalysisConfig) -> CycleResult:
    """Assign stations and group observations into elapsed-time cycles.

    A new cycle starts when the gap to the previous observation at the same
    station exceeds ``config.cycles.gap_minutes.get(label,
    config.cycles.default_gap_minutes)`` minutes (strictly greater). Repeated
    observations share a cycle and are retained. The returned observation frame
    is the input copy with ``station_id``, ``cycle_id``, ``segment_id`` and the
    ALGORITHMS.md section 2 diagnostics appended, in the original row order.
    """
    assignment = assign_stations(observations, config=config)
    work = assignment.observations
    required = {"point_id", "time", "hz_deg", "v_deg", "d_m",
                "te_m", "tn_m", "tz_m"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"observations missing required columns: {sorted(missing)}")

    work = work.copy()
    work["time"] = pd.to_datetime(work["time"])
    work["_row"] = np.arange(len(work))
    _attach_observation_diagnostics(work)

    break_time = processing_break(config)
    work["segment_id"] = [
        segment_label(station, time, break_time)
        for station, time in zip(work["station_id"], work["time"], strict=True)
    ]

    work = _assign_cycle_ids(work, config)
    cycle_table = _cycle_table(work)
    segments = _segments_table(work)
    changes = detect_processing_changes(cycle_table, config=config)

    candidate_events = _empty(CANDIDATE_EVENT_COLUMNS)
    candidate_events.attrs["note"] = (
        "candidate frame events require configured detection thresholds; "
        "measured processing-change deltas are in the changes table"
    )
    output = work.sort_values("_row", kind="stable").drop(columns=["_row"])
    return CycleResult(
        observations=output,
        cycles=cycle_table,
        assignment=assignment,
        segments=segments,
        candidate_events=candidate_events,
        changes=changes,
        status=STATUS_OK,
        reason=None,
    )


def _attach_observation_diagnostics(work: pd.DataFrame) -> None:
    d_east = (pd.to_numeric(work["te_m"], errors="coerce")
              - pd.to_numeric(work["se_m"], errors="coerce"))
    d_north = (pd.to_numeric(work["tn_m"], errors="coerce")
               - pd.to_numeric(work["sn_m"], errors="coerce"))
    d_z = (pd.to_numeric(work["tz_m"], errors="coerce")
           - pd.to_numeric(work["sh_m"], errors="coerce"))
    with np.errstate(invalid="ignore"):
        hd_c = np.hypot(d_east, d_north)
        sd_c = np.hypot(hd_c, d_z)
        zen = np.degrees(np.arctan2(hd_c, d_z))
    hz = pd.to_numeric(work["hz_deg"], errors="coerce")
    v = pd.to_numeric(work["v_deg"], errors="coerce")
    slope = pd.to_numeric(work["d_m"], errors="coerce")

    azimuth = np.asarray(azimuth_deg(d_east.to_numpy(dtype=float),
                                     d_north.to_numpy(dtype=float)))
    work["dE"] = d_east
    work["dN"] = d_north
    work["dZ"] = d_z
    work["hd_c"] = hd_c
    work["sd_c"] = sd_c
    work["az_deg"] = azimuth
    work["zen_deg"] = zen
    work["dHz_arcsec"] = angular_difference_deg(
        hz.to_numpy(dtype=float), azimuth) * 3600.0
    work["dV_arcsec"] = (v - zen) * 3600.0
    work["dSD_mm"] = (slope - sd_c) * 1000.0
    if "horz_dist_m" in work.columns:
        work["dHDrep_mm"] = (pd.to_numeric(work["horz_dist_m"], errors="coerce")
                             - hd_c) * 1000.0
    else:
        work["dHDrep_mm"] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = (sd_c.to_numpy(dtype=float) / slope.to_numpy(dtype=float) - 1.0) * 1e6
    work["scale_ppm"] = np.where(np.isfinite(scale), scale, np.nan)


def _assign_cycle_ids(work: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    work = work.copy()
    work["cycle_id"] = pd.NA
    gap_map = {str(key): float(value)
               for key, value in (config.cycles.gap_minutes or {}).items()}
    default_gap = float(config.cycles.default_gap_minutes)

    valid = work["time"].notna() & work["station_id"].notna()
    if not valid.any():
        return work
    ordered = (work.loc[valid]
               .sort_values(["station_id", "time"], kind="stable")
               .copy())
    station = ordered["station_id"].astype(str)
    gap_minutes = (ordered.groupby("station_id", sort=False)["time"]
                   .diff().dt.total_seconds() / 60.0)
    thresholds = station.map(gap_map).astype(float).fillna(default_gap)
    new_cycle = gap_minutes.isna() | (gap_minutes.to_numpy() > thresholds.to_numpy())
    counter = new_cycle.astype(int).groupby(station).cumsum() - 1
    ordered["cycle_id"] = [
        f"{label}{number:03d}" for label, number in zip(station, counter, strict=True)
    ]
    work.loc[ordered.index, "cycle_id"] = ordered["cycle_id"]
    return work


def _cycle_table(work: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = work["cycle_id"].notna()
    if valid.any():
        subset = work.loc[valid]
        iterator = subset.groupby(["station_id", "cycle_id"], sort=False, dropna=False)
        for (station, cycle), group in iterator:
            t0 = group["time"].min()
            t1 = group["time"].max()
            d_hz_arcsec = pd.to_numeric(group["dHz_arcsec"], errors="coerce")
            d_hz_deg = d_hz_arcsec.to_numpy(dtype=float) / 3600.0
            heights = pd.to_numeric(group["sh_m"], errors="coerce").dropna()
            rows.append({
                "cycle_id": cycle,
                "station_id": station,
                "t0": t0,
                "t1": t1,
                "n": int(len(group)),
                "npid": int(group["point_id"].nunique()),
                "se_m": _median(group, "se_m"),
                "sn_m": _median(group, "sn_m"),
                "sh_m": _median(group, "sh_m"),
                "n_distinct_heights": int(heights.nunique()),
                "orientation_deg": circular_mean_deg(d_hz_deg),
                "orientation_sd_arcsec": circular_std_deg(d_hz_deg) * 3600.0,
                "scale_ppm": _median(group, "scale_ppm"),
                "dv_arcsec": _median(group, "dV_arcsec"),
                "dur_min": float((t1 - t0).total_seconds() / 60.0),
            })
    frame = pd.DataFrame(rows, columns=list(CYCLE_COLUMNS))
    if not frame.empty:
        frame = (frame.sort_values(["station_id", "t0"], kind="stable")
                 .reset_index(drop=True))
    return frame


def _segments_table(work: pd.DataFrame) -> pd.DataFrame:
    valid = work["segment_id"].notna()
    if not valid.any():
        return _empty(SEGMENT_COLUMNS)
    subset = work.loc[valid]
    segments = (subset.groupby(["point_id", "segment_id", "station_id"],
                               sort=False, dropna=False)
                .agg(t_start=("time", "min"),
                     t_end=("time", "max"),
                     n_obs=("time", "size"))
                .reset_index())
    segments = (segments[list(SEGMENT_COLUMNS)]
                .sort_values(["station_id", "segment_id", "point_id"], kind="stable")
                .reset_index(drop=True))
    return segments


# ---------------------------------------------------------------------------
#  C. Processing-change diagnostics (never a classification)
# ---------------------------------------------------------------------------
def detect_processing_changes(cycles: pd.DataFrame | CycleResult, *,
                              config: AnalysisConfig) -> ChangeResult:
    """Measure every per-station cycle transition; classify nothing.

    The station-coordinate/orientation jump thresholds are not configured, so
    the result status is ``not_configured``. The ``changes`` table still lists
    every consecutive transition at a station with the measured
    ``d_east_mm``, ``d_north_mm``, ``d_height_mm`` and
    ``d_orientation_arcsec`` deltas.
    """
    table = getattr(cycles, "cycles", cycles)
    view = _station_cycle_view(table)
    if view.empty:
        changes = _empty(CHANGE_COLUMNS)
    else:
        view = (view.sort_values(["station_id", "t0"], kind="stable")
                .reset_index(drop=True))
        grouped = view.groupby("station_id", sort=False)
        previous_t0 = grouped["t0"].shift(1)
        previous_se = grouped["se_m"].shift(1)
        previous_sn = grouped["sn_m"].shift(1)
        previous_sh = grouped["sh_m"].shift(1)
        previous_orientation = grouped["orientation_deg"].shift(1)
        transition = previous_t0.notna()
        orientation_delta = pd.Series(
            angular_difference_deg(view["orientation_deg"].to_numpy(dtype=float),
                                   previous_orientation.to_numpy(dtype=float)) * 3600.0,
            index=view.index)
        changes = pd.DataFrame({
            "t": view.loc[transition, "t0"].to_numpy(),
            "station_id": view.loc[transition, "station_id"].to_numpy(),
            "d_east_mm": ((view["se_m"] - previous_se) * 1000.0)
                         .loc[transition].to_numpy(),
            "d_north_mm": ((view["sn_m"] - previous_sn) * 1000.0)
                          .loc[transition].to_numpy(),
            "d_height_mm": ((view["sh_m"] - previous_sh) * 1000.0)
                           .loc[transition].to_numpy(),
            "d_orientation_arcsec": orientation_delta.loc[transition].to_numpy(),
        }, columns=list(CHANGE_COLUMNS))
    return ChangeResult(changes=changes, status=NOT_CONFIGURED,
                        reason=_NOT_CONFIGURED_REASON)


def _station_cycle_view(cycles: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(cycles, pd.DataFrame):
        raise TypeError("cycles must be a DataFrame or a CycleResult")
    required = {"station_id", "t0", "se_m", "sn_m", "sh_m"}
    if required <= set(cycles.columns):
        view = cycles.copy()
        if "orientation_deg" not in view.columns:
            view["orientation_deg"] = np.nan
        view["t0"] = pd.to_datetime(view["t0"])
        return view[["station_id", "t0", "se_m", "sn_m", "sh_m", "orientation_deg"]]

    observation_level = {"station_id", "cycle_id", "time", "se_m", "sn_m", "sh_m"}
    if observation_level <= set(cycles.columns):
        work = cycles.copy()
        work["time"] = pd.to_datetime(work["time"])
        view = (work.groupby(["station_id", "cycle_id"], sort=False, dropna=False)
                .agg(t0=("time", "min"),
                     se_m=("se_m", "median"),
                     sn_m=("sn_m", "median"),
                     sh_m=("sh_m", "median"))
                .reset_index())
        if "dHz_arcsec" in work.columns:
            orientation = (work.groupby(["station_id", "cycle_id"], sort=False,
                                        dropna=False)["dHz_arcsec"]
                           .agg(lambda values: circular_mean_deg(
                               pd.to_numeric(values, errors="coerce").to_numpy()
                               / 3600.0))
                           .reset_index(name="orientation_deg"))
            view = view.merge(orientation, on=["station_id", "cycle_id"], how="left")
        else:
            view["orientation_deg"] = np.nan
        return view

    raise ValueError(
        "cannot derive station transitions from the supplied cycles object: "
        "expected the cycle table (station_id, t0, se_m, sn_m, sh_m) or an "
        "observation-level frame with station_id/cycle_id/time")
