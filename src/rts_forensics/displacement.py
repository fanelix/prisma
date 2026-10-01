"""
displacement.py — prism-cycle aggregation, segment baselines and displacements.

:func:`aggregate_prism_cycles` collapses repeats into one row per
``(point_id, station_id, cycle_id)``: circular median of Hz, medians of the
other values, within-cycle repeat SD and the tuple of contributing observation
IDs. Repeats are retained, never treated as duplicates.

:func:`compute_displacements` builds one baseline per prism per processing
segment (the first ``config.baseline.window_hours`` of the segment, falling
back to the first ``config.baseline.min_observations`` rows when the window is
too sparse) and emits, per prism-cycle:

* raw polar products in millimetres: ``d_rad_mm`` (LOS range change, positive
  away from the instrument), ``d_tan_mm``, ``d_vert_mm``;
* delivered coordinate changes ``d_east_mm``/``d_north_mm``/``d_up_mm`` from
  the same segment baseline. Delivered coordinates are never bridged across a
  processing segment;
* provenance: the prism-cycle observation IDs and the exact baseline
  observation IDs, exposed through :func:`provenance.build_source_links` with
  the roles ``observation`` and ``baseline``.

Segment labels come from :mod:`rts_forensics.cycles` (``{label}1`` without a
configured processing break, ``{label}-S1``/``{label}-S2`` around it). All
thresholds come from the effective configuration; an unconfigured baseline
window yields an explicit ``not_configured`` status, never a guessed value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig, is_not_configured
from .cycles import build_cycles, processing_break, segment_label
from .geometry import angular_difference_deg, circular_mean_deg, circular_std_deg
from .models import (
    BASELINE_COLUMNS,
    DISPLACEMENT_COLUMNS,
    NOT_CONFIGURED,
    PRISM_CYCLE_COLUMNS,
    SOURCE_LINK_COLUMNS,
    STATUS_INSUFFICIENT,
    STATUS_OK,
    DisplacementResult,
)
from .provenance import build_source_links

_NUMERIC_OBSERVATION_COLUMNS = (
    "hz_deg", "v_deg", "d_m", "te_m", "tn_m", "tz_m", "se_m", "sn_m", "sh_m",
)

_PRISM_CYCLE_KEYS = ("point_id", "station_id", "cycle_id")

_BASELINE_MATH_COLUMNS = (
    "point_id", "segment_id", "hz0_deg", "v0_deg", "d0_m",
    "te0_m", "tn0_m", "tz0_m", "start", "end", "n_obs", "complete",
    "status", "obs_ids",
)

_BASELINE_MATH_RENAMES = {
    "start": "baseline_start",
    "end": "baseline_end",
    "n_obs": "baseline_n_obs",
    "complete": "baseline_complete",
    "status": "baseline_status",
    "obs_ids": "baseline_obs_ids",
}


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _as_tuple(value: Any) -> tuple:
    if isinstance(value, (tuple, list, np.ndarray, set)):
        return tuple(value)
    if value is None:
        return ()
    try:
        if pd.isna(value):
            return ()
    except (TypeError, ValueError):
        pass
    return (value,)


def _flatten_obs_ids(values: pd.Series) -> tuple:
    collected: list[Any] = []
    for value in values:
        collected.extend(_as_tuple(value))
    return tuple(collected)


def _circular_median_deg(values: Any) -> float:
    """Circular median ``(ref + median(wrapped differences)) % 360``."""
    numbers = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    if numbers.size == 0:
        return float("nan")
    reference = circular_mean_deg(numbers)
    if not np.isfinite(reference):
        reference = float(numbers[0])
    wrapped = (numbers - reference + 180.0) % 360.0 - 180.0
    return float((reference + float(np.median(wrapped))) % 360.0)


def _circular_median_series(values: Any) -> float:
    return _circular_median_deg(values)


def _circular_std_series(values: Any) -> float:
    return circular_std_deg(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float))


# ---------------------------------------------------------------------------
#  D. Prism-cycle aggregation
# ---------------------------------------------------------------------------
def aggregate_prism_cycles(observations: pd.DataFrame, *,
                           config: AnalysisConfig) -> pd.DataFrame:
    """Aggregate repeated observations into prism-cycles.

    The input may be canonical observations carrying ``station_id`` and
    ``cycle_id`` (as produced by :func:`rts_forensics.cycles.build_cycles`);
    otherwise the cycles are derived first. Hz is aggregated with a circular
    median and the within-cycle repeat SD is reported in ``sd_hz_deg``,
    ``sd_v_deg`` and ``sd_d_m`` (NaN for a single repeat). The returned frame
    has exactly :data:`rts_forensics.models.PRISM_CYCLE_COLUMNS`; the input is
    never mutated.
    """
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame")
    if not {"station_id", "cycle_id"} <= set(observations.columns):
        observations = build_cycles(observations, config=config).observations
    required = {"point_id", "station_id", "cycle_id", "time",
                "hz_deg", "v_deg", "d_m"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"observations missing required columns: {sorted(missing)}")

    work = observations.copy()
    work["time"] = pd.to_datetime(work["time"])
    work = work.dropna(subset=["point_id", "station_id", "cycle_id", "time"])
    for column in _NUMERIC_OBSERVATION_COLUMNS:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    if work.empty:
        return _empty(PRISM_CYCLE_COLUMNS)

    grouped = work.groupby(list(_PRISM_CYCLE_KEYS), sort=False, dropna=False)
    spec: dict[str, tuple[str, Any]] = {
        "cycle_start": ("time", "min"),
        "cycle_end": ("time", "max"),
        "n_obs": ("time", "size"),
        "hz_deg": ("hz_deg", _circular_median_series),
        "v_deg": ("v_deg", "median"),
        "d_m": ("d_m", "median"),
        "sd_hz_deg": ("hz_deg", _circular_std_series),
        "sd_v_deg": ("v_deg", "std"),
        "sd_d_m": ("d_m", "std"),
    }
    for column in ("te_m", "tn_m", "tz_m", "se_m", "sn_m", "sh_m"):
        if column in work.columns:
            spec[column] = (column, "median")
    aggregated = grouped.agg(**spec)
    if "obs_id" in work.columns:
        aggregated["obs_ids"] = grouped["obs_id"].agg(lambda values: tuple(values))
    else:
        aggregated["obs_ids"] = pd.Series(
            [() for _ in range(len(aggregated))], index=aggregated.index, dtype="object")

    out = aggregated.reset_index()
    for column in PRISM_CYCLE_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    out = out.reindex(columns=list(PRISM_CYCLE_COLUMNS))
    out = (out.sort_values(["point_id", "station_id", "cycle_start"], kind="stable")
           .reset_index(drop=True))
    out["obs_ids"] = [_as_tuple(value) for value in out["obs_ids"]]
    return out


# ---------------------------------------------------------------------------
#  E. Segment baselines and displacements
# ---------------------------------------------------------------------------
def compute_displacements(prism_cycles: pd.DataFrame, *, segments=None,
                          config: AnalysisConfig) -> DisplacementResult:
    """Compute raw polar and segmented delivered displacements.

    ``segments`` is ``None`` to derive the processing segments from
    ``config.cycles.processing_break`` (the rule used by
    :func:`rts_forensics.cycles.build_cycles`), or a frame with at least
    ``point_id``, ``segment_id`` and ``t_start`` (``station_id`` and ``t_end``
    optional). A baseline is the first ``config.baseline.window_hours`` of each
    segment; when fewer than ``config.baseline.min_observations`` prism-cycles
    fall in that window the first ``min_observations`` rows are used with
    ``complete=False``, and fewer than two rows yields
    ``insufficient_data``. Delivered coordinates always use their own segment
    baseline and are never bridged.
    """
    if prism_cycles is None or (isinstance(prism_cycles, pd.DataFrame)
                                and prism_cycles.empty):
        return DisplacementResult(
            prism_cycles=_empty(PRISM_CYCLE_COLUMNS),
            displacements=_empty(DISPLACEMENT_COLUMNS),
            baselines=_empty(BASELINE_COLUMNS),
            source_links=_empty(SOURCE_LINK_COLUMNS),
            status=STATUS_INSUFFICIENT,
            reason="no prism cycles supplied",
        )

    prepared = _prepare_prism_cycles(prism_cycles)
    segment_frame = _resolve_segments(prepared, segments, config)
    attached = _attach_segments(prepared, segment_frame)
    if attached.empty:
        return DisplacementResult(
            prism_cycles=_empty(PRISM_CYCLE_COLUMNS),
            displacements=_empty(DISPLACEMENT_COLUMNS),
            baselines=_empty(BASELINE_COLUMNS),
            source_links=_empty(SOURCE_LINK_COLUMNS),
            status=STATUS_INSUFFICIENT,
            reason="no prism cycle could be assigned to a processing segment",
        )

    baselines, baseline_math = _build_baselines(attached, config)
    evaluated = _evaluate_displacements(attached, baseline_math)
    displacements = evaluated.reindex(columns=list(DISPLACEMENT_COLUMNS))
    enriched = evaluated.drop(columns=["_row"]).reset_index(drop=True)
    source_links = _build_source_links(displacements)

    if not len(baseline_math):
        status, reason = STATUS_INSUFFICIENT, "no baseline could be formed"
    elif (baseline_math["status"] == STATUS_OK).any():
        status, reason = STATUS_OK, None
    elif (baseline_math["status"] == NOT_CONFIGURED).all():
        status = NOT_CONFIGURED
        reason = "baseline window/completeness is not configured"
    else:
        status = STATUS_INSUFFICIENT
        reason = "no segment had at least two observations for a baseline"
    return DisplacementResult(
        prism_cycles=enriched,
        displacements=displacements,
        baselines=baselines,
        source_links=source_links,
        status=status,
        reason=reason,
    )


def _prepare_prism_cycles(prism_cycles: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prism_cycles, pd.DataFrame):
        raise TypeError("prism_cycles must be a pandas DataFrame")
    required = {"point_id", "station_id", "cycle_id", "cycle_start",
                "hz_deg", "v_deg", "d_m", "te_m", "tn_m", "tz_m"}
    missing = required - set(prism_cycles.columns)
    if missing:
        raise ValueError(f"prism_cycles missing required columns: {sorted(missing)}")

    prepared = prism_cycles.copy()
    prepared["cycle_start"] = pd.to_datetime(prepared["cycle_start"])
    if "cycle_end" in prepared.columns:
        prepared["cycle_end"] = pd.to_datetime(prepared["cycle_end"])
    else:
        prepared["cycle_end"] = prepared["cycle_start"]
    for column in ("hz_deg", "v_deg", "d_m", "te_m", "tn_m", "tz_m"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if "obs_ids" not in prepared.columns:
        prepared["obs_ids"] = [() for _ in range(len(prepared))]
    prepared["obs_ids"] = [_as_tuple(value) for value in prepared["obs_ids"]]
    prepared["_row"] = np.arange(len(prepared))
    return prepared


def _resolve_segments(prepared: pd.DataFrame, segments,
                      config: AnalysisConfig) -> pd.DataFrame:
    if segments is None:
        break_time = processing_break(config)
        labels = [segment_label(station, time, break_time)
                  for station, time in zip(prepared["station_id"],
                                           prepared["cycle_start"], strict=True)]
        derived = prepared[["point_id", "station_id", "cycle_start"]].copy()
        derived["segment_id"] = labels
        derived = derived.dropna(subset=["segment_id"])
        return (derived.groupby(["point_id", "station_id", "segment_id"],
                                sort=False, dropna=False)
                .agg(t_start=("cycle_start", "min"), t_end=("cycle_start", "max"))
                .reset_index()
                .loc[:, ["point_id", "station_id", "segment_id", "t_start", "t_end"]])

    if not isinstance(segments, pd.DataFrame):
        raise TypeError("segments must be None or a pandas DataFrame")
    missing = {"point_id", "segment_id", "t_start"} - set(segments.columns)
    if missing:
        raise ValueError(f"segments missing required columns: {sorted(missing)}")
    resolved = segments.copy()
    resolved["t_start"] = pd.to_datetime(resolved["t_start"])
    if "t_end" in resolved.columns:
        resolved["t_end"] = pd.to_datetime(resolved["t_end"])
    keep = ["point_id", "segment_id", "t_start"]
    if "station_id" in resolved.columns:
        keep.append("station_id")
    if "t_end" in resolved.columns:
        keep.append("t_end")
    return resolved.loc[:, keep]


def _attach_segments(prepared: pd.DataFrame, segment_frame: pd.DataFrame) -> pd.DataFrame:
    join_keys = ["point_id"]
    if "station_id" in segment_frame.columns:
        join_keys.append("station_id")
    columns = list(join_keys) + ["segment_id", "t_start"]
    if "t_end" in segment_frame.columns:
        columns.append("t_end")
    merged = prepared.merge(segment_frame.loc[:, columns], on=join_keys, how="inner")
    merged = merged.loc[merged["cycle_start"] >= merged["t_start"]]
    if "t_end" in merged.columns:
        merged = merged.loc[merged["cycle_start"] <= merged["t_end"]]
    merged = (merged.sort_values(["_row", "t_start"], kind="stable")
              .groupby("_row", sort=False, as_index=False)
              .tail(1))
    return merged.sort_values("_row", kind="stable").reset_index(drop=True)


def _build_baselines(attached: pd.DataFrame,
                     config: AnalysisConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_hours = config.baseline.window_hours
    min_observations = config.baseline.min_observations
    configured = (not is_not_configured(window_hours)
                  and not is_not_configured(min_observations))
    window = None
    minimum = 0
    if configured:
        window = pd.Timedelta(hours=float(window_hours))
        minimum = int(min_observations)
        if minimum < 1:
            raise ValueError("baseline.min_observations must be at least 1")

    export_rows: list[dict[str, Any]] = []
    math_rows: list[dict[str, Any]] = []
    iterator = attached.groupby(["point_id", "segment_id"], sort=False, dropna=False)
    for (point_id, segment_id), group in iterator:
        group = group.sort_values("cycle_start", kind="stable")
        if not configured:
            status, complete = NOT_CONFIGURED, False
            baseline_rows = group.iloc[0:0]
        else:
            window_end = group["t_start"].iloc[0] + window
            in_window = group.loc[group["cycle_start"] <= window_end]
            if len(in_window) >= minimum:
                baseline_rows, complete = in_window, True
            else:
                baseline_rows, complete = group.head(minimum), False
            status = STATUS_OK if len(baseline_rows) >= 2 else STATUS_INSUFFICIENT

        n_baseline = int(len(baseline_rows))
        if status == STATUS_OK:
            hz0 = _circular_median_deg(baseline_rows["hz_deg"])
            v0 = float(baseline_rows["v_deg"].median())
            d0 = float(baseline_rows["d_m"].median())
            te0 = float(baseline_rows["te_m"].median())
            tn0 = float(baseline_rows["tn_m"].median())
            tz0 = float(baseline_rows["tz_m"].median())
            obs_ids = _flatten_obs_ids(baseline_rows["obs_ids"])
            start = baseline_rows["cycle_start"].min()
            end = baseline_rows["cycle_end"].max()
        else:
            hz0 = v0 = d0 = te0 = tn0 = tz0 = float("nan")
            obs_ids = ()
            start = baseline_rows["cycle_start"].min() if n_baseline else pd.NaT
            end = baseline_rows["cycle_end"].max() if n_baseline else pd.NaT

        export_rows.append({
            "point_id": point_id,
            "segment_id": segment_id,
            "start": start,
            "end": end,
            "n_obs": n_baseline,
            "complete": bool(complete),
            "te_m": te0,
            "tn_m": tn0,
            "tz_m": tz0,
            "obs_ids": obs_ids,
            "status": status,
        })
        math_rows.append({
            "point_id": point_id,
            "segment_id": segment_id,
            "hz0_deg": hz0,
            "v0_deg": v0,
            "d0_m": d0,
            "te0_m": te0,
            "tn0_m": tn0,
            "tz0_m": tz0,
            "start": start,
            "end": end,
            "n_obs": n_baseline,
            "complete": bool(complete),
            "status": status,
            "obs_ids": obs_ids,
        })

    baselines = pd.DataFrame(export_rows, columns=list(BASELINE_COLUMNS))
    baseline_math = pd.DataFrame(math_rows, columns=list(_BASELINE_MATH_COLUMNS))
    if not baselines.empty:
        baselines = (baselines.sort_values(["point_id", "segment_id"], kind="stable")
                     .reset_index(drop=True))
    return baselines, baseline_math


def _evaluate_displacements(attached: pd.DataFrame,
                            baseline_math: pd.DataFrame) -> pd.DataFrame:
    math = baseline_math.rename(columns=_BASELINE_MATH_RENAMES)
    work = attached.merge(math, on=["point_id", "segment_id"], how="left")

    hz = work["hz_deg"].to_numpy(dtype=float)
    v = np.radians(work["v_deg"].to_numpy(dtype=float))
    d = work["d_m"].to_numpy(dtype=float)
    hz0 = work["hz0_deg"].to_numpy(dtype=float)
    v0 = np.radians(work["v0_deg"].to_numpy(dtype=float))
    d0 = work["d0_m"].to_numpy(dtype=float)

    d_hz_arcsec = angular_difference_deg(hz, hz0) * 3600.0
    d_rad_mm = (d - d0) * 1000.0
    d_tan_mm = d0 * np.sin(v0) * np.radians(d_hz_arcsec / 3600.0) * 1000.0
    rad_raw_mm = (d * np.sin(v) - d0 * np.sin(v0)) * 1000.0
    d_vert_mm = (d * np.cos(v) - d0 * np.cos(v0)) * 1000.0
    d_east_mm = (work["te_m"] - work["te0_m"]) * 1000.0
    d_north_mm = (work["tn_m"] - work["tn0_m"]) * 1000.0
    d_up_mm = (work["tz_m"] - work["tz0_m"]) * 1000.0

    baseline_status = work["baseline_status"].fillna(STATUS_INSUFFICIENT)
    evaluated = work.copy()
    evaluated["ts"] = evaluated["cycle_start"]
    evaluated["d_hz_arcsec"] = d_hz_arcsec
    evaluated["d_rad_mm"] = d_rad_mm
    evaluated["d_tan_mm"] = d_tan_mm
    evaluated["rad_raw_mm"] = rad_raw_mm
    evaluated["d_vert_mm"] = d_vert_mm
    evaluated["d_east_mm"] = d_east_mm
    evaluated["d_north_mm"] = d_north_mm
    evaluated["d_up_mm"] = d_up_mm
    evaluated["baseline_status"] = baseline_status
    evaluated["obs_ids"] = [_as_tuple(value) for value in evaluated["obs_ids"]]
    evaluated["baseline_obs_ids"] = [
        _as_tuple(value) for value in evaluated["baseline_obs_ids"]]
    return evaluated.sort_values("_row", kind="stable").reset_index(drop=True)


def _build_source_links(displacements: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in displacements.itertuples(index=False):
        row_key = f"{record.point_id}#{record.segment_id}@{record.cycle_id}"
        rows.append({"row_key": row_key, "role": "observation",
                     "obs_ids": _as_tuple(record.obs_ids)})
        rows.append({"row_key": row_key, "role": "baseline",
                     "obs_ids": _as_tuple(record.baseline_obs_ids)})
    return build_source_links("displacements", rows)
