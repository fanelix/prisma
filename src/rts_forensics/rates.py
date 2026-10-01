"""
rates.py — backwards 24 h blocks and Theil-Sen displacement rates.

Blocks are counted backwards from the final record so calendar-day boundaries
cannot bias diurnal signals (PROJECT_HANDOFF.md section 3.7 / section 4.4).
The oldest block is normally partial; partial blocks are never silently
included in a rate. Theil-Sen uses elapsed days computed from the block
medians, never a row index. The 72 h window uses all cycles and its
confidence interval ignores autocorrelation.

All numerical settings come from the effective configuration: the block
length (``rates.block_hours``), the windows (``rates.windows_days`` /
``rates.windows_hours``) and the confidence level (``rates.confidence``).
The two scaffold rules "fewer than 6 points" and "span shorter than one day"
are constants here because the configuration contract does not define them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from .config import AnalysisConfig
from .models import (
    RATE_COLUMNS,
    STATUS_INSUFFICIENT,
    STATUS_OK,
)

#: Supplied scaffold rules: a rate needs at least this many points and one day
#: of elapsed span, otherwise the result is ``insufficient_data``.
MIN_RATE_POINTS = 6
MIN_RATE_SPAN_DAYS = 1.0

BLOCK_COLUMNS = ("block_id", "point_id", "metric", "start", "end",
                 "n_obs", "obs_ids", "partial")

#: Displacement column -> metric name used by :func:`theil_sen_rate`.
DISPLACEMENT_METRIC_COLUMNS = {
    "d_rad_mm": "d_rad",
    "d_vert_mm": "d_ver",
    "d_ver_mm": "d_ver",
    "d_tan_mm": "d_tan",
    "d_east_mm": "d_east",
    "d_north_mm": "d_north",
    "d_up_mm": "d_up",
}

#: Canonical rate metrics, in report order.
RATE_METRICS = ("d_rad", "d_ver", "d_tan", "d_east", "d_north", "d_up")


@dataclass
class RateEstimate:
    """One rate estimate; fields mirror :data:`rts_forensics.models.RATE_COLUMNS`."""

    point_id: str = ""
    segment_id: str = ""
    metric: str = ""
    window: str = ""
    n_blocks: int = 0
    n_obs: int = 0
    t_start: pd.Timestamp | None = None
    t_end: pd.Timestamp | None = None
    rate_mm_day: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    method: str = "theil_sen"
    partial_block: bool = False
    status: str = STATUS_INSUFFICIENT
    reason: str | None = None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
#  D. Backwards 24 h blocks
# ---------------------------------------------------------------------------
def backward_blocks(series: pd.DataFrame, *, duration, anchor=None,
                    config: AnalysisConfig) -> pd.DataFrame:
    """Blocks of ``duration`` counted backwards from ``anchor``.

    ``block_id`` 0 is the block ending at the anchor and ids increase into the
    past. ``start``/``end`` are the actual bounds; the oldest block is partial
    when it does not span the full duration. ``obs_ids`` is populated when the
    series carries an ``obs_id`` column. Empty full blocks are kept so data
    gaps stay visible; partial blocks are flagged, never hidden.

    ``anchor`` defaults to the final record of each ``(point_id, metric)``
    group (identical to ``series.ts.max()`` for a single series).
    """
    if "ts" not in series.columns or "value" not in series.columns:
        raise ValueError("series must provide ts and value columns")
    if duration is None:
        duration = config.rates.block_hours
    if not isinstance(duration, pd.Timedelta):
        duration = pd.Timedelta(hours=float(duration))
    if duration <= pd.Timedelta(0):
        raise ValueError("duration must be positive")

    work = series.copy()
    work["ts"] = pd.to_datetime(work["ts"])
    work = work.dropna(subset=["ts"])
    if work.empty:
        return _empty_blocks()

    has_obs = "obs_id" in work.columns
    keys = [c for c in ("point_id", "metric") if c in work.columns]
    if keys:
        iterator = work.groupby(keys, sort=False, dropna=False)
    else:
        iterator = [((), work)]

    rows: list[dict[str, Any]] = []
    for key, group in iterator:
        if keys:
            key_values = list(key) if isinstance(key, tuple) else [key]
            point_id = key_values[keys.index("point_id")] if "point_id" in keys else ""
            metric = key_values[keys.index("metric")] if "metric" in keys else ""
        else:
            point_id, metric = "", ""
        g = group.sort_values("ts")
        t_min = g["ts"].min()
        t_max = g["ts"].max()
        base = pd.Timestamp(anchor) if anchor is not None else t_max
        if t_min > base:
            continue
        ids = _block_index(g["ts"], base, duration)
        k_max = int(ids.max())
        for k in range(k_max + 1):
            end = base - k * duration
            nominal_start = end - duration
            mask = ids == k
            if k == k_max:
                start = t_min
                partial = bool(t_min > nominal_start)
            else:
                start = nominal_start
                partial = False
            rows.append({
                "block_id": k,
                "point_id": point_id,
                "metric": metric,
                "start": start,
                "end": end,
                "n_obs": int(mask.sum()),
                "obs_ids": tuple(g.loc[mask, "obs_id"]) if has_obs else (),
                "partial": partial,
            })
    if not rows:
        return _empty_blocks()
    return pd.DataFrame(rows, columns=list(BLOCK_COLUMNS))


def _block_index(ts: pd.Series, anchor: pd.Timestamp,
                 duration: pd.Timedelta) -> np.ndarray:
    """Block index 0 at the anchor, increasing backwards (reference formula)."""
    elapsed = (anchor - ts) / duration
    values = np.asarray(elapsed, dtype=float) + 1e-9
    return np.maximum(np.floor(values), 0.0).astype(int)


def _empty_blocks() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in BLOCK_COLUMNS})


def _block_medians(series: pd.DataFrame, anchor: pd.Timestamp,
                   duration: pd.Timedelta) -> pd.Series:
    """Median ``value`` per block index for one already-filtered series."""
    ids = _block_index(series["ts"], anchor, duration)
    return (series.assign(_block=ids)
            .groupby("_block", sort=True)["value"].median())


# ---------------------------------------------------------------------------
#  E. Theil-Sen rate
# ---------------------------------------------------------------------------
def theil_sen_rate(series: pd.DataFrame, *, metric: str, window: str,
                   config: AnalysisConfig) -> RateEstimate:
    """Theil-Sen rate for one prism/metric over one window.

    ``30d`` uses all complete 24 h block medians, ``7d`` the newest complete
    blocks covering seven days and ``72h`` every cycle in the last 72 hours
    (method ``theil_sen``; its CI ignores autocorrelation). Fewer than
    :data:`MIN_RATE_POINTS` points or less than a day of span yields
    ``insufficient_data`` with missing values, never a fabricated number.
    """
    kind, value = _parse_window(window)
    work = _prepare_rate_series(series, metric)
    point_id, segment_id = _identity(work)
    if work.empty:
        return RateEstimate(
            point_id=point_id, segment_id=segment_id, metric=metric,
            window=window, status=STATUS_INSUFFICIENT,
            reason="no observations for this metric")

    if kind == "hours":
        return _rate_from_cycles(work, metric=metric, window=window,
                                 hours=value, point_id=point_id,
                                 segment_id=segment_id, config=config)
    return _rate_from_blocks(work, metric=metric, window=window, days=value,
                             point_id=point_id, segment_id=segment_id,
                             config=config)


def _rate_from_blocks(series: pd.DataFrame, *, metric: str, window: str,
                      days: float, point_id: str, segment_id: str,
                      config: AnalysisConfig) -> RateEstimate:
    duration = pd.Timedelta(hours=float(config.rates.block_hours))
    anchor = series["ts"].max()
    blocks = backward_blocks(series, duration=duration, anchor=None,
                             config=config)
    if blocks.empty:
        return RateEstimate(point_id=point_id, segment_id=segment_id,
                            metric=metric, window=window,
                            status=STATUS_INSUFFICIENT,
                            reason="no 24 h blocks could be formed")
    full = blocks.loc[~blocks["partial"]]
    all_data = str(window).strip().lower() == "30d"
    if all_data:
        selected = full
        partial_block = bool(blocks["partial"].any())
    else:
        max_blocks = max(1, int(round(days * 24.0 / float(config.rates.block_hours))))
        selected = full.loc[full["block_id"] < max_blocks]
        partial_block = bool(blocks["partial"].any()) and len(full) < max_blocks

    medians = _block_medians(series, anchor, duration)
    selected = selected.assign(median=selected["block_id"].map(medians))
    used = selected.dropna(subset=["median"])

    n_blocks = int(len(used))
    if n_blocks == 0:
        return RateEstimate(
            point_id=point_id, segment_id=segment_id, metric=metric,
            window=window, n_blocks=0, n_obs=0,
            t_start=None, t_end=None, partial_block=partial_block,
            status=STATUS_INSUFFICIENT,
            reason="no complete 24 h block has any observation")
    t_start = used["start"].min()
    t_end = used["end"].max()
    n_obs = int(used["n_obs"].sum())
    if n_blocks < MIN_RATE_POINTS:
        return RateEstimate(
            point_id=point_id, segment_id=segment_id, metric=metric,
            window=window, n_blocks=n_blocks, n_obs=n_obs,
            t_start=t_start, t_end=t_end, partial_block=partial_block,
            status=STATUS_INSUFFICIENT,
            reason=f"{n_blocks} usable block median(s); "
                   f"at least {MIN_RATE_POINTS} required")

    x = (used["end"] - used["end"].min()).dt.total_seconds().to_numpy() / 86400.0
    span = float(x.max() - x.min()) if len(x) else 0.0
    if span < MIN_RATE_SPAN_DAYS:
        return RateEstimate(
            point_id=point_id, segment_id=segment_id, metric=metric,
            window=window, n_blocks=n_blocks, n_obs=n_obs,
            t_start=t_start, t_end=t_end, partial_block=partial_block,
            status=STATUS_INSUFFICIENT,
            reason=f"block span {span:.2f} day(s); at least 1 day required")
    values = used["median"].to_numpy(dtype=float)
    slope, _intercept, low, high = theilslopes(
        values, x, alpha=float(config.rates.confidence))
    return RateEstimate(
        point_id=point_id, segment_id=segment_id, metric=metric,
        window=window, n_blocks=n_blocks, n_obs=n_obs,
        t_start=t_start, t_end=t_end,
        rate_mm_day=float(slope), ci_low=float(low), ci_high=float(high),
        method="theil_sen", partial_block=partial_block,
        status=STATUS_OK, reason=None)


def _rate_from_cycles(series: pd.DataFrame, *, metric: str, window: str,
                      hours: float, point_id: str, segment_id: str,
                      config: AnalysisConfig) -> RateEstimate:
    anchor = series["ts"].max()
    subset = series.loc[series["ts"] >= anchor - pd.Timedelta(hours=hours)]
    subset = subset.sort_values("ts")
    n_points = int(len(subset))
    note = (f"{window} window uses all cycles; the Theil-Sen CI ignores "
            "autocorrelation")
    if n_points == 0:
        return RateEstimate(point_id=point_id, segment_id=segment_id,
                            metric=metric, window=window, n_blocks=0, n_obs=0,
                            status=STATUS_INSUFFICIENT,
                            reason=f"no cycles in the {window} window")
    t_start = subset["ts"].min()
    t_end = subset["ts"].max()
    span = (t_end - t_start).total_seconds() / 86400.0
    if n_points < MIN_RATE_POINTS:
        return RateEstimate(
            point_id=point_id, segment_id=segment_id, metric=metric,
            window=window, n_blocks=n_points, n_obs=n_points,
            t_start=t_start, t_end=t_end, method="theil_sen",
            partial_block=False, status=STATUS_INSUFFICIENT,
            reason=f"{n_points} cycle(s); at least {MIN_RATE_POINTS} required. "
                   f"{note}")
    if span < MIN_RATE_SPAN_DAYS:
        return RateEstimate(
            point_id=point_id, segment_id=segment_id, metric=metric,
            window=window, n_blocks=n_points, n_obs=n_points,
            t_start=t_start, t_end=t_end, method="theil_sen",
            partial_block=False, status=STATUS_INSUFFICIENT,
            reason=f"cycle span {span:.2f} day(s); at least 1 day required. "
                   f"{note}")
    x = (subset["ts"] - t_start).dt.total_seconds().to_numpy() / 86400.0
    values = subset["value"].to_numpy(dtype=float)
    slope, _intercept, low, high = theilslopes(
        values, x, alpha=float(config.rates.confidence))
    return RateEstimate(
        point_id=point_id, segment_id=segment_id, metric=metric,
        window=window, n_blocks=n_points, n_obs=n_points,
        t_start=t_start, t_end=t_end,
        rate_mm_day=float(slope), ci_low=float(low), ci_high=float(high),
        method="theil_sen", partial_block=False, status=STATUS_OK,
        reason=note)


def _prepare_rate_series(series: pd.DataFrame, metric: str) -> pd.DataFrame:
    missing = {"ts", "value"} - set(series.columns)
    if missing:
        raise ValueError(f"series missing required columns: {sorted(missing)}")
    work = series.copy()
    work["ts"] = pd.to_datetime(work["ts"])
    if "metric" in work.columns:
        work = work.loc[work["metric"] == metric]
    work = work.dropna(subset=["ts", "value"]).sort_values("ts")
    return work


def _identity(series: pd.DataFrame) -> tuple[str, str]:
    point = _single_identity(series, "point_id")
    segment = _single_identity(series, "segment_id")
    return point, segment


def _single_identity(series: pd.DataFrame, column: str) -> str:
    if column not in series.columns:
        return ""
    values = series[column].dropna().unique()
    if len(values) == 0:
        return ""
    if len(values) > 1:
        raise ValueError(
            f"theil_sen_rate expects one {column}; got {len(values)}. "
            "Call it per prism/segment.")
    return str(values[0])


def _parse_window(window: str) -> tuple[str, float]:
    text = str(window).strip().lower()
    try:
        if text.endswith("d"):
            return "days", float(text[:-1])
        if text.endswith("h"):
            return "hours", float(text[:-1])
    except ValueError as exc:
        raise ValueError(f"unsupported window: {window!r}") from exc
    raise ValueError(f"unsupported window: {window!r}")


# ---------------------------------------------------------------------------
#  F. Pipeline convenience
# ---------------------------------------------------------------------------
def compute_rates(displacements: pd.DataFrame, *,
                  config: AnalysisConfig) -> pd.DataFrame:
    """Run every configured window for every prism/segment and metric.

    Accepts the wide displacement table (``d_rad_mm``, ``d_vert_mm``,
    ``d_tan_mm`` and optional ``d_east_mm``/``d_north_mm``/``d_up_mm``) or an
    already-long frame with ``metric``/``value`` columns. Returns a DataFrame
    with :data:`rts_forensics.models.RATE_COLUMNS`.
    """
    if displacements is None or len(displacements) == 0:
        return _empty_rates()
    long = _to_long_displacements(displacements)
    if long.empty:
        return _empty_rates()

    windows = [f"{days}d" for days in config.rates.windows_days]
    windows += [f"{hours}h" for hours in config.rates.windows_hours]
    group_columns = [c for c in ("point_id", "segment_id") if c in long.columns]
    if group_columns:
        iterator = long.groupby(group_columns, sort=True, dropna=False)
    else:
        iterator = [((), long)]

    rows: list[dict[str, Any]] = []
    for _key, group in iterator:
        for metric in RATE_METRICS:
            subset = group.loc[group["metric"] == metric]
            if subset.empty:
                continue
            for window in windows:
                estimate = theil_sen_rate(subset, metric=metric,
                                          window=window, config=config)
                rows.append(estimate.to_row())
    if not rows:
        return _empty_rates()
    return pd.DataFrame(rows).reindex(columns=list(RATE_COLUMNS))


def _to_long_displacements(displacements: pd.DataFrame) -> pd.DataFrame:
    if "metric" in displacements.columns and "value" in displacements.columns:
        long = displacements.copy()
        long["metric"] = long["metric"].astype(str)
        return long.loc[long["metric"].isin(RATE_METRICS)].reset_index(drop=True)

    rename: dict[str, str] = {}
    for column, metric in DISPLACEMENT_METRIC_COLUMNS.items():
        if column in displacements.columns and metric not in rename.values():
            rename[column] = metric
    for metric in RATE_METRICS:
        if metric in displacements.columns and metric not in rename.values():
            rename[metric] = metric
    if not rename:
        raise ValueError(
            "displacements provide none of the known metric columns "
            f"{sorted(DISPLACEMENT_METRIC_COLUMNS)}")
    metric_columns = list(rename.values())
    id_columns = [c for c in ("point_id", "segment_id", "ts", "cycle_id")
                  if c in displacements.columns]
    long = displacements.rename(columns=rename).melt(
        id_vars=id_columns, value_vars=metric_columns,
        var_name="metric", value_name="value")
    return long.dropna(subset=["value"]).reset_index(drop=True)


def _empty_rates() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in RATE_COLUMNS})
