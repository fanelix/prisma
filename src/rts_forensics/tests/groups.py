"""
groups.py — spatial, temporal, change-point, pairwise and vector tests.

This module hosts the shared analytical-table helpers used by every other
``rts_forensics.tests`` sub-module (``prepare_series``, ``resolve_config``,
``require_columns``, ``window_hours``, ``block_medians``). The helpers are
private because the public surface of the package is the hypothesis tests
exported by :mod:`rts_forensics.tests`.

``series`` input schema
-----------------------
One row per prism per cycle. Required columns:

``point_id``
    str — prism identifier as exported.
``ts``
    datetime64[ns] — site-local naive cycle time.

The displacement value is supplied in one of two documented forms:

* **long form** — a ``metric`` column plus one of ``value``,
  ``metric_value``, ``value_mm``, ``metric_value_mm``. Canonical metric names
  are ``los``, ``vert``, ``tan``, ``east``, ``north``, ``up``, ``hz`` and
  ``v`` (alias spellings such as ``d_rad`` or ``ver_raw`` are accepted);
* **wide form** — any of the documented aliases: ``los_mm``, ``los_fc_mm``,
  ``d_rad_mm``, ``dD``; ``vert_mm``, ``ver_fc_mm``, ``ver_raw_mm``,
  ``d_vert_mm``; ``tan_mm``, ``tan_fc_mm``, ``tan_raw_mm``, ``d_tan_mm``;
  ``hz_deg``; ``v_deg``/``zenith_deg``.

Optional context columns are carried through unchanged: ``cycle_id``,
``station_id``, ``segment_id``, ``n_obs`` (int), ``obs_ids`` (iterable of
contributing observation IDs), ``az_deg``/``azimuth_deg``,
``v_deg``/``zenith_deg``, ``d_m``. Displacement values are millimetres; angles
are degrees. Functions never mutate their inputs.

Honesty rules obeyed here: every numeric decision threshold is read from
``config.forensics`` (or a documented supplied setting); a statistic may still
be computed when the threshold is missing, but the decision column and the
result status are then ``not_configured`` with an explicit reason. Group
statistics record the member IDs and the actual contributors of every epoch,
so a group shrinking to one prism is visible. Unobserved periods are never
described as stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from scipy.stats import theilslopes

from ..config import AnalysisConfig, default_config, is_not_configured
from ..models import NOT_CONFIGURED, STATUS_INSUFFICIENT, STATUS_OK
from ..rates import MIN_RATE_POINTS, MIN_RATE_SPAN_DAYS

#: Status used by the exploratory missingness/step disclosures (no configured
#: significance rule exists, so no accept/reject decision is made).
STATUS_EXPLORATORY = "exploratory"

CANONICAL_METRICS = ("los", "vert", "tan", "east", "north", "up", "hz", "v")

#: Wide-form column aliases per canonical metric, in preference order.
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "los": ("los_mm", "los_fc_mm", "d_rad_mm", "dD", "d_rad", "los"),
    "vert": ("vert_mm", "ver_fc_mm", "ver_raw_mm", "d_vert_mm", "d_vert",
             "ver_raw", "vert"),
    "tan": ("tan_mm", "tan_fc_mm", "tan_raw_mm", "d_tan_mm", "d_tan",
            "tan_raw", "tan"),
    "east": ("d_east_mm", "d_east", "east"),
    "north": ("d_north_mm", "d_north", "north"),
    "up": ("d_up_mm", "d_up", "up"),
    "hz": ("hz_deg", "hz"),
    "v": ("v_deg", "zenith_deg", "zenith", "v"),
}

#: Long-form metric spellings mapped to canonical names.
_METRIC_SYNONYMS: dict[str, str] = {
    "los": "los", "radial": "los", "rad": "los", "d_rad": "los", "drad": "los",
    "dd": "los",
    "vert": "vert", "vertical": "vert", "ver": "vert", "d_ver": "vert",
    "d_vert": "vert", "ver_raw": "vert",
    "tan": "tan", "tangential": "tan", "d_tan": "tan", "tan_raw": "tan",
    "east": "east", "de": "east", "d_east": "east",
    "north": "north", "dn": "north", "d_north": "north",
    "up": "up", "dz": "up", "d_up": "up",
    "hz": "hz", "hz_deg": "hz",
    "v": "v", "v_deg": "v", "zenith": "v", "zenith_deg": "v",
}

_LONG_VALUE_COLUMNS = ("value", "metric_value", "value_mm", "metric_value_mm")

_CONTEXT_COLUMNS = (
    "cycle_id", "station_id", "segment_id", "n_obs", "obs_ids",
    "az_deg", "azimuth_deg", "v_deg", "zenith_deg", "hz_deg", "d_m",
)


# ---------------------------------------------------------------------------
#  Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ClusterResult:
    """Candidate spatial clusters found without a hard-coded prism list."""

    clusters: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


@dataclass
class PlaceboResult:
    """Spatial-placebo null distribution and its membership."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    null_distribution: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float))
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


@dataclass
class TemporalSplitResult:
    """First-half versus second-half net change, with epoch contributors."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    contributors: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


@dataclass
class ChangePointResult:
    """Hinge and three-segment change-point fits on block medians."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    sensitivity: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


@dataclass
class PairwiseLOSResult:
    """Pairwise LOS differences and their Theil-Sen rates."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


@dataclass
class MovementVectorResult:
    """Three-dimensional movement vectors reconstructed from LOS/tan/vert."""

    vectors: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


# ---------------------------------------------------------------------------
#  Shared schema helpers
# ---------------------------------------------------------------------------
def resolve_config(config) -> AnalysisConfig:
    """Accept an :class:`AnalysisConfig`, a plain dict or ``None`` (defaults)."""
    if config is None:
        return default_config()
    if isinstance(config, dict):
        return AnalysisConfig.from_dict(config)
    return config


def require_columns(frame: pd.DataFrame, required: tuple[str, ...],
                    name: str) -> None:
    """Raise ``ValueError`` when ``frame`` lacks documented columns."""
    if frame is None:
        raise ValueError(f"{name} is required")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} is missing documented columns: {sorted(missing)}")


def canonical_metric(name) -> str | None:
    """Map a metric spelling to a canonical name, or ``None`` when unknown."""
    text = str(name).strip().lower()
    if text in CANONICAL_METRICS:
        return text
    return _METRIC_SYNONYMS.get(text)


def prepare_series(series: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return a tidy ``point_id``/``ts``/``value`` frame for one metric.

    Accepts the documented long or wide ``series`` schema and never mutates the
    input. Rows with an unparseable timestamp are dropped; values are numeric
    (NaN when absent). ``hour`` is derived from the site-local timestamp and
    documented context columns are carried through.
    """
    canonical = canonical_metric(metric)
    if canonical is None:
        raise ValueError(
            f"unknown metric {metric!r}; documented metrics are "
            f"{list(CANONICAL_METRICS)}")
    require_columns(series, ("point_id", "ts"), "series")
    work = series.copy()
    work["point_id"] = work["point_id"].astype(str)
    work["ts"] = pd.to_datetime(work["ts"], errors="coerce")
    work = work.dropna(subset=["ts"])
    out_columns = ["point_id", "ts", "value", "hour"] + [
        column for column in _CONTEXT_COLUMNS if column in work.columns]

    if "metric" in work.columns:
        metric_values = work["metric"].map(canonical_metric)
        sub = work.loc[metric_values == canonical].copy()
        value_column = next(
            (column for column in _LONG_VALUE_COLUMNS if column in sub.columns),
            None)
        if value_column is None and not sub.empty:
            raise ValueError(
                "series has a metric column but no documented value column; "
                f"expected one of {list(_LONG_VALUE_COLUMNS)}")
    else:
        value_column = next(
            (column for column in METRIC_ALIASES[canonical]
             if column in work.columns), None)
        if value_column is None:
            raise ValueError(
                f"series provides none of the documented {canonical!r} "
                f"columns {list(METRIC_ALIASES[canonical])}")
        sub = work.copy()

    if sub.empty:
        return pd.DataFrame(
            {column: pd.Series(dtype="object") for column in out_columns})
    sub["value"] = pd.to_numeric(sub[value_column], errors="coerce")
    sub["hour"] = sub["ts"].dt.hour.astype(int)
    return (sub[out_columns]
            .sort_values(["point_id", "ts"])
            .reset_index(drop=True))


def window_hours(config: AnalysisConfig) -> float | None:
    """Documented comparison-window length in hours, or ``None``.

    Preference order: ``baseline.window_hours`` (supplied 48 h) and then
    ``forensics.missingness.final_window_hours`` (supplied 48 h).
    """
    hours = config.baseline.window_hours
    if is_not_configured(hours):
        hours = (config.forensics.missingness or {}).get("final_window_hours")
    if is_not_configured(hours):
        return None
    return float(hours)


def add_hour_class(work: pd.DataFrame,
                   config: AnalysisConfig) -> pd.DataFrame:
    """Add the configured ``hour_class`` (default 4 h) to a prepared frame."""
    step = config.forensics.hour_class_hours
    if is_not_configured(step) or int(step) <= 0:
        raise ValueError("forensics.hour_class_hours is not configured")
    out = work.copy()
    out["hour_class"] = (out["hour"].astype(int) // int(step)) * int(step)
    return out


def band_classes(config: AnalysisConfig, band: str) -> set[int]:
    """Configured hour-class starts for ``night``/``day`` (may be empty)."""
    bands = config.forensics.day_bands or {}
    return {int(value) for value in bands.get(band, [])}


def collect_ids(rows: pd.DataFrame) -> tuple[str, ...]:
    """Flatten a documented ``obs_ids`` column into a de-duplicated tuple."""
    if "obs_ids" not in rows.columns:
        return ()
    ids: list[str] = []
    for value in rows["obs_ids"]:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
            ids.extend(str(item) for item in value)
        elif isinstance(value, str):
            ids.append(value)
    seen: dict[str, None] = {}
    for item in ids:
        seen.setdefault(item, None)
    return tuple(seen)


def epoch_contributors(rows: pd.DataFrame, *, epoch: str,
                       start: pd.Timestamp | None = None,
                       end: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per prism present in an epoch, with contributing IDs.

    Recording the actual contributors per epoch makes a group shrinking to a
    single prism directly visible in the output table.
    """
    records: list[dict] = []
    for point_id, group in rows.groupby("point_id", sort=True):
        records.append({
            "epoch": epoch,
            "point_id": str(point_id),
            "start": start,
            "end": end,
            "n_obs": int(len(group)),
            "obs_ids": collect_ids(group),
        })
    return pd.DataFrame(
        records,
        columns=["epoch", "point_id", "start", "end", "n_obs", "obs_ids"])


def net_window_change(work: pd.DataFrame, *,
                      hours: float) -> pd.DataFrame:
    """Per-prism final-window minus baseline-window median change.

    Baseline = first ``hours`` of the prism's own record; final = last
    ``hours``. The contributing observation IDs of both windows are recorded.
    A prism without a baseline or final observation gets ``insufficient_data``
    with a reason; it is never reported as stable.
    """
    columns = ["point_id", "first", "last", "n_baseline", "n_final",
               "baseline_median_mm", "final_median_mm", "net_mm",
               "obs_ids_baseline", "obs_ids_final", "status", "reason"]
    rows: list[dict] = []
    window = pd.Timedelta(hours=float(hours))
    for point_id, group in work.groupby("point_id", sort=False):
        g = group.sort_values("ts")
        first = g["ts"].min()
        last = g["ts"].max()
        baseline = g.loc[g["ts"] <= first + window]
        final = g.loc[g["ts"] >= last - window]
        row = {
            "point_id": str(point_id),
            "first": first,
            "last": last,
            "n_baseline": int(len(baseline)),
            "n_final": int(len(final)),
            "baseline_median_mm": float("nan"),
            "final_median_mm": float("nan"),
            "net_mm": float("nan"),
            "obs_ids_baseline": collect_ids(baseline),
            "obs_ids_final": collect_ids(final),
            "status": STATUS_INSUFFICIENT,
            "reason": "",
        }
        if baseline.empty or final.empty:
            missing = []
            if baseline.empty:
                missing.append("baseline")
            if final.empty:
                missing.append("final")
            row["reason"] = (
                f"no observations in the {' and '.join(missing)} "
                f"{hours:g} h window; unobserved periods are not evidence "
                "of stability")
        else:
            row["baseline_median_mm"] = float(baseline["value"].median())
            row["final_median_mm"] = float(final["value"].median())
            row["net_mm"] = row["final_median_mm"] - row["baseline_median_mm"]
            row["status"] = STATUS_OK
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def block_medians(ts, values, *, hours: float = 24.0) -> pd.DataFrame:
    """Medians per ``hours`` block counted backwards from the final record.

    Returns a frame sorted by ``x_days`` with columns ``block_id``,
    ``x_days`` (days since the oldest block), ``value`` (median) and
    ``block_end``. The anchor is the final timestamp (never a calendar day).
    """
    frame = pd.DataFrame({
        "ts": pd.to_datetime(pd.Series(ts).reset_index(drop=True)),
        "value": pd.to_numeric(pd.Series(values).reset_index(drop=True),
                               errors="coerce"),
    }).dropna(subset=["ts", "value"])
    columns = ["block_id", "x_days", "value", "block_end"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    anchor = frame["ts"].max()
    delta_hours = (anchor - frame["ts"]).dt.total_seconds() / 3600.0
    frame["block_id"] = np.floor(delta_hours / float(hours) + 1e-9).astype(int)
    medians = (frame.groupby("block_id", sort=False)["value"].median()
               .reset_index())
    medians["block_end"] = anchor - pd.to_timedelta(
        medians["block_id"] * float(hours), unit="h")
    medians["x_days"] = ((medians["block_id"].max() - medians["block_id"])
                         * float(hours) / 24.0)
    return (medians.sort_values("x_days").reset_index(drop=True)[columns])


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object")
                         for column in columns})


def _coordinate_table(coordinates: pd.DataFrame) -> pd.DataFrame:
    require_columns(coordinates, ("point_id", "east", "north"), "coordinates")
    coords = coordinates.copy()
    coords["point_id"] = coords["point_id"].astype(str)
    coords["east"] = pd.to_numeric(coords["east"], errors="coerce")
    coords["north"] = pd.to_numeric(coords["north"], errors="coerce")
    return coords.dropna(subset=["east", "north"]).drop_duplicates("point_id")


def _cluster_coherence(work: pd.DataFrame,
                       member_ids: list[str]) -> float:
    """Median pairwise Spearman correlation of member LOS series."""
    sub = work.loc[work["point_id"].isin(member_ids)]
    if sub.empty or len(member_ids) < 2:
        return float("nan")
    index = ["cycle_id"] if "cycle_id" in sub.columns else ["ts"]
    if "station_id" in sub.columns and index == ["cycle_id"]:
        index = ["station_id", "cycle_id"]
    pivot = sub.pivot_table(index=index, columns="point_id", values="value",
                            aggfunc="median")
    if pivot.shape[1] < 2 or len(pivot) < 3:
        return float("nan")
    correlation = pivot.corr(method="spearman")
    mask = np.triu(np.ones(correlation.shape, dtype=bool), 1)
    values = correlation.where(mask).stack().dropna()
    if values.empty:
        return float("nan")
    return float(values.median())


# ---------------------------------------------------------------------------
#  A. Candidate spatial clusters
# ---------------------------------------------------------------------------
CLUSTER_COLUMNS = (
    "cluster_id", "n_members", "coherence", "centroid_east", "centroid_north",
    "members", "n_obs", "first", "last", "obs_ids", "status", "reason")


def candidate_clusters(series: pd.DataFrame, *,
                       coordinates: pd.DataFrame,
                       config: AnalysisConfig) -> ClusterResult:
    """Spatially coherent candidate clusters without a fixed prism list.

    ``coordinates`` schema: ``point_id``, ``east``, ``north`` (metres,
    Cartesian site grid); optional ``up``/``z`` are ignored. Clusters are the
    connected components of ``distance <= radius_m`` on coordinates, restricted
    to prisms that have a LOS series, kept when they have at least
    ``min_members`` and a within-cluster Spearman coherence of at least
    ``coherence``. All three values come from ``config.forensics.cluster``; if
    any is ``not_configured`` (or absent) the result status is
    ``not_configured`` and no radius, member count or coherence floor is
    invented.
    """
    cfg = resolve_config(config)
    cluster_cfg = dict(cfg.forensics.cluster or {})
    radius = cluster_cfg.get("radius_m")
    min_members = cluster_cfg.get("min_members")
    coherence_floor = cluster_cfg.get("coherence")
    missing = [name for name, value in (
        ("radius_m", radius), ("min_members", min_members),
        ("coherence", coherence_floor)) if is_not_configured(value)]
    if missing:
        return ClusterResult(
            clusters=_empty(CLUSTER_COLUMNS), members={},
            status=NOT_CONFIGURED,
            reason=("forensics.cluster."
                    + ", forensics.cluster.".join(missing)
                    + " not configured; no spatial radius, member count or "
                    "coherence floor is invented"))

    work = prepare_series(series, "los")
    coords = _coordinate_table(coordinates)
    present = set(work["point_id"])
    coords = coords.loc[coords["point_id"].isin(present)].reset_index(drop=True)
    if coords.empty or work.empty:
        return ClusterResult(
            clusters=_empty(CLUSTER_COLUMNS), members={},
            status=STATUS_INSUFFICIENT,
            reason="no prism has both coordinates and a LOS series")

    xy = coords[["east", "north"]].to_numpy(dtype=float)
    distance = cdist(xy, xy)
    adjacency = distance <= float(radius)
    _n_components, labels = connected_components(adjacency, directed=False)

    ids = coords["point_id"].tolist()
    rows: list[dict] = []
    members: dict[str, list[str]] = {}
    kept = 0
    for label in range(int(labels.max()) + 1):
        member_ids = [ids[index] for index in np.where(labels == label)[0]]
        if len(member_ids) < int(min_members):
            continue
        coherence = _cluster_coherence(work, member_ids)
        if not np.isfinite(coherence) or coherence < float(coherence_floor):
            continue
        kept += 1
        cluster_id = f"C{kept}"
        member_coords = coords.loc[coords["point_id"].isin(member_ids)]
        member_rows = work.loc[work["point_id"].isin(member_ids)]
        members[cluster_id] = sorted(member_ids)
        rows.append({
            "cluster_id": cluster_id,
            "n_members": len(member_ids),
            "coherence": coherence,
            "centroid_east": float(member_coords["east"].mean()),
            "centroid_north": float(member_coords["north"].mean()),
            "members": tuple(sorted(member_ids)),
            "n_obs": int(len(member_rows)),
            "first": member_rows["ts"].min(),
            "last": member_rows["ts"].max(),
            "obs_ids": collect_ids(member_rows),
            "status": STATUS_OK,
            "reason": (f"connected component within {float(radius):g} m with "
                       f"{len(member_ids)} members and coherence "
                       f"{coherence:.3f}"),
        })
    status = STATUS_OK if rows else STATUS_INSUFFICIENT
    reason = (None if rows else
              "no component met the configured member/coherence floors; "
              "singletons are not promoted to candidate clusters")
    return ClusterResult(
        clusters=pd.DataFrame(rows, columns=list(CLUSTER_COLUMNS)),
        members=members, status=status, reason=reason)


# ---------------------------------------------------------------------------
#  B. Spatial placebo
# ---------------------------------------------------------------------------
PLACEBO_COLUMNS = (
    "point_id", "net_mm", "placebo_median_mm", "placebo_p", "n_neighbors",
    "neighbor_ids", "status", "reason")


def spatial_placebo(series: pd.DataFrame, *,
                    coordinates: pd.DataFrame,
                    config: AnalysisConfig) -> PlaceboResult:
    """Median net change of the k nearest neighbours, excluding the prism.

    ``k = config.forensics.placebo_neighbors`` (supplied: 7). Each prism's net
    change is its final minus baseline window median (:func:`net_window_change`,
    window = ``baseline.window_hours``). The returned ``null_distribution`` is
    the array of per-prism placebo medians; ``table.neighbor_ids`` records
    exactly which prisms entered each null value. No decision threshold exists
    for this test, so no movement decision is produced. A missing ``k`` yields
    ``not_configured``; no neighbourhood size is invented.
    """
    cfg = resolve_config(config)
    k = cfg.forensics.placebo_neighbors
    if is_not_configured(k) or int(k) < 1:
        return PlaceboResult(
            table=_empty(PLACEBO_COLUMNS), members={},
            status=NOT_CONFIGURED,
            reason="forensics.placebo_neighbors is not configured; the "
                   "neighbourhood size is never invented")
    hours = window_hours(cfg)
    if hours is None:
        return PlaceboResult(
            table=_empty(PLACEBO_COLUMNS), members={},
            status=NOT_CONFIGURED,
            reason="baseline.window_hours is not configured; no comparison "
                   "window is invented")

    work = prepare_series(series, "los")
    coords = _coordinate_table(coordinates)
    present = set(work["point_id"])
    coords = coords.loc[coords["point_id"].isin(present)].reset_index(drop=True)
    if coords.empty:
        return PlaceboResult(
            table=_empty(PLACEBO_COLUMNS), members={},
            status=STATUS_INSUFFICIENT,
            reason="no prism has both coordinates and a LOS series")

    net = net_window_change(work, hours=hours)
    net_map = dict(zip(net["point_id"], net["net_mm"], strict=True))
    xy = coords[["east", "north"]].to_numpy(dtype=float)
    ids = coords["point_id"].tolist()
    distance = cdist(xy, xy)
    np.fill_diagonal(distance, np.inf)

    rows: list[dict] = []
    null: list[float] = []
    finite_null: list[float] = []
    for index, point_id in enumerate(ids):
        order = [position for position in np.argsort(distance[index],
                                                     kind="stable")
                 if np.isfinite(distance[index, position])]
        neighbors = [ids[position] for position in order[:int(k)]]
        values = np.array([net_map.get(neighbor, np.nan)
                           for neighbor in neighbors], dtype=float)
        placebo = float(np.nanmedian(values)) if np.isfinite(values).any() else float("nan")
        null.append(placebo)
        if np.isfinite(placebo):
            finite_null.append(placebo)
        observed = float(net_map.get(point_id, np.nan))
        if np.isfinite(observed) and np.isfinite(placebo) and finite_null:
            extreme = sum(1 for value in finite_null
                          if abs(value) >= abs(observed))
            placebo_p = (1 + extreme) / (1 + len(finite_null))
        else:
            placebo_p = float("nan")
        rows.append({
            "point_id": point_id,
            "net_mm": observed,
            "placebo_median_mm": placebo,
            "placebo_p": placebo_p,
            "n_neighbors": len(neighbors),
            "neighbor_ids": tuple(neighbors),
            "status": (STATUS_OK if np.isfinite(observed) and np.isfinite(placebo)
                       else STATUS_INSUFFICIENT),
            "reason": ("median net change of the configured k nearest "
                       "neighbours, excluding the prism itself; no decision "
                       "threshold is configured"),
        })
    members = {"null": ids}
    return PlaceboResult(
        table=pd.DataFrame(rows, columns=list(PLACEBO_COLUMNS)),
        null_distribution=np.asarray(null, dtype=float),
        members=members, status=STATUS_OK,
        reason=("null distribution of neighbourhood medians; a placebo value "
                "is not a movement decision"))


# ---------------------------------------------------------------------------
#  C. Temporal split
# ---------------------------------------------------------------------------
TEMPORAL_SPLIT_COLUMNS = (
    "point_id", "midpoint", "first_mm", "second_mm", "net_mm",
    "n_first", "n_second", "obs_ids_first", "obs_ids_second",
    "status", "reason")


def temporal_split(series: pd.DataFrame, *,
                   config: AnalysisConfig) -> TemporalSplitResult:
    """First-half versus second-half net change per prism.

    The split is the midpoint in elapsed time between the prism's own first and
    last record (never a row index). Values are the window medians. Every
    prism's contributors are recorded for both halves in the ``contributors``
    table, so a group that shrinks to a single prism remains visible.
    """
    resolve_config(config)
    work = prepare_series(series, "los")
    rows: list[dict] = []
    contributor_frames: list[pd.DataFrame] = []
    members: dict[str, list[str]] = {"first_half": [], "second_half": []}
    for point_id, group in work.groupby("point_id", sort=True):
        g = group.sort_values("ts")
        first = g["ts"].min()
        last = g["ts"].max()
        midpoint = first + (last - first) / 2
        first_half = g.loc[g["ts"] < midpoint]
        second_half = g.loc[g["ts"] >= midpoint]
        row = {
            "point_id": str(point_id),
            "midpoint": midpoint,
            "first_mm": float("nan"),
            "second_mm": float("nan"),
            "net_mm": float("nan"),
            "n_first": int(len(first_half)),
            "n_second": int(len(second_half)),
            "obs_ids_first": collect_ids(first_half),
            "obs_ids_second": collect_ids(second_half),
            "status": STATUS_INSUFFICIENT,
            "reason": "",
        }
        if first_half.empty or second_half.empty:
            row["reason"] = ("no observations in one half of the record; "
                             "unobserved periods are not evidence of stability")
        else:
            row["first_mm"] = float(first_half["value"].median())
            row["second_mm"] = float(second_half["value"].median())
            row["net_mm"] = row["second_mm"] - row["first_mm"]
            row["status"] = STATUS_OK
        rows.append(row)
        if not first_half.empty:
            members["first_half"].append(str(point_id))
            contributor_frames.append(
                epoch_contributors(first_half, epoch="first_half",
                                   start=first, end=midpoint))
        if not second_half.empty:
            members["second_half"].append(str(point_id))
            contributor_frames.append(
                epoch_contributors(second_half, epoch="second_half",
                                   start=midpoint, end=last))
    table = pd.DataFrame(rows, columns=list(TEMPORAL_SPLIT_COLUMNS))
    contributors = (pd.concat(contributor_frames, ignore_index=True)
                    if contributor_frames else
                    epoch_contributors(work.iloc[0:0], epoch="first_half"))
    return TemporalSplitResult(
        table=table, contributors=contributors,
        members={key: sorted(value) for key, value in members.items()},
        status=STATUS_OK if not table.empty else STATUS_INSUFFICIENT,
        reason="midpoint of each prism's own elapsed record")


# ---------------------------------------------------------------------------
#  D. Change point on block medians
# ---------------------------------------------------------------------------
CHANGE_POINT_COLUMNS = (
    "point_id", "n_blocks", "sse_null", "hinge_breakpoint", "hinge_x_days",
    "hinge_sse", "hinge_improvement", "hinge_improvement_fraction",
    "three_bp1", "three_bp2", "three_x1_days", "three_x2_days", "three_sse",
    "three_improvement", "three_improvement_fraction",
    "sensitivity_start", "sensitivity_end", "significant", "status", "reason")
SENSITIVITY_COLUMNS = (
    "point_id", "model", "breakpoint_1", "breakpoint_2", "x1_days", "x2_days",
    "sse", "sse_null", "improvement")


def _hinge_sse(x: np.ndarray, y: np.ndarray, breakpoint: float) -> float:
    design = np.column_stack([np.ones_like(x),
                              np.maximum(0.0, x - breakpoint)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    return float(residual @ residual)


def _three_segment_sse(x: np.ndarray, y: np.ndarray,
                       first: float, second: float) -> float:
    design = np.column_stack([
        np.ones_like(x), x,
        np.maximum(0.0, x - first), np.maximum(0.0, x - second)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    return float(residual @ residual)


def change_point(series: pd.DataFrame, *,
                 config: AnalysisConfig) -> ChangePointResult:
    """Hinge and three-segment least-squares change points on 24 h blocks.

    Blocks are ``config.rates.block_hours`` (supplied 24 h) medians anchored at
    the final record, exactly as the rate estimator uses them. For each prism
    the hinge model ``c + s·max(0, t - bp)`` and the continuous three-segment
    piecewise-linear model with two knots are minimised over every candidate
    block bound that leaves at least two blocks in each segment. SSE
    improvements are relative to the no-change (constant) model. The
    ``sensitivity`` table stores the full SSE profile; the reported sensitivity
    range is the span of candidate breakpoints that beat the constant model
    (a data comparison, not an invented numeric threshold). ``step_significance``
    is not configured in the supplied configuration, so the decision status is
    ``not_configured`` and ``significant`` is False unless the user configures
    a minimum fractional improvement.
    """
    cfg = resolve_config(config)
    events_cfg = dict(cfg.forensics.events or {})
    decision = events_cfg.get("step_significance")
    decision_configured = not is_not_configured(decision)
    block_hours = cfg.rates.block_hours
    if is_not_configured(block_hours):
        return ChangePointResult(
            table=_empty(CHANGE_POINT_COLUMNS),
            sensitivity=_empty(SENSITIVITY_COLUMNS), members={},
            status=NOT_CONFIGURED,
            reason="rates.block_hours is not configured; no block length is "
                   "invented")

    work = prepare_series(series, "los")
    rows: list[dict] = []
    profiles: list[dict] = []
    computed = 0
    for point_id, group in work.groupby("point_id", sort=True):
        g = group.sort_values("ts")
        blocks = block_medians(g["ts"], g["value"], hours=float(block_hours))
        row = {
            "point_id": str(point_id),
            "n_blocks": int(len(blocks)),
            "sse_null": float("nan"),
            "hinge_breakpoint": pd.NaT,
            "hinge_x_days": float("nan"),
            "hinge_sse": float("nan"),
            "hinge_improvement": float("nan"),
            "hinge_improvement_fraction": float("nan"),
            "three_bp1": pd.NaT,
            "three_bp2": pd.NaT,
            "three_x1_days": float("nan"),
            "three_x2_days": float("nan"),
            "three_sse": float("nan"),
            "three_improvement": float("nan"),
            "three_improvement_fraction": float("nan"),
            "sensitivity_start": pd.NaT,
            "sensitivity_end": pd.NaT,
            "significant": False,
            "status": STATUS_INSUFFICIENT,
            "reason": "",
        }
        if len(blocks) < MIN_RATE_POINTS:
            row["reason"] = (f"{len(blocks)} 24 h block medians; at least "
                             f"{MIN_RATE_POINTS} required for a change-point "
                             "scan")
            rows.append(row)
            continue
        x = blocks["x_days"].to_numpy(dtype=float)
        y = blocks["value"].to_numpy(dtype=float)
        ends = blocks["block_end"].tolist()
        sse_null = float(np.sum((y - y.mean()) ** 2))
        row["sse_null"] = sse_null

        candidates = [index for index in range(2, len(x) - 2)]
        hinge_results = [(bp, _hinge_sse(x, y, x[bp])) for bp in candidates]
        improving = [bp for bp, sse in hinge_results if sse <= sse_null]
        for bp, sse in hinge_results:
            profiles.append({
                "point_id": str(point_id), "model": "hinge",
                "breakpoint_1": ends[bp], "breakpoint_2": pd.NaT,
                "x1_days": x[bp], "x2_days": float("nan"),
                "sse": sse, "sse_null": sse_null,
                "improvement": sse_null - sse,
            })
        best_bp, best_hinge_sse = min(hinge_results, key=lambda item: item[1])
        row["hinge_breakpoint"] = ends[best_bp]
        row["hinge_x_days"] = x[best_bp]
        row["hinge_sse"] = best_hinge_sse
        row["hinge_improvement"] = sse_null - best_hinge_sse
        row["hinge_improvement_fraction"] = (
            (sse_null - best_hinge_sse) / sse_null if sse_null > 0 else 0.0)
        if improving:
            row["sensitivity_start"] = ends[min(improving)]
            row["sensitivity_end"] = ends[max(improving)]

        best_three: tuple[float, int, int] | None = None
        for first_index, second_index in combinations(candidates, 2):
            if second_index - first_index < 2:
                continue
            if first_index < 2 or len(x) - second_index < 3:
                continue
            sse = _three_segment_sse(x, y, x[first_index], x[second_index])
            profiles.append({
                "point_id": str(point_id), "model": "three_segment",
                "breakpoint_1": ends[first_index],
                "breakpoint_2": ends[second_index],
                "x1_days": x[first_index], "x2_days": x[second_index],
                "sse": sse, "sse_null": sse_null,
                "improvement": sse_null - sse,
            })
            if best_three is None or sse < best_three[0]:
                best_three = (sse, first_index, second_index)
        if best_three is not None:
            sse, first_index, second_index = best_three
            row["three_bp1"] = ends[first_index]
            row["three_bp2"] = ends[second_index]
            row["three_x1_days"] = x[first_index]
            row["three_x2_days"] = x[second_index]
            row["three_sse"] = sse
            row["three_improvement"] = sse_null - sse
            row["three_improvement_fraction"] = (
                (sse_null - sse) / sse_null if sse_null > 0 else 0.0)

        if decision_configured and np.isfinite(row["hinge_improvement_fraction"]):
            row["significant"] = bool(
                row["hinge_improvement_fraction"] >= float(decision))
            row["status"] = STATUS_OK
        else:
            row["status"] = NOT_CONFIGURED
        row["reason"] = (
            "step_significance is not configured; the breakpoint and SSE "
            "improvement are reported but no significance decision is made"
            if not decision_configured else
            f"significant at the configured minimum fractional improvement "
            f"{float(decision):g}")
        computed += 1
        rows.append(row)

    table = pd.DataFrame(rows, columns=list(CHANGE_POINT_COLUMNS))
    sensitivity = pd.DataFrame(profiles, columns=list(SENSITIVITY_COLUMNS))
    if not decision_configured:
        status = NOT_CONFIGURED
        reason = ("significance threshold (forensics.events.step_significance) "
                  "is not configured; breakpoints and SSE improvements are "
                  "statistics only")
    else:
        status = STATUS_OK if computed else STATUS_INSUFFICIENT
        reason = "minimum fractional SSE improvement supplied by configuration"
    return ChangePointResult(
        table=table, sensitivity=sensitivity,
        members={"prisms": sorted(table["point_id"].tolist())},
        status=status, reason=reason)


# ---------------------------------------------------------------------------
#  E. Pairwise LOS differences
# ---------------------------------------------------------------------------
PAIRWISE_COLUMNS = (
    "point_a", "point_b", "members", "n_pairs", "n_blocks", "span_days",
    "median_diff_mm", "rate_mm_day", "ci_low", "ci_high", "distance_m",
    "status", "reason")


def pairwise_los(series: pd.DataFrame, *,
                 coordinates: pd.DataFrame,
                 config: AnalysisConfig) -> PairwiseLOSResult:
    """Pairwise LOS differences with a Theil-Sen rate and separation distance.

    ``coordinates`` schema: ``point_id``, ``east``, ``north`` (metres).
    Series are aligned on ``cycle_id`` when available (optionally with
    ``station_id``) and otherwise on the exact timestamp. The difference
    ``a - b`` is reduced to 24 h block medians (anchored at the pair's final
    record) before Theil-Sen; fewer than six blocks or less than a day of span
    yields ``insufficient_data`` with no fabricated rate. Pair members are
    recorded on every row.
    """
    cfg = resolve_config(config)
    work = prepare_series(series, "los")
    coords = _coordinate_table(coordinates)
    distance_map: dict[frozenset[str], float] = {}
    for row in coords.to_dict("records"):
        distance_map[str(row["point_id"])] = float("nan")
    records = coords.to_dict("records")
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            key = frozenset((str(records[left]["point_id"]),
                             str(records[right]["point_id"])))
            separation = float(np.hypot(
                records[left]["east"] - records[right]["east"],
                records[left]["north"] - records[right]["north"]))
            distance_map[key] = separation

    if "cycle_id" in work.columns and "station_id" in work.columns:
        index = ["station_id", "cycle_id"]
    elif "cycle_id" in work.columns:
        index = ["cycle_id"]
    else:
        index = ["ts"]
    pivot = work.pivot_table(index=index, columns="point_id", values="value",
                             aggfunc="median")
    time_lookup = work.groupby(index, sort=True)["ts"].max()

    rows: list[dict] = []
    members: dict[str, list[str]] = {}
    for point_a, point_b in combinations(sorted(pivot.columns), 2):
        pair = pivot[[point_a, point_b]].dropna()
        label = f"{point_a}|{point_b}"
        row = {
            "point_a": str(point_a), "point_b": str(point_b),
            "members": (str(point_a), str(point_b)),
            "n_pairs": int(len(pair)), "n_blocks": 0,
            "span_days": float("nan"), "median_diff_mm": float("nan"),
            "rate_mm_day": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "distance_m": float("nan"),
            "status": STATUS_INSUFFICIENT, "reason": "",
        }
        key = frozenset((str(point_a), str(point_b)))
        if key in distance_map:
            row["distance_m"] = distance_map[key]
        if len(pair) < 2:
            row["reason"] = "fewer than two aligned cycle pairs"
            rows.append(row)
            continue
        difference = pair[point_a] - pair[point_b]
        timestamps = time_lookup.reindex(difference.index)
        blocks = block_medians(timestamps, difference,
                               hours=float(cfg.rates.block_hours))
        row["n_blocks"] = int(len(blocks))
        row["median_diff_mm"] = float(difference.median())
        if len(blocks) < 2:
            row["reason"] = "no usable 24 h block medians"
            rows.append(row)
            continue
        span = float(blocks["x_days"].max() - blocks["x_days"].min())
        row["span_days"] = span
        if len(blocks) < MIN_RATE_POINTS:
            row["reason"] = (f"{len(blocks)} block medians; at least "
                             f"{MIN_RATE_POINTS} required")
            rows.append(row)
            continue
        if span < MIN_RATE_SPAN_DAYS:
            row["reason"] = (f"block span {span:.2f} day(s); at least "
                             f"{MIN_RATE_SPAN_DAYS:g} day required")
            rows.append(row)
            continue
        slope, _intercept, low, high = theilslopes(
            blocks["value"].to_numpy(dtype=float),
            blocks["x_days"].to_numpy(dtype=float),
            alpha=float(cfg.rates.confidence))
        row["rate_mm_day"] = float(slope)
        row["ci_low"] = float(low)
        row["ci_high"] = float(high)
        row["status"] = STATUS_OK
        rows.append(row)
        members[label] = [str(point_a), str(point_b)]

    table = pd.DataFrame(rows, columns=list(PAIRWISE_COLUMNS))
    return PairwiseLOSResult(
        table=table, members=members,
        status=STATUS_OK if not table.empty else STATUS_INSUFFICIENT,
        reason=("pairwise differences cancel any signal shared by both "
                "prisms; no decision threshold is configured"))


# ---------------------------------------------------------------------------
#  F. Movement vectors
# ---------------------------------------------------------------------------
MOVEMENT_VECTOR_COLUMNS = (
    "point_id", "ts", "los_mm", "tan_mm", "vert_mm", "az_deg", "zenith_deg",
    "rad_h_mm", "d_east_mm", "d_north_mm", "d_up_mm", "horizontal_mm",
    "vector_azimuth_deg", "towards_station_mm", "n_obs", "obs_ids",
    "status", "reason")

_AZIMUTH_ALIASES = ("az_deg", "azimuth_deg", "az")
_ZENITH_ALIASES = ("v_deg", "zenith_deg", "zenith")


def movement_vectors(series: pd.DataFrame, *,
                     coordinates: pd.DataFrame | None,
                     config: AnalysisConfig) -> MovementVectorResult:
    """Reconstruct 3-D movement vectors from LOS, tangential and vertical.

    Sign conventions (identical to :mod:`rts_forensics.geometry`): grid azimuth
    ``a`` is clockwise from North; radial is positive *away* from the station;
    ``los`` is positive away, ``vert`` positive up, ``tan`` positive clockwise
    from above (looking from the station towards the prism); ``v`` is the
    zenith angle. The horizontal radial component removes the vertical
    projection first::

        rad_h = (los - vert·cos V) / sin V
        dE    = rad_h·sin a + tan·cos a
        dN    = rad_h·cos a - tan·sin a

    An azimuth and a zenith column are required; they may be supplied on
    ``series`` (``az_deg``/``azimuth_deg`` and ``v_deg``/``zenith_deg``) or on
    ``coordinates`` keyed by ``point_id``. Displacements are millimetres.
    """
    resolve_config(config)
    require_columns(series, ("point_id", "ts"), "series")

    def _narrow(metric: str, output: str) -> pd.DataFrame:
        frame = prepare_series(series, metric)
        keys = ["point_id", "ts"]
        keep = keys + ["value"]
        frame = frame[keep].rename(columns={"value": output})
        return frame

    base = _narrow("los", "los_mm")
    base = base.merge(_narrow("tan", "tan_mm"), on=["point_id", "ts"],
                      how="inner")
    base = base.merge(_narrow("vert", "vert_mm"), on=["point_id", "ts"],
                      how="inner")
    if base.empty:
        return MovementVectorResult(
            vectors=_empty(MOVEMENT_VECTOR_COLUMNS), members={},
            status=STATUS_INSUFFICIENT,
            reason="no cycle provides all of los, tan and vert")

    azimuth_column = next(
        (column for column in _AZIMUTH_ALIASES if column in series.columns),
        None)
    zenith_column = next(
        (column for column in _ZENITH_ALIASES if column in series.columns),
        None)
    context_columns = [column for column in ("cycle_id", "station_id", "n_obs",
                                              "obs_ids", *_AZIMUTH_ALIASES,
                                              *_ZENITH_ALIASES)
                       if column in series.columns]
    if context_columns:
        context = (series[["point_id", "ts", *context_columns]]
                   .drop_duplicates(["point_id", "ts"]))
        base = base.merge(context, on=["point_id", "ts"], how="left")

    if coordinates is not None:
        coords = coordinates.copy()
        coords["point_id"] = coords["point_id"].astype(str)
        geometry_columns = [
            column for column in (*_AZIMUTH_ALIASES, *_ZENITH_ALIASES)
            if column in coords.columns and column not in base.columns]
        if geometry_columns:
            coords = coords[["point_id", *geometry_columns]].drop_duplicates(
                "point_id")
            base = base.merge(coords, on="point_id", how="left")

    if azimuth_column is None:
        azimuth_column = next(
            (column for column in _AZIMUTH_ALIASES if column in base.columns),
            None)
        if azimuth_column is not None and azimuth_column != "az_deg":
            base["az_deg"] = base[azimuth_column]
            azimuth_column = "az_deg"
    if zenith_column is None:
        zenith_column = next(
            (column for column in _ZENITH_ALIASES if column in base.columns),
            None)
        if zenith_column is not None and zenith_column != "v_deg":
            base["v_deg"] = base[zenith_column]
            zenith_column = "v_deg"

    if azimuth_column is None or zenith_column is None:
        raise ValueError(
            "movement_vectors requires a documented azimuth column "
            f"({list(_AZIMUTH_ALIASES)}) and zenith column "
            f"({list(_ZENITH_ALIASES)}), on series or coordinates")

    az = pd.to_numeric(base[azimuth_column], errors="coerce").to_numpy(float)
    zenith = pd.to_numeric(base[zenith_column], errors="coerce").to_numpy(float)
    los = pd.to_numeric(base["los_mm"], errors="coerce").to_numpy(float)
    tan = pd.to_numeric(base["tan_mm"], errors="coerce").to_numpy(float)
    vert = pd.to_numeric(base["vert_mm"], errors="coerce").to_numpy(float)

    azimuth_rad = np.radians(az)
    sin_v = np.sin(np.radians(zenith))
    cos_v = np.cos(np.radians(zenith))
    with np.errstate(divide="ignore", invalid="ignore"):
        rad_h = (los - vert * cos_v) / sin_v
    d_east = rad_h * np.sin(azimuth_rad) + tan * np.cos(azimuth_rad)
    d_north = rad_h * np.cos(azimuth_rad) - tan * np.sin(azimuth_rad)
    horizontal = np.hypot(d_east, d_north)
    vector_azimuth = np.degrees(np.arctan2(d_east, d_north)) % 360.0
    towards_station = -(d_east * np.sin(azimuth_rad)
                        + d_north * np.cos(azimuth_rad))

    out = base.copy()
    out["az_deg"] = az
    out["zenith_deg"] = zenith
    out["rad_h_mm"] = rad_h
    out["d_east_mm"] = d_east
    out["d_north_mm"] = d_north
    out["d_up_mm"] = vert
    out["horizontal_mm"] = horizontal
    out["vector_azimuth_deg"] = vector_azimuth
    out["towards_station_mm"] = towards_station
    finite = np.isfinite(rad_h) & np.isfinite(d_east) & np.isfinite(d_north)
    out["status"] = np.where(finite, STATUS_OK, STATUS_INSUFFICIENT)
    out["reason"] = np.where(
        finite,
        "reconstructed from frame-corrected LOS/tan/vert; sign conventions in "
        "the movement_vectors docstring",
        "missing azimuth/zenith or a degenerate sin V; component unavailable")
    if "n_obs" not in out.columns:
        out["n_obs"] = np.nan
    if "obs_ids" not in out.columns:
        out["obs_ids"] = [() for _ in range(len(out))]
    vectors = out[[column for column in MOVEMENT_VECTOR_COLUMNS
                   if column in out.columns]]
    vectors = (vectors.sort_values(["point_id", "ts"])
               .reset_index(drop=True))
    return MovementVectorResult(
        vectors=vectors,
        members={"prisms": sorted(vectors["point_id"].unique().tolist())},
        status=STATUS_OK if bool(finite.any()) else STATUS_INSUFFICIENT,
        reason="horizontal radial component reconstructed as "
               "(los - vert·cos V)/sin V")
