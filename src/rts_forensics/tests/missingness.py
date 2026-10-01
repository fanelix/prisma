"""
missingness.py — coverage, lost-versus-retained, pre-loss trends, observability.

The final no-data window (``forensics.missingness.final_window_hours``, supplied
48 h) defines a dropout only against the network's final record. A prism with no
observation in that window is reported as unobserved; unobserved is never
described as stable. The permutation test uses the configured
``permutation_n`` (supplied 5000) and ``seed`` (supplied 1) and is therefore
deterministic. Its result status is ``exploratory``: no decision significance is
configured, so both the permutation and Mann–Whitney p-values are reported
without an accept/reject verdict.

Input schema is documented in :mod:`rts_forensics.tests.groups`; ``coordinates``
requires ``point_id``, ``east`` and ``north``. An optional boolean column
``candidate_zone``/``candidate`` on ``coordinates`` names the candidate cluster
used for the distance permutation; otherwise the configured
``forensics.missingness.candidate_zone`` or the configured cluster search is
used, and finally the network centroid (recorded in ``candidate_source``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, theilslopes

from ..config import is_not_configured
from ..models import NOT_CONFIGURED, STATUS_INSUFFICIENT, STATUS_OK
from .groups import (
    MIN_RATE_POINTS,
    STATUS_EXPLORATORY,
    add_hour_class,
    band_classes,
    candidate_clusters,
    collect_ids,
    prepare_series,
    resolve_config,
)

#: Arithmetic scaffold minimum for a quartile-based pre-loss contrast; it is
#: not a configured decision threshold.
MIN_WINDOW_POINTS = 4

COVERAGE_COLUMNS = (
    "point_id", "n_obs", "first", "last", "span_hours", "longest_gap_hours",
    "median_interval_hours", "expected_slots", "coverage", "n_final_window",
    "in_final_window", "obs_ids", "status", "reason")
LOST_TABLE_COLUMNS = (
    "point_id", "east", "north", "distance_m", "lost", "n_final_window",
    "role", "status", "reason")
LOST_TEST_COLUMNS = (
    "statistic_mm", "p_permutation", "p_mannwhitney", "n_lost", "n_retained",
    "n_permutations", "seed", "candidate_source", "status", "reason")
PRE_LOSS_COLUMNS = (
    "point_id", "n_window", "window_start", "window_end", "first_median_mm",
    "last_median_mm", "net_mm", "trend_mm_day", "trend_ci_low",
    "trend_ci_high", "rank_percentile", "n_peers", "obs_ids", "status",
    "reason")
OBSERVABILITY_COLUMNS = (
    "point_id", "night_obs_first", "night_obs_last", "night_success_first",
    "night_success_last", "night_blind", "status", "reason")
OBSERVABILITY_HOUR_COLUMNS = (
    "point_id", "window", "hour_class", "n_observed", "n_expected",
    "success", "status", "reason")


@dataclass
class CoverageResult:
    """Coverage, gaps and final-window observations per prism."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


@dataclass
class LostVsRetainedResult:
    """Lost-versus-retained distance tests (exploratory)."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    tests: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = STATUS_EXPLORATORY
    reason: str | None = None


@dataclass
class PreLossTrendResult:
    """Pre-loss trend and peer rank per prism."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


@dataclass
class ObservabilityResult:
    """Observation success by hour class over the first and final windows."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_hour: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


def _final_window_hours(config) -> float | None:
    value = (config.forensics.missingness or {}).get("final_window_hours")
    if is_not_configured(value):
        return None
    return float(value)


def _cycle_key_columns(work: pd.DataFrame) -> list[str]:
    if "cycle_id" in work.columns:
        return (["station_id", "cycle_id"] if "station_id" in work.columns
                else ["cycle_id"])
    return ["ts"]


# ---------------------------------------------------------------------------
#  A. Coverage
# ---------------------------------------------------------------------------
def coverage(series: pd.DataFrame, *, config) -> CoverageResult:
    """Observations, first/last, longest gap, coverage fraction, final window.

    ``coverage`` is ``n_obs / expected_slots`` where expected slots are the
    network cycles of the same station between the prism's first and last
    record (when ``station_id``/``cycle_id`` are available) or the span divided
    by the network median sampling interval. Observations in the final
    ``final_window_hours`` window are counted against the network final record.
    If that window is not configured the counts are still computed for the
    documented fallback but the status is ``not_configured`` and no dropout
    decision is made.
    """
    cfg = resolve_config(config)
    final_hours = _final_window_hours(cfg)
    window_configured = final_hours is not None
    work = prepare_series(series, "los")
    if work.empty:
        return CoverageResult(
            table=pd.DataFrame(columns=list(COVERAGE_COLUMNS)), members={},
            status=STATUS_INSUFFICIENT, reason="no LOS observations")

    network_last = work["ts"].max()
    intervals = []
    for _point_id, group in work.groupby("point_id", sort=False):
        differences = (group["ts"].sort_values().diff()
                       .dt.total_seconds().to_numpy() / 3600.0)
        intervals.extend(float(value) for value in differences
                         if np.isfinite(value) and value > 0)
    median_interval = float(np.median(intervals)) if intervals else float("nan")

    key_columns = _cycle_key_columns(work)
    network_cycles = work.drop_duplicates(key_columns)

    rows: list[dict] = []
    for point_id, group in work.groupby("point_id", sort=True):
        first = group["ts"].min()
        last = group["ts"].max()
        span_hours = float((last - first).total_seconds() / 3600.0)
        differences = (group["ts"].sort_values().diff()
                       .dt.total_seconds().to_numpy() / 3600.0)
        positive = [float(value) for value in differences
                    if np.isfinite(value) and value > 0]
        longest_gap = max(positive) if positive else float("nan")

        if "station_id" in work.columns and "cycle_id" in work.columns:
            station_id = group["station_id"].iloc[0]
            candidates = network_cycles.loc[
                (network_cycles["station_id"] == station_id)
                & (network_cycles["ts"] >= first)
                & (network_cycles["ts"] <= last)]
            expected_slots = int(len(candidates))
        elif np.isfinite(median_interval) and median_interval > 0:
            expected_slots = int(np.floor(span_hours / median_interval) + 1)
        else:
            expected_slots = int(len(group))
        coverage_fraction = (min(1.0, len(group) / expected_slots)
                             if expected_slots > 0 else float("nan"))
        n_final = int((group["ts"] >= network_last
                       - pd.Timedelta(hours=final_hours)).sum()) \
            if window_configured else -1

        row = {
            "point_id": str(point_id),
            "n_obs": int(len(group)),
            "first": first,
            "last": last,
            "span_hours": span_hours,
            "longest_gap_hours": longest_gap,
            "median_interval_hours": median_interval,
            "expected_slots": expected_slots,
            "coverage": coverage_fraction,
            "n_final_window": n_final if window_configured else np.nan,
            "in_final_window": bool(n_final > 0) if window_configured else False,
            "obs_ids": collect_ids(group),
            "status": STATUS_OK if window_configured else NOT_CONFIGURED,
            "reason": "",
        }
        if not window_configured:
            row["reason"] = ("forensics.missingness.final_window_hours is not "
                             "configured; coverage is reported but no dropout "
                             "decision is made")
        elif n_final <= 0:
            row["reason"] = (
                f"no observations in the final {final_hours:g} h window; the "
                "period is unobserved and is not evidence of stability")
        elif np.isfinite(coverage_fraction) and coverage_fraction < 1.0:
            row["reason"] = ("gaps within the record; coverage is relative to "
                             "the observed schedule, not a completeness grade")
        else:
            row["reason"] = ("observations span the record and the final "
                             "window; this is coverage, not a stability claim")
        rows.append(row)

    table = pd.DataFrame(rows, columns=list(COVERAGE_COLUMNS))
    return CoverageResult(
        table=table,
        members={"prisms": sorted(table["point_id"].tolist())},
        status=STATUS_OK if window_configured else NOT_CONFIGURED,
        reason=(f"final window {final_hours:g} h"
                if window_configured else
                "final window not configured"))


# ---------------------------------------------------------------------------
#  B. Lost versus retained
# ---------------------------------------------------------------------------
def _candidate_centre(coordinates: pd.DataFrame, work: pd.DataFrame,
                      config, network_last: pd.Timestamp,
                      final_hours: float) -> tuple[tuple[float, float], str]:
    """Resolve the candidate zone centre without hard-coding a prism list."""
    coords = coordinates.copy()
    for column in ("candidate_zone", "candidate"):
        if column in coords.columns:
            mask = coords[column].astype(bool)
            if mask.any():
                subset = coords.loc[mask]
                return ((float(subset["east"].mean()),
                         float(subset["north"].mean())),
                        f"coordinates.{column}")

    zone = (config.forensics.missingness or {}).get("candidate_zone")
    if isinstance(zone, dict):
        if "center_east" in zone and "center_north" in zone:
            return ((float(zone["center_east"]), float(zone["center_north"])),
                    "forensics.missingness.candidate_zone.center")
        if zone.get("point_ids"):
            subset = coords.loc[coords["point_id"].isin(
                [str(value) for value in zone["point_ids"]])]
            if not subset.empty:
                return ((float(subset["east"].mean()),
                         float(subset["north"].mean())),
                        "forensics.missingness.candidate_zone.point_ids")

    try:
        result = candidate_clusters(work.rename(columns={"value": "los_mm"}),
                                    coordinates=coords, config=config)
        if result.status == STATUS_OK and not result.clusters.empty:
            largest = result.clusters.sort_values(
                "n_members", ascending=False).iloc[0]
            return ((float(largest["centroid_east"]),
                     float(largest["centroid_north"])),
                    "forensics.cluster")
    except ValueError:
        pass

    return ((float(coords["east"].mean()), float(coords["north"].mean())),
            "network_centroid")


def lost_vs_retained(series: pd.DataFrame, *, coordinates: pd.DataFrame,
                     config) -> LostVsRetainedResult:
    """Lost (no final-window observation) versus retained distance tests.

    Distances are measured to the resolved candidate-zone centre. The
    permutation test shuffles the lost/retained labels ``permutation_n`` times
    with ``np.random.default_rng(seed)`` (deterministic for the configured
    seed); Mann–Whitney is reported alongside. Both p-values are exploratory:
    no decision significance is configured and proximity to a zone is not a
    cause.
    """
    cfg = resolve_config(config)
    final_hours = _final_window_hours(cfg)
    if final_hours is None:
        return LostVsRetainedResult(
            table=pd.DataFrame(columns=list(LOST_TABLE_COLUMNS)),
            tests=pd.DataFrame(columns=list(LOST_TEST_COLUMNS)), members={},
            status=NOT_CONFIGURED,
            reason=("forensics.missingness.final_window_hours is not "
                    "configured; no dropout is defined"))
    work = prepare_series(series, "los")
    if work.empty:
        return LostVsRetainedResult(
            table=pd.DataFrame(columns=list(LOST_TABLE_COLUMNS)),
            tests=pd.DataFrame(columns=list(LOST_TEST_COLUMNS)), members={},
            status=STATUS_INSUFFICIENT, reason="no LOS observations")
    network_last = work["ts"].max()
    cutoff = network_last - pd.Timedelta(hours=final_hours)

    coords = coordinates.copy()
    coords["point_id"] = coords["point_id"].astype(str)
    final_counts = (work.assign(in_window=work["ts"] >= cutoff)
                    .groupby("point_id", sort=False)["in_window"].sum())
    rows: list[dict] = []
    for record in coords.to_dict("records"):
        point_id = str(record["point_id"])
        n_final = int(final_counts.get(point_id, 0))
        rows.append({
            "point_id": point_id,
            "east": float(record["east"]),
            "north": float(record["north"]),
            "distance_m": float("nan"),
            "lost": bool(n_final <= 0),
            "n_final_window": n_final,
            "role": ("no observation in the final window; unobserved, not "
                     "stable" if n_final <= 0 else "retained"),
            "status": STATUS_OK,
            "reason": ("network final record minus the configured drop-out "
                       "window"),
        })
    table = pd.DataFrame(rows, columns=list(LOST_TABLE_COLUMNS))
    if table.empty:
        return LostVsRetainedResult(
            table=table,
            tests=pd.DataFrame(columns=list(LOST_TEST_COLUMNS)), members={},
            status=STATUS_INSUFFICIENT, reason="no coordinates supplied")

    (center_east, center_north), source = _candidate_centre(
        table, work, cfg, network_last, final_hours)
    table["distance_m"] = np.hypot(table["east"] - center_east,
                                   table["north"] - center_north)

    lost_mask = table["lost"].to_numpy(dtype=bool)
    distances = table["distance_m"].to_numpy(dtype=float)
    lost_distances = distances[lost_mask]
    retained_distances = distances[~lost_mask]
    seed = (cfg.forensics.missingness or {}).get("seed", 1)
    permutation_n = (cfg.forensics.missingness or {}).get("permutation_n", 5000)

    statistic = float("nan")
    p_permutation = float("nan")
    p_mannwhitney = float("nan")
    if lost_distances.size and retained_distances.size:
        statistic = float(np.mean(lost_distances)
                          - np.mean(retained_distances))
        rng = np.random.default_rng(seed)
        extreme = 0
        for _ in range(int(permutation_n)):
            permuted = rng.permutation(lost_mask)
            difference = (float(np.mean(distances[permuted]))
                          - float(np.mean(distances[~permuted])))
            if abs(difference) >= abs(statistic) - 1e-12:
                extreme += 1
        p_permutation = (1 + extreme) / (1 + int(permutation_n))
        try:
            with np.errstate(invalid="ignore"):
                p_mannwhitney = float(mannwhitneyu(
                    lost_distances, retained_distances, alternative="two-sided"
                ).pvalue)
        except ValueError:
            p_mannwhitney = float("nan")

    tests = pd.DataFrame([{
        "statistic_mm": statistic,
        "p_permutation": p_permutation,
        "p_mannwhitney": p_mannwhitney,
        "n_lost": int(lost_mask.sum()),
        "n_retained": int((~lost_mask).sum()),
        "n_permutations": int(permutation_n),
        "seed": int(seed) if seed is not None else None,
        "candidate_source": source,
        "status": STATUS_EXPLORATORY,
        "reason": ("exploratory only: no decision significance is configured "
                   "and distance to the candidate zone is not a cause"),
    }], columns=list(LOST_TEST_COLUMNS))
    members = {
        "lost": sorted(table.loc[table["lost"], "point_id"].tolist()),
        "retained": sorted(table.loc[~table["lost"], "point_id"].tolist()),
    }
    return LostVsRetainedResult(
        table=table, tests=tests, members=members,
        status=STATUS_EXPLORATORY,
        reason=("permutation and Mann–Whitney p-values reported without a "
                "significance decision; candidate source: " + source))


# ---------------------------------------------------------------------------
#  C. Pre-loss trends
# ---------------------------------------------------------------------------
def pre_loss_trends(series: pd.DataFrame, *, config) -> PreLossTrendResult:
    """Theil-Sen trend and peer rank in the window before each prism's loss.

    The window is ``final_window_hours`` ending at the prism's own final record.
    ``net_mm`` is the median of the last quarter minus the median of the first
    quarter of the window; ``trend_mm_day`` is the Theil-Sen slope. The rank
    percentile compares ``net_mm`` with every other prism's own same-length
    window (an unavailable trend is excluded, never imputed).
    """
    cfg = resolve_config(config)
    final_hours = _final_window_hours(cfg)
    if final_hours is None:
        return PreLossTrendResult(
            table=pd.DataFrame(columns=list(PRE_LOSS_COLUMNS)), members={},
            status=NOT_CONFIGURED,
            reason=("forensics.missingness.final_window_hours is not "
                    "configured; no pre-loss window is invented"))
    work = prepare_series(series, "los")
    rows: list[dict] = []
    for point_id, group in work.groupby("point_id", sort=True):
        g = group.sort_values("ts")
        last = g["ts"].max()
        window = g.loc[g["ts"] >= last - pd.Timedelta(hours=final_hours)]
        row = {
            "point_id": str(point_id),
            "n_window": int(len(window)),
            "window_start": window["ts"].min() if not window.empty else pd.NaT,
            "window_end": last,
            "first_median_mm": float("nan"),
            "last_median_mm": float("nan"),
            "net_mm": float("nan"),
            "trend_mm_day": float("nan"),
            "trend_ci_low": float("nan"),
            "trend_ci_high": float("nan"),
            "rank_percentile": float("nan"),
            "n_peers": 0,
            "obs_ids": collect_ids(window),
            "status": STATUS_INSUFFICIENT,
            "reason": "",
        }
        if len(window) < MIN_WINDOW_POINTS:
            row["reason"] = (f"{len(window)} observation(s) in the "
                             f"{final_hours:g} h window before the final "
                             f"record; at least {MIN_WINDOW_POINTS} required")
            rows.append(row)
            continue
        quarter = max(1, len(window) // 4)
        ordered = window.sort_values("ts")
        row["first_median_mm"] = float(ordered["value"].head(quarter).median())
        row["last_median_mm"] = float(ordered["value"].tail(quarter).median())
        row["net_mm"] = row["last_median_mm"] - row["first_median_mm"]
        if len(window) >= MIN_RATE_POINTS:
            x = ((ordered["ts"] - ordered["ts"].min())
                 .dt.total_seconds().to_numpy() / 86400.0)
            slope, _intercept, low, high = theilslopes(
                ordered["value"].to_numpy(dtype=float), x,
                alpha=float(cfg.rates.confidence))
            row["trend_mm_day"] = float(slope)
            row["trend_ci_low"] = float(low)
            row["trend_ci_high"] = float(high)
        row["status"] = STATUS_OK
        row["reason"] = ("window ends at the prism's final record; an "
                         "unobserved later period is not stability")
        rows.append(row)

    table = pd.DataFrame(rows, columns=list(PRE_LOSS_COLUMNS))
    finite = table["net_mm"].to_numpy(dtype=float)
    for index in table.index:
        value = finite[index]
        if not np.isfinite(value):
            continue
        less = int(np.sum(finite < value))
        equal = int(np.sum(finite == value))
        n_peers = int(np.isfinite(finite).sum())
        if n_peers > 1:
            table.at[index, "rank_percentile"] = (
                100.0 * (less + 0.5 * equal) / n_peers)
            table.at[index, "n_peers"] = n_peers
    return PreLossTrendResult(
        table=table,
        members={"prisms": sorted(table["point_id"].tolist())},
        status=STATUS_OK if not table.empty else STATUS_INSUFFICIENT,
        reason=("comparison windows end at each prism's own final record; "
                "no decision threshold is configured"))


# ---------------------------------------------------------------------------
#  D. Observability by hour
# ---------------------------------------------------------------------------
def observability_by_hour(series: pd.DataFrame, *,
                          config) -> ObservabilityResult:
    """Observation success by hour class in the first and final windows.

    Expected cycles per class come from the network (distinct cycles observed
    by any prism), so a prism-only gap is visible. ``night_blind`` is exact and
    threshold-free: a prism had at least one night observation in the first
    window and none in the final window. Night-blindness is a pattern; the
    cause is not asserted and no external driver is supplied.
    """
    cfg = resolve_config(config)
    final_hours = _final_window_hours(cfg)
    if final_hours is None:
        return ObservabilityResult(
            table=pd.DataFrame(columns=list(OBSERVABILITY_COLUMNS)),
            by_hour=pd.DataFrame(columns=list(OBSERVABILITY_HOUR_COLUMNS)),
            members={}, status=NOT_CONFIGURED,
            reason=("forensics.missingness.final_window_hours is not "
                    "configured; no observability window is invented"))
    work = add_hour_class(prepare_series(series, "los"), cfg)
    if work.empty:
        return ObservabilityResult(
            table=pd.DataFrame(columns=list(OBSERVABILITY_COLUMNS)),
            by_hour=pd.DataFrame(columns=list(OBSERVABILITY_HOUR_COLUMNS)),
            members={}, status=STATUS_INSUFFICIENT, reason="no LOS observations")

    key_columns = _cycle_key_columns(work)
    network_first = work["ts"].min()
    network_last = work["ts"].max()
    window = pd.Timedelta(hours=final_hours)
    windows = {
        "first": work.loc[work["ts"] <= network_first + window],
        "final": work.loc[work["ts"] >= network_last - window],
    }
    expected: dict[str, dict[int, int]] = {}
    for name, subset in windows.items():
        cycles = subset.drop_duplicates([*key_columns, "hour_class"])
        counts = cycles.groupby("hour_class").size()
        expected[name] = {int(key): int(value)
                          for key, value in counts.items()}
    night = band_classes(cfg, "night")
    first_cutoff = network_first + window
    final_cutoff = network_last - window

    by_hour_rows: list[dict] = []
    rows: list[dict] = []
    for point_id, group in work.groupby("point_id", sort=True):
        observed: dict[str, dict[int, int]] = {}
        for name in ("first", "final"):
            if name == "first":
                subset = group.loc[group["ts"] <= first_cutoff]
            else:
                subset = group.loc[group["ts"] >= final_cutoff]
            counts = subset.groupby("hour_class").size()
            observed[name] = {int(key): int(value)
                              for key, value in counts.items()}
        for name in ("first", "final"):
            for hour_class in sorted(set(expected[name]) | set(observed[name])):
                n_expected = expected[name].get(hour_class, 0)
                n_observed = observed[name].get(hour_class, 0)
                success = (n_observed / n_expected
                           if n_expected > 0 else float("nan"))
                by_hour_rows.append({
                    "point_id": str(point_id),
                    "window": name,
                    "hour_class": hour_class,
                    "n_observed": n_observed,
                    "n_expected": n_expected,
                    "success": success,
                    "status": STATUS_OK if n_expected > 0
                    else STATUS_INSUFFICIENT,
                    "reason": ("network cycles observed in this class and "
                               "window form the denominator"),
                })
        first_night = sum(observed["first"].get(hour_class, 0)
                          for hour_class in night)
        final_night = sum(observed["final"].get(hour_class, 0)
                          for hour_class in night)
        first_expected = sum(expected["first"].get(hour_class, 0)
                             for hour_class in night)
        final_expected = sum(expected["final"].get(hour_class, 0)
                             for hour_class in night)
        night_blind = bool(first_night > 0 and final_night == 0)
        rows.append({
            "point_id": str(point_id),
            "night_obs_first": int(first_night),
            "night_obs_last": int(final_night),
            "night_success_first": (first_night / first_expected
                                    if first_expected else float("nan")),
            "night_success_last": (final_night / final_expected
                                   if final_expected else float("nan")),
            "night_blind": night_blind,
            "status": STATUS_OK,
            "reason": ("night observations present in the first window and "
                       "absent in the final window; a pattern only, the "
                       "cause is not asserted" if night_blind else
                       "no exact night-blind pattern detected"),
        })
    table = pd.DataFrame(rows, columns=list(OBSERVABILITY_COLUMNS))
    by_hour = pd.DataFrame(by_hour_rows,
                           columns=list(OBSERVABILITY_HOUR_COLUMNS))
    return ObservabilityResult(
        table=table, by_hour=by_hour,
        members={"prisms": sorted(table["point_id"].tolist())},
        status=STATUS_OK if not table.empty else STATUS_INSUFFICIENT,
        reason=("observability is a sampling pattern; no cause is asserted "
                "and no decision threshold is configured"))
