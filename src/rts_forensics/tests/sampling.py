"""
sampling.py — time-of-day sampling tests (day/night, hour matching, bias).

Every window length is read from configuration: the comparison window is
``baseline.window_hours`` (supplied 48 h), the hour classes come from
``forensics.hour_class_hours`` (supplied 4 h) and the bands from
``forensics.day_bands`` (supplied night 20/00/04 and day 08/12/16 cycle
starts). The daytime-bias test uses the configured Theil-Sen confidence for
its night-only trend. No decision threshold is invented: the tests report
statistics and composition, and the reason text states that an association is
not a cause. Periods with no observations are described as unobserved, never
as stable.

Input schema is documented in :mod:`rts_forensics.tests.groups`
(``prepare_series``): one row per prism per cycle with ``point_id``, ``ts`` and
a documented LOS column, optionally ``cycle_id``/``station_id``, ``obs_ids``,
``az_deg``/``azimuth_deg`` and ``v_deg``/``zenith_deg``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from ..models import NOT_CONFIGURED, STATUS_INSUFFICIENT, STATUS_OK
from .groups import (
    MIN_RATE_POINTS,
    add_hour_class,
    band_classes,
    collect_ids,
    prepare_series,
    resolve_config,
    window_hours,
)


# ---------------------------------------------------------------------------
#  Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DayNightResult:
    """Day/night net change plus the night share of both windows."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    composition: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


@dataclass
class HourMatchedResult:
    """Hour-class-matched net change per prism and the classes used."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    classes: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_classes: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


@dataclass
class CompositionResult:
    """Observed-hour composition of the first and final windows."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


@dataclass
class DaytimeBiasResult:
    """Cycle-detrended day-minus-night bias with weekly and geometry controls."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    weekly: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


DAY_NIGHT_COLUMNS = (
    "point_id", "band", "classes", "n_baseline", "n_final",
    "baseline_median_mm", "final_median_mm", "net_mm",
    "obs_ids_baseline", "obs_ids_final", "status", "reason")
COMPOSITION_COLUMNS = (
    "point_id", "n_baseline", "n_final", "night_classes",
    "night_share_baseline", "night_share_final", "night_share_delta",
    "classes_baseline", "classes_final", "classes_added", "classes_removed",
    "composition_changed", "status", "reason")
HOUR_MATCHED_COLUMNS = (
    "point_id", "n_classes", "classes_used", "class_nets", "net_mm",
    "n_baseline", "n_final", "obs_ids_baseline", "obs_ids_final",
    "status", "reason")
HOUR_CLASS_COLUMNS = (
    "point_id", "hour_class", "n_baseline", "n_final",
    "baseline_median_mm", "final_median_mm", "net_mm", "status", "reason")
DAYTIME_BIAS_COLUMNS = (
    "point_id", "n_day", "n_night", "day_median_mm", "night_median_mm",
    "day_minus_night_mm", "weekly_median_mm", "n_weeks",
    "weekly_sign_consistency", "night_trend_mm_day", "night_trend_ci_low",
    "night_trend_ci_high", "control_bias_mm", "control_delta_mm",
    "control_ids", "status", "reason")
WEEKLY_COLUMNS = (
    "point_id", "week", "n_day", "n_night", "day_median_mm",
    "night_median_mm", "day_minus_night_mm", "status", "reason")


def _window_masks(work: pd.DataFrame, hours: float) -> tuple[pd.Series,
                                                             pd.Series]:
    first = work.groupby("point_id", sort=False)["ts"].transform("min")
    last = work.groupby("point_id", sort=False)["ts"].transform("max")
    window = pd.Timedelta(hours=float(hours))
    baseline = work["ts"] <= first + window
    final = work["ts"] >= last - window
    return baseline, final


# ---------------------------------------------------------------------------
#  A. Day/night comparison
# ---------------------------------------------------------------------------
def day_night_comparison(series: pd.DataFrame, *,
                         config) -> DayNightResult:
    """Net change per prism by configured night/day bands.

    For each band the net change is the median of the final window minus the
    median of the baseline window restricted to the band's hour classes.
    ``composition`` records the night share of the first and final windows so a
    comparison based on changing sampling is visible. A band with no final
    observations is ``insufficient_data``: unobserved hours are not evidence of
    stability. No decision threshold is configured.
    """
    cfg = resolve_config(config)
    hours = window_hours(cfg)
    bands = dict(cfg.forensics.day_bands or {})
    if hours is None or not bands:
        return DayNightResult(
            table=pd.DataFrame(columns=list(DAY_NIGHT_COLUMNS)),
            composition=pd.DataFrame(columns=list(COMPOSITION_COLUMNS)),
            members={}, status=NOT_CONFIGURED,
            reason="window/band configuration missing; no day/night decision "
                   "is invented")

    work = add_hour_class(prepare_series(series, "los"), cfg)
    baseline_mask, final_mask = _window_masks(work, hours)
    night = band_classes(cfg, "night")

    rows: list[dict] = []
    composition_rows: list[dict] = []
    for point_id, group in work.groupby("point_id", sort=True):
        group_baseline = baseline_mask.reindex(group.index, fill_value=False)
        group_final = final_mask.reindex(group.index, fill_value=False)
        baseline_all = group.loc[group_baseline]
        final_all = group.loc[group_final]
        baseline_night = baseline_all.loc[
            baseline_all["hour_class"].isin(night)]
        final_night = final_all.loc[final_all["hour_class"].isin(night)]
        baseline_share = (
            float(len(baseline_night) / len(baseline_all))
            if len(baseline_all) else float("nan"))
        final_share = (
            float(len(final_night) / len(final_all))
            if len(final_all) else float("nan"))
        delta = (final_share - baseline_share
                 if np.isfinite(final_share) and np.isfinite(baseline_share)
                 else float("nan"))
        composition_rows.append({
            "point_id": str(point_id),
            "n_baseline": int(len(baseline_all)),
            "n_final": int(len(final_all)),
            "night_classes": tuple(sorted(night)),
            "night_share_baseline": baseline_share,
            "night_share_final": final_share,
            "night_share_delta": delta,
            "classes_baseline": tuple(sorted(
                baseline_all["hour_class"].unique().tolist())),
            "classes_final": tuple(sorted(
                final_all["hour_class"].unique().tolist())),
            "classes_added": tuple(sorted(
                set(final_all["hour_class"]) - set(baseline_all["hour_class"]))),
            "classes_removed": tuple(sorted(
                set(baseline_all["hour_class"]) - set(final_all["hour_class"]))),
            "composition_changed": bool(
                set(baseline_all["hour_class"]) != set(final_all["hour_class"])),
            "status": STATUS_OK,
            "reason": ("night share of the first and final windows; a change "
                       "means day/night comparisons are not hour-matched"),
        })
        for band in ("night", "day"):
            classes = band_classes(cfg, band)
            baseline_band = group.loc[
                group_baseline & group["hour_class"].isin(classes)]
            final_band = group.loc[
                group_final & group["hour_class"].isin(classes)]
            row = {
                "point_id": str(point_id),
                "band": band,
                "classes": tuple(sorted(classes)),
                "n_baseline": int(len(baseline_band)),
                "n_final": int(len(final_band)),
                "baseline_median_mm": float("nan"),
                "final_median_mm": float("nan"),
                "net_mm": float("nan"),
                "obs_ids_baseline": collect_ids(baseline_band),
                "obs_ids_final": collect_ids(final_band),
                "status": STATUS_INSUFFICIENT,
                "reason": "",
            }
            if baseline_band.empty or final_band.empty:
                missing = []
                if baseline_band.empty:
                    missing.append("baseline")
                if final_band.empty:
                    missing.append("final")
                row["reason"] = (
                    f"no {band} observations in the "
                    f"{' and '.join(missing)} {hours:g} h window; unobserved "
                    "hours are not evidence of stability")
            else:
                row["baseline_median_mm"] = float(
                    baseline_band["value"].median())
                row["final_median_mm"] = float(final_band["value"].median())
                row["net_mm"] = (row["final_median_mm"]
                                 - row["baseline_median_mm"])
                row["status"] = STATUS_OK
            rows.append(row)

    table = pd.DataFrame(rows, columns=list(DAY_NIGHT_COLUMNS))
    composition = pd.DataFrame(composition_rows,
                               columns=list(COMPOSITION_COLUMNS))
    return DayNightResult(
        table=table, composition=composition,
        members={"prisms": sorted(table["point_id"].unique().tolist())},
        status=STATUS_OK if not table.empty else STATUS_INSUFFICIENT,
        reason=("band medians within the same window; changing night/day "
                "composition is reported and is not treated as movement"))


# ---------------------------------------------------------------------------
#  B. Hour-matched change
# ---------------------------------------------------------------------------
def hour_matched_change(series: pd.DataFrame, *,
                        config) -> HourMatchedResult:
    """Net change within each configured hour class, then median across classes.

    The baseline window is the first ``baseline.window_hours`` of the prism and
    the final window is its last ``baseline.window_hours``; a class is used only
    when it has observations in both windows. ``n_classes`` exposes how many
    classes entered the median, and ``classes`` records the per-class net
    changes and contributors. No decision threshold is configured.
    """
    cfg = resolve_config(config)
    hours = window_hours(cfg)
    if hours is None:
        return HourMatchedResult(
            table=pd.DataFrame(columns=list(HOUR_MATCHED_COLUMNS)),
            classes=pd.DataFrame(columns=list(HOUR_CLASS_COLUMNS)),
            n_classes=pd.DataFrame(columns=["point_id", "n_classes"]),
            members={}, status=NOT_CONFIGURED,
            reason="baseline.window_hours is not configured; no comparison "
                   "window is invented")

    work = add_hour_class(prepare_series(series, "los"), cfg)
    baseline_mask, final_mask = _window_masks(work, hours)
    class_rows: list[dict] = []
    rows: list[dict] = []
    for point_id, group in work.groupby("point_id", sort=True):
        group_baseline = baseline_mask.reindex(group.index, fill_value=False)
        group_final = final_mask.reindex(group.index, fill_value=False)
        nets: list[float] = []
        used: list[int] = []
        baseline_all = group.loc[group_baseline]
        final_all = group.loc[group_final]
        for hour_class in sorted(group["hour_class"].unique().tolist()):
            baseline_class = group.loc[
                group_baseline & (group["hour_class"] == hour_class)]
            final_class = group.loc[
                group_final & (group["hour_class"] == hour_class)]
            class_row = {
                "point_id": str(point_id),
                "hour_class": int(hour_class),
                "n_baseline": int(len(baseline_class)),
                "n_final": int(len(final_class)),
                "baseline_median_mm": float("nan"),
                "final_median_mm": float("nan"),
                "net_mm": float("nan"),
                "status": STATUS_INSUFFICIENT,
                "reason": ("no observations in the baseline or final window "
                           "for this class"),
            }
            if not baseline_class.empty and not final_class.empty:
                class_row["baseline_median_mm"] = float(
                    baseline_class["value"].median())
                class_row["final_median_mm"] = float(
                    final_class["value"].median())
                class_row["net_mm"] = (class_row["final_median_mm"]
                                       - class_row["baseline_median_mm"])
                class_row["status"] = STATUS_OK
                class_row["reason"] = "same-hour baseline and final windows"
                nets.append(class_row["net_mm"])
                used.append(int(hour_class))
            class_rows.append(class_row)
        row = {
            "point_id": str(point_id),
            "n_classes": len(nets),
            "classes_used": tuple(used),
            "class_nets": tuple(nets),
            "net_mm": float(np.median(nets)) if nets else float("nan"),
            "n_baseline": int(len(baseline_all)),
            "n_final": int(len(final_all)),
            "obs_ids_baseline": collect_ids(baseline_all),
            "obs_ids_final": collect_ids(final_all),
            "status": STATUS_OK if nets else STATUS_INSUFFICIENT,
            "reason": (f"median across {len(nets)} hour class(es); classes "
                       "without observations in both windows are excluded, "
                       "never interpolated"),
        }
        rows.append(row)

    table = pd.DataFrame(rows, columns=list(HOUR_MATCHED_COLUMNS))
    classes = pd.DataFrame(class_rows, columns=list(HOUR_CLASS_COLUMNS))
    n_classes = table[["point_id", "n_classes"]].copy()
    return HourMatchedResult(
        table=table, classes=classes, n_classes=n_classes,
        members={"prisms": sorted(table["point_id"].unique().tolist())},
        status=STATUS_OK if not table.empty else STATUS_INSUFFICIENT,
        reason=("hour-matched comparison removes sampling-composition "
                "artefacts; no decision threshold is configured"))


# ---------------------------------------------------------------------------
#  C. Sampling composition
# ---------------------------------------------------------------------------
def sampling_composition(series: pd.DataFrame, *,
                         config) -> CompositionResult:
    """Night share and observed-hour composition of the first/final windows.

    ``composition_changed`` is exact and threshold-free: it is True when the
    set of observed hour classes differs between the prism's first and final
    windows. Night shares are reported alongside. No decision significance is
    configured, so the change flag is descriptive only.
    """
    cfg = resolve_config(config)
    hours = window_hours(cfg)
    if hours is None:
        return CompositionResult(
            table=pd.DataFrame(columns=list(COMPOSITION_COLUMNS)), members={},
            status=NOT_CONFIGURED,
            reason="baseline.window_hours is not configured; no comparison "
                   "window is invented")
    work = add_hour_class(prepare_series(series, "los"), cfg)
    baseline_mask, final_mask = _window_masks(work, hours)
    night = band_classes(cfg, "night")
    rows: list[dict] = []
    for point_id, group in work.groupby("point_id", sort=True):
        baseline = group.loc[baseline_mask.reindex(group.index,
                                                   fill_value=False)]
        final = group.loc[final_mask.reindex(group.index, fill_value=False)]
        baseline_classes = set(baseline["hour_class"].tolist())
        final_classes = set(final["hour_class"].tolist())
        baseline_night = baseline.loc[baseline["hour_class"].isin(night)]
        final_night = final.loc[final["hour_class"].isin(night)]
        baseline_share = (len(baseline_night) / len(baseline)
                          if len(baseline) else float("nan"))
        final_share = (len(final_night) / len(final)
                       if len(final) else float("nan"))
        delta = (final_share - baseline_share
                 if np.isfinite(final_share) and np.isfinite(baseline_share)
                 else float("nan"))
        changed = baseline_classes != final_classes
        rows.append({
            "point_id": str(point_id),
            "n_baseline": int(len(baseline)),
            "n_final": int(len(final)),
            "night_classes": tuple(sorted(night)),
            "night_share_baseline": baseline_share,
            "night_share_final": final_share,
            "night_share_delta": delta,
            "classes_baseline": tuple(sorted(baseline_classes)),
            "classes_final": tuple(sorted(final_classes)),
            "classes_added": tuple(sorted(final_classes - baseline_classes)),
            "classes_removed": tuple(sorted(baseline_classes - final_classes)),
            "composition_changed": bool(changed),
            "status": STATUS_OK,
            "reason": ("observed-hour composition changed between the first "
                       "and final windows; a raw net change is not "
                       "hour-matched" if changed else
                       "observed-hour composition is unchanged; a raw net "
                       "change is still not a significance decision"),
        })
    table = pd.DataFrame(rows, columns=list(COMPOSITION_COLUMNS))
    return CompositionResult(
        table=table,
        members={"prisms": sorted(table["point_id"].unique().tolist())},
        status=STATUS_OK if not table.empty else STATUS_INSUFFICIENT,
        reason=("descriptive composition comparison; no significance "
                "threshold is configured"))


# ---------------------------------------------------------------------------
#  D. Daytime bias
# ---------------------------------------------------------------------------
def _geometry_control(work: pd.DataFrame, biases: dict[str, float],
                      config) -> tuple[dict[str, float], dict[str, tuple]]:
    """Same-geometry control: nearest neighbours in (azimuth, zenith) space."""
    k = config.forensics.placebo_neighbors
    if k is None or int(k) < 1:
        return {}, {}
    azimuth_column = next(
        (column for column in ("az_deg", "azimuth_deg")
         if column in work.columns), None)
    zenith_column = next(
        (column for column in ("v_deg", "zenith_deg") if column in work.columns),
        None)
    if azimuth_column is None or zenith_column is None:
        return {}, {}
    geometry = (work.groupby("point_id", sort=True)[[azimuth_column,
                                                     zenith_column]]
                .median())
    ids = geometry.index.astype(str).tolist()
    az = np.radians(geometry[azimuth_column].to_numpy(dtype=float))
    zenith = np.radians(geometry[zenith_column].to_numpy(dtype=float))
    unit = np.column_stack([np.sin(zenith) * np.sin(az),
                            np.sin(zenith) * np.cos(az),
                            np.cos(zenith)])
    dot = np.clip(unit @ unit.T, -1.0, 1.0)
    angular = np.arccos(dot)
    np.fill_diagonal(angular, np.inf)
    control: dict[str, float] = {}
    control_ids: dict[str, tuple] = {}
    for index, point_id in enumerate(ids):
        order = [position for position in np.argsort(angular[index],
                                                     kind="stable")
                 if np.isfinite(angular[index, position])]
        neighbours = [ids[position] for position in order[:int(k)]]
        values = np.array([biases.get(neighbour, np.nan)
                           for neighbour in neighbours], dtype=float)
        control[point_id] = (float(np.nanmedian(values))
                             if np.isfinite(values).any() else float("nan"))
        control_ids[point_id] = tuple(neighbours)
    return control, control_ids


def daytime_bias(series: pd.DataFrame, *, config) -> DaytimeBiasResult:
    """Cycle-detrended day-minus-night LOS bias per prism.

    Each cycle's median across prisms is subtracted before the per-prism
    day-minus-night median is taken (``day_minus_night_mm``). Weekly stability
    is the distribution of the same contrast in 7-day epochs; the night-only
    trend is a Theil-Sen slope on night residuals; the same-geometry control is
    the median bias of the nearest neighbours in azimuth/zenith space. The
    result reports associations only: a daytime reading difference is never
    asserted to be the cause of movement, and external drivers are not
    supplied. No decision threshold is configured.
    """
    cfg = resolve_config(config)
    work = add_hour_class(prepare_series(series, "los"), cfg)
    if work.empty:
        return DaytimeBiasResult(
            table=pd.DataFrame(columns=list(DAYTIME_BIAS_COLUMNS)),
            weekly=pd.DataFrame(columns=list(WEEKLY_COLUMNS)), members={},
            status=STATUS_INSUFFICIENT, reason="no LOS observations")

    if "cycle_id" in work.columns:
        keys = ["station_id", "cycle_id"] if "station_id" in work.columns \
            else ["cycle_id"]
    else:
        keys = ["ts"]
    cycle_median = work.groupby(keys, sort=False)["value"].transform("median")
    work = work.assign(detrended=work["value"] - cycle_median)
    night = band_classes(cfg, "night")
    day = band_classes(cfg, "day")

    first_time = work["ts"].min()
    work = work.assign(
        week=((work["ts"] - first_time).dt.total_seconds() / 86400.0
              // 7).astype("Int64"))

    weekly_rows: list[dict] = []
    biases: dict[str, float] = {}
    for point_id, group in work.groupby("point_id", sort=True):
        for week, week_group in group.groupby("week", sort=True):
            day_group = week_group.loc[week_group["hour_class"].isin(day)]
            night_group = week_group.loc[
                week_group["hour_class"].isin(night)]
            if day_group.empty or night_group.empty:
                weekly_rows.append({
                    "point_id": str(point_id), "week": int(week),
                    "n_day": int(len(day_group)), "n_night": int(len(night_group)),
                    "day_median_mm": float("nan"),
                    "night_median_mm": float("nan"),
                    "day_minus_night_mm": float("nan"),
                    "status": STATUS_INSUFFICIENT,
                    "reason": "no day or no night observations in this week",
                })
            else:
                weekly_rows.append({
                    "point_id": str(point_id), "week": int(week),
                    "n_day": int(len(day_group)), "n_night": int(len(night_group)),
                    "day_median_mm": float(day_group["detrended"].median()),
                    "night_median_mm": float(night_group["detrended"].median()),
                    "day_minus_night_mm": float(
                        day_group["detrended"].median()
                        - night_group["detrended"].median()),
                    "status": STATUS_OK,
                    "reason": "7-day epoch of the cycle-detrended contrast",
                })

    weekly = pd.DataFrame(weekly_rows, columns=list(WEEKLY_COLUMNS))
    rows: list[dict] = []
    for point_id, group in work.groupby("point_id", sort=True):
        day_group = group.loc[group["hour_class"].isin(day)]
        night_group = group.loc[group["hour_class"].isin(night)]
        bias = (float(day_group["detrended"].median()
                      - night_group["detrended"].median())
                if not day_group.empty and not night_group.empty
                else float("nan"))
        if np.isfinite(bias):
            biases[str(point_id)] = bias
        weekly_group = weekly.loc[
            (weekly["point_id"] == str(point_id))
            & weekly["day_minus_night_mm"].notna()]
        weekly_median = (float(weekly_group["day_minus_night_mm"].median())
                         if not weekly_group.empty else float("nan"))
        if not weekly_group.empty and np.isfinite(bias) and bias != 0.0:
            consistency = float(
                (np.sign(weekly_group["day_minus_night_mm"]) == np.sign(bias))
                .mean())
        else:
            consistency = float("nan")

        night_values = night_group.loc[night_group["detrended"].notna()]
        if len(night_values) >= MIN_RATE_POINTS:
            x = ((night_values["ts"] - night_values["ts"].min())
                 .dt.total_seconds().to_numpy() / 86400.0)
            slope, _intercept, low, high = theilslopes(
                night_values["detrended"].to_numpy(dtype=float), x,
                alpha=float(cfg.rates.confidence))
        else:
            slope = low = high = float("nan")

        rows.append({
            "point_id": str(point_id),
            "n_day": int(len(day_group)),
            "n_night": int(len(night_group)),
            "day_median_mm": (float(day_group["detrended"].median())
                              if not day_group.empty else float("nan")),
            "night_median_mm": (float(night_group["detrended"].median())
                                if not night_group.empty else float("nan")),
            "day_minus_night_mm": bias,
            "weekly_median_mm": weekly_median,
            "n_weeks": int(len(weekly_group)),
            "weekly_sign_consistency": consistency,
            "night_trend_mm_day": float(slope),
            "night_trend_ci_low": float(low),
            "night_trend_ci_high": float(high),
            "control_bias_mm": float("nan"),
            "control_delta_mm": float("nan"),
            "control_ids": (),
            "status": STATUS_OK if np.isfinite(bias) else STATUS_INSUFFICIENT,
            "reason": ("cycle-detrended association only: a daytime reading "
                       "difference is not asserted to be the cause and no "
                       "external driver is supplied"),
        })

    table = pd.DataFrame(rows, columns=list(DAYTIME_BIAS_COLUMNS))
    control, control_ids = _geometry_control(work, biases, cfg)
    if control:
        for index in table.index:
            point_id = table.at[index, "point_id"]
            if point_id in control:
                table.at[index, "control_bias_mm"] = control[point_id]
                table.at[index, "control_ids"] = control_ids[point_id]
                if (np.isfinite(table.at[index, "day_minus_night_mm"])
                        and np.isfinite(control[point_id])):
                    table.at[index, "control_delta_mm"] = float(
                        table.at[index, "day_minus_night_mm"]
                        - control[point_id])
    else:
        table["reason"] = table["reason"] + (
            "; same-geometry control unavailable (azimuth/zenith geometry or "
            "placebo_neighbors not supplied)")
    return DaytimeBiasResult(
        table=table, weekly=weekly,
        members={"prisms": sorted(table["point_id"].unique().tolist())},
        status=STATUS_OK if not table.empty else STATUS_INSUFFICIENT,
        reason=("cycle-detrending removes a per-cycle common mode; the "
                "day-minus-night contrast remains associative, not causal"))
