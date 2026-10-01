"""
events.py — time-of-day-balanced step tests and network common-mode step scan.

``balanced_step_test`` compares a statistic before and after a candidate time
using the configured ``forensics.events.balanced_windows`` (supplied
``[[16, 20, 0, 4], [20, 0, 4, 8]]``). Each side takes the closest sampling cycle
of every listed hour class, so the two windows have the same time-of-day
composition; the pair table exposes the actual classes and counts, and
``balanced`` is True only when both configured class lists were fully
available. Candidate times are configuration inputs
(``forensics.events.candidate_times``) or a boolean ``candidate``/
``candidate_event`` column on the series — never a hard-coded event date.

``common_mode_steps`` scans network 24 h block medians for common-mode steps and
reports candidate times, magnitudes and the competing explanations for each.
It never selects a single cause. The decision threshold
``forensics.events.step_significance`` is not configured in the supplied
configuration, so the status is ``not_configured`` and no step is declared
significant.

Input schema: documented in :mod:`rts_forensics.tests.groups`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import is_not_configured
from ..models import (
    INVESTIGATION_COLUMNS,
    NOT_CONFIGURED,
    STATUS_INSUFFICIENT,
    STATUS_OK,
)
from .groups import (
    add_hour_class,
    collect_ids,
    prepare_series,
    resolve_config,
)

BALANCED_STEP_COLUMNS = (
    "candidate_time", "point_id", "n_before", "n_after", "hours_before",
    "hours_after", "stat_before_mm", "stat_after_mm", "difference_mm",
    "obs_ids_before", "obs_ids_after", "status", "reason")
BALANCED_PAIR_COLUMNS = (
    "candidate_time", "n_prisms", "n_before", "n_after", "hours_before",
    "hours_after", "balanced", "median_before_mm", "median_after_mm",
    "difference_mm", "significant", "status", "reason")
COMMON_MODE_COLUMNS = (
    "from_time", "to_time", "magnitude_mm", "robust_z", "n_prisms_prev",
    "n_prisms_curr", "contributors_prev", "contributors_curr",
    "candidate_explanation", "competing_explanation", "significant",
    "status", "reason")

_CANDIDATE_EXPLANATION = (
    "candidate common-mode step in the network block medians; an instrument "
    "or frame change, a resection/station change and a real common movement "
    "are indistinguishable from these data alone")
_COMPETING_EXPLANATION = (
    "competing explanations: atmospheric/refraction change, changing prism "
    "membership across the blocks, a processing change, or a real "
    "common-mode ground movement")


@dataclass
class BalancedStepResult:
    """Balanced before/after step comparison; no cause is asserted."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    pairs: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


@dataclass
class CommonModeResult:
    """Network common-mode step candidates with competing explanations."""

    steps: pd.DataFrame = field(default_factory=pd.DataFrame)
    investigations: pd.DataFrame = field(default_factory=pd.DataFrame)
    members: dict[str, list[str]] = field(default_factory=dict)
    status: str = NOT_CONFIGURED
    reason: str | None = None


def _resolve_candidates(series: pd.DataFrame, events_cfg: dict) -> list:
    configured = events_cfg.get("candidate_times")
    if configured is not None and not is_not_configured(configured):
        if isinstance(configured, dict):
            configured = configured.get("times", [])
        return [pd.Timestamp(value) for value in configured]
    column = next((name for name in ("candidate", "candidate_event")
                   if name in series.columns), None)
    if column is not None:
        flagged = series.loc[series[column].astype(bool), "ts"]
        return [pd.Timestamp(value)
                for value in pd.to_datetime(flagged).dropna().unique()]
    return []


def _window_sample(group: pd.DataFrame, *, classes, before: bool,
                   candidate: pd.Timestamp) -> pd.DataFrame:
    """Closest cycle of each listed hour class on one side of the candidate."""
    selected = []
    for hour_class in classes:
        eligible = group.loc[(group["hour_class"] == hour_class)
                             & ((group["ts"] < candidate) if before
                                else (group["ts"] >= candidate))]
        if eligible.empty:
            continue
        if before:
            index = eligible["ts"].idxmax()
        else:
            index = eligible["ts"].idxmin()
        selected.append(eligible.loc[index])
    if not selected:
        return group.iloc[0:0]
    return pd.DataFrame(selected)


def balanced_step_test(series: pd.DataFrame, *,
                       config) -> BalancedStepResult:
    """Before/after comparison in time-of-day-balanced windows.

    ``forensics.events.balanced_windows`` supplies the two lists of hour-class
    starts (supplied ``[[16, 20, 0, 4], [20, 0, 4, 8]]``); each window takes
    the closest cycle of every listed class on its side of the candidate time.
    ``pairs.balanced`` requires every configured class to be present on both
    sides. The statistic is the window median of the configured metric (LOS by
    default) per prism and its network median. ``step_significance`` is not
    configured, so every pair carries ``status='not_configured'`` and no step
    is declared; when a minimum absolute difference is configured it is used
    exactly.
    """
    cfg = resolve_config(config)
    events_cfg = dict(cfg.forensics.events or {})
    windows = events_cfg.get("balanced_windows")
    if is_not_configured(windows) or not isinstance(windows, (list, tuple)) \
            or len(windows) < 2:
        return BalancedStepResult(
            table=pd.DataFrame(columns=list(BALANCED_STEP_COLUMNS)),
            pairs=pd.DataFrame(columns=list(BALANCED_PAIR_COLUMNS)),
            members={}, status=NOT_CONFIGURED,
            reason=("forensics.events.balanced_windows is not configured; no "
                    "time-of-day matching is invented"))
    before_classes = [int(value) for value in windows[0]]
    after_classes = [int(value) for value in windows[1]]
    metric = events_cfg.get("metric", "los")
    candidates = _resolve_candidates(series, events_cfg)
    if not candidates:
        return BalancedStepResult(
            table=pd.DataFrame(columns=list(BALANCED_STEP_COLUMNS)),
            pairs=pd.DataFrame(columns=list(BALANCED_PAIR_COLUMNS)),
            members={}, status=NOT_CONFIGURED,
            reason=("no candidate time supplied (forensics.events."
                    "candidate_times or a candidate column); no event date is "
                    "hard-coded"))

    work = add_hour_class(prepare_series(series, metric), cfg)
    decision = events_cfg.get("step_significance")
    decision_configured = not is_not_configured(decision)

    table_rows: list[dict] = []
    pair_rows: list[dict] = []
    for candidate in candidates:
        per_prism: list[dict] = []
        for point_id, group in work.groupby("point_id", sort=True):
            before = _window_sample(group, classes=before_classes,
                                    before=True, candidate=candidate)
            after = _window_sample(group, classes=after_classes,
                                   before=False, candidate=candidate)
            row = {
                "candidate_time": candidate,
                "point_id": str(point_id),
                "n_before": int(len(before)),
                "n_after": int(len(after)),
                "hours_before": tuple(sorted(
                    before["hour_class"].tolist())),
                "hours_after": tuple(sorted(after["hour_class"].tolist())),
                "stat_before_mm": (float(before["value"].median())
                                   if not before.empty else float("nan")),
                "stat_after_mm": (float(after["value"].median())
                                  if not after.empty else float("nan")),
                "difference_mm": float("nan"),
                "obs_ids_before": collect_ids(before),
                "obs_ids_after": collect_ids(after),
                "status": STATUS_OK,
                "reason": ("closest cycle of each configured hour class on "
                           "each side of the candidate time"),
            }
            if (not before.empty and not after.empty):
                row["difference_mm"] = (row["stat_after_mm"]
                                        - row["stat_before_mm"])
            else:
                row["status"] = STATUS_INSUFFICIENT
                row["reason"] = (
                    "one side has no observation in the configured hour "
                    "classes; the difference is not computed")
            per_prism.append(row)
            table_rows.append(row)
        differences = [row["difference_mm"] for row in per_prism
                       if np.isfinite(row["difference_mm"])]
        hours_before = {hour for row in per_prism for hour in row["hours_before"]}
        hours_after = {hour for row in per_prism for hour in row["hours_after"]}
        balanced = bool(
            per_prism
            and all(set(row["hours_before"]) == set(before_classes)
                    for row in per_prism)
            and all(set(row["hours_after"]) == set(after_classes)
                    for row in per_prism)
            and len(before_classes) == len(after_classes))
        difference = float(np.median(differences)) if differences else float("nan")
        if decision_configured and np.isfinite(difference):
            significant = bool(abs(difference) >= float(decision))
            status = STATUS_OK
            reason = (f"minimum absolute difference {float(decision):g} "
                      "supplied by configuration")
        else:
            significant = False
            status = NOT_CONFIGURED
            reason = ("forensics.events.step_significance is not configured; "
                      "the balanced difference is a statistic only")
        pair_rows.append({
            "candidate_time": candidate,
            "n_prisms": len(per_prism),
            "n_before": int(sum(row["n_before"] for row in per_prism)),
            "n_after": int(sum(row["n_after"] for row in per_prism)),
            "hours_before": tuple(sorted(hours_before)),
            "hours_after": tuple(sorted(hours_after)),
            "balanced": balanced,
            "median_before_mm": (float(np.median(
                [row["stat_before_mm"] for row in per_prism
                 if np.isfinite(row["stat_before_mm"])]))
                if any(np.isfinite(row["stat_before_mm"])
                       for row in per_prism) else float("nan")),
            "median_after_mm": (float(np.median(
                [row["stat_after_mm"] for row in per_prism
                 if np.isfinite(row["stat_after_mm"])]))
                if any(np.isfinite(row["stat_after_mm"])
                       for row in per_prism) else float("nan")),
            "difference_mm": difference,
            "significant": significant,
            "status": status,
            "reason": reason,
        })

    table = pd.DataFrame(table_rows, columns=list(BALANCED_STEP_COLUMNS))
    pairs = pd.DataFrame(pair_rows, columns=list(BALANCED_PAIR_COLUMNS))
    if decision_configured:
        overall = STATUS_OK
        reason = "configured step comparison; time-of-day composition recorded"
    else:
        overall = NOT_CONFIGURED
        reason = ("step_significance is not configured; the balanced step "
                  "difference is reported without a decision")
    return BalancedStepResult(
        table=table, pairs=pairs,
        members={"prisms": sorted(table["point_id"].unique().tolist())},
        status=overall, reason=reason)


def common_mode_steps(series: pd.DataFrame, *,
                      config) -> CommonModeResult:
    """Network common-mode step scan on 24 h block medians.

    Each block is the median of a prism's LOS; the network common mode is the
    median across prisms present in that block. Steps between consecutive
    blocks are standardised by ``1.4826·MAD`` of the common-mode series, and
    every step records the contributing prisms of both blocks so changing
    membership is visible. Candidate times, magnitudes and the competing
    explanations are returned; no single cause is selected. Without a
    configured ``step_significance`` the status is ``not_configured``.
    """
    cfg = resolve_config(config)
    events_cfg = dict(cfg.forensics.events or {})
    block_hours = cfg.rates.block_hours
    if is_not_configured(block_hours):
        return CommonModeResult(
            steps=pd.DataFrame(columns=list(COMMON_MODE_COLUMNS)),
            investigations=pd.DataFrame(),
            members={}, status=NOT_CONFIGURED,
            reason="rates.block_hours is not configured; no block length is "
                   "invented")
    metric = events_cfg.get("metric", "los")
    work = prepare_series(series, metric)
    if work.empty:
        return CommonModeResult(
            steps=pd.DataFrame(columns=list(COMMON_MODE_COLUMNS)),
            investigations=pd.DataFrame(),
            members={}, status=STATUS_INSUFFICIENT, reason="no observations")

    anchor = work["ts"].max()
    delta_hours = (anchor - work["ts"]).dt.total_seconds() / 3600.0
    work = work.assign(
        block_id=np.floor(delta_hours / float(block_hours) + 1e-9).astype(int))
    pivot = work.pivot_table(index="block_id", columns="point_id",
                             values="value", aggfunc="median")
    pivot = pivot.sort_index(ascending=False)
    common = pivot.median(axis=1, skipna=True)
    scale_values = common.to_numpy(dtype=float)
    if len(scale_values) >= 3:
        robust_scale = float(1.4826 * np.median(
            np.abs(scale_values - np.median(scale_values))))
    else:
        robust_scale = float("nan")

    decision = events_cfg.get("step_significance")
    decision_configured = not is_not_configured(decision)
    rows: list[dict] = []
    investigations: list[dict] = []
    indices = list(common.index)
    for previous, current in zip(indices[:-1], indices[1:], strict=True):
        magnitude = float(common.loc[current] - common.loc[previous])
        robust_z = (magnitude / robust_scale
                    if np.isfinite(robust_scale) and robust_scale > 0
                    else float("nan"))
        previous_series = pivot.loc[previous].dropna()
        current_series = pivot.loc[current].dropna()
        from_time = anchor - pd.to_timedelta(previous * float(block_hours),
                                             unit="h")
        to_time = anchor - pd.to_timedelta(current * float(block_hours),
                                           unit="h")
        if decision_configured and np.isfinite(robust_z):
            significant = bool(abs(robust_z) >= float(decision))
            status = STATUS_OK
            reason = (f"minimum |robust z| {float(decision):g} supplied by "
                      "configuration")
        else:
            significant = False
            status = NOT_CONFIGURED
            reason = ("forensics.events.step_significance is not configured; "
                      "the common-mode step is a candidate, not a decision")
        rows.append({
            "from_time": from_time,
            "to_time": to_time,
            "magnitude_mm": magnitude,
            "robust_z": robust_z,
            "n_prisms_prev": int(len(previous_series)),
            "n_prisms_curr": int(len(current_series)),
            "contributors_prev": tuple(previous_series.index.astype(str)),
            "contributors_curr": tuple(current_series.index.astype(str)),
            "candidate_explanation": _CANDIDATE_EXPLANATION,
            "competing_explanation": _COMPETING_EXPLANATION,
            "significant": significant,
            "status": status,
            "reason": reason,
        })
        investigations.append({
            "investigation_id": f"common_mode_step:{to_time}",
            "opened": to_time,
            "topic": "network common-mode step candidate",
            "candidate_explanation": _CANDIDATE_EXPLANATION,
            "competing_explanation": _COMPETING_EXPLANATION,
            "evidence": f"events.common_mode_steps@{to_time}",
            "status": "open" if decision_configured else NOT_CONFIGURED,
        })
    steps = pd.DataFrame(rows, columns=list(COMMON_MODE_COLUMNS))
    if not steps.empty:
        steps = (steps.assign(_abs_z=steps["robust_z"].abs())
                 .sort_values("_abs_z", ascending=False)
                 .drop(columns="_abs_z")
                 .reset_index(drop=True))
    investigation_frame = pd.DataFrame(
        investigations, columns=list(INVESTIGATION_COLUMNS))
    status = (STATUS_OK if decision_configured and not steps.empty
              else STATUS_INSUFFICIENT if steps.empty else NOT_CONFIGURED)
    reason = ("step significance is not configured; every step is a candidate "
              "with competing explanations and no single cause is selected"
              if not decision_configured else
              "configured step comparison reported")
    return CommonModeResult(
        steps=steps, investigations=investigation_frame,
        members={"prisms": sorted(work["point_id"].unique().tolist())},
        status=status, reason=reason)
