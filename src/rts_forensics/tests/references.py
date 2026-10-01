"""
references.py — resection-reference inference and resection validation.

Both tests work from the documented ``series`` schema (see
:mod:`rts_forensics.tests.groups`) and a documented ``station_series``:

``station_series`` schema
-------------------------
One row per station cycle. It must provide ``ts`` and/or ``cycle_id``; when the
series also carries ``station_id`` the merge additionally matches on it.

* height — one of ``height``, ``sh_m``, ``height_m``, ``station_height``,
  ``station_height_m`` (metres, converted to millimetres internally) or
  ``height_mm``/``sh_mm`` (already millimetres);
* orientation — one of ``orientation_deg``, ``orientation``, ``hz_deg``
  (degrees).

``reference_inference`` subtracts the per-cycle median of the vertical series
(and the per-cycle circular mean of the Hz series) before correlating each
prism with the station height and orientation. Bonferroni correction uses
``forensics.references.alpha`` and ``forensics.references.bonferroni``.
``resection_validation`` regresses the raw common vertical on the exported
station height and compares the slope with the configured expected slope
(supplied: ``-1.0``) — a real station rise lowers the raw vertical by the same
amount. No threshold is invented; the supplied alpha and expected slope are
read from configuration.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr, t

from ..config import is_not_configured
from ..geometry import angular_difference_deg
from ..models import NOT_CONFIGURED, STATUS_INSUFFICIENT, STATUS_OK
from .groups import (
    prepare_series,
    require_columns,
    resolve_config,
)

_HEIGHT_ALIASES: dict[str, float] = {
    "height": 1000.0,
    "sh_m": 1000.0,
    "height_m": 1000.0,
    "station_height": 1000.0,
    "station_height_m": 1000.0,
    "height_mm": 1.0,
    "sh_mm": 1.0,
}
_ORIENTATION_ALIASES = ("orientation_deg", "orientation", "hz_deg")

REFERENCE_COLUMNS = (
    "point_id", "metric", "station_field", "rho", "n", "p_value",
    "p_adjusted", "n_tests", "alpha", "significant", "status", "reason")


@dataclass
class ReferenceInferenceResult:
    """Cycle-detrended Spearman correlations with Bonferroni correction."""

    correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: str = NOT_CONFIGURED
    reason: str | None = None


@dataclass
class ResectionResult:
    """Raw common vertical versus exported station height regression."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    slope: float = float("nan")
    intercept: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    r_value: float = float("nan")
    p_value: float = float("nan")
    n: int = 0
    expected_slope: float = -1.0
    comparison: str = STATUS_INSUFFICIENT
    status: str = STATUS_INSUFFICIENT
    reason: str | None = None


def _station_frame(station_series: pd.DataFrame) -> pd.DataFrame:
    """Normalise the documented station series (height in mm, orientation deg)."""
    require_columns(station_series, (), "station_series")
    if "ts" not in station_series.columns and "cycle_id" not in station_series.columns:
        raise ValueError(
            "station_series must provide 'ts' and/or 'cycle_id'")
    frame = station_series.copy()
    if "ts" in frame.columns:
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
    height_column = next(
        (column for column in _HEIGHT_ALIASES if column in frame.columns), None)
    orientation_column = next(
        (column for column in _ORIENTATION_ALIASES if column in frame.columns),
        None)
    if height_column is None and orientation_column is None:
        raise ValueError(
            "station_series provides neither a documented height column "
            f"{list(_HEIGHT_ALIASES)} nor an orientation column "
            f"{list(_ORIENTATION_ALIASES)}")
    if height_column is not None:
        frame["station_height_mm"] = (
            pd.to_numeric(frame[height_column], errors="coerce")
            * _HEIGHT_ALIASES[height_column])
    if orientation_column is not None:
        frame["orientation_deg"] = pd.to_numeric(
            frame[orientation_column], errors="coerce")
    keep = [column for column in ("ts", "cycle_id", "station_id",
                                  "station_height_mm", "orientation_deg")
            if column in frame.columns]
    return frame[keep].dropna(subset=[column for column in ("ts", "cycle_id")
                                      if column in frame.columns],
                              how="all")


def _merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    if "cycle_id" in left.columns and "cycle_id" in right.columns:
        if "station_id" in left.columns and "station_id" in right.columns:
            return ["station_id", "cycle_id"]
        return ["cycle_id"]
    return ["ts"]


def _cycle_detrend(work: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Subtract the per-cycle median (vertical) or circular mean (Hz)."""
    if "cycle_id" in work.columns:
        keys = (["station_id", "cycle_id"] if "station_id" in work.columns
                else ["cycle_id"])
    else:
        keys = ["ts"]
    out = work.copy()
    if metric == "hz":
        radians = np.radians(out["value"].to_numpy(dtype=float))
        out["_cos"] = np.cos(radians)
        out["_sin"] = np.sin(radians)
        means = out.groupby(keys, sort=False)[["_cos", "_sin"]].transform("mean")
        reference = np.degrees(np.arctan2(means["_sin"], means["_cos"])) % 360.0
        out["residual"] = angular_difference_deg(out["value"], reference)
        out = out.drop(columns=["_cos", "_sin"])
    else:
        median = out.groupby(keys, sort=False)["value"].transform("median")
        out["residual"] = out["value"] - median
    return out


def _spearman(x, y) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def reference_inference(series: pd.DataFrame, *,
                        station_series: pd.DataFrame,
                        config) -> ReferenceInferenceResult:
    """Cycle-detrended Spearman correlations and Bonferroni correction.

    For every prism, the cycle-detrended raw vertical is correlated with the
    exported station height and the cycle-detrended Hz with the station
    orientation. When ``forensics.references.bonferroni`` is true, ``p_adjusted
    = min(1, p·n_tests)`` with ``n_tests`` the number of valid correlations in
    the same family (vertical or Hz). ``significant`` uses the configured
    ``forensics.references.alpha``; if alpha is missing the correlations are
    still returned but the result status is ``not_configured`` and no
    significance decision is made. Correlation with a reference signature is
    not proof that a prism is a resection reference.
    """
    cfg = resolve_config(config)
    references = dict(cfg.forensics.references or {})
    alpha = references.get("alpha")
    bonferroni = bool(references.get("bonferroni", False))
    alpha_configured = not is_not_configured(alpha)

    station = _station_frame(station_series)
    rows: list[dict] = []
    families = (("vert", "vertical", "station_height_mm"),
                ("hz", "hz", "orientation_deg"))
    for metric, label, station_field in families:
        if station_field not in station.columns:
            continue
        work = prepare_series(series, metric)
        if work.empty:
            continue
        detrended = _cycle_detrend(work, metric)
        keys = _merge_keys(detrended, station)
        merged = detrended.merge(
            station[[*keys, station_field]], on=keys, how="inner")
        family: list[dict] = []
        for point_id, group in merged.groupby("point_id", sort=True):
            rho, p_value = _spearman(group["residual"].to_numpy(dtype=float),
                                     group[station_field].to_numpy(dtype=float))
            family.append({
                "point_id": str(point_id),
                "metric": label,
                "station_field": station_field,
                "rho": rho,
                "n": int(len(group)),
                "p_value": p_value,
            })
        n_tests = sum(1 for row in family if np.isfinite(row["p_value"]))
        n_tests = max(n_tests, 1)
        for row in family:
            if not np.isfinite(row["p_value"]):
                adjusted = float("nan")
            elif bonferroni:
                adjusted = min(1.0, row["p_value"] * n_tests)
            else:
                adjusted = row["p_value"]
            row["p_adjusted"] = adjusted
            row["n_tests"] = n_tests
            row["alpha"] = float(alpha) if alpha_configured else float("nan")
            if not alpha_configured:
                row["significant"] = False
                row["status"] = NOT_CONFIGURED
                row["reason"] = ("forensics.references.alpha is not "
                                 "configured; the correlation is reported but "
                                 "no significance decision is made")
            else:
                row["significant"] = bool(
                    np.isfinite(adjusted) and adjusted < float(alpha))
                row["status"] = STATUS_OK
                row["reason"] = (
                    "cycle-detrended Spearman correlation; a significant "
                    "correlation suggests association with the station "
                    "quantity, not confirmed reference identity")
            rows.append(row)

    table = pd.DataFrame(rows, columns=list(REFERENCE_COLUMNS))
    if not alpha_configured:
        status = NOT_CONFIGURED
        reason = ("forensics.references.alpha is not configured; "
                  "correlations are statistics only")
    elif table.empty:
        status = STATUS_INSUFFICIENT
        reason = ("station_series provides no height/orientation column that "
                  "matches the series")
    else:
        status = STATUS_OK
        reason = ("Bonferroni correction applied within each family"
                  if bonferroni else "Bonferroni correction disabled by "
                  "configuration; raw p-values reported")
    return ReferenceInferenceResult(
        correlations=table, status=status, reason=reason)


def resection_validation(series: pd.DataFrame, *,
                         station_series: pd.DataFrame,
                         config) -> ResectionResult:
    """Regress the raw common vertical on the exported station height.

    A real station rise lowers the raw common vertical one-for-one, so the
    supplied expected slope is ``forensics.references.
    expected_rise_vertical_slope`` (default ``-1.0``, millimetres of vertical
    per millimetre of station height). The confidence interval uses
    ``rates.confidence`` (supplied 0.95). ``comparison`` is ``consistent`` when
    the configured expected slope lies inside the interval and ``inconsistent``
    otherwise; with too few cycles the comparison is ``insufficient_data`` and
    no slope verdict is fabricated.
    """
    cfg = resolve_config(config)
    references = dict(cfg.forensics.references or {})
    expected = references.get("expected_rise_vertical_slope", -1.0)
    if expected is None:
        return ResectionResult(
            status=NOT_CONFIGURED,
            reason=("forensics.references.expected_rise_vertical_slope is not "
                    "configured; no expected slope is invented"))
    expected = float(expected)
    confidence = cfg.rates.confidence

    work = prepare_series(series, "vert")
    station = _station_frame(station_series)
    if "station_height_mm" not in station.columns:
        return ResectionResult(
            status=NOT_CONFIGURED, expected_slope=expected,
            reason="station_series provides no documented height column")
    if work.empty:
        return ResectionResult(
            status=STATUS_INSUFFICIENT, expected_slope=expected,
            reason="no vertical observations")

    keys = _merge_keys(work, station)
    common = (work.groupby(keys, sort=True)["value"].median()
              .reset_index(name="common_vertical_mm"))
    merged = common.merge(station[[*keys, "station_height_mm"]],
                          on=keys, how="inner")
    table = merged[[*keys, "common_vertical_mm", "station_height_mm"]].copy()
    if len(merged) < 3 or np.ptp(merged["station_height_mm"]) == 0:
        return ResectionResult(
            table=table, n=int(len(merged)), expected_slope=expected,
            status=STATUS_INSUFFICIENT,
            comparison=STATUS_INSUFFICIENT,
            reason=("fewer than three cycles or no station-height variation; "
                    "the slope is not estimated"))

    x = merged["station_height_mm"].to_numpy(dtype=float)
    y = merged["common_vertical_mm"].to_numpy(dtype=float)
    result = linregress(x, y)
    if confidence is None:
        ci_low = ci_high = float("nan")
        comparison = STATUS_INSUFFICIENT
    else:
        degrees = len(merged) - 2
        critical = float(t.ppf(1.0 - (1.0 - float(confidence)) / 2.0, degrees))
        ci_low = float(result.slope - critical * result.stderr)
        ci_high = float(result.slope + critical * result.stderr)
        comparison = ("consistent"
                      if ci_low <= expected <= ci_high else "inconsistent")
    status = (STATUS_OK if confidence is not None else NOT_CONFIGURED)
    reason = ("raw common vertical versus exported station height; a real "
              "station rise lowers the raw vertical one-for-one"
              if confidence is not None else
              "rates.confidence is not configured; the slope is reported but "
              "no comparison interval is formed")
    return ResectionResult(
        table=table, slope=float(result.slope),
        intercept=float(result.intercept), ci_low=ci_low, ci_high=ci_high,
        r_value=float(result.rvalue), p_value=float(result.pvalue),
        n=int(len(merged)), expected_slope=expected, comparison=comparison,
        status=status, reason=reason)
