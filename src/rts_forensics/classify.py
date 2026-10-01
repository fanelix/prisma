"""
classify.py — movement concern, measurement reliability and TARP, kept separate.

Three outputs that must never be merged:

* :func:`screen_movement` — exploratory screening of net change against
  ``max(sigma_multiplier·sigma, floor)`` plus the confidence-interval sign
  rule. It never raises an alarm and carries no TARP meaning.
* :func:`grade_reliability` — A–D data-reliability grade from configured
  boundaries (coverage, gaps, final-window observations, noise, spikes). The
  grade never changes the movement concern.
* :func:`apply_tarp` — applies user-supplied thresholds exactly as configured
  and nothing more; without a TARP no state beyond ``no_tarp_configured`` is
  produced, because a slope is never declared safe or unsafe.

:func:`classify` combines the three tables plus the reliability flags and a
status dictionary; concern and reliability remain separate fields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AnalysisConfig, default_config, is_not_configured
from .models import (
    CONCERN_COLUMNS,
    NOT_CONFIGURED,
    RELIABILITY_COLUMNS,
    STATUS_INSUFFICIENT,
    STATUS_OK,
    TARP_COLUMNS,
    ClassificationResult,
)

_concern_columns = list(CONCERN_COLUMNS)
_concern_columns.insert(_concern_columns.index("status"), "possible")
_CONCERN_OUTPUT_COLUMNS = tuple(_concern_columns)
_RELIABILITY_OUTPUT_COLUMNS = tuple(RELIABILITY_COLUMNS) + ("flags",)

_GRADE_LABELS = {"A": "good", "B": "moderate", "C": "weak", "D": "insufficient"}

_TARP_META_HINTS = (
    "averag", "persist", "window", "missing", "minimum", "required",
    "days", "hours", "n_", "min_", "max_",
)
_METRIC_SYNONYMS = {
    "los": {"los", "radial", "d_rad", "dd", "d_rad_mm", "rad"},
    "vertical": {"vert", "vertical", "d_vert", "ver_raw", "ver_raw_mm",
                 "up", "d_up", "dz", "d_z", "height"},
    "tangential": {"tan", "tangential", "d_tan", "tan_raw", "tan_raw_mm"},
}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _resolve_config(config) -> AnalysisConfig:
    """Accept an :class:`AnalysisConfig`, a plain dict or ``None`` (defaults)."""
    if config is None:
        return default_config()
    if isinstance(config, dict):
        return AnalysisConfig.from_dict(config)
    return config


def _point_table(summary) -> pd.DataFrame:
    if summary is None:
        return pd.DataFrame(columns=["point_id", "segment_id"])
    table = summary.copy() if isinstance(summary, pd.DataFrame) else pd.DataFrame(summary)
    if "point_id" not in table.columns:
        index_name = table.index.name
        table = table.reset_index()
        if "point_id" not in table.columns:
            if index_name is not None and index_name in table.columns:
                table = table.rename(columns={index_name: "point_id"})
            elif len(table.columns):
                table = table.rename(columns={table.columns[0]: "point_id"})
    if "segment_id" not in table.columns:
        table = table.assign(segment_id="")
    return table.reset_index(drop=True)


def _record_float(record: dict, names: tuple[str, ...]) -> float | None:
    for name in names:
        if name not in record:
            continue
        value = record[name]
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _record_value(record: dict, names: tuple[str, ...]):
    for name in names:
        if name in record:
            value = record[name]
            if value is None:
                continue
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                return value
            return value
    return None


def _text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _is_vertical(metric: str) -> bool:
    text = str(metric).strip().lower()
    if not text:
        return False
    canonical = _canonical_metric(text)
    return canonical == "vertical" or "vert" in text


def _canonical_metric(metric: str) -> str:
    text = str(metric).strip().lower()
    for suffix in ("_mm", "_arcsec", "_ppm", "_deg", "_m"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    for canonical, aliases in _METRIC_SYNONYMS.items():
        if text in aliases:
            return canonical
    return text


def _finite(value: float | None) -> bool:
    return value is not None and np.isfinite(value)


# ---------------------------------------------------------------------------
#  Movement concern
# ---------------------------------------------------------------------------
def screen_movement(summary: pd.DataFrame, *, config=None) -> pd.DataFrame:
    """Exploratory movement screening with :data:`CONCERN_COLUMNS` (+``possible``).

    For every prism/segment/metric row: fewer than
    ``config.screening.min_prism_cycles`` cycles gives
    ``status="insufficient_data"``. Otherwise ``exceeds`` is True when
    ``|net| >= max(sigma_multiplier·sigma, floor_mm)``, the confidence interval
    excludes zero and (when ``require_same_sign_ci``) the interval signs match
    ``sign(net)``. ``possible`` uses the separate ``possible_*`` settings.
    Fields ``net_mm``/``sigma_mm``/``n_prism_cycles`` are accepted, as are the
    short aliases ``net``/``sigma``/``n_cycles``.

    The output is exploratory: no alarm is ever produced and the reason text
    states that the screening carries no TARP meaning.
    """
    cfg = _resolve_config(config)
    screening = cfg.screening
    table = _point_table(summary)
    rows: list[dict] = []

    for record in table.to_dict("records"):
        point_id = _text(record.get("point_id"))
        segment_id = _text(record.get("segment_id"))
        metric = _text(record.get("metric"))
        net = _record_float(record, ("net_mm", "net", "delta_mm"))
        sigma = _record_float(record, ("sigma_mm", "sigma", "sd_mm"))
        ci_low = _record_float(record, ("ci_low", "rate_ci_low"))
        ci_high = _record_float(record, ("ci_high", "rate_ci_high"))
        n_cycles = _record_float(record, ("n_prism_cycles", "n_cycles"))

        vertical = _is_vertical(metric)
        floor = float(screening.vertical_floor_mm if vertical else screening.los_floor_mm)
        possible_floor = float(
            screening.possible_vertical_floor_mm if vertical
            else screening.possible_los_floor_mm)

        row = {
            "point_id": point_id,
            "segment_id": segment_id,
            "metric": metric,
            "net_mm": net if net is not None else np.nan,
            "sigma_mm": sigma if sigma is not None else np.nan,
            "floor_mm": floor,
            "ci_low": ci_low if ci_low is not None else np.nan,
            "ci_high": ci_high if ci_high is not None else np.nan,
            "exceeds": False,
            "possible": False,
            "status": STATUS_INSUFFICIENT,
            "reason": "",
        }

        if not _finite(n_cycles):
            row["reason"] = ("insufficient prism-cycles: count unavailable "
                             f"(min_prism_cycles={screening.min_prism_cycles})")
            rows.append(row)
            continue
        if n_cycles < float(screening.min_prism_cycles):
            row["reason"] = (
                f"insufficient prism-cycles ({n_cycles:g} < "
                f"{screening.min_prism_cycles}); exploratory screen not available")
            rows.append(row)
            continue
        if not all(_finite(value) for value in (net, sigma, ci_low, ci_high)):
            row["reason"] = ("insufficient_data: net change, uncertainty or confidence "
                             "interval unavailable")
            rows.append(row)
            continue

        threshold = max(float(screening.sigma_multiplier) * sigma, floor)
        possible_threshold = max(
            float(screening.possible_sigma_multiplier) * sigma, possible_floor)
        ci_excludes_zero = bool(ci_low * ci_high > 0.0)
        if screening.require_same_sign_ci:
            same_sign = bool(
                np.sign(ci_low) == np.sign(ci_high) == np.sign(net) and net != 0.0)
        else:
            same_sign = True
        exceeds = bool(abs(net) >= threshold and ci_excludes_zero and same_sign)
        possible = bool(
            abs(net) >= possible_threshold and ci_excludes_zero and same_sign)

        row["exceeds"] = exceeds
        row["possible"] = possible
        row["status"] = STATUS_OK
        row["reason"] = (
            "exploratory screening only: not an alarm and no TARP meaning"
            + ("; movement concern flagged for review" if exceeds
               else "; possible movement (weaker evidence)" if possible
               else "; no credible movement detected within available "
                    "observations and sensitivity"))
        rows.append(row)

    out = pd.DataFrame(rows, columns=list(_CONCERN_OUTPUT_COLUMNS))
    for column in ("net_mm", "sigma_mm", "floor_mm", "ci_low", "ci_high"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    out["exceeds"] = out["exceeds"].astype(bool)
    out["possible"] = out["possible"].astype(bool)
    return out


# ---------------------------------------------------------------------------
#  Reliability
# ---------------------------------------------------------------------------
def grade_reliability(summary: pd.DataFrame, *, config=None) -> pd.DataFrame:
    """Grade data reliability A–D from configured boundaries.

    With ``config.reliability.boundaries is None`` every row is
    ``grade="not_configured"``. Otherwise: fewer than
    ``insufficient_below_cycles`` is D; any hard flag (coverage below
    ``coverage_weak_pct``, longest gap above ``max_gap_hours``, no observation
    in the final window, ``sigma_dD`` above ``noisy_los_mm`` or vertical sigma
    above ``noisy_vertical_mm``) is C; coverage below ``coverage_moderate_pct``
    or at least ``spike_flags_moderate`` spikes is B; else A. The flag string
    is retained on every row. Reliability never changes movement concern.
    """
    cfg = _resolve_config(config)
    boundaries = cfg.reliability.boundaries
    table = _point_table(summary)
    configured = isinstance(boundaries, dict) and any(
        value is not None for value in boundaries.values())

    insufficient_below = boundaries.get("insufficient_below_cycles") if configured else None
    weak_pct = boundaries.get("coverage_weak_pct") if configured else None
    moderate_pct = boundaries.get("coverage_moderate_pct") if configured else None
    max_gap = boundaries.get("max_gap_hours") if configured else None
    noisy_los = boundaries.get("noisy_los_mm") if configured else None
    noisy_vertical = boundaries.get("noisy_vertical_mm") if configured else None
    spike_moderate = boundaries.get("spike_flags_moderate") if configured else None
    observation_columns = ("obs_last48h", "obs_in_last48h", "n_final48h", "n_obs_last48h")
    observation_column = next(
        (column for column in observation_columns if column in table.columns), None)

    rows: list[dict] = []
    for record in table.to_dict("records"):
        point_id = _text(record.get("point_id"))
        segment_id = _text(record.get("segment_id"))
        n_cycles = _record_float(record, ("n_prism_cycles", "n_cycles"))
        coverage = _record_float(record, ("coverage_pct", "coverage"))
        gap_hours = _record_float(record, ("max_gap_hours", "gap_hours"))
        observations = _record_value(
            record, ("obs_last48h", "obs_in_last48h", "n_final48h", "n_obs_last48h"))
        sigma_los = _record_float(record, ("sigma_los_mm", "sig_dD", "sigma_dD"))
        sigma_vert = _record_float(
            record, ("sigma_vert_mm", "sig_ver_raw", "sigma_ver_raw"))
        spikes = _record_float(record, ("n_spike_flags", "n_spikes"))
        frame_member = bool(record.get("frame_member", False))

        row = {
            "point_id": point_id,
            "segment_id": segment_id,
            "n_prism_cycles": int(n_cycles) if _finite(n_cycles) else None,
            "frame_member": frame_member,
            "noise_mm": sigma_los if sigma_los is not None else np.nan,
            "coverage": coverage if coverage is not None else np.nan,
            "grade": NOT_CONFIGURED,
            "status": NOT_CONFIGURED,
            "reason": "",
            "flags": "",
        }

        if not configured:
            row["reason"] = ("reliability boundaries are not configured; "
                             "no grade is invented")
            rows.append(row)
            continue

        flags: list[str] = []
        if not _finite(n_cycles) or (
                insufficient_below is not None and n_cycles < float(insufficient_below)):
            grade = "D"
            flags.append("insufficient_cycles")
            status = STATUS_INSUFFICIENT
        else:
            hard_flags = []
            if (weak_pct is not None and _finite(coverage)
                    and coverage < float(weak_pct)):
                hard_flags.append("coverage_weak")
            if max_gap is not None and _finite(gap_hours) and gap_hours > float(max_gap):
                hard_flags.append("gap_gt_max")
            if observation_column is not None:
                if observations is None:
                    hard_flags.append("final_window_unknown")
                elif isinstance(observations, (bool, np.bool_)):
                    if not bool(observations):
                        hard_flags.append("no_obs_final_window")
                elif isinstance(observations, (int, float, np.integer, np.floating)):
                    if float(observations) <= 0:
                        hard_flags.append("no_obs_final_window")
                else:
                    hard_flags.append("final_window_unknown")
            if (noisy_los is not None and _finite(sigma_los)
                    and sigma_los > float(noisy_los)):
                hard_flags.append("noisy_los")
            if (noisy_vertical is not None and _finite(sigma_vert)
                    and sigma_vert > float(noisy_vertical)):
                hard_flags.append("noisy_vertical")

            if hard_flags:
                grade = "C"
                status = STATUS_OK
                flags.extend(hard_flags)
            else:
                moderate = (
                    (moderate_pct is not None and _finite(coverage)
                     and coverage < float(moderate_pct))
                    or (spike_moderate is not None and _finite(spikes)
                        and spikes >= float(spike_moderate)))
                if moderate:
                    grade = "B"
                    if (moderate_pct is not None and _finite(coverage)
                            and coverage < float(moderate_pct)):
                        flags.append("coverage_moderate")
                    if (spike_moderate is not None and _finite(spikes)
                            and spikes >= float(spike_moderate)):
                        flags.append("spike_flags")
                else:
                    grade = "A"
                status = STATUS_OK

        row["grade"] = grade
        row["status"] = status
        row["flags"] = "; ".join(flags)
        label = _GRADE_LABELS[grade]
        row["reason"] = (
            f"reliability grade {grade} ({label}); flags: "
            + (", ".join(flags) if flags else "none"))
        rows.append(row)

    out = pd.DataFrame(rows, columns=list(_RELIABILITY_OUTPUT_COLUMNS))
    out["coverage"] = pd.to_numeric(out["coverage"], errors="coerce").astype(float)
    out["noise_mm"] = pd.to_numeric(out["noise_mm"], errors="coerce").astype(float)
    out["frame_member"] = out["frame_member"].astype(bool)
    return out


# ---------------------------------------------------------------------------
#  TARP
# ---------------------------------------------------------------------------
def _tarp_thresholds(tarp) -> dict[str, float]:
    if not isinstance(tarp, dict):
        return {}
    raw = tarp.get("thresholds") if isinstance(tarp.get("thresholds"), dict) else tarp
    thresholds: dict[str, float] = {}
    for key, value in raw.items():
        text = str(key)
        if any(hint in text.lower() for hint in _TARP_META_HINTS):
            continue
        if isinstance(value, (bool, str)) or value is None:
            continue
        try:
            thresholds[text] = float(value)
        except (TypeError, ValueError):
            continue
    return thresholds


def _metrics_match(threshold_key: str, metric: str) -> bool:
    if not metric:
        return False
    left = _canonical_metric(threshold_key)
    right = _canonical_metric(metric)
    return left == right


def apply_tarp(series: pd.DataFrame, *, tarp) -> pd.DataFrame:
    """Produce :data:`TARP_COLUMNS`; apply only user-supplied thresholds.

    ``tarp is None`` or ``"not_configured"`` yields one row per input row with
    ``state="no_tarp_configured"``, ``status="not_configured"`` and the reason
    ``"a slope is never declared safe or unsafe"``. When thresholds are
    supplied (``{metric: value_mm}``, optionally nested under ``thresholds``);
    only those metrics are evaluated, auxiliary averaging/persistence keys are
    metadata and never become thresholds.
    """
    table = _point_table(series)
    rows: list[dict] = []

    if is_not_configured(tarp):
        for record in table.to_dict("records"):
            value = _record_float(record, ("value_mm", "value", "net_mm"))
            rows.append({
                "point_id": _text(record.get("point_id")),
                "segment_id": _text(record.get("segment_id")),
                "metric": _text(record.get("metric")),
                "value_mm": value if value is not None else np.nan,
                "tarp_mm": np.nan,
                "state": "no_tarp_configured",
                "status": NOT_CONFIGURED,
                "reason": "a slope is never declared safe or unsafe",
            })
        out = pd.DataFrame(rows, columns=list(TARP_COLUMNS))
        out["value_mm"] = pd.to_numeric(out["value_mm"], errors="coerce").astype(float)
        return out

    thresholds = _tarp_thresholds(tarp)
    if not thresholds:
        for record in table.to_dict("records"):
            rows.append({
                "point_id": _text(record.get("point_id")),
                "segment_id": _text(record.get("segment_id")),
                "metric": _text(record.get("metric")),
                "value_mm": np.nan,
                "tarp_mm": np.nan,
                "state": "no_tarp_configured",
                "status": NOT_CONFIGURED,
                "reason": "TARP supplied without numeric thresholds; nothing to apply",
            })
        out = pd.DataFrame(rows, columns=list(TARP_COLUMNS))
        out["value_mm"] = pd.to_numeric(out["value_mm"], errors="coerce").astype(float)
        return out

    for record in table.to_dict("records"):
        point_id = _text(record.get("point_id"))
        segment_id = _text(record.get("segment_id"))
        metric = _text(record.get("metric"))
        value = _record_float(record, ("value_mm", "value", "net_mm"))
        if metric:
            matches = [key for key in thresholds if _metrics_match(key, metric)]
        else:
            matches = list(thresholds)
        if not matches:
            rows.append({
                "point_id": point_id,
                "segment_id": segment_id,
                "metric": metric,
                "value_mm": value if value is not None else np.nan,
                "tarp_mm": np.nan,
                "state": "not_configured",
                "status": NOT_CONFIGURED,
                "reason": f"no TARP threshold supplied for metric {metric!r}",
            })
            continue
        for key in matches:
            threshold = thresholds[key]
            if _finite(value):
                state = "exceeded" if abs(value) >= threshold else "within"
                status = "applied"
                reason = ("user-supplied TARP applied exactly as configured; "
                          "no automatic safety conclusion")
            else:
                state = "not_configured"
                status = NOT_CONFIGURED
                reason = "value unavailable; TARP threshold not evaluated"
            rows.append({
                "point_id": point_id,
                "segment_id": segment_id,
                "metric": key if not metric else metric,
                "value_mm": value if value is not None else np.nan,
                "tarp_mm": threshold,
                "state": state,
                "status": status,
                "reason": reason,
            })

    out = pd.DataFrame(rows, columns=list(TARP_COLUMNS))
    out["value_mm"] = pd.to_numeric(out["value_mm"], errors="coerce").astype(float)
    out["tarp_mm"] = pd.to_numeric(out["tarp_mm"], errors="coerce").astype(float)
    return out


# ---------------------------------------------------------------------------
#  Combined classification
# ---------------------------------------------------------------------------
def _tarp_input(evidence) -> pd.DataFrame:
    if evidence is None:
        return pd.DataFrame()
    if isinstance(evidence, pd.DataFrame):
        return evidence
    if isinstance(evidence, dict):
        for key in ("series", "displacements", "tarp", "rows"):
            value = evidence.get(key)
            if isinstance(value, pd.DataFrame):
                return value
    return pd.DataFrame()


def classify(summary: pd.DataFrame, evidence, *, config=None) -> ClassificationResult:
    """Combine concern, reliability, flags and TARP without merging fields.

    ``evidence`` is the long series used by :func:`apply_tarp` (a table of
    prism/segment/metric rows); it may also be a dict containing a ``series``
    entry. Concern and reliability are separate result fields and neither is
    derived from the other; ``statuses`` reports each family independently.
    """
    cfg = _resolve_config(config)
    concern = screen_movement(summary, config=cfg)
    reliability = grade_reliability(summary, config=cfg)
    if "flags" in reliability.columns:
        flags = reliability[["point_id", "segment_id", "flags"]].copy()
    else:
        flags = pd.DataFrame(columns=["point_id", "segment_id", "flags"])
    tarp = apply_tarp(_tarp_input(evidence), tarp=cfg.tarp)

    if len(concern) == 0:
        concern_status = NOT_CONFIGURED
    elif (concern["status"] == STATUS_OK).any():
        concern_status = STATUS_OK
    else:
        concern_status = STATUS_INSUFFICIENT

    boundaries = cfg.reliability.boundaries
    reliability_configured = isinstance(boundaries, dict) and any(
        value is not None for value in boundaries.values())
    if not reliability_configured:
        reliability_status = NOT_CONFIGURED
    elif len(reliability) == 0:
        reliability_status = STATUS_INSUFFICIENT
    else:
        reliability_status = STATUS_OK

    tarp_configured = (not is_not_configured(cfg.tarp)
                       and bool(_tarp_thresholds(cfg.tarp)))
    tarp_status = "applied" if (
        tarp_configured and not tarp.empty and tarp["status"].eq("applied").any()
    ) else NOT_CONFIGURED

    statuses = {
        "concern": concern_status,
        "reliability": reliability_status,
        "tarp": tarp_status,
    }
    return ClassificationResult(
        concern=concern, reliability=reliability, flags=flags,
        tarp=tarp, statuses=statuses)
