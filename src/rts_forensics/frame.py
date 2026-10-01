"""
frame.py — per-cycle robust frame reconstruction (experimental).

Faithful port of the reference per-cycle frame fit (ALGORITHMS.md section 5).
For every cycle a weighted Huber IRLS joint fit is run over a *frame set* of
prisms (reliability A, at least ``reliability.frame_min_cycles`` cycles and not
excluded). Frame-corrected values are always published beside the raw values
and are experimental: they are relative to the chosen network, never absolute
ground movement.

Model and signs (angles in radians internally, rotation in arcsec):

* ``a``   grid azimuth from station to prism (degrees), ``HD = D·sin V`` (m),
  ``Dk = D/1000`` (km), ``sinV = sin V``.
* Horizontal design (2n x 4), rows scaled by ``1/sigma``; unknowns
  ``[rotation_arcsec, tE_mm, tN_mm, scale_ppm]``::

      Hz  [1, 206.265·(-cos a)/HD, 206.265·(sin a)/HD, 0]   y = dHz_as
      LOS [0, -sin a·sinV,          -cos a·sinV,         Dk]  y = dD

* Vertical design (n x 4), rows scaled by ``1/sigma``; unknowns
  ``[vert_offset_mm, vert_index_mm_per_km, tilt_sin, tilt_cos]``::

      Ver [1, Dk, Dk·sin a, Dk·cos a]                       y = ver_raw

* Predictions and corrections (ALGORITHMS section 5)::

      pred_hz  = w + 206.265·(-cos a·tE + sin a·tN)/HD
      tan_fc   = (dHz_as - pred_hz)·HD/206.265
      pred_los = -(sin a·tE + cos a·tN)·sinV + s·Dk
      los_fc   = dD - pred_los
      pred_ver = h + Dk·(iota + tilt_s·sin a + tilt_c·cos a)
      ver_fc   = ver_raw - pred_ver

Numerical conventions
---------------------
* Huber IRLS: cutoff ``frame.huber_cutoff`` (1.5), at most
  ``frame.convergence.max_iterations`` (30) iterations, robust scale
  ``1.4826·MAD(residual)`` floored at ``frame.convergence.scale_floor`` (1.0)
  in sigma-normalised units, weights ``clip(k·scale/max(|r|, 1e-12), 0, 1)``.
* Covariance ``s² · pinv(Awᵀ Aw)`` with ``s² = weighted SSE / dof`` and
  ``dof = max(n - 4, 1)``; standard errors are ``sqrt(diag)``.
* Reported ``rms_hz``/``rms_D``/``rms_v`` are ``1.4826·MAD(residual)·sigma`` of
  the respective block (physical units: arcsec, mm, mm). Because
  :class:`~rts_forensics.models.CycleFrameFit` has no ``diagnostics`` field,
  they are carried inside ``coefficients`` (they are not fitted parameters).
* A rank-deficient, insufficient or otherwise failed fit returns empty
  ``coefficients``/``std_errors`` and never a fabricated zero correction; the
  caller (and :func:`fit_frames`) leaves the corrected values NaN.

Leave-one-out
-------------
``select_frame_set`` only proposes members. The caller is responsible for the
leave-one-out rule: a tested prism must never be part of its own frame set.
Explicit exclusions (configuration, site list or the ``exclusions`` argument)
are returned in ``FrameSelection.exclusions`` for audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AnalysisConfig, default_config
from .models import (
    NOT_CONFIGURED,
    STATUS_FAILED,
    STATUS_INSUFFICIENT,
    STATUS_OK,
    STATUS_RANK_DEFICIENT,
    CycleFrameFit,
    FrameResult,
    FrameSelection,
)

#: arcsec per (mm of translation / m of horizontal distance)
_HZ_ARCSEC = 206.265

HORIZONTAL_PARAMETERS = ("rotation_arcsec", "tE_mm", "tN_mm", "scale_ppm")
VERTICAL_PARAMETERS = ("vert_offset_mm", "vert_index_mm_per_km", "tilt_sin", "tilt_cos")
RMS_PARAMETERS = ("rms_hz", "rms_D", "rms_v")

_STD_ERROR_NAMES = {
    "rotation_arcsec": "rotation_se",
    "tE_mm": "tE_se",
    "tN_mm": "tN_se",
    "scale_ppm": "scale_se",
    "vert_offset_mm": "vert_offset_se",
    "vert_index_mm_per_km": "vert_index_se",
    "tilt_sin": "tilt_sin_se",
    "tilt_cos": "tilt_cos_se",
}

_ID_COLUMNS = ("point_id", "station_id", "segment_id", "cycle_id")
_GROUP_COLUMNS = ("station_id", "segment_id", "cycle_id")

CYCLE_COEFFICIENT_COLUMNS = (
    "station_id", "segment_id", "cycle_id",
    "rotation_arcsec", "rotation_se",
    "tE_mm", "tE_se", "tN_mm", "tN_se",
    "scale_ppm", "scale_se",
    "vert_offset_mm", "vert_offset_se",
    "vert_index_mm_per_km", "vert_index_se",
    "tilt_sin", "tilt_sin_se", "tilt_cos", "tilt_cos_se",
    "rank", "condition_number",
    "rms_hz", "rms_D", "rms_v",
    "n_frame_members", "n_used", "status", "reason",
)
CYCLE_DIAGNOSTIC_COLUMNS = (
    "station_id", "segment_id", "cycle_id",
    "n_frame_members", "n_rows", "n_used",
    "rank", "condition_number", "rms_hz", "rms_D", "rms_v",
    "frame_members", "status", "reason",
)
CORRECTED_COLUMNS = (
    "point_id", "station_id", "segment_id", "cycle_id",
    "los_raw_mm", "los_fc_mm",
    "tan_raw_mm", "tan_fc_mm",
    "ver_raw_mm", "ver_fc_mm",
    "status", "reason",
)

_COEFFICIENT_NUMERIC_COLUMNS = (
    "rotation_arcsec", "rotation_se", "tE_mm", "tE_se", "tN_mm", "tN_se",
    "scale_ppm", "scale_se", "vert_offset_mm", "vert_offset_se",
    "vert_index_mm_per_km", "vert_index_se", "tilt_sin", "tilt_sin_se",
    "tilt_cos", "tilt_cos_se", "rank", "condition_number",
    "rms_hz", "rms_D", "rms_v", "n_frame_members", "n_used",
)
_CORRECTED_NUMERIC_COLUMNS = (
    "los_raw_mm", "los_fc_mm", "tan_raw_mm", "tan_fc_mm",
    "ver_raw_mm", "ver_fc_mm",
)


# ---------------------------------------------------------------------------
#  Small dataframe helpers
# ---------------------------------------------------------------------------
def _resolve_config(config) -> AnalysisConfig:
    """Accept an :class:`AnalysisConfig`, a plain dict or ``None`` (defaults)."""
    if config is None:
        return default_config()
    if isinstance(config, dict):
        return AnalysisConfig.from_dict(config)
    return config


def _with_point_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a string ``point_id`` column (index promoted if needed)."""
    if "point_id" in df.columns:
        out = df.copy()
        out["point_id"] = out["point_id"].astype(str)
        return out
    out = df.copy()
    index_name = out.index.name
    out = out.reset_index()
    if "point_id" in out.columns:
        out["point_id"] = out["point_id"].astype(str)
        return out
    if index_name is not None and index_name in out.columns:
        out = out.rename(columns={index_name: "point_id"})
    else:
        out = out.rename(columns={out.columns[0]: "point_id"})
    out["point_id"] = out["point_id"].astype(str)
    return out


def _numeric(df: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray | None:
    """First present column among ``names`` as a float array, else ``None``."""
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    return None


def _text(record: dict, names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = record.get(name)
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        return str(value)
    return default


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(np.abs(values - np.median(values))))


def _robust_sd(normalised_residual: np.ndarray, sigma: float) -> float:
    return 1.4826 * _mad(normalised_residual) * float(sigma)


def _cycle_id(frame: pd.DataFrame) -> int | str | None:
    if "cycle_id" not in frame.columns or len(frame) == 0:
        return None
    value = frame["cycle_id"].iloc[0]
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _sort_key(key):
    """Deterministic ordering for group keys (numbers before text per position)."""
    values = key if isinstance(key, tuple) else (key,)
    parts = []
    for value in values:
        if isinstance(value, (int, float, np.integer, np.floating)) \
                and not isinstance(value, bool):
            parts.append((0, float(value)))
        else:
            parts.append((1, str(value)))
    return tuple(parts)


# ---------------------------------------------------------------------------
#  Frame-set selection
# ---------------------------------------------------------------------------
def select_frame_set(
    summary: pd.DataFrame,
    *,
    exclusions: list[str] | None = None,
    config=None,
) -> FrameSelection:
    """Select frame-set members from a per-prism summary table.

    A member must have ``reliability`` starting with
    ``config.frame.frame_set_reliability`` (default ``"A"``) and at least
    ``config.reliability.frame_min_cycles`` cycles, and must not appear in
    ``config.frame.exclusions``, ``config.frame.site_exclusions`` or the
    ``exclusions`` argument.

    Leave-one-out is the caller's responsibility: a tested prism must never be
    part of its own frame set. All explicit exclusions (configuration, site
    list and argument) are returned in ``FrameSelection.exclusions`` for audit;
    ``diagnostics`` records the reason for every evaluated prism.

    Parameters
    ----------
    summary:
        One row per prism. Needs ``point_id`` (or a meaningfully named index),
        ``n_cycles`` (or ``n_prism_cycles``) and ``reliability`` strings such
        as ``"A (good)"``.
    exclusions:
        Additional prism identifiers to remove from the frame set, e.g. the
        prism under test for a leave-one-out run.
    config:
        :class:`~rts_forensics.config.AnalysisConfig`.

    Returns
    -------
    FrameSelection
        ``status="ok"`` when members exist; ``"insufficient_data"`` with an
        explicit reason otherwise.
    """
    cfg = _resolve_config(config)
    frame_cfg = cfg.frame
    table = _with_point_id(summary)
    point_id = table["point_id"].astype(str)

    n_cycles = _numeric(table, ("n_cycles", "n_prism_cycles"))
    if n_cycles is None:
        n_cycles = np.zeros(len(table), dtype=float)
    reliability = table["reliability"].astype(str) if "reliability" in table.columns \
        else pd.Series("", index=table.index, dtype=object)
    reliability = reliability.reset_index(drop=True)
    point_id = point_id.reset_index(drop=True)
    n_cycles = np.asarray(n_cycles, dtype=float)

    target = str(frame_cfg.frame_set_reliability).strip()
    minimum_cycles = float(cfg.reliability.frame_min_cycles)
    prefix = reliability.str.strip().str.split("(").str[0].str.strip()
    eligible = prefix.eq(target).to_numpy() & (n_cycles >= minimum_cycles)

    explicit: list[str] = []
    for group in (frame_cfg.exclusions, frame_cfg.site_exclusions, exclusions):
        if not group:
            continue
        if isinstance(group, str):
            explicit.append(group)
        else:
            explicit.extend(str(value) for value in group)
    explicit = sorted(dict.fromkeys(explicit))
    explicit_set = set(explicit)

    member_mask = eligible & ~point_id.isin(explicit_set).to_numpy()
    members = sorted(point_id[member_mask].tolist())

    rows = []
    for pid, rel, n_value, is_member in zip(
            point_id, reliability, n_cycles, member_mask, strict=True):
        if is_member:
            reason = "member"
        elif pid in explicit_set:
            reason = "excluded by configuration or caller"
        elif str(rel).strip().split("(")[0].strip() != target:
            reason = f"reliability {rel!r} does not start with {target!r}"
        else:
            reason = f"n_cycles {n_value:g} < frame_min_cycles {minimum_cycles:g}"
        rows.append({
            "point_id": pid,
            "reliability": rel,
            "n_cycles": n_value,
            "member": bool(is_member),
            "reason": reason,
        })
    diagnostics = pd.DataFrame(
        rows, columns=["point_id", "reliability", "n_cycles", "member", "reason"])
    if not diagnostics.empty:
        diagnostics["n_cycles"] = diagnostics["n_cycles"].astype(float)

    if members:
        return FrameSelection(members=members, exclusions=explicit,
                              diagnostics=diagnostics, status=STATUS_OK, reason=None)
    reason = (
        f"no prism passes the frame-set rules (reliability prefix {target!r}, "
        f"n_cycles >= {minimum_cycles:g}); {len(explicit)} explicit exclusion(s)"
    )
    return FrameSelection(members=[], exclusions=explicit, diagnostics=diagnostics,
                          status=STATUS_INSUFFICIENT, reason=reason)


# ---------------------------------------------------------------------------
#  Huber IRLS
# ---------------------------------------------------------------------------
@dataclass
class _IrlsFit:
    x: np.ndarray
    se: np.ndarray
    rank: int
    condition_number: float
    residual_normalised: np.ndarray
    converged: bool


def _huber_irls(
    design: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    *,
    cutoff: float,
    max_iterations: int,
    scale_floor: float,
) -> _IrlsFit:
    """Weighted Huber IRLS for a sigma-normalised linear system.

    ``design``/``observed`` are physical; rows are divided by ``sigma`` before
    fitting. Returns sigma-normalised residuals; multiply by the relevant
    ``sigma`` to obtain physical residuals.
    """
    design = np.asarray(design, dtype=float)
    observed = np.asarray(observed, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    n_rows, n_params = design.shape
    if n_rows == 0:
        return _IrlsFit(np.full(n_params, np.nan), np.full(n_params, np.nan),
                        0, float("nan"), np.full(0, np.nan), False)

    normalised = design / sigma[:, None]
    target = observed / sigma
    rank = int(np.linalg.matrix_rank(normalised)) if n_rows else 0
    try:
        condition_number = float(np.linalg.cond(normalised))
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        condition_number = float("nan")
    if rank < n_params:
        return _IrlsFit(np.full(n_params, np.nan), np.full(n_params, np.nan),
                        rank, condition_number,
                        np.full(n_rows, np.nan), False)

    x, *_ = np.linalg.lstsq(normalised, target, rcond=None)
    weights = np.ones(n_rows, dtype=float)
    weighted = normalised
    weighted_target = target
    converged = False
    for _ in range(max(int(max_iterations), 1)):
        residual = normalised @ x - target
        scale = max(1.4826 * _mad(residual), float(scale_floor))
        weights = np.clip(cutoff * scale / np.maximum(np.abs(residual), 1e-12), 0.0, 1.0)
        weighted = normalised * weights[:, None]
        weighted_target = target * weights
        x_new, *_ = np.linalg.lstsq(weighted, weighted_target, rcond=None)
        if float(np.max(np.abs(x_new - x))) < 1e-10:
            x = x_new
            converged = True
            break
        x = x_new

    residual_weighted = weighted @ x - weighted_target
    sse = float(np.sum(residual_weighted**2))
    dof = max(n_rows - n_params, 1)
    covariance = (sse / dof) * np.linalg.pinv(weighted.T @ weighted)
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    residual_normalised = normalised @ x - target
    return _IrlsFit(x, se, rank, condition_number, residual_normalised, converged)


# ---------------------------------------------------------------------------
#  Per-cycle fit
# ---------------------------------------------------------------------------
def _iter_members(frame_set) -> list[str]:
    if frame_set is None:
        return []
    if isinstance(frame_set, FrameSelection):
        return [str(m) for m in frame_set.members]
    if isinstance(frame_set, str):
        return [frame_set]
    return [str(m) for m in frame_set]


def _frame_members(frame_set) -> list[str]:
    return sorted(set(_iter_members(frame_set)))


def fit_cycle_frame(
    prism_cycle: pd.DataFrame,
    *,
    frame_set=None,
    config=None,
) -> CycleFrameFit:
    """Fit the eight frame terms for one cycle over the frame-set members.

    Parameters
    ----------
    prism_cycle:
        One row per prism in the cycle with ``point_id``, ``az_deg``, ``v``
        (zenith, degrees), ``D`` (slope, metres), ``dD`` (mm), ``dHz_as``
        (arcsec), ``ver_raw`` (mm). ``v_deg``/``d_m``/``dD_mm`` are accepted as
        aliases.
    frame_set:
        :class:`FrameSelection` or an iterable of prism IDs used as references.
    config:
        :class:`~rts_forensics.config.AnalysisConfig`.

    Returns
    -------
    CycleFrameFit
        On success ``coefficients`` holds the seven terms plus ``rms_hz``,
        ``rms_D`` and ``rms_v`` (documented diagnostics, not fitted
        parameters). Failed fits return empty coefficients and status
        ``insufficient_data``, ``rank_deficient`` or ``failed``.
    """
    cfg = _resolve_config(config)
    frame_cfg = cfg.frame
    cycle_id = _cycle_id(prism_cycle) if isinstance(prism_cycle, pd.DataFrame) else None

    convergence = frame_cfg.convergence or {}
    max_iterations = convergence.get("max_iterations")
    scale_floor = convergence.get("scale_floor")
    weights = frame_cfg.weights or {}
    sigma_hz = weights.get("hz_arcsec")
    sigma_los = weights.get("los_mm")
    sigma_vertical = weights.get("vertical_mm")
    if (frame_cfg.huber_cutoff is None or max_iterations is None
            or scale_floor is None or sigma_hz is None
            or sigma_los is None or sigma_vertical is None):
        return CycleFrameFit(
            cycle_id=cycle_id, rank=0, status=NOT_CONFIGURED,
            reason="frame Huber cutoff, convergence settings and weights are not configured")

    members = _frame_members(frame_set)
    if not members:
        return CycleFrameFit(
            cycle_id=cycle_id, rank=0, status=STATUS_INSUFFICIENT,
            reason="frame set has no members")

    table = _with_point_id(prism_cycle)
    in_set = table["point_id"].isin(members).to_numpy()
    sub = table.loc[in_set]

    az = _numeric(sub, ("az_deg", "az"))
    zenith = _numeric(sub, ("v", "v_deg"))
    slope = _numeric(sub, ("D", "d_m"))
    d_delta = _numeric(sub, ("dD", "dD_mm", "d_rad_mm", "los_raw_mm"))
    d_hz = _numeric(sub, ("dHz_as", "d_hz_as"))
    ver_raw = _numeric(sub, ("ver_raw", "ver_raw_mm", "d_vert_mm"))
    required = {
        "az_deg": az, "v": zenith, "D": slope,
        "dD": d_delta, "dHz_as": d_hz, "ver_raw": ver_raw,
    }
    missing = [name for name, values in required.items() if values is None]
    if missing:
        return CycleFrameFit(
            cycle_id=cycle_id, rank=0, status=STATUS_INSUFFICIENT,
            reason="missing required columns: " + ", ".join(missing))

    with np.errstate(divide="ignore", invalid="ignore"):
        azimuth = np.radians(az)
        vertical = np.radians(zenith)
        sin_v = np.sin(vertical)
        horizontal = slope * sin_v
    valid = np.isfinite(az) & np.isfinite(zenith) & np.isfinite(slope)
    valid &= np.isfinite(d_delta) & np.isfinite(d_hz) & np.isfinite(ver_raw)
    valid &= np.isfinite(horizontal) & (horizontal > 1e-9)

    minimum = int(frame_cfg.min_prism_cycles_per_fit or 1)
    n_valid = int(valid.sum())
    if n_valid < minimum:
        return CycleFrameFit(
            cycle_id=cycle_id, rank=0, status=STATUS_INSUFFICIENT,
            reason=(f"{n_valid} usable frame members < "
                    f"min_prism_cycles_per_fit={minimum}"))

    a = azimuth[valid]
    sin_v = sin_v[valid]
    hd = horizontal[valid]
    dk = slope[valid] / 1000.0
    y_hz = d_hz[valid]
    y_los = d_delta[valid]
    y_ver = ver_raw[valid]
    n = a.size

    horizontal_design = np.vstack([
        np.column_stack([
            np.ones(n),
            _HZ_ARCSEC * (-np.cos(a)) / hd,
            _HZ_ARCSEC * np.sin(a) / hd,
            np.zeros(n),
        ]),
        np.column_stack([
            np.zeros(n),
            -np.sin(a) * sin_v,
            -np.cos(a) * sin_v,
            dk,
        ]),
    ])
    horizontal_target = np.concatenate([y_hz, y_los])
    horizontal_sigma = np.concatenate([
        np.full(n, float(sigma_hz)), np.full(n, float(sigma_los))])
    fit_h = _huber_irls(
        horizontal_design, horizontal_target, horizontal_sigma,
        cutoff=float(frame_cfg.huber_cutoff),
        max_iterations=int(max_iterations),
        scale_floor=float(scale_floor),
    )

    vertical_design = np.column_stack([
        np.ones(n), dk, dk * np.sin(a), dk * np.cos(a)])
    vertical_sigma = np.full(n, float(sigma_vertical))
    fit_v = _huber_irls(
        vertical_design, y_ver, vertical_sigma,
        cutoff=float(frame_cfg.huber_cutoff),
        max_iterations=int(max_iterations),
        scale_floor=float(scale_floor),
    )

    rank = min(fit_h.rank, fit_v.rank)
    conditions = [value for value in (fit_h.condition_number, fit_v.condition_number)
                  if np.isfinite(value)]
    condition_number = max(conditions) if conditions else None

    if fit_h.rank < len(HORIZONTAL_PARAMETERS) or fit_v.rank < len(VERTICAL_PARAMETERS):
        return CycleFrameFit(
            cycle_id=cycle_id, rank=rank, condition_number=condition_number,
            status=STATUS_RANK_DEFICIENT,
            reason=(f"design rank {fit_h.rank} (horizontal) / {fit_v.rank} "
                    f"(vertical) is below 4; no correction fabricated"))

    limit = frame_cfg.conditioning_limit
    limit_configured = limit is not None and str(limit).strip().lower() != "not_configured"
    if limit_configured and condition_number is not None and condition_number > float(limit):
        return CycleFrameFit(
            cycle_id=cycle_id, rank=rank, condition_number=condition_number,
            status=STATUS_FAILED,
            reason=(f"condition number {condition_number:.3g} exceeds "
                    f"conditioning_limit {float(limit):.3g}"))

    coefficients = dict(zip(
        HORIZONTAL_PARAMETERS, (float(v) for v in fit_h.x), strict=True))
    coefficients.update(dict(zip(
        VERTICAL_PARAMETERS, (float(v) for v in fit_v.x), strict=True)))
    coefficients["rms_hz"] = _robust_sd(fit_h.residual_normalised[:n], float(sigma_hz))
    coefficients["rms_D"] = _robust_sd(fit_h.residual_normalised[n:], float(sigma_los))
    coefficients["rms_v"] = _robust_sd(fit_v.residual_normalised, float(sigma_vertical))

    std_errors = dict(zip(
        HORIZONTAL_PARAMETERS, (float(v) for v in fit_h.se), strict=True))
    std_errors.update(dict(zip(
        VERTICAL_PARAMETERS, (float(v) for v in fit_v.se), strict=True)))

    residuals = pd.DataFrame({
        "point_id": sub["point_id"].to_numpy()[valid],
        "residual_hz_arcsec": fit_h.residual_normalised[:n] * float(sigma_hz),
        "residual_los_mm": fit_h.residual_normalised[n:] * float(sigma_los),
        "residual_ver_mm": fit_v.residual_normalised * float(sigma_vertical),
    })
    return CycleFrameFit(
        cycle_id=cycle_id, coefficients=coefficients, std_errors=std_errors,
        rank=rank, condition_number=condition_number, residuals=residuals,
        status=STATUS_OK, reason=None)


# ---------------------------------------------------------------------------
#  All-cycle fit
# ---------------------------------------------------------------------------
def _as_frame_selection(frame_set) -> FrameSelection:
    if isinstance(frame_set, FrameSelection):
        return frame_set
    members = sorted(set(_iter_members(frame_set)))
    if members:
        return FrameSelection(members=members, status=STATUS_OK, reason=None)
    return FrameSelection(members=[], status=STATUS_INSUFFICIENT,
                          reason="frame set has no members")


def _model_terms(
    frame: pd.DataFrame, coefficients: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Predicted Hz (arcsec), LOS (mm), vertical (mm) and HD (m) for rows."""
    az = _numeric(frame, ("az_deg", "az"))
    zenith = _numeric(frame, ("v", "v_deg"))
    slope = _numeric(frame, ("D", "d_m"))
    if az is None or zenith is None or slope is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.radians(az)
        vertical = np.radians(zenith)
        sin_v = np.sin(vertical)
        hd = slope * sin_v
        dk = slope / 1000.0
        pred_hz = (coefficients["rotation_arcsec"]
                   + _HZ_ARCSEC * (-np.cos(a) * coefficients["tE_mm"]
                                   + np.sin(a) * coefficients["tN_mm"]) / hd)
        pred_los = (-(np.sin(a) * coefficients["tE_mm"]
                      + np.cos(a) * coefficients["tN_mm"]) * sin_v
                    + coefficients["scale_ppm"] * dk)
        pred_ver = (coefficients["vert_offset_mm"]
                    + dk * (coefficients["vert_index_mm_per_km"]
                            + coefficients["tilt_sin"] * np.sin(a)
                            + coefficients["tilt_cos"] * np.cos(a)))
    return pred_hz, pred_los, pred_ver, hd


def _group_corrected(frame: pd.DataFrame, fit: CycleFrameFit) -> pd.DataFrame:
    n = len(frame)
    out = pd.DataFrame(index=np.arange(n))
    out["point_id"] = frame["point_id"].astype(str).to_numpy()
    for column in _GROUP_COLUMNS:
        out[column] = frame[column].to_numpy() if column in frame.columns else None
    if "_order" in frame.columns:
        out["_order"] = frame["_order"].to_numpy()

    los_raw = _numeric(frame, ("dD", "dD_mm", "los_raw_mm", "d_rad_mm"))
    tan_raw_existing = _numeric(frame, ("tan_raw_mm",))
    d_hz = _numeric(frame, ("dHz_as", "d_hz_as"))
    ver_raw = _numeric(frame, ("ver_raw", "ver_raw_mm", "d_vert_mm"))
    terms = _model_terms(frame, fit.coefficients) if fit.status == STATUS_OK else None
    hd = terms[3] if terms is not None else _numeric(frame, ("D", "d_m"))
    if terms is None and hd is not None:
        zenith = _numeric(frame, ("v", "v_deg"))
        if zenith is not None:
            with np.errstate(invalid="ignore"):
                hd = hd * np.sin(np.radians(zenith))

    if los_raw is None:
        los_raw = np.full(n, np.nan)
    if ver_raw is None:
        ver_raw = np.full(n, np.nan)
    if tan_raw_existing is not None:
        tan_raw = tan_raw_existing
    elif d_hz is not None and hd is not None:
        with np.errstate(invalid="ignore"):
            tan_raw = d_hz * hd / _HZ_ARCSEC
    else:
        tan_raw = np.full(n, np.nan)

    out["los_raw_mm"] = los_raw
    out["tan_raw_mm"] = tan_raw
    out["ver_raw_mm"] = ver_raw
    if terms is not None:
        pred_hz, pred_los, pred_ver, _ = terms
        with np.errstate(invalid="ignore"):
            out["los_fc_mm"] = los_raw - pred_los
            out["tan_fc_mm"] = tan_raw - pred_hz * hd / _HZ_ARCSEC
            out["ver_fc_mm"] = ver_raw - pred_ver
            out["status"] = fit.status
            out["reason"] = fit.reason or ""
            return out
    out["los_fc_mm"] = np.full(n, np.nan)
    out["tan_fc_mm"] = np.full(n, np.nan)
    out["ver_fc_mm"] = np.full(n, np.nan)
    out["status"] = fit.status
    out["reason"] = fit.reason or ""
    return out


def fit_frames(series: pd.DataFrame, *, frame_set=None, config=None) -> FrameResult:
    """Fit the frame for every cycle of a long prism-cycle series.

    Parameters
    ----------
    series:
        Long per prism per cycle with ``point_id``, ``cycle_id`` (plus
        ``station_id``/``segment_id`` when available), raw columns ``dD``,
        ``dHz_as``, ``ver_raw`` and geometry ``az_deg``, ``v``, ``D``.
    frame_set:
        :class:`FrameSelection` or iterable of reference prism IDs. The
        selection is published unchanged as the result membership.
    config:
        :class:`~rts_forensics.config.AnalysisConfig`.

    Returns
    -------
    FrameResult
        ``coefficients`` has one row per cycle (deterministic ordering by
        group key), ``corrected`` keeps raw and corrected values side by side
        in the input row order, and ``diagnostics`` records per-cycle rank,
        conditioning and frame-set membership. Cycles whose fit failed carry
        NaN corrected values with an explicit status/reason — never zero.
        ``experimental`` is always True; the raw input frame is never mutated.
    """
    cfg = _resolve_config(config)
    frame_cfg = cfg.frame
    membership = _as_frame_selection(frame_set)

    convergence = frame_cfg.convergence or {}
    tuning_missing = (
        frame_cfg.huber_cutoff is None
        or convergence.get("max_iterations") is None
        or convergence.get("scale_floor") is None
        or not frame_cfg.weights)
    work = _with_point_id(series)
    key_columns = [c for c in _GROUP_COLUMNS if c in work.columns]
    if "cycle_id" not in key_columns:
        return FrameResult(
            membership=membership, correction_status=NOT_CONFIGURED,
            experimental=True, status=NOT_CONFIGURED,
            reason="series has no cycle_id column")

    if tuning_missing:
        corrected = _group_corrected(
            work.assign(_order=np.arange(len(work))),
            CycleFrameFit(
                status=NOT_CONFIGURED,
                reason="frame Huber cutoff and convergence settings are not configured"))
        corrected = corrected.sort_values("_order", ignore_index=True)
        return FrameResult(
            corrected=corrected.loc[:, list(CORRECTED_COLUMNS)],
            membership=membership, correction_status=NOT_CONFIGURED,
            experimental=True, status=NOT_CONFIGURED,
            reason="frame Huber cutoff and convergence settings are not configured")

    work = work.reset_index(drop=True)
    work["_order"] = np.arange(len(work))
    groups = list(work.groupby(key_columns, sort=False, dropna=False))
    groups.sort(key=lambda item: _sort_key(item[0]))

    members = set(membership.members)
    coefficient_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    corrected_parts: list[pd.DataFrame] = []
    residual_parts: list[pd.DataFrame] = []
    statuses: list[str] = []

    for key, group in groups:
        key_values = key if isinstance(key, tuple) else (key,)
        base = {column: None for column in _GROUP_COLUMNS}
        base.update(dict(zip(key_columns, key_values, strict=True)))
        fit = fit_cycle_frame(group, frame_set=membership, config=cfg)
        statuses.append(fit.status)

        in_frame = group["point_id"].isin(members)
        n_frame_members = int(in_frame.sum())
        n_used = len(fit.residuals) if isinstance(fit.residuals, pd.DataFrame) else 0

        coefficient_row = dict(base)
        for name, se_name in _STD_ERROR_NAMES.items():
            coefficient_row[name] = np.nan
            coefficient_row[se_name] = np.nan
        for name in RMS_PARAMETERS:
            coefficient_row[name] = np.nan
        coefficient_row["rank"] = fit.rank
        coefficient_row["condition_number"] = (
            float(fit.condition_number) if fit.condition_number is not None else np.nan)
        coefficient_row["n_frame_members"] = n_frame_members
        coefficient_row["n_used"] = n_used
        coefficient_row["status"] = fit.status
        coefficient_row["reason"] = fit.reason or ""
        if fit.status == STATUS_OK:
            for name in HORIZONTAL_PARAMETERS + VERTICAL_PARAMETERS:
                coefficient_row[name] = float(fit.coefficients[name])
                coefficient_row[_STD_ERROR_NAMES[name]] = float(fit.std_errors[name])
            for name in RMS_PARAMETERS:
                coefficient_row[name] = float(fit.coefficients.get(name, np.nan))
        coefficient_rows.append(coefficient_row)

        diagnostic_row = dict(base)
        diagnostic_row.update({
            "n_frame_members": n_frame_members,
            "n_rows": int(len(group)),
            "n_used": n_used,
            "rank": fit.rank,
            "condition_number": coefficient_row["condition_number"],
            "rms_hz": coefficient_row["rms_hz"],
            "rms_D": coefficient_row["rms_D"],
            "rms_v": coefficient_row["rms_v"],
            "frame_members": ";".join(membership.members),
            "status": fit.status,
            "reason": fit.reason or "",
        })
        diagnostic_rows.append(diagnostic_row)

        corrected_parts.append(_group_corrected(group, fit))
        if isinstance(fit.residuals, pd.DataFrame) and not fit.residuals.empty:
            residuals = fit.residuals.copy()
            for column, value in base.items():
                residuals[column] = value
            residuals["status"] = fit.status
            residual_parts.append(residuals)

    coefficients = pd.DataFrame(coefficient_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    if corrected_parts:
        corrected = pd.concat(corrected_parts, ignore_index=True)
        corrected = corrected.sort_values("_order", ignore_index=True)
        corrected = corrected.loc[:, list(CORRECTED_COLUMNS)]
    else:
        corrected = pd.DataFrame(columns=CORRECTED_COLUMNS)
    residuals = pd.concat(residual_parts, ignore_index=True) \
        if residual_parts else pd.DataFrame(
            columns=["point_id", *_GROUP_COLUMNS, "residual_hz_arcsec",
                     "residual_los_mm", "residual_ver_mm", "status"])

    numeric_columns = [c for c in _COEFFICIENT_NUMERIC_COLUMNS if c in coefficients.columns]
    for column in numeric_columns:
        coefficients[column] = pd.to_numeric(coefficients[column], errors="coerce").astype(float)
    for column in _CORRECTED_NUMERIC_COLUMNS:
        corrected[column] = pd.to_numeric(corrected[column], errors="coerce").astype(float)

    if statuses and all(status == STATUS_OK for status in statuses):
        correction_status = STATUS_OK
        reason = None
    elif any(status == STATUS_OK for status in statuses):
        correction_status = "partial"
        reason = "some cycles could not be fitted; corrected values are NaN there"
    else:
        correction_status = NOT_CONFIGURED
        reasons = []
        for row in coefficient_rows:
            if row["reason"] and row["reason"] not in reasons:
                reasons.append(row["reason"])
        reason = "; ".join(reasons) or "no cycle could be fitted"

    return FrameResult(
        coefficients=coefficients, diagnostics=diagnostics, residuals=residuals,
        membership=membership, corrected=corrected,
        correction_status=correction_status, experimental=True,
        status=correction_status, reason=reason)


def apply_frame(series: pd.DataFrame, frames: FrameResult) -> pd.DataFrame:
    """Merge corrected frame columns onto a copy of ``series``.

    All raw columns are preserved; existing columns are never overwritten.
    Colliding corrected names are prefixed with ``frame_`` and the per-row
    correction ``status``/``reason`` become ``frame_status``/``frame_reason``.
    """
    out = series.copy()
    corrected = frames.corrected if frames is not None else None
    if not isinstance(corrected, pd.DataFrame) or corrected.empty:
        return out
    keys = [column for column in _ID_COLUMNS
            if column in out.columns and column in corrected.columns]
    if "point_id" not in keys or "cycle_id" not in keys:
        return out
    payload = corrected.drop_duplicates(subset=keys).copy()
    rename = {}
    for column in payload.columns:
        if column in keys:
            continue
        if column == "status":
            rename[column] = "frame_status"
        elif column == "reason":
            rename[column] = "frame_reason"
        elif column in out.columns:
            rename[column] = f"frame_{column}"
    payload = payload.rename(columns=rename)
    columns = keys + [c for c in payload.columns if c not in keys]
    merged = out.merge(payload[columns], on=keys, how="left", sort=False)
    merged.index = out.index
    return merged
