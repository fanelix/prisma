"""
rts_forensics.tests — analytical hypothesis tests (not pytest software tests).

This sub-package is deliberately separate from the repository's root ``tests/``
directory: here "tests" means statistical hypothesis tests of the monitoring
record. Each module exposes named-input functions whose thresholds come from
``config.forensics`` (or another documented supplied setting); where a decision
threshold is not configured the statistic is still computed but the decision
column and status are ``not_configured`` with an explicit reason.

Documented ``series`` schema
----------------------------
One row per prism per cycle: ``point_id`` (str) and ``ts`` (datetime64[ns],
site-local naive) are required. Values are supplied either long-form
(``metric`` plus ``value``/``metric_value``/``value_mm``) or wide-form using the
documented aliases (``los_mm``/``los_fc_mm``/``d_rad_mm``, ``vert_mm``/
``ver_fc_mm``/``ver_raw_mm``/``d_vert_mm``, ``tan_mm``/``tan_fc_mm``/
``d_tan_mm``). Optional context columns are carried through: ``cycle_id``,
``station_id``, ``segment_id``, ``n_obs``, ``obs_ids``,
``az_deg``/``azimuth_deg``, ``v_deg``/``zenith_deg``, ``hz_deg``, ``d_m``.

``coordinates`` schema: ``point_id``, ``east``, ``north`` (metres, supplied
Cartesian site grid); optional geometry columns (``az_deg``/``azimuth_deg``,
``v_deg``/``zenith_deg``) and an optional boolean ``candidate``/``candidate_zone``
column naming the candidate cluster.

``station_series`` schema: ``ts`` and/or ``cycle_id``; a height column
(``height``/``sh_m``/``height_m``/``station_height`` in metres or
``height_mm``/``sh_mm`` in millimetres) and/or an orientation column
(``orientation_deg``/``orientation``/``hz_deg``).

Public API: :func:`run_forensics` runs every test whose inputs are available and
returns a :class:`~rts_forensics.models.ForensicResult` with an evidence
register, per-test detail tables, an investigation register and statuses. No
slope is ever declared safe or unsafe, no cause is asserted, and unobserved
periods are never described as stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import (
    EVIDENCE_COLUMNS,
    INVESTIGATION_COLUMNS,
    NOT_CONFIGURED,
    STATUS_UNAVAILABLE,
    ForensicResult,
)
from . import events, groups, missingness, references, sampling
from .groups import resolve_config

__all__ = [
    "events",
    "groups",
    "missingness",
    "references",
    "run_forensics",
    "sampling",
]

_METRIC = "los"

_EXPLANATIONS: dict[str, tuple[str, str]] = {
    "candidate spatial clusters": (
        "spatially coherent movement shared by neighbouring prisms",
        "shared frame/atmospheric/geometry artefact; a cluster is a candidate, "
        "not a movement decision"),
    "spatial placebo neighbours": (
        "prism net change differs from its neighbourhood",
        "network-wide (common-mode) change; the placebo has no configured "
        "decision threshold"),
    "temporal split": (
        "step or change concentrated in one half of the record",
        "sampling change or an instrument event inside one half"),
    "change point on block medians": (
        "hinge/segment change in the block-median series",
        "block quantisation and noise; significance is not configured"),
    "pairwise LOS differences": (
        "relative movement between the two prisms",
        "signals shared by the pair cancel; a pair result is relative, not "
        "absolute"),
    "3-D movement vectors": (
        "radial, tangential and vertical components agree on a 3-D vector",
        "frame-correction artefact; tangential-only signals need independent "
        "consistency evidence"),
    "day/night net change": (
        "movement detected in one time-of-day band only",
        "changing night/day sampling composition"),
    "hour-matched net change": (
        "movement present with matched observation hours",
        "sampling bias removed by matching; a remaining change still needs a "
        "significance rule that is not configured"),
    "sampling composition": (
        "observed-hour composition changed between the windows",
        "scheduling change; composition is not itself movement"),
    "daytime bias": (
        "daytime reading difference (possible measurement artefact)",
        "same-geometry control change or real movement correlated with day; no "
        "cause is asserted"),
    "reference inference": (
        "prism correlates with station height/orientation (possible resection "
        "reference)",
        "coincidental common trend; correlation is not proof of reference "
        "identity"),
    "resection validation": (
        "raw common vertical responds to exported station height one-for-one",
        "station-height estimates unsupported by raw data; frame change"),
    "coverage and final window": (
        "prism has no observation in the final window",
        "scheduling or observability change; an unobserved period is not "
        "evidence of stability"),
    "lost versus retained distance": (
        "lost prisms are closer to the candidate zone",
        "chance, geometry or scheduling; exploratory only"),
    "pre-loss trends": (
        "prism trend before loss differs from its peers",
        "sampling change or sparse window; no significance is configured"),
    "observability by hour": (
        "prism is night-blind in the final window",
        "scheduling/targeting change; the cause is not asserted"),
    "balanced step test": (
        "step at the candidate time with matched observation hours",
        "sampling composition or a common-mode instrument change"),
    "common-mode steps": (
        "network common-mode step candidate",
        "instrument/frame change versus real common movement versus a "
        "processing change"),
}


def _finite_statistic(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _slug(text: str) -> str:
    return "".join(character if character.isalnum() else "_"
                   for character in str(text)).strip("_")


def _reference_alpha(config) -> float | None:
    value = (config.forensics.references or {}).get("alpha")
    if value is None or isinstance(value, str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_forensics(series: pd.DataFrame, *,
                  coordinates: pd.DataFrame | None = None,
                  station_series: pd.DataFrame | None = None,
                  config=None) -> ForensicResult:
    """Run the available forensic tests and assemble a ``ForensicResult``.

    Coordinate-dependent tests run only when ``coordinates`` is supplied;
    reference tests run only when ``station_series`` is supplied. Every test
    contributes exactly one row to the evidence register with an explicit
    ``status`` and ``reason``; unavailable inputs and unconfigured decision
    thresholds are recorded rather than guessed. The investigation register
    carries only ``open`` (result produced, review needed) and
    ``not_configured`` statuses. Detail tables are returned under the test's
    key in ``ForensicResult.evidence``; the long register is under
    ``"register"``.
    """
    cfg = resolve_config(config)
    register: list[dict] = []
    details: dict[str, pd.DataFrame] = {}

    def record(test: str, hypothesis: str, metric: str, statistic,
               result, *, p_value=None, alpha=None, members=None) -> None:
        status = getattr(result, "status", NOT_CONFIGURED)
        reason = getattr(result, "reason", None) or ""
        register.append({
            "test": test,
            "hypothesis": hypothesis,
            "metric": metric,
            "statistic": _finite_statistic(statistic),
            "p_value": p_value,
            "alpha": alpha,
            "members": members,
            "status": status,
            "reason": reason,
        })

    def unavailable(test: str, hypothesis: str, metric: str,
                    reason: str) -> None:
        register.append({
            "test": test, "hypothesis": hypothesis, "metric": metric,
            "statistic": float("nan"), "p_value": None, "alpha": None,
            "members": None, "status": STATUS_UNAVAILABLE, "reason": reason,
        })

    def attempt(test: str, hypothesis: str, metric: str, call,
                statistic, *, p_value=None, alpha=None, members=None):
        try:
            result = call()
        except ValueError as error:
            unavailable(test, hypothesis, metric,
                        f"test could not run with the supplied tables: {error}")
            return None
        record(test, hypothesis, metric, statistic(result), result,
               p_value=p_value(result) if p_value else None,
               alpha=alpha, members=members(result) if members else None)
        return result

    # -- groups -------------------------------------------------------------
    if coordinates is None:
        unavailable("groups", "candidate spatial clusters", _METRIC,
                    "coordinates not supplied; no spatial cluster search is "
                    "invented")
        unavailable("groups", "spatial placebo neighbours", _METRIC,
                    "coordinates not supplied; the neighbourhood cannot be "
                    "built")
        unavailable("groups", "pairwise LOS differences", _METRIC,
                    "coordinates not supplied; separation distance is a "
                    "documented input")
    else:
        cluster_result = attempt(
            "groups", "candidate spatial clusters", _METRIC,
            lambda: groups.candidate_clusters(series, coordinates=coordinates,
                                              config=cfg),
            lambda result: len(result.members),
            members=lambda result: result.members)
        if cluster_result is not None:
            details["candidate_clusters"] = cluster_result.clusters

        placebo_result = attempt(
            "groups", "spatial placebo neighbours", _METRIC,
            lambda: groups.spatial_placebo(series, coordinates=coordinates,
                                           config=cfg),
            lambda result: float(pd.Series(
                result.null_distribution).abs().median()))
        if placebo_result is not None:
            details["spatial_placebo"] = placebo_result.table

        pair_result = attempt(
            "groups", "pairwise LOS differences", _METRIC,
            lambda: groups.pairwise_los(series, coordinates=coordinates,
                                        config=cfg),
            lambda result: float(result.table["rate_mm_day"].abs().median())
            if not result.table.empty else float("nan"),
            members=lambda result: result.members)
        if pair_result is not None:
            details["pairwise_los"] = pair_result.table

    split_result = attempt(
        "groups", "temporal split", _METRIC,
        lambda: groups.temporal_split(series, config=cfg),
        lambda result: float(result.table["net_mm"].median())
        if not result.table.empty else float("nan"))
    if split_result is not None:
        details["temporal_split"] = split_result.table
        details["temporal_split_contributors"] = split_result.contributors

    change_result = attempt(
        "groups", "change point on block medians", _METRIC,
        lambda: groups.change_point(series, config=cfg),
        lambda result: float(result.table["hinge_improvement_fraction"].median())
        if not result.table.empty else float("nan"))
    if change_result is not None:
        details["change_point"] = change_result.table
        details["change_point_sensitivity"] = change_result.sensitivity

    vector_result = attempt(
        "groups", "3-D movement vectors", _METRIC,
        lambda: groups.movement_vectors(series, coordinates=coordinates,
                                        config=cfg),
        lambda result: int(len(result.vectors)))
    if vector_result is not None:
        details["movement_vectors"] = vector_result.vectors

    # -- sampling -----------------------------------------------------------
    day_night_result = attempt(
        "sampling", "day/night net change", _METRIC,
        lambda: sampling.day_night_comparison(series, config=cfg),
        lambda result: float(result.table["net_mm"].median())
        if not result.table.empty else float("nan"))
    if day_night_result is not None:
        details["day_night"] = day_night_result.table
        details["day_night_composition"] = day_night_result.composition

    hour_result = attempt(
        "sampling", "hour-matched net change", _METRIC,
        lambda: sampling.hour_matched_change(series, config=cfg),
        lambda result: float(result.table["net_mm"].median())
        if not result.table.empty else float("nan"))
    if hour_result is not None:
        details["hour_matched"] = hour_result.table
        details["hour_matched_classes"] = hour_result.classes

    composition_result = attempt(
        "sampling", "sampling composition", _METRIC,
        lambda: sampling.sampling_composition(series, config=cfg),
        lambda result: int(result.table["composition_changed"].sum())
        if not result.table.empty else 0)
    if composition_result is not None:
        details["sampling_composition"] = composition_result.table

    bias_result = attempt(
        "sampling", "daytime bias", _METRIC,
        lambda: sampling.daytime_bias(series, config=cfg),
        lambda result: float(result.table["day_minus_night_mm"].abs().median())
        if not result.table.empty else float("nan"))
    if bias_result is not None:
        details["daytime_bias"] = bias_result.table
        details["daytime_bias_weekly"] = bias_result.weekly

    # -- references ---------------------------------------------------------
    if station_series is None:
        unavailable("references", "reference inference", "vert",
                    "station_series not supplied; no station correlation is "
                    "invented")
        unavailable("references", "resection validation", "vert",
                    "station_series not supplied; the expected slope cannot "
                    "be compared")
    else:
        reference_result = attempt(
            "references", "reference inference", "vert",
            lambda: references.reference_inference(
                series, station_series=station_series, config=cfg),
            lambda result: float(result.correlations["p_adjusted"].min())
            if not result.correlations.empty else float("nan"),
            p_value=lambda result: (
                float(result.correlations["p_adjusted"].min())
                if not result.correlations.empty else None),
            alpha=_reference_alpha(cfg))
        if reference_result is not None:
            details["reference_inference"] = reference_result.correlations

        resection_result = attempt(
            "references", "resection validation", "vert",
            lambda: references.resection_validation(
                series, station_series=station_series, config=cfg),
            lambda result: result.slope)
        if resection_result is not None:
            details["resection_validation"] = resection_result.table

    # -- missingness --------------------------------------------------------
    coverage_result = attempt(
        "missingness", "coverage and final window", _METRIC,
        lambda: missingness.coverage(series, config=cfg),
        lambda result: int((~result.table["in_final_window"]).sum())
        if not result.table.empty else 0)
    if coverage_result is not None:
        details["coverage"] = coverage_result.table

    if coordinates is None:
        unavailable("missingness", "lost versus retained distance", _METRIC,
                    "coordinates not supplied; distances to the candidate "
                    "zone cannot be computed")
    else:
        lost_result = attempt(
            "missingness", "lost versus retained distance", _METRIC,
            lambda: missingness.lost_vs_retained(
                series, coordinates=coordinates, config=cfg),
            lambda result: float(result.tests["statistic_mm"].iloc[0])
            if not result.tests.empty else float("nan"),
            p_value=lambda result: (
                float(result.tests["p_permutation"].iloc[0])
                if not result.tests.empty else None))
        if lost_result is not None:
            details["lost_vs_retained"] = lost_result.table
            details["lost_vs_retained_tests"] = lost_result.tests

    trend_result = attempt(
        "missingness", "pre-loss trends", _METRIC,
        lambda: missingness.pre_loss_trends(series, config=cfg),
        lambda result: float(result.table["net_mm"].abs().max())
        if not result.table.empty else float("nan"))
    if trend_result is not None:
        details["pre_loss_trends"] = trend_result.table

    observability_result = attempt(
        "missingness", "observability by hour", _METRIC,
        lambda: missingness.observability_by_hour(series, config=cfg),
        lambda result: int(result.table["night_blind"].sum())
        if not result.table.empty else 0)
    if observability_result is not None:
        details["observability_by_hour"] = observability_result.table
        details["observability_by_hour_classes"] = observability_result.by_hour

    # -- events -------------------------------------------------------------
    balanced_result = attempt(
        "events", "balanced step test", _METRIC,
        lambda: events.balanced_step_test(series, config=cfg),
        lambda result: float(result.pairs["difference_mm"].abs().median())
        if not result.pairs.empty else float("nan"))
    if balanced_result is not None:
        details["balanced_step"] = balanced_result.table
        details["balanced_step_pairs"] = balanced_result.pairs

    common_result = attempt(
        "events", "common-mode steps", _METRIC,
        lambda: events.common_mode_steps(series, config=cfg),
        lambda result: float(result.steps["robust_z"].abs().max())
        if not result.steps.empty else float("nan"))
    if common_result is not None:
        details["common_mode_steps"] = common_result.steps

    register_frame = pd.DataFrame(register, columns=list(EVIDENCE_COLUMNS))
    register_frame["p_value"] = pd.to_numeric(register_frame["p_value"],
                                              errors="coerce")
    register_frame["statistic"] = pd.to_numeric(register_frame["statistic"],
                                                errors="coerce")
    register_frame["alpha"] = pd.to_numeric(register_frame["alpha"],
                                            errors="coerce")

    if "ts" in series.columns and len(series):
        opened = pd.to_datetime(series["ts"]).max()
    else:
        opened = pd.NaT
    investigations: list[dict] = []
    for row in register:
        candidate, competing = _EXPLANATIONS.get(
            row["hypothesis"],
            ("candidate explanation not enumerated",
             "competing explanations not enumerated"))
        investigations.append({
            "investigation_id": f"{row['test']}:{_slug(row['hypothesis'])}",
            "opened": opened,
            "topic": row["hypothesis"],
            "candidate_explanation": candidate,
            "competing_explanation": competing,
            "evidence": f"{row['test']}.{_slug(row['hypothesis'])}",
            "status": ("not_configured" if row["status"] == NOT_CONFIGURED
                       else "open"),
        })
    investigation_frame = pd.DataFrame(
        investigations, columns=list(INVESTIGATION_COLUMNS))

    statuses = (register_frame[["test", "hypothesis", "status", "reason"]]
                .copy())
    evidence = {"register": register_frame, **details}
    return ForensicResult(evidence=evidence, investigations=investigation_frame,
                          statuses=statuses)
