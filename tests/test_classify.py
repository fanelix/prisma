"""
test_classify.py — concern, reliability and TARP boundaries and separation.

All boundary values come from the test configuration, never from production
constants. The tests assert strict/eq behaviour exactly at the configured
cut-offs, the confidence-interval sign rule, insufficient-data handling, the
absence of any alarm state without a TARP and that concern and reliability are
never merged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rts_forensics.classify import (
    apply_tarp,
    classify,
    grade_reliability,
    screen_movement,
)
from rts_forensics.config import load_config
from rts_forensics.models import (
    CONCERN_COLUMNS,
    NOT_CONFIGURED,
    RELIABILITY_COLUMNS,
    STATUS_INSUFFICIENT,
    STATUS_OK,
    TARP_COLUMNS,
)

_CONFIG_YAML = Path(__file__).resolve().parents[1] / "config.yaml"


def _config(**screening_overrides):
    return load_config(overrides={"screening": screening_overrides})


def _effective_config(**overrides):
    return load_config(_CONFIG_YAML, overrides=overrides or None)


def _summary(**row) -> pd.DataFrame:
    base = {
        "point_id": "P1",
        "segment_id": "E-S1",
        "metric": "d_rad",
        "net_mm": 0.0,
        "sigma_mm": 1.0,
        "ci_low": -1.0,
        "ci_high": 1.0,
        "n_prism_cycles": 50,
    }
    base.update(row)
    return pd.DataFrame([base])


def _row(frame: pd.DataFrame, point_id: str) -> pd.Series:
    return frame.set_index("point_id").loc[point_id]


# ---------------------------------------------------------------------------
#  Movement concern
# ---------------------------------------------------------------------------
def test_screen_sigma_threshold_boundary():
    config = _config()
    above = screen_movement(
        _summary(net_mm=6.5, sigma_mm=2.0, ci_low=1.0, ci_high=10.0), config=config)
    assert bool(above.loc[0, "exceeds"])
    assert above.loc[0, "floor_mm"] == pytest.approx(2.0)

    just_below = screen_movement(
        _summary(net_mm=5.999, sigma_mm=2.0, ci_low=1.0, ci_high=10.0), config=config)
    assert not bool(just_below.loc[0, "exceeds"])

    exact = screen_movement(
        _summary(net_mm=6.0, sigma_mm=2.0, ci_low=1.0, ci_high=10.0), config=config)
    assert bool(exact.loc[0, "exceeds"])


def test_screen_floor_dominates_when_sigma_is_small():
    config = _config()
    above = screen_movement(
        _summary(net_mm=2.01, sigma_mm=0.5, ci_low=0.5, ci_high=5.0), config=config)
    assert bool(above.loc[0, "exceeds"])
    below = screen_movement(
        _summary(net_mm=1.99, sigma_mm=0.5, ci_low=0.5, ci_high=5.0), config=config)
    assert not bool(below.loc[0, "exceeds"])


def test_screen_vertical_floor_and_possible():
    config = _config()
    above = screen_movement(
        _summary(metric="d_vert", net_mm=5.0, sigma_mm=1.0,
                 ci_low=1.0, ci_high=8.0), config=config)
    assert bool(above.loc[0, "exceeds"])
    assert above.loc[0, "floor_mm"] == pytest.approx(5.0)

    below = screen_movement(
        _summary(metric="d_vert", net_mm=4.99, sigma_mm=1.0,
                 ci_low=1.0, ci_high=8.0), config=config)
    assert not bool(below.loc[0, "exceeds"])

    possible = screen_movement(
        _summary(metric="d_vert", net_mm=4.0, sigma_mm=1.0,
                 ci_low=1.0, ci_high=8.0), config=config)
    assert not bool(possible.loc[0, "exceeds"])
    assert bool(possible.loc[0, "possible"])
    assert possible.loc[0, "status"] == STATUS_OK


def test_screen_confidence_interval_sign_rule():
    same_sign = screen_movement(
        _summary(net_mm=-10.0, sigma_mm=1.0, ci_low=-9.0, ci_high=-1.0), config=_config())
    assert bool(same_sign.loc[0, "exceeds"])

    opposite_sign = screen_movement(
        _summary(net_mm=-10.0, sigma_mm=1.0, ci_low=1.0, ci_high=9.0), config=_config())
    assert not bool(opposite_sign.loc[0, "exceeds"])

    no_rule = screen_movement(
        _summary(net_mm=-10.0, sigma_mm=1.0, ci_low=1.0, ci_high=9.0),
        config=_config(require_same_sign_ci=False))
    assert bool(no_rule.loc[0, "exceeds"])

    spans_zero = screen_movement(
        _summary(net_mm=10.0, sigma_mm=1.0, ci_low=-1.0, ci_high=9.0), config=_config())
    assert not bool(spans_zero.loc[0, "exceeds"])


def test_screen_insufficient_cycles():
    config = _config()
    frame = screen_movement(
        _summary(net_mm=50.0, sigma_mm=0.1, ci_low=49.0, ci_high=51.0,
                 n_prism_cycles=19), config=config)
    assert frame.loc[0, "status"] == STATUS_INSUFFICIENT
    assert not bool(frame.loc[0, "exceeds"])
    assert not bool(frame.loc[0, "possible"])
    assert "20" in frame.loc[0, "reason"]


def test_screen_reason_is_exploratory_and_column_contract():
    config = _config()
    frame = screen_movement(
        _summary(net_mm=6.5, sigma_mm=2.0, ci_low=1.0, ci_high=10.0), config=config)
    assert set(CONCERN_COLUMNS).issubset(frame.columns)
    assert "exploratory" in frame.loc[0, "reason"]
    assert "TARP" in frame.loc[0, "reason"]


# ---------------------------------------------------------------------------
#  Reliability
# ---------------------------------------------------------------------------
def _reliability_summary(rows: list[dict]) -> pd.DataFrame:
    base = {
        "segment_id": "E-S1",
        "n_prism_cycles": 20,
        "coverage_pct": 80.0,
        "max_gap_hours": 72.0,
        "obs_last48h": True,
        "sigma_los_mm": 2.0,
        "sigma_vert_mm": 6.0,
        "n_spike_flags": 4,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_reliability_boundaries_exactly_at_cutoffs():
    config = _effective_config()
    summary = _reliability_summary([
        {"point_id": "A", "coverage_pct": 80.0},
        {"point_id": "B_coverage", "coverage_pct": 79.9},
        {"point_id": "B_spikes", "n_spike_flags": 5},
        {"point_id": "C_coverage", "coverage_pct": 49.9},
        {"point_id": "C_gap", "max_gap_hours": 72.1},
        {"point_id": "C_final", "obs_last48h": False},
        {"point_id": "C_los", "sigma_los_mm": 2.1},
        {"point_id": "C_vert", "sigma_vert_mm": 6.1},
        {"point_id": "D_cycles", "n_prism_cycles": 19},
        {"point_id": "B_weak_exact", "coverage_pct": 50.0},
    ])
    graded = grade_reliability(summary, config=config)
    assert _row(graded, "A")["grade"] == "A"
    assert _row(graded, "B_coverage")["grade"] == "B"
    assert _row(graded, "B_spikes")["grade"] == "B"
    assert _row(graded, "C_coverage")["grade"] == "C"
    assert _row(graded, "C_gap")["grade"] == "C"
    assert _row(graded, "C_final")["grade"] == "C"
    assert _row(graded, "C_los")["grade"] == "C"
    assert _row(graded, "C_vert")["grade"] == "C"
    assert _row(graded, "D_cycles")["grade"] == "D"
    assert _row(graded, "D_cycles")["status"] == STATUS_INSUFFICIENT
    assert _row(graded, "B_weak_exact")["grade"] == "B"
    assert "coverage_weak" not in _row(graded, "B_weak_exact")["flags"]
    assert "coverage_moderate" in _row(graded, "B_weak_exact")["flags"]
    assert set(RELIABILITY_COLUMNS).issubset(graded.columns)
    assert (graded["flags"].fillna("") != "").any()


def test_reliability_not_configured():
    config = _effective_config(reliability={"boundaries": None})
    graded = grade_reliability(_reliability_summary([{"point_id": "A"}]), config=config)
    assert graded.loc[0, "grade"] == NOT_CONFIGURED
    assert graded.loc[0, "status"] == NOT_CONFIGURED
    assert graded.loc[0, "reason"]


# ---------------------------------------------------------------------------
#  TARP
# ---------------------------------------------------------------------------
def _series() -> pd.DataFrame:
    return pd.DataFrame({
        "point_id": ["P1", "P1"],
        "segment_id": ["E-S1", "E-S1"],
        "metric": ["los", "vertical"],
        "value_mm": [12.0, 3.0],
    })


def test_tarp_absent_produces_no_alarm_state():
    absent = apply_tarp(_series(), tarp=None)
    assert list(absent.columns) == list(TARP_COLUMNS)
    assert set(absent["state"]) == {"no_tarp_configured"}
    assert set(absent["status"]) == {NOT_CONFIGURED}
    assert all("never declared safe or unsafe" in reason for reason in absent["reason"])
    assert not absent["state"].str.contains("alarm", case=False).any()

    unconfigured = apply_tarp(_series(), tarp="not_configured")
    assert set(unconfigured["state"]) == {"no_tarp_configured"}


def test_tarp_applies_only_supplied_thresholds():
    tarp = {"los_mm": 10.0, "averaging_days": 3, "persistence_hours": 24}
    applied = apply_tarp(_series(), tarp=tarp)
    los = applied[applied["metric"] == "los"].iloc[0]
    vertical = applied[applied["metric"] == "vertical"].iloc[0]
    assert los["state"] == "exceeded"
    assert los["tarp_mm"] == pytest.approx(10.0)
    assert los["status"] == "applied"
    assert vertical["state"] == "not_configured"
    assert vertical["status"] == NOT_CONFIGURED
    assert "averaging_days" not in set(applied["metric"])


# ---------------------------------------------------------------------------
#  Combined classification
# ---------------------------------------------------------------------------
def test_classify_keeps_concern_and_reliability_separate():
    config = _effective_config()
    summary = _summary(
        point_id="M1", metric="los", net_mm=20.0, sigma_mm=1.0,
        ci_low=18.0, ci_high=22.0, n_prism_cycles=50)
    result = classify(summary, _series(), config=config)

    assert set(CONCERN_COLUMNS).issubset(result.concern.columns)
    assert set(RELIABILITY_COLUMNS).issubset(result.reliability.columns)
    assert "grade" not in result.concern.columns
    assert "exceeds" not in result.reliability.columns
    assert bool(result.concern.loc[0, "exceeds"])
    assert result.concern.loc[0, "status"] == STATUS_OK
    assert result.reliability.loc[0, "grade"] in {"A", "B", "C", "D"}
    assert set(result.flags.columns) == {"point_id", "segment_id", "flags"}
    assert set(result.statuses) == {"concern", "reliability", "tarp"}
    assert result.statuses["tarp"] == NOT_CONFIGURED
    assert set(result.tarp["state"]) == {"no_tarp_configured"}
