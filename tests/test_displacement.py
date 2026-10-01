"""
test_displacement.py — prism-cycle aggregation, baselines and displacements.

Fixtures come from the independent forward model in ``tests/synthetic.py``
(parsed through :func:`rts_forensics.io.read_geomos`) and from hand-built
prism-cycle frames with known values. The injected motion, rotation and Hz
repeats are explicit test parameters; the configuration under test is passed
by every call and never defaulted silently.

The rotation test exercises the documented artefact: rotation enters the raw
Hz observations but not the delivered coordinates, so the raw tangential
series shows it while ``d_east_mm``/``d_north_mm`` stay clean, and the break
re-anchors the baseline instead of bridging it.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import synthetic
from synthetic import Effects, Prism, Station

from rts_forensics import cycles
from rts_forensics.config import load_config
from rts_forensics.displacement import aggregate_prism_cycles, compute_displacements
from rts_forensics.io import read_geomos
from rts_forensics.models import (
    BASELINE_COLUMNS,
    DISPLACEMENT_COLUMNS,
    NOT_CONFIGURED,
    PRISM_CYCLE_COLUMNS,
    SOURCE_LINK_COLUMNS,
    STATUS_OK,
)

_START = datetime(2026, 1, 1)
_BREAK = pd.Timestamp("2026-01-02 00:00")


def _anchored_config(**overrides):
    data = {
        "stations": {
            "anchors": [
                {"label": "E", "easting_ref": 1000.0},
                {"label": "W", "easting_ref": 0.0},
            ]
        }
    }
    data.update(overrides)
    return load_config(overrides=data)


def _single_station() -> dict:
    return {"E": Station("E", 1000.0, 2000.0, 50.0, 0.0, 30,
                         [0, 4, 8, 12, 16, 20])}


def _two_prisms(*, mover: str = "M", rate_mm_per_day: float = 3.0) -> dict:
    motion = {mover: rate_mm_per_day} if rate_mm_per_day else {}
    return {
        "M": Prism("M", "E", 1400.0, 2600.0, 58.0),
        "S": Prism("S", "E", 1200.0, 1800.0, 45.0),
    }, motion


def _synthetic(tmp_path, *, days: int, effects: Effects,
               stations: dict | None = None, prisms: dict | None = None,
               name: str = "export.csv"):
    path, _ = synthetic.write_fixture(
        tmp_path / name, days=days, stations=stations, prisms=prisms,
        effects=effects)
    return read_geomos(path).observations


def _prism_cycle_row(point_id: str, cycle: int, hours: float, *, station: str = "E",
                     hz_deg: float = 100.0, v_deg: float = 90.0, d_m: float = 1000.0,
                     te_m: float = 0.0, tn_m: float = 1000.0, tz_m: float = 0.0,
                     obs_id: str = "src:1", start: datetime = _START) -> dict:
    timestamp = start + timedelta(hours=hours)
    return {
        "point_id": point_id,
        "station_id": station,
        "cycle_id": f"{station}{cycle:03d}",
        "cycle_start": pd.Timestamp(timestamp),
        "cycle_end": pd.Timestamp(timestamp) + timedelta(minutes=1),
        "n_obs": 1,
        "hz_deg": hz_deg,
        "v_deg": v_deg,
        "d_m": d_m,
        "te_m": te_m,
        "tn_m": tn_m,
        "tz_m": tz_m,
        "se_m": 1000.0,
        "sn_m": 2000.0,
        "sh_m": 50.0,
        "sd_hz_deg": 0.0,
        "sd_v_deg": 0.0,
        "sd_d_m": 0.0,
        "obs_ids": (obs_id,),
    }


# ---------------------------------------------------------------------------
#  D. Prism-cycle aggregation
# ---------------------------------------------------------------------------
def test_aggregate_prism_cycles_returns_documented_columns(tmp_path):
    observations = _synthetic(tmp_path, days=2, effects=Effects(),
                              stations=_single_station())
    result = cycles.build_cycles(observations, config=_anchored_config())
    prism_cycles = aggregate_prism_cycles(result.observations, config=_anchored_config())

    assert list(prism_cycles.columns) == list(PRISM_CYCLE_COLUMNS)
    assert len(prism_cycles) == len(result.cycles) * 6
    assert prism_cycles["n_obs"].eq(1).all()
    assert prism_cycles["cycle_start"].le(prism_cycles["cycle_end"]).all()

    from_raw = aggregate_prism_cycles(observations, config=_anchored_config())
    pd.testing.assert_frame_equal(from_raw, prism_cycles)


def test_circular_median_and_wrap_in_aggregation():
    observations = pd.DataFrame({
        "point_id": ["P", "P"],
        "station_id": ["E", "E"],
        "cycle_id": ["E000", "E000"],
        "time": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")],
        "hz_deg": [359.9, 0.1],
        "v_deg": [90.0, 90.0],
        "d_m": [1000.0, 1000.0],
        "te_m": [0.0, 0.0],
        "tn_m": [1000.0, 1000.0],
        "tz_m": [0.0, 0.0],
        "se_m": [0.0, 0.0],
        "sn_m": [0.0, 0.0],
        "sh_m": [0.0, 0.0],
        "obs_id": ["src:1", "src:2"],
    })
    prism_cycles = aggregate_prism_cycles(observations, config=load_config())
    assert len(prism_cycles) == 1
    assert prism_cycles["hz_deg"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert prism_cycles["sd_hz_deg"].iloc[0] == pytest.approx(0.1, abs=1e-6)
    assert prism_cycles["obs_ids"].iloc[0] == ("src:1", "src:2")


# ---------------------------------------------------------------------------
#  E. Displacements: motion, rotation, wrap, hand-computed values
# ---------------------------------------------------------------------------
def test_radial_local_motion_gives_positive_dd(tmp_path):
    prisms, motion = _two_prisms(mover="M", rate_mm_per_day=3.0)
    effects = Effects(seed=0, noise_d_mm=0.3, local_motion_mm_per_day=motion)
    observations = _synthetic(tmp_path, days=4, effects=effects,
                              stations=_single_station(), prisms=prisms)
    config = _anchored_config()
    result = cycles.build_cycles(observations, config=config)
    prism_cycles = aggregate_prism_cycles(result.observations, config=config)
    displaced = compute_displacements(prism_cycles, segments=None, config=config)

    assert displaced.status == STATUS_OK
    mover = displaced.displacements.loc[
        displaced.displacements["point_id"] == "M"].sort_values("ts")
    stable = displaced.displacements.loc[
        displaced.displacements["point_id"] == "S"]

    assert mover["d_rad_mm"].iloc[-1] > 2.0
    assert mover["d_rad_mm"].iloc[-1] > stable["d_rad_mm"].abs().max() + 1.0
    assert stable["d_rad_mm"].abs().max() < 2.0
    assert mover["baseline_status"].eq(STATUS_OK).all()
    complete = displaced.baselines.set_index(["point_id", "segment_id"])["complete"]
    assert bool(complete.loc[("M", "E1")]) is True


def test_rotation_artifact_resets_at_break_and_never_enters_post_break_baseline(
        tmp_path):
    prisms, _ = _two_prisms()
    effects = Effects(seed=0, rotation_drift_arcsec_per_day=-5.0,
                      rotation_applied_in_coordinates=False)
    observations = _synthetic(tmp_path, days=3, effects=effects,
                              stations=_single_station(), prisms=prisms)
    config = _anchored_config(
        cycles={"processing_break": str(_BREAK)},
        baseline={"window_hours": 5, "min_observations": 2})
    result = cycles.build_cycles(observations, config=config)
    prism_cycles = aggregate_prism_cycles(result.observations, config=config)
    displaced = compute_displacements(prism_cycles, segments=None, config=config)

    for point_id in ("M", "S"):
        rows = displaced.displacements.loc[
            displaced.displacements["point_id"] == point_id].sort_values("ts")
        first = rows.loc[rows["segment_id"] == "E-S1"].sort_values("ts")
        second = rows.loc[rows["segment_id"] == "E-S2"].sort_values("ts")

        assert first["ts"].max() < _BREAK
        assert second["ts"].min() >= _BREAK
        assert abs(first["d_tan_mm"].iloc[-1]) > 4.0
        assert abs(second["d_tan_mm"].iloc[0]) < 2.0
        assert abs(second["d_tan_mm"].iloc[-1]) > abs(second["d_tan_mm"].iloc[0])

        assert rows["d_east_mm"].abs().max() < 0.5
        assert rows["d_north_mm"].abs().max() < 0.5

    baselines = displaced.baselines.set_index(["point_id", "segment_id"])
    for point_id in ("M", "S"):
        s1 = set(baselines.loc[(point_id, "E-S1"), "obs_ids"])
        s2 = baselines.loc[(point_id, "E-S2")]
        assert s2["start"] >= _BREAK
        assert s2["end"] >= _BREAK
        assert not (s1 & set(s2["obs_ids"]))
    assert displaced.status == STATUS_OK


def test_explicit_cycle_segments_match_derived_segments(tmp_path):
    observations = _synthetic(tmp_path, days=3, effects=Effects(),
                              stations=_single_station())
    config = _anchored_config(cycles={"processing_break": str(_BREAK)})
    result = cycles.build_cycles(observations, config=config)
    prism_cycles = aggregate_prism_cycles(result.observations, config=config)

    derived = compute_displacements(prism_cycles, segments=None, config=config)
    explicit = compute_displacements(prism_cycles, segments=result.segments,
                                     config=config)

    assert set(derived.displacements["segment_id"]) == {"E-S1", "E-S2"}
    pd.testing.assert_frame_equal(derived.displacements, explicit.displacements)
    pd.testing.assert_frame_equal(derived.baselines, explicit.baselines)
    pd.testing.assert_frame_equal(derived.source_links, explicit.source_links)


def test_circular_hz_wrap_is_not_ordinary_subtraction():
    rows = [
        _prism_cycle_row("P", 0, 0.0, hz_deg=359.95, obs_id="a:1"),
        _prism_cycle_row("P", 1, 4.0, hz_deg=0.05, obs_id="a:2"),
    ]
    prism_cycles = pd.DataFrame(rows)
    config = load_config()
    displaced = compute_displacements(prism_cycles, segments=None, config=config)

    ordered = displaced.displacements.sort_values("ts")
    expected = 1000.0 * math.sin(math.radians(90.0)) * math.radians(0.05) * 1000.0
    assert ordered["d_tan_mm"].iloc[0] < 0
    assert ordered["d_tan_mm"].iloc[1] > 0
    assert abs(ordered["d_tan_mm"].iloc[0]) == pytest.approx(expected, rel=1e-6)
    assert abs(ordered["d_tan_mm"].iloc[1]) == pytest.approx(expected, rel=1e-6)


def test_hand_computed_displacement_mm():
    rows = [_prism_cycle_row("P", index, float(index), obs_id=f"b:{index}")
            for index in range(6)]
    rows.append(_prism_cycle_row("P", 100, 100.0, hz_deg=100.01, d_m=1001.5,
                                 te_m=0.2, tn_m=999.5, tz_m=-0.3,
                                 obs_id="t:1"))
    prism_cycles = pd.DataFrame(rows)
    displaced = compute_displacements(prism_cycles, segments=None,
                                      config=load_config())

    baseline = displaced.baselines.iloc[0]
    assert baseline["status"] == STATUS_OK
    assert baseline["complete"] is True or bool(baseline["complete"]) is True
    assert baseline["n_obs"] == 6
    assert baseline["obs_ids"] == tuple(f"b:{index}" for index in range(6))

    row = displaced.displacements.sort_values("ts").iloc[-1]
    assert row["point_id"] == "P" and row["segment_id"] == "E1"
    assert row["d_rad_mm"] == pytest.approx(1500.0)
    expected_tan = 1000.0 * math.sin(math.radians(90.0)) * math.radians(0.01) * 1000.0
    assert row["d_tan_mm"] == pytest.approx(expected_tan, rel=1e-9)
    assert row["d_vert_mm"] == pytest.approx(0.0, abs=1e-9)
    assert row["d_east_mm"] == pytest.approx(200.0)
    assert row["d_north_mm"] == pytest.approx(-500.0)
    assert row["d_up_mm"] == pytest.approx(-300.0)
    assert row["baseline_status"] == STATUS_OK

    enriched = displaced.prism_cycles.sort_values("cycle_start").iloc[-1]
    assert enriched["segment_id"] == "E1"
    assert enriched["hz0_deg"] == pytest.approx(100.0)
    assert enriched["d0_m"] == pytest.approx(1000.0)
    assert enriched["baseline_obs_ids"] == tuple(f"b:{index}" for index in range(6))
    assert enriched["rad_raw_mm"] == pytest.approx(1500.0)


# ---------------------------------------------------------------------------
#  Baselines and provenance
# ---------------------------------------------------------------------------
def test_short_segment_baseline_fallback_and_insufficient_data():
    rows = [_prism_cycle_row("SHORT", index, 4.0 * index, obs_id=f"s:{index}")
            for index in range(4)]
    rows += [_prism_cycle_row("FULL", index, 4.0 * index, obs_id=f"f:{index}")
             for index in range(7)]
    rows += [_prism_cycle_row("ONE", 0, 0.0, obs_id="o:1")]
    prism_cycles = pd.DataFrame(rows)
    displaced = compute_displacements(prism_cycles, segments=None,
                                      config=load_config())

    baselines = displaced.baselines.set_index("point_id")
    assert baselines.loc["SHORT", "n_obs"] == 4
    assert bool(baselines.loc["SHORT", "complete"]) is False
    assert baselines.loc["SHORT", "status"] == STATUS_OK
    assert baselines.loc["FULL", "n_obs"] == 7
    assert bool(baselines.loc["FULL", "complete"]) is True
    assert baselines.loc["ONE", "status"] == "insufficient_data"
    assert baselines.loc["ONE", "obs_ids"] == ()
    assert displaced.status == STATUS_OK

    counts = displaced.displacements.groupby("point_id")["d_rad_mm"].apply(
        lambda values: values.notna().sum())
    assert counts.to_dict() == {"FULL": 7, "ONE": 0, "SHORT": 4}


def test_unconfigured_baseline_window_reports_status_never_values(tmp_path):
    observations = _synthetic(tmp_path, days=2, effects=Effects(),
                              stations=_single_station())
    config = _anchored_config(baseline={"window_hours": None})
    result = cycles.build_cycles(observations, config=config)
    prism_cycles = aggregate_prism_cycles(result.observations, config=config)
    displaced = compute_displacements(prism_cycles, segments=None, config=config)

    assert displaced.status == NOT_CONFIGURED
    assert (displaced.baselines["status"] == NOT_CONFIGURED).all()
    assert (displaced.baselines["obs_ids"].map(len) == 0).all()
    assert displaced.displacements["d_rad_mm"].isna().all()
    assert displaced.displacements["baseline_status"].eq(NOT_CONFIGURED).all()


def test_source_links_link_every_row_to_exact_baseline_rows(tmp_path):
    prisms, motion = _two_prisms()
    effects = Effects(seed=4, noise_d_mm=0.2, local_motion_mm_per_day=motion)
    observations = _synthetic(tmp_path, days=2, effects=effects,
                              stations=_single_station(), prisms=prisms)
    config = _anchored_config()
    result = cycles.build_cycles(observations, config=config)
    prism_cycles = aggregate_prism_cycles(result.observations, config=config)
    displaced = compute_displacements(prism_cycles, segments=None, config=config)

    links = displaced.source_links
    assert list(links.columns) == list(SOURCE_LINK_COLUMNS)
    assert set(links["table_name"]) == {"displacements"}
    assert set(links["role"]) == {"observation", "baseline"}

    expected: dict[tuple[str, str], set] = {}
    for row in displaced.displacements.itertuples(index=False):
        key = f"{row.point_id}#{row.segment_id}@{row.cycle_id}"
        expected[(key, "observation")] = set(row.obs_ids)
        expected[(key, "baseline")] = set(row.baseline_obs_ids)
    got: dict[tuple[str, str], set] = {}
    for row in links.itertuples(index=False):
        got.setdefault((row.row_key, row.role), set()).add(row.obs_id)
    assert got == expected
    assert len(expected) == 2 * len(displaced.displacements)

    baselines = displaced.baselines.set_index(["point_id", "segment_id"])["obs_ids"]
    for row in displaced.displacements.itertuples(index=False):
        key = f"{row.point_id}#{row.segment_id}@{row.cycle_id}"
        assert got[(key, "baseline")] == set(
            baselines.loc[(row.point_id, row.segment_id)])


def test_repeats_retained_through_aggregation_and_displacement(tmp_path):
    prisms, _ = _two_prisms()
    effects = Effects(seed=1, repeats=3, noise_d_mm=0.4, noise_hz_arcsec=0.3)
    observations = _synthetic(tmp_path, days=1, effects=effects,
                              stations=_single_station(), prisms=prisms)
    config = _anchored_config()
    result = cycles.build_cycles(observations, config=config)
    prism_cycles = aggregate_prism_cycles(result.observations, config=config)

    assert prism_cycles["n_obs"].eq(3).all()
    assert prism_cycles["obs_ids"].map(len).eq(3).all()
    assert prism_cycles["sd_hz_deg"].notna().all()
    assert prism_cycles["sd_d_m"].notna().all()
    assert prism_cycles["sd_d_m"].max() > 0.0

    displaced = compute_displacements(prism_cycles, segments=None, config=config)
    assert len(displaced.displacements) == len(prism_cycles)
    assert displaced.displacements["obs_ids"].map(len).eq(3).all()
    assert list(displaced.displacements.columns) == list(DISPLACEMENT_COLUMNS)
    assert list(displaced.baselines.columns) == list(BASELINE_COLUMNS)


def test_changes_table_status_not_configured(tmp_path):
    observations = _synthetic(tmp_path, days=2, effects=Effects(),
                              stations=_single_station())
    config = _anchored_config()
    result = cycles.build_cycles(observations, config=config)

    changes = result.changes
    assert changes.status == NOT_CONFIGURED
    assert changes.reason is not None
    assert "not configured" in changes.reason
    assert list(changes.changes.columns) == list(cycles.CHANGE_COLUMNS)
    assert len(changes.changes) == len(result.cycles) - 1
    assert np.isfinite(changes.changes["d_east_mm"]).all()
    assert np.isfinite(changes.changes["d_orientation_arcsec"]).all()
