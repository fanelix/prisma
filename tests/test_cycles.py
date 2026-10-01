"""
test_cycles.py — station assignment, elapsed-time cycles and change deltas.

Fixtures come from the independent forward model in ``tests/synthetic.py``
(parsed through :func:`rts_forensics.io.read_geomos`) plus small hand-built
canonical frames for gap boundaries and out-of-order rows. Every test passes
the configuration it relies on; no production constant is injected.

The orientation test places prisms on the station-orientation ray on purpose:
the synthetic model writes delivered coordinates along
``station.orientation_deg``, so the coordinate azimuth equals the true azimuth
only on that ray and the circular mean of ``Hz - azimuth`` recovers
``-orientation_deg`` in a clean export.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import synthetic
from synthetic import Effects, Prism, Station

from rts_forensics.config import load_config
from rts_forensics.cycles import (
    CHANGE_COLUMNS,
    CYCLE_COLUMNS,
    SEGMENT_COLUMNS,
    assign_stations,
    build_cycles,
    detect_processing_changes,
)
from rts_forensics.geometry import angular_difference_deg
from rts_forensics.io import read_geomos
from rts_forensics.models import NOT_CONFIGURED, STATUS_OK

_START = datetime(2026, 1, 1)


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


def _fixture(tmp_path: Path, name: str = "export.csv", **kwargs):
    path, rows = synthetic.write_fixture(tmp_path / name, **kwargs)
    return read_geomos(path).observations, rows


def _manual_observations(times, *, point_id: str = "P1", easting: float = 1000.0,
                         north: float = 2000.0, height: float = 50.0,
                         hz_deg: float = 0.0, v_deg: float = 90.0,
                         d_m: float = 400.0) -> pd.DataFrame:
    rows = []
    for index, minutes in enumerate(times):
        rows.append({
            "obs_id": f"manual:{index + 1}",
            "source_sha256": "manual",
            "src_line": index + 1,
            "source_name": "manual",
            "point_id": point_id,
            "time": _START + timedelta(minutes=minutes),
            "hz_deg": hz_deg,
            "v_deg": v_deg,
            "d_m": d_m,
            "te_m": easting + 100.0,
            "tn_m": north,
            "tz_m": height,
            "se_m": easting,
            "sn_m": north,
            "sh_m": height,
            "horz_dist_m": 100.0,
            "null_meas_m": 100.0,
            "temp_c": 10.0,
            "ppm": 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  A. Station assignment
# ---------------------------------------------------------------------------
def test_assign_stations_with_anchors(tmp_path):
    observations, _ = _fixture(tmp_path, days=1)
    before = observations.copy(deep=True)

    result = assign_stations(observations, config=_anchored_config())

    assert result.status == STATUS_OK
    assert result.reason is None
    assert "station_id" not in observations.columns
    assert observations.equals(before)
    assigned = result.observations
    assert "station_id" in assigned.columns and not assigned.equals(observations)

    e_points = {"P_A", "P_B", "P_C", "P_D", "P_E", "P_F"}
    got_e = set(assigned.loc[assigned["station_id"] == "E", "point_id"])
    got_w = set(assigned.loc[assigned["station_id"] == "W", "point_id"])
    assert got_e == e_points
    assert got_w == {"P_W1", "P_W2"}

    stations = result.stations.set_index("label")
    assert set(stations.index) == {"E", "W"}
    assert stations.loc["E", "easting_ref"] == pytest.approx(1000.0)
    assert stations.loc["E", "n_obs"] == 6 * 6
    assert stations.loc["W", "n_obs"] == 14 * 2
    assert stations.loc["E", "median_se_m"] == pytest.approx(1000.0)
    assert stations.loc["E", "median_sn_m"] == pytest.approx(2000.0)
    assert stations.loc["E", "median_sh_m"] == pytest.approx(50.0)
    assert stations.loc["W", "median_se_m"] == pytest.approx(0.0)

    messages = " ".join(result.diagnostics["message"])
    assert "tolerance" in messages
    assert "not configured" in messages


def test_assign_stations_auto_detect_labels_are_automatic(tmp_path):
    observations, _ = _fixture(tmp_path, days=1)
    result = assign_stations(observations, config=load_config())

    assert result.status == STATUS_OK
    assigned = result.observations
    assert set(assigned["station_id"]) == {"ST1", "ST2"}
    assert set(assigned.loc[assigned["station_id"] == "ST1", "point_id"]) == {
        "P_A", "P_B", "P_C", "P_D", "P_E", "P_F"}
    assert set(assigned.loc[assigned["station_id"] == "ST2", "point_id"]) == {
        "P_W1", "P_W2"}

    stations = result.stations
    assert stations["label"].tolist() == ["ST1", "ST2"]
    assert stations["easting_ref"].iloc[0] > stations["easting_ref"].iloc[1]
    messages = " ".join(result.diagnostics["message"])
    assert "automatic" in messages
    assert "no anchors" in messages


# ---------------------------------------------------------------------------
#  B. Cycle grouping
# ---------------------------------------------------------------------------
def test_build_cycles_tables_and_cycle_counts(tmp_path):
    observations, _ = _fixture(tmp_path, days=3)
    result = build_cycles(observations, config=_anchored_config())

    assert result.status == STATUS_OK
    assert list(result.cycles.columns) == list(CYCLE_COLUMNS)
    assert list(result.segments.columns) == list(SEGMENT_COLUMNS)

    counts = result.cycles.groupby("station_id", sort=True).size().to_dict()
    assert counts == {"E": 3 * 6, "W": 3 * 14}
    assert result.cycles.loc[result.cycles["station_id"] == "E", "n"].eq(6).all()
    assert result.cycles.loc[result.cycles["station_id"] == "W", "n"].eq(2).all()
    assert result.cycles.loc[result.cycles["station_id"] == "E", "npid"].eq(6).all()
    assert result.cycles.loc[result.cycles["station_id"] == "W", "npid"].eq(2).all()
    assert result.cycles["n_distinct_heights"].eq(1).all()
    assert result.cycles["dur_min"].eq(0.0).all()

    assignment = result.assignment
    assert assignment is not None and assignment.status == STATUS_OK
    assert list(assignment.stations["label"]) == ["E", "W"]
    assert result.changes is not None
    assert result.changes.status == NOT_CONFIGURED


def test_gap_thresholds_are_per_station_and_elapsed_minutes():
    observations = _manual_observations([0, 20, 40, 60])
    wide = build_cycles(
        observations,
        config=_anchored_config(cycles={"gap_minutes": {"E": 30},
                                        "default_gap_minutes": 30}))
    assert len(wide.cycles) == 1
    assert wide.cycles["n"].iloc[0] == 4

    narrow = build_cycles(
        observations,
        config=_anchored_config(cycles={"gap_minutes": {"E": 15},
                                        "default_gap_minutes": 30}))
    assert len(narrow.cycles) == 4
    assert narrow.cycles["cycle_id"].tolist() == ["E000", "E001", "E002", "E003"]


def test_gap_boundary_is_strictly_greater_than_threshold():
    config = _anchored_config(cycles={"gap_minutes": {"E": 30},
                                      "default_gap_minutes": 30})
    exactly = build_cycles(_manual_observations([0, 30]), config=config)
    over = build_cycles(_manual_observations([0, 31]), config=config)
    assert len(exactly.cycles) == 1
    assert len(over.cycles) == 2


def test_out_of_order_rows_are_grouped_by_elapsed_time_and_order_preserved():
    times = [100, 10, 0, 110]
    observations = _manual_observations(times)
    config = _anchored_config(cycles={"gap_minutes": {"E": 30},
                                      "default_gap_minutes": 30})
    result = build_cycles(observations, config=config)
    output = result.observations

    assert len(result.cycles) == 2
    assert [value - _START for value in output["time"].tolist()] == [
        timedelta(minutes=value) for value in times]
    cycle_of = dict(zip(output["time"], output["cycle_id"], strict=True))
    assert cycle_of[_START] == "E000"
    assert cycle_of[_START + timedelta(minutes=10)] == "E000"
    assert cycle_of[_START + timedelta(minutes=100)] == "E001"
    assert cycle_of[_START + timedelta(minutes=110)] == "E001"


def test_same_cycle_repeats_are_retained(tmp_path):
    observations, rows = _fixture(tmp_path, days=1, effects=Effects(repeats=3))
    result = build_cycles(observations, config=_anchored_config())

    e_cycles = result.cycles.loc[result.cycles["station_id"] == "E"]
    assert len(e_cycles) == 6
    assert e_cycles["n"].eq(6 * 3).all()
    assert e_cycles["npid"].eq(6).all()
    assert len(result.observations) == len(rows)


def test_cycle_ids_are_zero_padded_from_zero():
    observations = _manual_observations([0, 100, 200, 300])
    config = _anchored_config(cycles={"gap_minutes": {"E": 30},
                                      "default_gap_minutes": 30})
    result = build_cycles(observations, config=config)
    assert result.cycles["cycle_id"].tolist() == ["E000", "E001", "E002", "E003"]


# ---------------------------------------------------------------------------
#  Diagnostics and orientation
# ---------------------------------------------------------------------------
def test_observation_diagnostics_and_orientation_recovered(tmp_path):
    orientation = 30.0
    station = Station("E", 1000.0, 2000.0, 50.0, orientation, 30,
                      [0, 4, 8, 12, 16, 20])
    prisms = {}
    for index, horizontal in enumerate((300.0, 400.0, 500.0, 600.0, 700.0)):
        east = station.east + horizontal * math.sin(math.radians(orientation))
        north = station.north + horizontal * math.cos(math.radians(orientation))
        pid = f"R{index}"
        prisms[pid] = Prism(pid, "E", round(east, 3), round(north, 3),
                            45.0 + 2.0 * index)
    path, _ = synthetic.write_fixture(
        tmp_path / "clean.csv", days=1, stations={"E": station}, prisms=prisms,
        effects=Effects())
    observations = read_geomos(path).observations
    config = load_config(overrides={"stations": {"anchors": [
        {"label": "E", "easting_ref": 1000.0}]}})

    result = build_cycles(observations, config=config)
    for column in ("dHz_arcsec", "dV_arcsec", "dSD_mm", "dHDrep_mm", "scale_ppm"):
        assert column in result.observations.columns

    for _, cycle in result.cycles.iterrows():
        delta = angular_difference_deg(
            cycle["orientation_deg"], -orientation) * 3600.0
        assert abs(delta) < 0.01
        assert cycle["orientation_sd_arcsec"] < 0.5
    assert abs(result.cycles["dv_arcsec"].median()) < 0.5
    assert abs(result.cycles["scale_ppm"].median()) < 5.0


# ---------------------------------------------------------------------------
#  Segments
# ---------------------------------------------------------------------------
def test_segments_split_at_configured_processing_break(tmp_path):
    observations, _ = _fixture(tmp_path, days=3)
    break_time = pd.Timestamp("2026-01-02 00:00")
    result = build_cycles(
        observations,
        config=_anchored_config(cycles={"processing_break": str(break_time)}))

    segments = result.segments
    assert set(segments["segment_id"]) == {"E-S1", "E-S2", "W-S1", "W-S2"}
    assert not segments.duplicated(["point_id", "segment_id"]).any()
    starts = segments.set_index("segment_id")["t_start"]
    ends = segments.set_index("segment_id")["t_end"]
    assert (starts.loc[starts.index.str.endswith("S2")] >= break_time).all()
    assert (ends.loc[ends.index.str.endswith("S1")] < break_time).all()

    first_segment = result.observations["segment_id"].str.endswith("S1")
    assert (result.observations.loc[first_segment, "time"] < break_time).all()
    assert (result.observations.loc[~first_segment, "time"] >= break_time).all()


def test_segments_default_single_label_without_break(tmp_path):
    observations, _ = _fixture(tmp_path, days=1)
    result = build_cycles(observations, config=_anchored_config())

    assert set(result.segments["segment_id"]) == {"E1", "W1"}
    assert not result.segments.duplicated(["point_id", "segment_id"]).any()
    assert list(result.segments.columns) == list(SEGMENT_COLUMNS)
    assert result.candidate_events.empty
    assert "candidate frame events" in result.candidate_events.attrs["note"]


# ---------------------------------------------------------------------------
#  C. Processing-change deltas
# ---------------------------------------------------------------------------
def test_detect_processing_changes_measures_but_never_classifies(tmp_path):
    observations, _ = _fixture(tmp_path, days=2)
    e_points = ["P_A", "P_B", "P_C", "P_D", "P_E", "P_F"]
    is_e = observations["point_id"].isin(e_points)
    coordinates_shifted = is_e & (observations["time"] >= pd.Timestamp("2026-01-02"))
    hz_shifted = is_e & (observations["time"] >= pd.Timestamp("2026-01-02 08:00"))
    observations.loc[coordinates_shifted, "se_m"] += 0.005
    observations.loc[coordinates_shifted, "sn_m"] -= 0.003
    observations.loc[coordinates_shifted, "sh_m"] += 0.004
    observations.loc[hz_shifted, "hz_deg"] = (
        observations.loc[hz_shifted, "hz_deg"] + 2.0 / 3600.0) % 360.0

    result = build_cycles(observations, config=_anchored_config())
    changes = result.changes

    assert changes.status == NOT_CONFIGURED
    assert changes.reason is not None and "not configured" in changes.reason
    assert list(changes.changes.columns) == list(CHANGE_COLUMNS)
    assert len(changes.changes) == len(result.cycles) - 2

    e_changes = changes.changes.loc[changes.changes["station_id"] == "E"]
    coordinate_jump = e_changes.loc[
        e_changes["t"] == pd.Timestamp("2026-01-02 00:00")].iloc[0]
    assert coordinate_jump["d_east_mm"] == pytest.approx(5.0)
    assert coordinate_jump["d_north_mm"] == pytest.approx(-3.0)
    assert coordinate_jump["d_height_mm"] == pytest.approx(4.0)

    orientation_jump = e_changes.loc[
        e_changes["t"] == pd.Timestamp("2026-01-02 08:00")].iloc[0]
    assert orientation_jump["d_east_mm"] == pytest.approx(0.0, abs=1e-9)
    assert orientation_jump["d_north_mm"] == pytest.approx(0.0, abs=1e-9)
    assert orientation_jump["d_height_mm"] == pytest.approx(0.0, abs=1e-9)
    assert orientation_jump["d_orientation_arcsec"] == pytest.approx(2.0, abs=0.02)

    direct = detect_processing_changes(result.cycles, config=_anchored_config())
    assert direct.status == NOT_CONFIGURED
    pd.testing.assert_frame_equal(direct.changes, changes.changes)

    before = e_changes.loc[e_changes["t"] < pd.Timestamp("2026-01-02")]
    assert (before["d_east_mm"].abs() < 1e-9).all()
    assert before["d_north_mm"].abs().max() < 1e-9
