"""
test_forensic_tests.py — software tests for ``rts_forensics.tests``.

The analytical hypothesis tests are exercised on small synthetic DataFrames
with generic IDs (A1, P0, F1, ...) and generic dates (January 2026); no site
prism ID and no known site event date appears anywhere in the fixtures. Known
signals (steps, hinges, common modes, daytime bias, dropouts) are injected with
explicit amplitudes and recovered within the tolerances stated per test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import rts_forensics.tests as forensics
from rts_forensics.config import default_config
from rts_forensics.models import (
    EVIDENCE_COLUMNS,
    INVESTIGATION_COLUMNS,
    NOT_CONFIGURED,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    ForensicResult,
)
from rts_forensics.tests import events, groups, missingness, references, sampling


def _config():
    return default_config()


def _timestamps(*, days: int = 10, hours: int = 4) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=int(days * 24 / hours),
                         freq=f"{hours}h")


def _series(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def _coordinates(pairs: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"point_id": point_id, "east": float(east), "north": float(north)}
        for point_id, (east, north) in pairs.items()
    ])


# ---------------------------------------------------------------------------
#  groups
# ---------------------------------------------------------------------------
def test_candidate_clusters_not_configured_without_radius():
    cfg = _config()
    series = pd.DataFrame({
        "point_id": ["A"], "ts": [pd.Timestamp("2026-01-01")], "los_mm": [0.0]})
    coordinates = _coordinates({"A": (0.0, 0.0)})
    result = groups.candidate_clusters(series, coordinates=coordinates,
                                       config=cfg)
    assert result.status == NOT_CONFIGURED
    assert result.clusters.empty
    assert result.members == {}
    assert "radius" in result.reason


def test_candidate_clusters_finds_two_configured_clusters():
    cfg = _config()
    cfg.forensics.cluster = {"radius_m": 5.0, "min_members": 3,
                             "coherence": 0.5}
    ts = _timestamps(days=10)
    rng = np.random.default_rng(11)
    signal = np.sin(np.linspace(0.0, 6.0, len(ts)))
    layout = {
        "A1": (0.0, 0.0), "A2": (1.0, 0.0), "A3": (0.0, 1.0),
        "B1": (100.0, 0.0), "B2": (101.0, 0.0), "B3": (100.0, 1.0),
    }
    frames = []
    for index, point_id in enumerate(layout):
        amplitude = 1.0 + 0.1 * index
        values = amplitude * signal + rng.normal(0.0, 0.01, len(ts))
        frames.append(pd.DataFrame({
            "point_id": point_id, "ts": ts, "los_mm": values}))
    result = groups.candidate_clusters(_series(frames),
                                       coordinates=_coordinates(layout),
                                       config=cfg)
    assert result.status == STATUS_OK
    assert len(result.clusters) == 2
    found = {tuple(sorted(members)) for members in result.members.values()}
    assert found == {("A1", "A2", "A3"), ("B1", "B2", "B3")}
    assert (result.clusters["n_members"] == 3).all()
    assert result.clusters["coherence"].min() > 0.5


def test_spatial_placebo_excludes_seed_and_uses_exact_k():
    cfg = _config()
    assert cfg.forensics.placebo_neighbors == 7
    ts = _timestamps(days=5)
    layout = {f"P{index}": (10.0 * index, 0.0) for index in range(8)}
    frames = []
    for index, point_id in enumerate(layout):
        values = np.zeros(len(ts))
        values[ts >= pd.Timestamp("2026-01-04")] = float(index)
        frames.append(pd.DataFrame({
            "point_id": point_id, "ts": ts, "los_mm": values}))
    result = groups.spatial_placebo(_series(frames),
                                    coordinates=_coordinates(layout),
                                    config=cfg)
    assert result.status == STATUS_OK
    assert len(result.null_distribution) == 8
    seed_row = result.table.set_index("point_id").loc["P0"]
    assert seed_row["n_neighbors"] == 7
    assert "P0" not in seed_row["neighbor_ids"]
    assert set(seed_row["neighbor_ids"]) == {f"P{i}" for i in range(1, 8)}
    assert seed_row["placebo_median_mm"] == pytest.approx(4.0)


def test_temporal_split_recovers_known_step_and_records_contributors():
    cfg = _config()
    ts = _timestamps(days=10)
    values = np.where(ts >= pd.Timestamp("2026-01-06"), 10.0, 0.0)
    kept = ts[ts < pd.Timestamp("2026-01-04")]
    frames = [
        pd.DataFrame({"point_id": "S1", "ts": ts, "los_mm": values}),
        pd.DataFrame({"point_id": "S2", "ts": kept,
                      "los_mm": np.zeros(len(kept))}),
    ]
    result = groups.temporal_split(_series(frames), config=cfg)
    row = result.table.set_index("point_id").loc["S1"]
    assert row["status"] == STATUS_OK
    assert row["net_mm"] == pytest.approx(10.0, abs=0.1)
    assert result.members["first_half"] == ["S1", "S2"]
    assert set(result.members["second_half"]) == {"S1", "S2"}
    second_epoch = result.contributors[
        result.contributors["epoch"] == "second_half"].set_index("point_id")
    assert second_epoch.loc["S2", "n_obs"] < second_epoch.loc["S1", "n_obs"]
    assert second_epoch.loc["S2", "n_obs"] > 0


def test_change_point_recovers_known_hinge():
    cfg = _config()
    ts = _timestamps(days=14)
    elapsed_days = (ts - ts[0]).total_seconds().to_numpy() / 86400.0
    values = np.where(elapsed_days <= 6.0, 0.0, (elapsed_days - 6.0))
    series = pd.DataFrame({"point_id": "H1", "ts": ts, "los_mm": values})
    result = groups.change_point(series, config=cfg)
    assert result.status == NOT_CONFIGURED
    assert "significance" in result.reason
    row = result.table.iloc[0]
    assert row["hinge_improvement"] > 0.0
    assert row["hinge_improvement_fraction"] > 0.9
    assert row["hinge_x_days"] == pytest.approx(6.0, abs=1.0)
    assert bool(row["significant"]) is False
    assert not result.sensitivity.empty


def test_pairwise_los_difference_cancels_common_injected_signal():
    cfg = _config()
    ts = _timestamps(days=10)
    common = np.where(ts >= pd.Timestamp("2026-01-06"), 5.0, 0.0)
    frames = [
        pd.DataFrame({"point_id": point_id, "ts": ts, "los_mm": common})
        for point_id in ("A", "B")
    ]
    result = groups.pairwise_los(_series(frames),
                                 coordinates=_coordinates(
                                     {"A": (0.0, 0.0), "B": (10.0, 0.0)}),
                                 config=cfg)
    row = result.table.iloc[0]
    assert row["status"] == STATUS_OK
    assert row["median_diff_mm"] == pytest.approx(0.0, abs=1e-9)
    assert row["rate_mm_day"] == pytest.approx(0.0, abs=1e-9)
    assert row["ci_low"] <= 0.0 <= row["ci_high"]
    assert row["distance_m"] == pytest.approx(10.0)
    assert result.members["A|B"] == ["A", "B"]


def test_movement_vector_sign_points_towards_station():
    cfg = _config()
    ts = pd.Timestamp("2026-01-01")
    series = pd.DataFrame({
        "point_id": ["M1", "M2"],
        "ts": [ts, ts],
        "los_mm": [-2.0, -2.0],
        "tan_mm": [0.0, 0.0],
        "vert_mm": [0.0, 0.0],
        "az_deg": [0.0, 90.0],
        "v_deg": [90.0, 90.0],
        "n_obs": [1, 1],
    })
    result = groups.movement_vectors(series, coordinates=None, config=cfg)
    assert result.status == STATUS_OK
    by_id = result.vectors.set_index("point_id")
    north = by_id.loc["M1"]
    east = by_id.loc["M2"]
    assert north["rad_h_mm"] == pytest.approx(-2.0)
    assert north["d_north_mm"] == pytest.approx(-2.0)
    assert north["d_east_mm"] == pytest.approx(0.0, abs=1e-12)
    assert north["horizontal_mm"] == pytest.approx(2.0)
    assert north["towards_station_mm"] == pytest.approx(2.0)
    assert east["d_east_mm"] == pytest.approx(-2.0)
    assert east["d_north_mm"] == pytest.approx(0.0, abs=1e-12)
    assert east["towards_station_mm"] > 0.0


def test_movement_vectors_requires_azimuth_and_zenith():
    cfg = _config()
    ts = pd.Timestamp("2026-01-01")
    series = pd.DataFrame({
        "point_id": ["M1"], "ts": [ts],
        "los_mm": [-1.0], "tan_mm": [0.0], "vert_mm": [0.0]})
    with pytest.raises(ValueError, match="azimuth"):
        groups.movement_vectors(series, coordinates=None, config=cfg)


# ---------------------------------------------------------------------------
#  references
# ---------------------------------------------------------------------------
def test_reference_inference_detects_anticorrelation_and_applies_bonferroni():
    cfg = _config()
    ts = pd.date_range("2026-01-01", periods=30, freq="12h")
    height = 50.0 + 0.002 * np.arange(30) + 0.0001 * np.sin(np.arange(30))
    rng = np.random.default_rng(5)
    frames = []
    for index in range(1, 9):
        coefficient = -float(index)
        values = (coefficient * (height - height[0]) * 1000.0
                  + rng.normal(0.0, 0.3, len(ts)))
        frames.append(pd.DataFrame({
            "point_id": f"R{index}", "ts": ts, "vert_mm": values}))
    station = pd.DataFrame({"ts": ts, "sh_m": height})
    result = references.reference_inference(
        _series(frames), station_series=station, config=cfg)
    assert result.status == STATUS_OK
    assert len(result.correlations) == 8
    assert set(result.correlations["metric"]) == {"vertical"}
    strongest = result.correlations.sort_values("rho").iloc[0]
    assert strongest["rho"] < -0.9
    assert bool(strongest["significant"]) is True
    expected = np.minimum(
        1.0,
        result.correlations["p_value"] * result.correlations["n_tests"])
    assert np.allclose(result.correlations["p_adjusted"], expected)
    assert (result.correlations["n_tests"] == 8).all()


def test_resection_validation_recovers_minus_one_for_station_rise():
    cfg = _config()
    ts = pd.date_range("2026-01-01", periods=20, freq="12h")
    height = 50.0 + 0.002 * np.arange(20)
    delta_mm = (height - height[0]) * 1000.0
    rng = np.random.default_rng(6)
    frames = []
    for index in range(3):
        values = -delta_mm + rng.normal(0.0, 0.01, len(ts))
        frames.append(pd.DataFrame({
            "point_id": f"V{index}", "ts": ts, "vert_mm": values}))
    station = pd.DataFrame({"ts": ts, "sh_m": height})
    result = references.resection_validation(
        _series(frames), station_series=station, config=cfg)
    assert result.status == STATUS_OK
    assert result.n == 20
    assert result.expected_slope == pytest.approx(-1.0)
    assert result.slope == pytest.approx(-1.0, abs=0.01)
    assert result.ci_low <= -1.0 <= result.ci_high
    assert result.comparison == "consistent"


# ---------------------------------------------------------------------------
#  missingness
# ---------------------------------------------------------------------------
def test_missingness_detects_dropped_prism_and_is_deterministic():
    cfg = _config()
    ts = _timestamps(days=10)
    cutoff = pd.Timestamp("2026-01-06")
    frames = []
    layout = {}
    for index in range(6):
        point_id = f"P{index}"
        kept = ts if point_id != "P0" else ts[ts < cutoff]
        frames.append(pd.DataFrame({
            "point_id": point_id, "ts": kept,
            "los_mm": np.linspace(0.0, 1.0, len(kept))}))
        layout[point_id] = ((500.0, 500.0) if point_id == "P0"
                            else (float(index), 0.0))
    series = _series(frames)
    coordinates = _coordinates(layout)

    coverage = missingness.coverage(series, config=cfg)
    lost = coverage.table.set_index("point_id").loc["P0"]
    retained = coverage.table.set_index("point_id").loc["P5"]
    assert bool(lost["in_final_window"]) is False
    assert int(lost["n_final_window"]) == 0
    assert lost["last"] < pd.Timestamp("2026-01-07")
    assert "not evidence of stability" in lost["reason"]
    assert bool(retained["in_final_window"]) is True
    assert coverage.status == STATUS_OK

    first = missingness.lost_vs_retained(series, coordinates=coordinates,
                                         config=cfg)
    second = missingness.lost_vs_retained(series, coordinates=coordinates,
                                          config=cfg)
    assert first.status == "exploratory"
    assert not first.table.loc[first.table["lost"]].empty
    test_row = first.tests.iloc[0]
    assert int(test_row["n_lost"]) == 1
    assert int(test_row["n_retained"]) == 5
    assert int(test_row["seed"]) == 1
    assert int(test_row["n_permutations"]) == 5000
    assert test_row["candidate_source"] == "network_centroid"
    assert test_row["p_permutation"] == second.tests.iloc[0]["p_permutation"]
    assert np.isfinite(test_row["p_permutation"])
    assert np.isfinite(test_row["p_mannwhitney"])


def test_pre_loss_trends_recovers_injected_slope():
    cfg = _config()
    ts = _timestamps(days=10)
    elapsed = (ts - ts[0]).total_seconds().to_numpy() / 86400.0
    frames = []
    for index in range(5):
        frames.append(pd.DataFrame({
            "point_id": f"T{index}", "ts": ts,
            "los_mm": (-0.2 * index) * elapsed}))
    result = missingness.pre_loss_trends(_series(frames), config=cfg)
    ordered = result.table.sort_values("trend_mm_day")
    assert ordered.iloc[0]["trend_mm_day"] == pytest.approx(-0.8, abs=0.05)
    assert ordered.iloc[-1]["trend_mm_day"] == pytest.approx(0.0, abs=0.05)
    assert ordered.iloc[0]["rank_percentile"] < 50.0
    assert ordered.iloc[-1]["rank_percentile"] > 50.0


def test_observability_by_hour_detects_night_blind_prism():
    cfg = _config()
    ts = _timestamps(days=10)
    night = np.isin(ts.hour, [20, 0, 4])
    kept = (ts < pd.Timestamp("2026-01-06")) | ~night
    frames = [
        pd.DataFrame({"point_id": "N1", "ts": ts[kept],
                      "los_mm": np.zeros(int(kept.sum()))}),
        pd.DataFrame({"point_id": "N2", "ts": ts,
                      "los_mm": np.zeros(len(ts))}),
    ]
    result = missingness.observability_by_hour(_series(frames), config=cfg)
    by_id = result.table.set_index("point_id")
    assert bool(by_id.loc["N1", "night_blind"]) is True
    assert int(by_id.loc["N1", "night_obs_last"]) == 0
    assert bool(by_id.loc["N2", "night_blind"]) is False
    assert "cause" in by_id.loc["N1", "reason"]
    assert not result.by_hour.empty


# ---------------------------------------------------------------------------
#  sampling
# ---------------------------------------------------------------------------
def test_hour_matched_change_removes_sampled_only_bias():
    cfg = _config()
    ts = _timestamps(days=10)
    day = np.isin(ts.hour, [8, 12, 16])
    values = np.where(day, 10.0, 0.0)
    kept = (ts < pd.Timestamp("2026-01-06")) | day
    series = pd.DataFrame({
        "point_id": "H1", "ts": ts[kept], "los_mm": values[kept]})
    naive = groups.net_window_change(groups.prepare_series(series, "los"),
                                     hours=48.0)
    result = sampling.hour_matched_change(series, config=cfg)
    row = result.table.iloc[0]
    assert abs(naive.iloc[0]["net_mm"]) >= 5.0
    assert int(row["n_classes"]) == 3
    assert set(row["classes_used"]) == {8, 12, 16}
    assert row["net_mm"] == pytest.approx(0.0, abs=0.01)
    assert result.status == STATUS_OK


def test_day_night_comparison_and_composition_report_sampling_change():
    cfg = _config()
    ts = _timestamps(days=10)
    day = np.isin(ts.hour, [8, 12, 16])
    values = np.where(day, 10.0, 0.0)
    kept = (ts < pd.Timestamp("2026-01-06")) | day
    series = pd.DataFrame({
        "point_id": "H1", "ts": ts[kept], "los_mm": values[kept]})
    result = sampling.day_night_comparison(series, config=cfg)
    by_band = result.table.set_index("band")
    assert by_band.loc["night", "status"] == "insufficient_data"
    assert "unobserved" in by_band.loc["night", "reason"]
    assert by_band.loc["day", "net_mm"] == pytest.approx(0.0, abs=0.01)
    composition = result.composition.iloc[0]
    assert composition["night_share_baseline"] > 0.5
    assert composition["night_share_final"] == pytest.approx(0.0)

    described = sampling.sampling_composition(series, config=cfg).table.iloc[0]
    assert bool(described["composition_changed"]) is True
    assert set(described["classes_removed"]) == {0, 4, 20}


def test_daytime_bias_reports_association_and_geometry_control():
    cfg = _config()
    ts = _timestamps(days=8)
    day = np.isin(ts.hour, [8, 12, 16])
    rng = np.random.default_rng(31)
    common = np.sin(np.linspace(0.0, 10.0, len(ts)))
    layout = {"B1": (30.0, 85.0), "B2": (40.0, 80.0), "B3": (50.0, 75.0),
              "B4": (60.0, 70.0)}
    frames = []
    for point_id, (azimuth, zenith) in layout.items():
        bias = 10.0 if point_id == "B1" else 0.0
        values = common + rng.normal(0.0, 0.01, len(ts)) + bias * day
        frames.append(pd.DataFrame({
            "point_id": point_id, "ts": ts, "los_mm": values,
            "az_deg": azimuth, "v_deg": zenith}))
    result = sampling.daytime_bias(_series(frames), config=cfg)
    row = result.table.set_index("point_id").loc["B1"]
    assert row["day_minus_night_mm"] == pytest.approx(10.0, abs=0.5)
    assert row["control_delta_mm"] == pytest.approx(10.0, abs=1.0)
    assert "not asserted" in row["reason"]
    assert not result.weekly.empty


# ---------------------------------------------------------------------------
#  events
# ---------------------------------------------------------------------------
def test_balanced_step_uses_equal_hour_composition():
    cfg = _config()
    cfg.forensics.events = {
        "balanced_windows": [[16, 20, 0, 4], [20, 0, 4, 8]],
        "candidate_times": ["2026-01-03 00:00"],
    }
    ts = pd.date_range("2026-01-01", periods=30, freq="4h")
    values = np.where(ts >= pd.Timestamp("2026-01-03"), 3.0, 0.0)
    series = pd.DataFrame({"point_id": "E1", "ts": ts, "los_mm": values})
    result = events.balanced_step_test(series, config=cfg)
    pair = result.pairs.iloc[0]
    assert bool(pair["balanced"]) is True
    assert int(pair["n_before"]) == 4
    assert int(pair["n_after"]) == 4
    assert set(pair["hours_before"]) == {16, 20, 0, 4}
    assert set(pair["hours_after"]) == {20, 0, 4, 8}
    assert pair["difference_mm"] == pytest.approx(3.0)
    assert result.status == NOT_CONFIGURED
    assert bool(pair["significant"]) is False
    assert "step_significance" in pair["reason"]


def test_common_mode_steps_reports_candidates_and_competing_explanations():
    cfg = _config()
    ts = _timestamps(days=10)
    rng = np.random.default_rng(21)
    frames = []
    for index in range(3):
        values = (np.where(ts >= pd.Timestamp("2026-01-06"), 5.0, 0.0)
                  + rng.normal(0.0, 0.05, len(ts)))
        frames.append(pd.DataFrame({
            "point_id": f"C{index}", "ts": ts, "los_mm": values}))
    result = events.common_mode_steps(_series(frames), config=cfg)
    assert result.status == NOT_CONFIGURED
    assert not result.steps.empty
    assert result.steps["magnitude_mm"].abs().max() == pytest.approx(5.0,
                                                                     abs=1.0)
    assert result.steps["candidate_explanation"].str.len().gt(0).all()
    assert result.steps["competing_explanation"].str.len().gt(0).all()
    assert set(result.investigations["status"]) <= {"open", "not_configured"}


# ---------------------------------------------------------------------------
#  not-configured policy and assembly
# ---------------------------------------------------------------------------
def test_unconfigured_thresholds_are_never_guessed():
    cfg = _config()
    series = pd.DataFrame({
        "point_id": ["A"], "ts": [pd.Timestamp("2026-01-01")], "los_mm": [0.0]})
    coordinates = _coordinates({"A": (0.0, 0.0)})
    assert (groups.candidate_clusters(series, coordinates=coordinates,
                                      config=cfg).status == NOT_CONFIGURED)
    assert (groups.change_point(series, config=cfg).status == NOT_CONFIGURED)
    no_neighbors = _config()
    no_neighbors.forensics.placebo_neighbors = None
    assert (groups.spatial_placebo(series, coordinates=coordinates,
                                   config=no_neighbors).status
            == NOT_CONFIGURED)
    no_window = _config()
    no_window.forensics.missingness = {}
    coverage = missingness.coverage(series, config=no_window)
    assert coverage.status == NOT_CONFIGURED
    assert np.isnan(coverage.table.iloc[0]["n_final_window"])
    no_expected = _config()
    no_expected.forensics.references = {"expected_rise_vertical_slope": None}
    station = pd.DataFrame({"ts": [pd.Timestamp("2026-01-01")], "sh_m": [50.0]})
    resection = references.resection_validation(
        series, station_series=station, config=no_expected)
    assert resection.status == NOT_CONFIGURED


def test_run_forensics_assembles_register_and_investigations():
    cfg = _config()
    ts = _timestamps(days=8)
    rng = np.random.default_rng(41)
    frames = []
    layout = {}
    for index in range(4):
        values = rng.normal(0.0, 0.1, len(ts))
        frames.append(pd.DataFrame({
            "point_id": f"F{index}", "ts": ts, "los_mm": values,
            "vert_mm": values * 0.5, "tan_mm": values * 0.1,
            "az_deg": 30.0 + 10.0 * index, "v_deg": 85.0,
        }))
        layout[f"F{index}"] = (10.0 * index, 0.0)
    station = pd.DataFrame({
        "ts": ts,
        "height": 50.0 + 0.001 * np.arange(len(ts)),
        "orientation_deg": 30.0 + 0.0001 * np.arange(len(ts)),
    })
    result = forensics.run_forensics(
        _series(frames), coordinates=_coordinates(layout),
        station_series=station, config=cfg)
    assert isinstance(result, ForensicResult)
    register = result.evidence["register"]
    assert list(register.columns) == list(EVIDENCE_COLUMNS)
    assert register["reason"].astype(str).str.len().gt(0).all()
    assert set(register["test"]) == {"groups", "sampling", "references",
                                     "missingness", "events"}
    cluster_row = register.loc[
        register["hypothesis"] == "candidate spatial clusters"].iloc[0]
    assert cluster_row["status"] == NOT_CONFIGURED
    assert "radius" in cluster_row["reason"]
    assert list(result.investigations.columns) == list(INVESTIGATION_COLUMNS)
    assert result.investigations["status"].isin(
        ["open", "not_configured"]).all()
    assert set(result.statuses.columns) == {"test", "hypothesis", "status",
                                            "reason"}


def test_run_forensics_records_unavailable_inputs():
    cfg = _config()
    series = pd.DataFrame({
        "point_id": ["A", "A"],
        "ts": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "los_mm": [0.0, 1.0]})
    result = forensics.run_forensics(series, config=cfg)
    register = result.evidence["register"]
    cluster_row = register.loc[
        register["hypothesis"] == "candidate spatial clusters"].iloc[0]
    reference_row = register.loc[
        register["hypothesis"] == "reference inference"].iloc[0]
    assert cluster_row["status"] == STATUS_UNAVAILABLE
    assert reference_row["status"] == STATUS_UNAVAILABLE
    assert "coordinates" in cluster_row["reason"]
    assert "station" in reference_row["reason"]


def test_package_source_has_no_site_ids_or_event_dates():
    package_dir = Path(forensics.__file__).resolve().parent
    forbidden = ("HLO-", "STG4N", "DAM5_", "BS_HL_5", "2026-09", "2026-10")
    for path in sorted(package_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token!r} found in {path.name}"
