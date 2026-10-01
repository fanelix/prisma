"""
test_noise.py — pooled repeatability, rolling-median noise and retained spikes.

Expected pooled values are hand-computed from the documented formula in each
test. The independent forward model in ``tests/synthetic.py`` confirms that
injected repeat noise is recovered from a source-shaped export. Rolling-median
tests build small long series directly so the window edge and the MAD scaling
are checkable by hand.
"""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from synthetic import Effects, build_observations

from rts_forensics import noise
from rts_forensics.config import default_config
from rts_forensics.models import (
    STATUS_INSUFFICIENT,
    STATUS_OK,
    CycleResult,
)

DMS_PATTERN = re.compile(r"^\s*(-?\d+)\D+(\d+)\D+([\d.]+)\D*$")


def _parse_dms(text: str) -> float:
    match = DMS_PATTERN.match(str(text).strip().strip('"'))
    assert match is not None, text
    degrees, minutes, seconds = match.groups()
    sign = -1.0 if degrees.startswith("-") else 1.0
    return sign * (abs(int(degrees)) + int(minutes) / 60.0 + float(seconds) / 3600.0)


def _canonical(rows: list[dict]) -> pd.DataFrame:
    """Source-shaped synthetic rows -> canonical observations with cycle ids."""
    records = []
    for index, row in enumerate(rows):
        station = "E" if abs(float(row["Station Easting [m]"]) - 1000.0) < 1.0 else "W"
        records.append({
            "obs_id": f"src:{index + 2}",
            "point_id": row["Point ID"],
            "station_id": station,
            "time": datetime.strptime(row["Time"], "%d/%m/%Y %H:%M"),
            "d_m": float(row["D [m]"]),
            "hz_deg": _parse_dms(row["Hz [dms]"]),
            "v_deg": _parse_dms(row["V [dms]"]),
            "te_m": float(row["Target Easting [m]"]),
            "tn_m": float(row["Target Northing [m]"]),
            "tz_m": float(row["Target Elevation [m]"]),
        })
    frame = pd.DataFrame(records)
    frame["cycle_id"] = frame.groupby(
        ["point_id", "station_id", "time"], sort=False).ngroup()
    return frame


def _repeat_frame() -> pd.DataFrame:
    """Two repeat groups plus one single observation that must be ignored."""
    return pd.DataFrame({
        "obs_id": ["o1", "o2", "o3", "o4", "o5", "o6"],
        "point_id": ["P1", "P1", "P1", "P1", "P1", "P2"],
        "station_id": ["E", "E", "E", "E", "E", "E"],
        "cycle_id": [1, 1, 1, 2, 2, 1],
        "time": pd.to_datetime([
            "2026-01-01 10:00:00", "2026-01-01 10:00:10",
            "2026-01-01 10:00:20", "2026-01-01 12:00:00",
            "2026-01-01 12:15:00", "2026-01-01 14:00:00",
        ]),
        "d_m": [1.000, 1.002, 1.004, 2.000, 2.000, 9.999],
        "hz_deg": [100.000, 100.001, 100.002, 100.0, 100.0, 50.0],
        "v_deg": [90.0, 90.0005, 90.0010, 90.0, 90.0, 80.0],
        "te_m": [10.0, 10.001, 10.002, 20.0, 20.0, 99.0],
        "tn_m": [30.0, 30.002, 30.004, 40.0, 40.0, 88.0],
        "tz_m": [5.0, 5.001, 5.002, 6.0, 6.0, 77.0],
    })


def _pooled_expected(groups: list[list[float]], factor: float) -> float:
    total = sum(len(group) for group in groups)
    k = len(groups)
    ss = sum(float(((np.asarray(group) - np.mean(group)) ** 2).sum())
             for group in groups)
    return float(np.sqrt(ss / (total - k)) * factor)


def test_pooled_repeatability_hand_computed():
    out = noise.pooled_repeatability(_repeat_frame(), cycles=None,
                                     config=default_config())
    assert out["span_class"].tolist() == ["all", "same_minute", "ge_10min"]
    assert out["n_obs"].tolist() == [5, 3, 2]
    assert out["n_groups"].tolist() == [2, 1, 1]

    group_a = [1.000, 1.002, 1.004]
    group_b = [2.000, 2.000]
    expected_all = _pooled_expected([group_a, group_b], 1000.0)
    expected_same = _pooled_expected([group_a], 1000.0)
    expected_ge = _pooled_expected([group_b], 1000.0)
    assert out.loc[0, "sd_d_mm"] == pytest.approx(expected_all)
    assert out.loc[1, "sd_d_mm"] == pytest.approx(expected_same)
    assert out.loc[2, "sd_d_mm"] == pytest.approx(expected_ge)
    assert out.loc[0, "sd_d_mm"] == pytest.approx(1.632993, abs=1e-5)
    assert out.loc[1, "sd_d_mm"] == pytest.approx(2.0, abs=1e-9)
    assert out.loc[2, "sd_d_mm"] == pytest.approx(0.0, abs=1e-12)

    hz_a = [100.000, 100.001, 100.002]
    hz_b = [100.0, 100.0]
    assert out.loc[0, "sd_hz_arcsec"] == pytest.approx(
        _pooled_expected([hz_a, hz_b], 3600.0))
    assert out.loc[0, "sd_v_arcsec"] == pytest.approx(
        _pooled_expected([[90.0, 90.0005, 90.0010], [90.0, 90.0]], 3600.0))
    assert out.loc[0, "sd_e_mm"] == pytest.approx(
        _pooled_expected([[10.0, 10.001, 10.002], [20.0, 20.0]], 1000.0))
    assert out.loc[0, "sd_n_mm"] == pytest.approx(
        _pooled_expected([[30.0, 30.002, 30.004], [40.0, 40.0]], 1000.0))
    assert out.loc[0, "sd_z_mm"] == pytest.approx(
        _pooled_expected([[5.0, 5.001, 5.002], [6.0, 6.0]], 1000.0))


def test_pooled_repeatability_recovers_injected_repeat_noise():
    effects = Effects(seed=3, noise_d_mm=2.0, noise_hz_arcsec=1.0,
                      noise_v_arcsec=1.0, repeats=4)
    observations = _canonical(build_observations(days=3, effects=effects))
    out = noise.pooled_repeatability(observations, cycles=None,
                                     config=default_config())
    row = out.set_index("span_class").loc["same_minute"]
    assert row["n_groups"] > 100
    assert row["sd_d_mm"] == pytest.approx(2.0, abs=0.3)
    assert row["sd_hz_arcsec"] == pytest.approx(1.0, abs=0.25)
    assert row["sd_v_arcsec"] == pytest.approx(1.0, abs=0.25)


def test_pooled_repeatability_accepts_cycle_assignment():
    frame = _repeat_frame().drop(columns=["cycle_id", "station_id"])
    out = noise.pooled_repeatability(frame, cycles=[1, 1, 1, 2, 2, 3],
                                     config=default_config())
    assert out.loc[0, "n_groups"] == 2
    assert out.loc[0, "sd_d_mm"] == pytest.approx(1.632993, abs=1e-5)


def test_pooled_repeatability_accepts_cycle_result_observations():
    frame = _repeat_frame().drop(columns=["cycle_id", "station_id"])
    assigned = _repeat_frame()[["obs_id", "point_id", "station_id", "cycle_id"]]
    out = noise.pooled_repeatability(frame, cycles=CycleResult(observations=assigned),
                                     config=default_config())
    assert out.loc[0, "n_groups"] == 2
    assert out.loc[0, "sd_d_mm"] == pytest.approx(1.632993, abs=1e-5)


def test_pooled_repeatability_accepts_cycle_table_with_time_bounds():
    frame = _repeat_frame().drop(columns=["cycle_id"])
    table = pd.DataFrame({
        "station_id": ["E", "E"],
        "cycle_id": [1, 2],
        "cycle_start": pd.to_datetime(["2026-01-01 09:50", "2026-01-01 11:50"]),
        "cycle_end": pd.to_datetime(["2026-01-01 10:30", "2026-01-01 12:30"]),
    })
    out = noise.pooled_repeatability(frame, cycles=table,
                                     config=default_config())
    assert out.loc[0, "n_groups"] == 2
    assert out.loc[0, "sd_d_mm"] == pytest.approx(1.632993, abs=1e-5)


def test_pooled_repeatability_requires_cycle_id():
    frame = _repeat_frame().drop(columns=["cycle_id", "station_id"])
    with pytest.raises(ValueError, match="cycle_id"):
        noise.pooled_repeatability(frame, cycles=None, config=default_config())


def _long_series(values, *, start: str = "2026-01-01", freq: str = "h",
                 metric: str = "d_rad", point_id: str = "P") -> pd.DataFrame:
    return pd.DataFrame({
        "point_id": point_id,
        "segment_id": "E-S1",
        "metric": metric,
        "ts": pd.date_range(start=start, periods=len(values), freq=freq),
        "value": np.asarray(values, dtype=float),
    })


def test_rolling_residual_spike_flagged_and_retained():
    rng = np.random.default_rng(11)
    n = 48
    values = 10.0 + 0.05 * np.arange(n) + rng.normal(0.0, 0.2, n)
    values[24] += 8.0
    series = _long_series(values)
    result = noise.estimate_noise(series, config=default_config())

    assert result.status == STATUS_OK
    flag = result.flags.loc[
        result.flags["ts"] == series.loc[24, "ts"]].iloc[0]
    assert abs(flag["robust_z"]) > 6.0
    assert bool(flag["spike"]) is True
    assert bool(flag["retained"]) is True
    assert result.flags["retained"].all()
    assert len(result.flags) == len(series)
    assert result.pooled.loc[0, "robust_sd_mm"] == pytest.approx(0.2, abs=0.08)
    assert result.pooled.loc[0, "status"] == STATUS_OK


def test_rolling_residual_uses_min_periods():
    series = _long_series([1.0, 1.0])
    result = noise.estimate_noise(series, config=default_config())
    assert result.series["residual_mm"].isna().all()
    assert result.pooled.loc[0, "status"] == STATUS_INSUFFICIENT
    assert np.isnan(result.pooled.loc[0, "robust_sd_mm"])
    assert result.status == STATUS_INSUFFICIENT
    assert result.reason is not None


def test_rolling_edge_uses_available_window_with_min_periods():
    config = default_config()
    config.noise.rolling_window_hours = 5.0
    config.noise.min_observations = 3
    series = _long_series(np.arange(10, dtype=float))
    result = noise.estimate_noise(series, config=config)
    assert result.series["residual_mm"].notna().all()
    assert result.series.loc[0, "rolling_median_mm"] == pytest.approx(1.0)
    assert result.series.loc[9, "rolling_median_mm"] == pytest.approx(8.0)


@pytest.mark.parametrize("metric,floor,small,big", [
    ("d_rad", 0.5, 0.4, 3.5),
    ("d_ver", 1.0, 0.9, 7.0),
    ("d_tan", 1.0, 0.9, 7.0),
    ("d_up", 1.0, 0.9, 7.0),
])
def test_spike_floor_uses_supplied_metric_value(metric, floor, small, big):
    config = default_config()
    baseline = np.full(30, 5.0)
    below = baseline.copy()
    below[10] += small
    result = noise.estimate_noise(_long_series(below, metric=metric),
                                  config=config)
    assert result.flags.loc[10, "robust_z"] == pytest.approx(small / floor)
    assert not bool(result.flags.loc[10, "spike"])

    above = baseline.copy()
    above[10] += big
    result = noise.estimate_noise(_long_series(above, metric=metric),
                                  config=config)
    assert bool(result.flags.loc[10, "spike"]) is True


def test_spike_floors_argument_enables_other_metrics():
    config = default_config()
    values = np.full(30, 5.0)
    values[10] += 7.0
    unknown = noise.estimate_noise(_long_series(values, metric="d_east"),
                                   config=config)
    assert unknown.flags["robust_z"].isna().all()
    assert not unknown.flags["spike"].any()

    enabled = noise.estimate_noise(_long_series(values, metric="d_east"),
                                   config=config, floors={"d_east": 1.0})
    assert bool(enabled.flags.loc[10, "spike"]) is True


def test_flag_spikes_is_idempotent():
    rng = np.random.default_rng(5)
    values = rng.normal(0.0, 0.5, 30)
    values[15] += 6.0
    series = _long_series(values)
    config = default_config()
    result = noise.estimate_noise(series, config=config)
    flags = noise.flag_spikes(series, noise=result, config=config)
    pd.testing.assert_frame_equal(flags, result.flags)

    recomputed = noise.flag_spikes(series, noise=None, config=config)
    assert len(recomputed) == len(series)
    assert recomputed["retained"].all()
