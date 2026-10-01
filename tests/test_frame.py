"""
test_frame.py — independent synthetic recovery of the per-cycle frame fit.

The fixture is built with the shared forward model in ``tests/synthetic.py``
(no pipeline code). Two exports are generated with identical geometry, one
clean and one carrying an injected rotation step and drift, a station
translation and a scale error; ``dD``, ``dHz_as`` and ``ver_raw`` are formed by
differencing the raw observations of the two exports. The frame fit must
recover the injected values, correct the stable prisms towards zero and keep
the local mover's signal.

The export serialises D and coordinates at 1 mm resolution, so recovered
translation/scale carry a quantization floor of a few tenths of a millimetre;
tolerances below state that limit explicitly.

No production constant is injected by the fixture: every test passes the
configuration it relies on.
"""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from synthetic import Effects, Prism, Station, build_observations

from rts_forensics.config import load_config
from rts_forensics.frame import (
    HORIZONTAL_PARAMETERS,
    VERTICAL_PARAMETERS,
    apply_frame,
    fit_cycle_frame,
    fit_frames,
    select_frame_set,
)
from rts_forensics.models import (
    STATUS_INSUFFICIENT,
    STATUS_OK,
    STATUS_RANK_DEFICIENT,
    FrameSelection,
)

_DMS = re.compile(r"^(-?)(\d+)\D+(\d+)\D+([\d.]+)\D*$")
_START = datetime(2026, 1, 1)
_DAYS = 3
_ROTATION_STEP = -4.4
_ROTATION_DRIFT = -0.7
_TRANSLATION = (25.0, -30.0)
_SCALE_PPM = 15.0
_MOVER = "M0"
_MOVER_MM_PER_DAY = -3.0


def _parse_dms(text) -> float:
    sign, degrees, minutes, seconds = _DMS.match(str(text).strip()).groups()
    value = int(degrees) + int(minutes) / 60.0 + float(seconds) / 3600.0
    return -value if sign else value


def _station() -> Station:
    return Station("E", 1000.0, 2000.0, 50.0, 30.0, 30, [0, 4, 8, 12, 16, 20])


def _prisms() -> dict[str, Prism]:
    station = _station()
    prisms: dict[str, Prism] = {}
    for index in range(12):
        azimuth = (index * 30.0 + 7.0) % 360.0
        horizontal = 400.0 + 90.0 * index
        east = station.east + horizontal * np.sin(np.radians(azimuth))
        north = station.north + horizontal * np.cos(np.radians(azimuth))
        pid = f"S{index:02d}"
        prisms[pid] = Prism(pid, "E", round(east, 4), round(north, 4), 40.0 + 2.0 * index)
    prisms[_MOVER] = Prism(_MOVER, "E", 1600.0, 2100.0, 46.0)
    return prisms


def _observations(rows: list[dict]) -> pd.DataFrame:
    out = []
    for row in rows:
        out.append({
            "point_id": row["Point ID"],
            "time": datetime.strptime(row["Time"], "%d/%m/%Y %H:%M"),
            "hz": _parse_dms(row["Hz [dms]"]),
            "v": _parse_dms(row["V [dms]"]),
            "D": float(row["D [m]"]),
        })
    return pd.DataFrame(out)


def _config(**frame_overrides):
    return load_config(overrides={"frame": frame_overrides})


@pytest.fixture(scope="module")
def recorded_change() -> dict:
    """Injected effects, raw long series and expected per-cycle values."""
    station = _station()
    prisms = _prisms()
    effects = Effects(
        seed=0,
        rotation_step_arcsec=_ROTATION_STEP,
        rotation_drift_arcsec_per_day=_ROTATION_DRIFT,
        translation_mm=_TRANSLATION,
        scale_ppm=_SCALE_PPM,
        local_motion_mm_per_day={_MOVER: _MOVER_MM_PER_DAY},
    )
    changed = _observations(build_observations(
        days=_DAYS, stations={"E": station}, prisms=prisms, effects=effects))
    clean = _observations(build_observations(
        days=_DAYS, stations={"E": station}, prisms=prisms, effects=Effects(seed=0)))
    merged = changed.merge(clean, on=["point_id", "time"], suffixes=("_e", "_r"))

    d_hz = ((merged["hz_e"] - merged["hz_r"] + 180.0) % 360.0 - 180.0) * 3600.0
    d_d = (merged["D_e"] - merged["D_r"]) * 1000.0
    ver_raw = (merged["D_e"] * np.cos(np.radians(merged["v_e"]))
               - merged["D_r"] * np.cos(np.radians(merged["v_r"]))) * 1000.0
    true_azimuth = {
        pid: float(np.degrees(np.arctan2(
            prism.east - station.east, prism.north - station.north)) % 360.0)
        for pid, prism in prisms.items()
    }
    azimuth = merged["point_id"].map(true_azimuth).to_numpy(dtype=float)
    times = sorted(merged["time"].unique())
    cycle_of = {time: index for index, time in enumerate(times)}

    series = pd.DataFrame({
        "point_id": merged["point_id"],
        "station_id": "E",
        "segment_id": "E-S1",
        "cycle_id": [cycle_of[time] for time in merged["time"]],
        "az_deg": azimuth,
        "v": merged["v_r"].to_numpy(dtype=float),
        "D": merged["D_r"].to_numpy(dtype=float),
        "dD": d_d.to_numpy(dtype=float),
        "dHz_as": d_hz.to_numpy(dtype=float),
        "ver_raw": ver_raw.to_numpy(dtype=float),
    })
    stable = sorted(pid for pid in prisms if pid != _MOVER)
    elapsed_days = {index: (time - times[0]).total_seconds() / 86400.0
                    for index, time in enumerate(times)}
    day_index = {index: (time - times[0]).days for index, time in enumerate(times)}
    summary = pd.DataFrame({
        "point_id": stable + [_MOVER],
        "n_cycles": [150] * len(stable) + [150],
        "reliability": ["A (good)"] * len(stable) + ["B (moderate)"],
    })
    return {
        "series": series,
        "stable": stable,
        "summary": summary,
        "elapsed_days": elapsed_days,
        "day_index": day_index,
    }


@pytest.fixture(scope="module")
def fitted(recorded_change) -> dict:
    config = _config(min_prism_cycles_per_fit=6)
    selection = select_frame_set(
        recorded_change["summary"], exclusions=[_MOVER], config=config)
    result = fit_frames(recorded_change["series"], frame_set=selection, config=config)
    return {"selection": selection, "result": result, "config": config}


# ---------------------------------------------------------------------------
#  Frame-set selection
# ---------------------------------------------------------------------------
def test_select_frame_set_filters_and_reports_exclusions():
    config = _config()
    summary = pd.DataFrame({
        "point_id": ["A1", "A2", "B1", "A3"],
        "n_cycles": [150, 150, 150, 50],
        "reliability": ["A (good)", "A (good)", "B (moderate)", "A (good)"],
    })
    selection = select_frame_set(summary, exclusions=["A2"], config=config)
    assert selection.status == STATUS_OK
    assert selection.members == ["A1"]
    assert selection.exclusions == ["A2"]
    diagnostics = selection.diagnostics.set_index("point_id")
    assert bool(diagnostics.loc["A1", "member"])
    assert "excluded" in diagnostics.loc["A2", "reason"]
    assert not bool(diagnostics.loc["B1", "member"])
    assert not bool(diagnostics.loc["A3", "member"])


def test_select_frame_set_insufficient_without_members():
    config = _config()
    summary = pd.DataFrame({
        "point_id": ["A1"],
        "n_cycles": [10],
        "reliability": ["A (good)"],
    })
    selection = select_frame_set(summary, exclusions=None, config=config)
    assert selection.members == []
    assert selection.status == STATUS_INSUFFICIENT
    assert selection.reason


# ---------------------------------------------------------------------------
#  Recovery of the injected frame
# ---------------------------------------------------------------------------
def test_fit_frames_recovers_injected_effects(recorded_change, fitted):
    result = fitted["result"]
    assert fitted["selection"].members == recorded_change["stable"]
    assert result.correction_status == STATUS_OK
    assert result.experimental is True
    coefficients = result.coefficients
    assert (coefficients["status"] == STATUS_OK).all()
    assert len(coefficients) == len(recorded_change["elapsed_days"])

    elapsed = coefficients["cycle_id"].map(recorded_change["elapsed_days"])
    expected_rotation = _ROTATION_STEP + _ROTATION_DRIFT * elapsed
    assert np.allclose(coefficients["rotation_arcsec"], expected_rotation, atol=0.15)

    slope = np.polyfit(elapsed.to_numpy(dtype=float),
                       coefficients["rotation_arcsec"].to_numpy(dtype=float), 1)[0]
    assert slope == pytest.approx(_ROTATION_DRIFT, abs=0.05)

    assert coefficients["tE_mm"].mean() == pytest.approx(_TRANSLATION[0], abs=0.5)
    assert coefficients["tN_mm"].mean() == pytest.approx(_TRANSLATION[1], abs=0.5)
    assert coefficients["scale_ppm"].mean() == pytest.approx(_SCALE_PPM, abs=1.0)
    assert (coefficients["rank"] == 4).all()


def test_cycle_fit_reports_coefficients_and_diagnostics(recorded_change, fitted):
    config = fitted["config"]
    selection = fitted["selection"]
    prism_cycle = recorded_change["series"][
        recorded_change["series"]["cycle_id"] == 0]
    fit = fit_cycle_frame(prism_cycle, frame_set=selection, config=config)
    assert fit.status == STATUS_OK
    assert set(HORIZONTAL_PARAMETERS + VERTICAL_PARAMETERS).issubset(fit.coefficients)
    assert {"rms_hz", "rms_D", "rms_v"}.issubset(fit.coefficients)
    assert set(HORIZONTAL_PARAMETERS + VERTICAL_PARAMETERS).issubset(fit.std_errors)
    assert set(fit.residuals.columns) >= {
        "point_id", "residual_hz_arcsec", "residual_los_mm", "residual_ver_mm"}
    assert fit.rank == 4
    assert fit.condition_number is not None


# ---------------------------------------------------------------------------
#  Corrected series
# ---------------------------------------------------------------------------
def test_corrected_stable_prisms_near_zero_mover_keeps_signal(recorded_change, fitted):
    corrected = fitted["result"].corrected
    stable = corrected[corrected["point_id"].isin(recorded_change["stable"])]
    assert len(stable) > 0
    assert np.nanmax(np.abs(stable["los_fc_mm"])) < 1.0
    assert np.nanmedian(np.abs(stable["los_fc_mm"])) < 0.5

    mover = corrected[corrected["point_id"] == _MOVER]
    expected = _MOVER_MM_PER_DAY * mover["cycle_id"].map(recorded_change["day_index"])
    assert np.allclose(mover["los_fc_mm"], expected, atol=0.8)
    assert mover["los_fc_mm"].iloc[-1] < -4.0


def test_raw_series_is_not_mutated(recorded_change, fitted):
    series = recorded_change["series"]
    before = series.copy(deep=True)
    fit_frames(series, frame_set=fitted["selection"], config=fitted["config"])
    assert series.equals(before)
    assert recorded_change["series"].equals(before)


def test_apply_frame_keeps_raw_columns_beside_corrected(recorded_change, fitted):
    series = recorded_change["series"]
    merged = apply_frame(series, fitted["result"])
    assert len(merged) == len(series)
    assert list(merged.index) == list(series.index)
    for column in ("dD", "dHz_as", "ver_raw", "az_deg", "v", "D"):
        assert np.array_equal(merged[column].to_numpy(), series[column].to_numpy())
    assert {"los_raw_mm", "los_fc_mm", "tan_raw_mm", "tan_fc_mm",
            "ver_raw_mm", "ver_fc_mm", "frame_status"}.issubset(merged.columns)
    assert "status" not in merged.columns
    assert not np.shares_memory(merged["dD"].to_numpy(), series["dD"].to_numpy())


# ---------------------------------------------------------------------------
#  Failure modes
# ---------------------------------------------------------------------------
def _degenerate_series(n: int = 24) -> pd.DataFrame:
    return pd.DataFrame({
        "point_id": [f"R{index:02d}" for index in range(n)],
        "station_id": "E",
        "segment_id": "E-S1",
        "cycle_id": 0,
        "az_deg": 45.0,
        "v": 90.0,
        "D": 500.0,
        "dD": np.linspace(-1.0, 1.0, n),
        "dHz_as": np.linspace(-0.5, 0.5, n),
        "ver_raw": np.linspace(-2.0, 2.0, n),
    })


def test_rank_deficient_geometry_returns_status_and_nan_corrections():
    config = _config(min_prism_cycles_per_fit=6)
    series = _degenerate_series()
    members = series["point_id"].tolist()
    selection = FrameSelection(members=members, status=STATUS_OK)
    direct = fit_cycle_frame(series, frame_set=selection, config=config)
    assert direct.status == STATUS_RANK_DEFICIENT
    assert direct.coefficients == {}
    assert direct.std_errors == {}

    result = fit_frames(series, frame_set=selection, config=config)
    assert (result.coefficients["status"] == STATUS_RANK_DEFICIENT).all()
    assert result.coefficients["rotation_arcsec"].isna().all()
    assert np.isnan(result.corrected["los_fc_mm"]).all()
    assert np.isnan(result.corrected["tan_fc_mm"]).all()
    assert np.isnan(result.corrected["ver_fc_mm"]).all()
    assert not (result.corrected["los_fc_mm"] == 0.0).any()
    assert (result.corrected["status"] == STATUS_RANK_DEFICIENT).all()


def test_insufficient_frame_members_leave_corrected_nan(recorded_change):
    config = _config(min_prism_cycles_per_fit=6)
    series = recorded_change["series"]
    selection = FrameSelection(members=recorded_change["stable"][:3], status=STATUS_OK)
    direct = fit_cycle_frame(
        series[series["cycle_id"] == 0], frame_set=selection, config=config)
    assert direct.status == STATUS_INSUFFICIENT
    assert direct.coefficients == {}
    assert direct.std_errors == {}

    result = fit_frames(series, frame_set=selection, config=config)
    assert result.correction_status == "not_configured"
    assert (result.coefficients["status"] == STATUS_INSUFFICIENT).all()
    assert np.isnan(result.corrected["los_fc_mm"]).all()
    assert np.isnan(result.corrected["ver_fc_mm"]).all()
