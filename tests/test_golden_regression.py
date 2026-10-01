"""
Real-data regression against golden_values.json.

The machine-readable fixture is committed at ``tests/fixtures/golden_values.json``
(the site confirmed the export uses a shifted dummy grid, and the owner restores
the real grid separately). Skipped unless the monthly export itself is available;
place it at ``tests/fixtures/HLO_Sept_2026.csv`` or point the environment at it:

    RTS_HLO_CSV=/path/HLO_Sept_2026.csv
    RTS_GOLDEN_VALUES=/path/golden_values.json   # optional override

Validation uses the explicit fixture definitions and tolerances below. The
rounded handoff table is never substituted for the machine-readable fixture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from rts_forensics.config import load_config
from rts_forensics.pipeline import run_analysis

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"


def _golden_path() -> Path | None:
    candidates = [os.environ.get("RTS_GOLDEN_VALUES"),
                  FIXTURES / "golden_values.json"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def _csv_path() -> Path | None:
    candidates = [os.environ.get("RTS_HLO_CSV"),
                  FIXTURES / "HLO_Sept_2026.csv",
                  REPO / "HLO_Sept_2026.csv"]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


pytestmark = pytest.mark.skipif(
    _golden_path() is None or _csv_path() is None,
    reason="monthly export is not present; drop it at tests/fixtures/ or set "
           "RTS_HLO_CSV")


@pytest.fixture(scope="module")
def golden():
    return json.loads(_golden_path().read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def hasil():
    cfg = load_config(REPO / "config.yaml")
    return run_analysis(str(_csv_path()), config=cfg)


def _close(value, target, tol, rel=True):
    assert np.isfinite(value), f"value not finite: {value} vs {target}"
    limit = tol * abs(target) if rel else tol
    assert abs(value - target) <= max(limit, 1e-9), f"{value} != {target}"


def test_input_counts(hasil, golden):
    assert len(hasil.observations) == golden["input"]["rows"]
    assert hasil.parsed.sources[0].n_rows == golden["input"]["rows"]
    assert hasil.observations.point_id.nunique() == golden["input"]["prisms"]
    cycles = hasil.cycles.cycles
    counts = cycles.groupby("station_id").size().to_dict()
    for station, expected in golden["input"]["stations"].items():
        assert counts.get(station) == expected["cycles"]


def test_repeatability(hasil, golden):
    pooled = hasil.noise.pooled
    all_rows = pooled[pooled.span_class == "all"].iloc[0]
    expected = golden["repeatability"]
    _close(all_rows.sd_d_mm, expected["D_mm"], 0.25, rel=False)
    _close(all_rows.sd_hz_arcsec, expected["Hz_arcsec"], 0.25, rel=False)
    _close(all_rows.sd_v_arcsec, expected["V_arcsec"], 0.25, rel=False)


def test_station_e_frame(hasil, golden):
    expected = golden["station_E"]
    coefficients = hasil.frames.coefficients
    assert len(coefficients), hasil.frames.reason
    east = coefficients[coefficients.station_id == "E"].sort_values("ts")
    assert len(east)

    change = str(expected["processing_change"])
    step = east[east.ts >= change].iloc[0].rotation_arcsec - \
        east[east.ts < change].iloc[-1].rotation_arcsec
    _close(east.iloc[-1].rotation_arcsec, expected["rotation_end_arcsec"], 0.15)

    rotation_step = expected["rotation_step_7Sep_arcsec"]
    before = east[east.ts < "2026-09-07 04:34"].iloc[-1].rotation_arcsec
    after = east[east.ts >= "2026-09-07 08:04"].iloc[0].rotation_arcsec
    _close(after - before, rotation_step, 0.2, rel=False)
    assert np.isfinite(step)


def test_g1_frame_corrected_net(hasil, golden):
    corrected = hasil.tables["frame_corrected"]
    assert len(corrected)
    anchor = corrected.ts.max()
    window = 48
    for pid, target in golden["G1"]["net_los_frame_corrected_mm"].items():
        rows = corrected[corrected.point_id == pid].sort_values("ts")
        assert len(rows), f"missing corrected rows for {pid}"
        first = rows[rows.ts <= rows.ts.min() + np.timedelta64(window, "h")]
        last = rows[rows.ts >= anchor - np.timedelta64(window, "h")]
        net = float(last.los_fc_mm.median() - first.los_fc_mm.median())
        _close(net, target, 0.3, rel=False)


def test_placebo_and_screening_not_invented(hasil):
    assert (hasil.tables["tarp"].status == "not_configured").all()
    assert hasil.classification.statuses.get("tarp") in (None, "not_configured")
