"""In-process integration tests for the shared pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from synthetic import Effects, Prism, build_observations, rows_to_bytes

from rts_forensics.config import load_config
from rts_forensics.pipeline import run_analysis

REPO = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path, *, days: int = 8, boundaries=None,
             local_motion: dict[str, float] | None = None):
    prisms: dict[str, Prism] = {}
    for i in range(24):
        pid = f"P_{i:02d}"
        prisms[pid] = Prism(pid, "E", 1150.0 + 38.0 * i, 2300.0 + 22.0 * i,
                            53.0 + 0.7 * i)
    prisms["Q_0"] = Prism("Q_0", "W", -120.0, 1250.0, 44.0)
    effects = Effects(seed=11, noise_hz_arcsec=0.5, noise_v_arcsec=0.8,
                      noise_d_mm=0.3, rotation_step_arcsec=-4.4,
                      rotation_drift_arcsec_per_day=-0.15,
                      translation_mm=(0.8, -0.4), scale_ppm=-1.5,
                      local_motion_mm_per_day=local_motion or {"P_03": -2.0},
                      repeats=2)
    csv = tmp_path / "synthetic.csv"
    csv.write_bytes(rows_to_bytes(build_observations(
        days=days, prisms=prisms, effects=effects)))
    overrides = {
        "cycles": {"gap_minutes": {"E": 30, "W": 15}},
        "stations": {"anchors": [{"label": "E", "easting_ref": 1000.0},
                                 {"label": "W", "easting_ref": 0.0}]},
        "reliability": {"frame_min_cycles": 20},
        "frame": {"min_prism_cycles_per_fit": 20},
    }
    if boundaries is not None:
        overrides["reliability"]["boundaries"] = boundaries
    cfg = load_config(REPO / "config.yaml", overrides=overrides)
    return csv, cfg


def test_pipeline_registers_tables_and_provenance(tmp_path):
    csv, cfg = _fixture(tmp_path)
    result = run_analysis(str(csv), config=cfg)

    for name in ("observations", "cycle_table", "prism_cycles", "baselines",
                 "displacements", "source_links", "rates", "prism_summary",
                 "frame_cycles", "frame_corrected", "timeseries_24h",
                 "concern", "reliability", "tarp", "evidence",
                 "investigation_register"):
        assert name in result.tables, name
        assert isinstance(result.tables[name], __import__("pandas").DataFrame)

    assert result.parsed.raw_bytes
    assert len(result.displacements.source_links)
    assert (result.tables["tarp"].status == "not_configured").all()
    assert result.audit.summary()["n_not_configured"] >= 1


def test_frame_recovery_and_raw_kept(tmp_path):
    csv, cfg = _fixture(tmp_path)
    result = run_analysis(str(csv), config=cfg)

    assert result.frames.status in ("ok", "partial"), result.frames.reason
    east = result.frames.coefficients
    east = east[east.station_id == "E"]
    assert len(east) and (east.status == "ok").all()
    assert np.isfinite(east.rotation_arcsec).all()
    # the first-48 h baseline absorbs the constant step and the absolute
    # orientation; the frame must still recover the injected drift rate.
    # Cycles are equally spaced (4 h), so the coefficient order is time order.
    elapsed = np.arange(len(east)) * (4.0 / 24.0)
    slope = float(np.polyfit(elapsed, east.rotation_arcsec, 1)[0])
    assert abs(slope - (-0.15)) < 0.05, slope

    corrected = result.tables["frame_corrected"]
    for column in ("los_raw_mm", "los_fc_mm", "ver_raw_mm", "ver_fc_mm",
                   "tan_raw_mm", "tan_fc_mm"):
        assert column in corrected.columns


def test_mover_flagged_and_stables_quiet(tmp_path):
    csv, cfg = _fixture(tmp_path)
    result = run_analysis(str(csv), config=cfg)

    concern = result.tables["concern"]
    mover = concern[(concern.point_id == "P_03") & (concern.metric == "d_rad")]
    assert not mover.empty
    assert mover.iloc[0].status == "ok"
    assert mover.iloc[0].net_mm < -5.0
    assert bool(mover.iloc[0].exceeds) is True

    corrected = result.tables["frame_corrected"]
    anchor = corrected.ts.max()
    stables = ["P_00", "P_01", "P_02"]
    nets = []
    for pid in stables:
        rows = corrected[corrected.point_id == pid].sort_values("ts")
        first = rows[rows.ts <= rows.ts.min() + np.timedelta64(48, "h")]
        last = rows[rows.ts >= anchor - np.timedelta64(48, "h")]
        nets.append(float(last.los_fc_mm.median() - first.los_fc_mm.median()))
    assert max(abs(np.array(nets))) < 1.5


def test_unconfigured_reliability_stays_explicit(tmp_path):
    csv, cfg = _fixture(tmp_path, boundaries=None)
    cfg.reliability.boundaries = None
    result = run_analysis(str(csv), config=cfg)
    assert set(result.tables["reliability"].grade) == {"not_configured"}
    assert set(result.tables["reliability"].status) == {"not_configured"}
    assert result.frames.status == "not_configured"
