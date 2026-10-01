"""End-to-end CLI test on a synthetic export (no private data)."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml
from synthetic import Effects, Prism, build_observations, rows_to_bytes

REPO = Path(__file__).resolve().parents[1]


def _prisms() -> dict[str, Prism]:
    prisms: dict[str, Prism] = {}
    for i in range(22):
        pid = f"P_{i:02d}"
        prisms[pid] = Prism(pid, "E", 1200.0 + 35.0 * i, 2350.0 + 18.0 * i,
                            54.0 + 0.6 * i)
    for i in range(4):
        pid = f"Q_{i}"
        prisms[pid] = Prism(pid, "W", 150.0 + 40.0 * i, 1150.0 + 25.0 * i,
                            43.0 + 0.4 * i)
    return prisms


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    effects = Effects(seed=7, noise_hz_arcsec=0.5, noise_v_arcsec=0.8,
                      noise_d_mm=0.3, rotation_step_arcsec=-4.4,
                      rotation_drift_arcsec_per_day=-0.2,
                      translation_mm=(0.8, -0.4), scale_ppm=-1.5,
                      local_motion_mm_per_day={"P_03": -2.0},
                      missing_night=["Q_2"], repeats=2)
    data = rows_to_bytes(build_observations(days=8, prisms=_prisms(),
                                            effects=effects))
    csv = tmp_path / "synthetic.csv"
    csv.write_bytes(data)
    config = {
        "cycles": {"gap_minutes": {"E": 30, "W": 15}},
        "stations": {"anchors": [{"label": "E", "easting_ref": 1000.0},
                                 {"label": "W", "easting_ref": 0.0}]},
        "reliability": {"frame_min_cycles": 20},
        "frame": {"min_prism_cycles_per_fit": 20},
    }
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return csv, cfg


def test_cli_run_end_to_end(tmp_path):
    csv, cfg = _fixture(tmp_path)
    out = tmp_path / "results"
    proc = subprocess.run(
        [sys.executable, "-m", "rts_forensics.cli", "run", str(csv),
         "--config", str(cfg), "--out", str(out)],
        capture_output=True, text=True, cwd=REPO, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    for name in ("effective_config.yaml", "report.md", "report.html",
                 "rts_forensics.gpkg", "manifest.json", "table_dictionary.csv",
                 "prism_summary.csv", "concern.csv", "reliability.csv",
                 "rates.csv", "source_links.csv", "frame_corrected.csv"):
        assert (out / name).is_file(), name
    assert (out / "raw" / csv.name).is_file()

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sources"] and manifest["config_sha256"]
    assert manifest["outputs"]
    assert "manifest.json" not in manifest["outputs"]

    con = sqlite3.connect(out / "rts_forensics.gpkg")
    try:
        assert con.execute("PRAGMA application_id").fetchone()[0] == 0x47504B47
        layers = {row[0] for row in con.execute(
            "SELECT table_name FROM gpkg_contents")}
    finally:
        con.close()
    assert {"prism_summary", "frame_cycles", "timeseries_24h", "sight_lines",
            "movement_vectors"} <= layers

    assert "safe or unsafe" in proc.stdout
    assert "tarp" in proc.stdout


def test_cli_missing_config(tmp_path):
    csv, _ = _fixture(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "rts_forensics.cli", "run", str(csv),
         "--config", str(tmp_path / "nope.yaml"), "--out", str(tmp_path / "o")],
        capture_output=True, text=True, cwd=REPO, timeout=300)
    assert proc.returncode == 2
    assert "config not found" in proc.stderr


def test_cli_parser_help():
    proc = subprocess.run(
        [sys.executable, "-m", "rts_forensics.cli", "--help"],
        capture_output=True, text=True, cwd=REPO, timeout=120)
    assert proc.returncode == 0
    assert "rts-forensics" in proc.stdout
