"""
test_report.py — CSV, figure, report and manifest contracts.

A small synthetic run mapping (and an equivalent ``RunResult``) is exported to
a temporary directory. The tests assert the honesty rules: raw figures exist,
the mandatory limitations are printed, movement concern and reliability are
separate sections, the TARP-absent exploratory notice is present and the
manifest hashes every output except itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from rts_forensics import report
from rts_forensics.models import (
    ClassificationResult,
    CycleResult,
    DisplacementResult,
    ForensicResult,
    FormatSpec,
    FrameResult,
    NoiseResult,
    ParsedExport,
    RunResult,
    SourceRecord,
)

SHA = "a" * 64


def _displacements() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=6, freq="D")
    rows = []
    for pid, base in (("P1", 0.0), ("P2", 1.0)):
        for index, ts in enumerate(times):
            rows.append({
                "point_id": pid,
                "segment_id": "E-S1",
                "cycle_id": index,
                "ts": ts,
                "d_rad_mm": base + index * 0.4,
                "los_fc_mm": base + index * 0.1,
                "d_vert_mm": -base - index * 0.2,
                "ver_fc_mm": -base - index * 0.05,
            })
    return pd.DataFrame(rows)


def _summary() -> pd.DataFrame:
    return pd.DataFrame({
        "point_id": ["P1", "P2"],
        "station_id": ["E", "E"],
        "east_m": [1000.0, 1100.0],
        "north_m": [2000.0, 2100.0],
        "movement_concern": ["possible", "no_credible_movement"],
        "reliability_grade": ["A", "B"],
    })


def _frame_coefficients() -> pd.DataFrame:
    return pd.DataFrame({
        "station_id": ["E", "E", "E"],
        "segment_id": ["E-S1", "E-S1", "E-S1"],
        "cycle_id": [0, 1, 2],
        "rotation_arcsec": [0.0, -1.0, -2.0],
        "tE_mm": [0.0, 0.5, 1.0],
        "tN_mm": [0.0, -0.5, -1.0],
        "scale_ppm": [0.0, 1.0, 2.0],
    })


def _concern() -> pd.DataFrame:
    return pd.DataFrame({
        "point_id": ["P1", "P2"],
        "segment_id": ["E-S1", "E-S1"],
        "metric": ["los", "los"],
        "net_mm": [7.0, 0.5],
        "status": ["ok", "ok"],
    })


def _reliability() -> pd.DataFrame:
    return pd.DataFrame({
        "point_id": ["P1", "P2"],
        "segment_id": ["E-S1", "E-S1"],
        "grade": ["A", "B"],
        "status": ["ok", "ok"],
    })


def _run_mapping() -> dict:
    return {
        "tables": {
            "displacements": _displacements(),
            "prism_summary": _summary(),
            "frame_cycles": _frame_coefficients(),
        },
        "observations": pd.DataFrame({
            "point_id": ["P1", "P2"],
            "source_sha256": [SHA, SHA],
            "src_line": [2, 3],
        }),
        "parsed": {"sources": [{"name": "export.csv", "sha256": SHA}]},
        "config": {"tarp": "not_configured", "gis": {"srs_id": -1}},
        "frames": {
            "correction_status": "ok",
            "experimental": True,
            "reason": None,
            "coefficients": _frame_coefficients(),
        },
        "classification": {
            "concern": _concern(),
            "reliability": _reliability(),
            "tarp": pd.DataFrame({"status": ["not_configured"]}),
        },
        "manifest": {},
        "audit": {},
    }


def _run_result() -> RunResult:
    observations = pd.DataFrame({
        "point_id": ["P1", "P2"],
        "source_sha256": [SHA, SHA],
        "src_line": [2, 3],
    })
    parsed = ParsedExport(
        observations=observations,
        format=FormatSpec(delimiter=";", encoding="cp1252",
                          time_format="%d/%m/%Y %H:%M"),
        sources=[SourceRecord(name="export.csv", sha256=SHA, n_bytes=10)],
    )
    return RunResult(
        observations=observations,
        parsed=parsed,
        cycles=CycleResult(observations=observations),
        displacements=DisplacementResult(
            prism_cycles=pd.DataFrame(), displacements=_displacements()),
        noise=NoiseResult(),
        frames=FrameResult(coefficients=_frame_coefficients(),
                           correction_status="ok", experimental=True),
        forensics=ForensicResult(),
        classification=ClassificationResult(concern=_concern(), reliability=_reliability()),
        tables={
            "displacements": _displacements(),
            "prism_summary": _summary(),
            "frame_cycles": _frame_coefficients(),
        },
        config={"tarp": "not_configured"},
    )


# ---------------------------------------------------------------------------
#  CSV tables
# ---------------------------------------------------------------------------
def test_write_tables_writes_csv_and_dictionary(tmp_path: Path):
    paths = report.write_tables(_run_mapping(), tmp_path)
    names = {path.name for path in paths}
    assert {"displacements.csv", "prism_summary.csv", "frame_cycles.csv",
            "table_dictionary.csv"} <= names
    for path in paths:
        assert path.exists()
    dictionary = pd.read_csv(tmp_path / "table_dictionary.csv")
    assert {"table", "column"} <= set(dictionary.columns)
    assert not dictionary.empty
    exported = pd.read_csv(tmp_path / "prism_summary.csv")
    assert list(exported["point_id"]) == ["P1", "P2"]


def test_write_tables_accepts_run_result(tmp_path: Path):
    report.write_tables(_run_result(), tmp_path)
    assert (tmp_path / "table_dictionary.csv").exists()
    assert (tmp_path / "displacements.csv").exists()


# ---------------------------------------------------------------------------
#  Figures
# ---------------------------------------------------------------------------
def test_plot_results_writes_figures(tmp_path: Path):
    paths = report.plot_results(_run_mapping(), tmp_path)
    names = {path.name for path in paths}
    assert {"fig_prism_timeseries.png", "fig_station_frame.png",
            "fig_map.png"} <= names
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_results_skips_station_frame_with_note(tmp_path: Path):
    run = _run_mapping()
    run["tables"] = {
        "displacements": _displacements(),
        "prism_summary": _summary(),
    }
    run["frames"] = {
        "correction_status": "not_configured",
        "reason": "frame correction not configured",
    }
    paths = report.plot_results(run, tmp_path)
    names = {path.name for path in paths}
    assert "fig_station_frame.png" not in names
    assert "fig_prism_timeseries.png" in names
    assert "fig_map.png" in names
    note = tmp_path / "fig_station_frame_note.txt"
    assert note.exists()
    assert "not available" in note.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
#  Markdown / HTML report
# ---------------------------------------------------------------------------
def test_render_report_has_separate_sections_and_limitations(tmp_path: Path):
    paths = report.render_report(_run_mapping(), tmp_path)
    assert paths.md.exists()
    assert paths.html.exists()
    text = paths.md.read_text(encoding="utf-8")
    assert "## Movement concern" in text
    assert "## Measurement reliability" in text
    concern_section, reliability_section = text.split("## Measurement reliability", 1)
    assert "## Movement concern" in concern_section
    assert "net_mm" in concern_section
    assert "grade" in reliability_section
    for limitation in report.LIMITATIONS:
        assert limitation in text
    assert "EXPERIMENTAL" in text
    assert "relative to the chosen network" in text
    assert "TARP is not configured" in text
    html = paths.html.read_text(encoding="utf-8")
    assert "Movement concern" in html
    assert "Measurement reliability" in html
    assert "EXPERIMENTAL" in html
    assert report.LIMITATIONS[0] in html


def test_render_report_with_tarp_omits_exploratory_notice(tmp_path: Path):
    run = _run_mapping()
    run["config"] = {"tarp": {"los_mm": 10.0}}
    text = report.render_report(run, tmp_path).md.read_text(encoding="utf-8")
    assert "TARP is not configured" not in text
    assert "EXPERIMENTAL" in text


def test_render_report_accepts_run_result(tmp_path: Path):
    text = report.render_report(_run_result(), tmp_path).md.read_text(encoding="utf-8")
    assert report.LIMITATIONS[-1] in text
    assert "## Movement concern" in text


# ---------------------------------------------------------------------------
#  Manifest
# ---------------------------------------------------------------------------
def test_write_manifest_excludes_itself_and_hashes_outputs(tmp_path: Path):
    run = _run_mapping()
    report.write_tables(run, tmp_path)
    report.render_report(run, tmp_path)
    path = report.write_manifest(run, tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "manifest.json"
    assert "manifest.json" not in manifest["outputs"]
    assert "report.md" in manifest["outputs"]
    assert "report.html" in manifest["outputs"]
    assert "displacements.csv" in manifest["outputs"]
    expected = hashlib.sha256((tmp_path / "report.md").read_bytes()).hexdigest()
    assert manifest["outputs"]["report.md"] == expected
    assert manifest["sources"]["export.csv"] == SHA
    assert manifest["config_sha256"]
    assert manifest["runtime"]["package"]


def test_write_manifest_rerun_does_not_hash_previous_manifest(tmp_path: Path):
    run = _run_mapping()
    report.write_tables(run, tmp_path)
    report.write_manifest(run, tmp_path)
    second = report.write_manifest(run, tmp_path)
    manifest = json.loads(second.read_text(encoding="utf-8"))
    assert "manifest.json" not in manifest["outputs"]
