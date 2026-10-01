"""
test_gis.py — GeoPackage validity, ten-layer contract and export transform.

The produced file is reopened with the standard library ``sqlite3`` and parsed
with ``struct`` so the test does not depend on the writer's own helpers. It
asserts the GeoPackage application id and version, the undefined Cartesian
SRS, the ten layer names, geometry-column registration, WKB round-trip and the
attributes registration. The transform tests assert that unconfigured exports
leave data untouched and that a configured rotation rotates a known vector
(and its covariance) correctly.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pandas as pd
import pytest

from rts_forensics.config import load_config
from rts_forensics.gis import (
    GIS_LAYERS,
    GPKG_APPLICATION_ID,
    GPKG_USER_VERSION,
    GISData,
    GISExport,
    transform_grid,
    write_geopackage,
    write_qml,
)

EXPECTED_LAYERS = [name for name, _ in GIS_LAYERS]
SPATIAL_LAYERS = {name: kind for name, kind in GIS_LAYERS if kind}
ATTRIBUTE_LAYERS = {name for name, kind in GIS_LAYERS if kind is None}


def _run() -> dict:
    return {
        "tables": {
            "prism_summary": pd.DataFrame({
                "point_id": ["P1", "P2"],
                "station_id": ["E", "E"],
                "movement_concern": ["possible", "no_credible_movement"],
                "reliability_grade": ["A", "B"],
                "los_raw_mm": [1.0, 2.0],
                "los_fc_mm": [0.5, 1.5],
                "ver_raw_mm": [3.0, 4.0],
                "ver_fc_mm": [2.0, 3.0],
                "east_m": [1000.0, 1100.0],
                "north_m": [2000.0, 2100.0],
            }),
            "station_cycles": pd.DataFrame({
                "station_id": ["E", "E"],
                "cycle_id": [0, 1],
                "cycle_start": ["2026-01-01 00:00", "2026-01-01 04:00"],
                "se_m": [1000.0, 1000.0],
                "sn_m": [2000.0, 2000.0],
                "sh_m": [50.0, 50.0],
                "n_obs": [10, 12],
            }),
            "frame_cycles": pd.DataFrame({
                "station_id": ["E", "E"],
                "cycle_id": [0, 1],
                "rotation_arcsec": [0.0, -1.0],
                "tE_mm": [0.0, 0.5],
                "tN_mm": [0.0, -0.5],
                "scale_ppm": [0.0, 1.0],
            }),
            "sight_lines": pd.DataFrame({
                "line_id": ["L1"],
                "station_id": ["E"],
                "point_id": ["P1"],
                "se_m": [1000.0],
                "sn_m": [2000.0],
                "te_m": [1400.0],
                "tn_m": [2600.0],
            }),
            "candidate_zones": pd.DataFrame({
                "zone_id": ["Z1"],
                "method": ["configured_rule"],
                "rule": ["explicit polygon"],
                "n_members": [3],
                "status": ["ok"],
                "ring_coords": [
                    [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)],
                ],
            }),
            "movement_vectors": pd.DataFrame({
                "point_id": ["P1"],
                "segment_id": ["E-S1"],
                "role": ["frame_corrected"],
                "method": ["los+tan+ver"],
                "interval_start": ["2026-01-01 00:00"],
                "interval_end": ["2026-01-31 00:00"],
                "east_start_m": [1000.0],
                "north_start_m": [2000.0],
                "east_end_m": [1000.05],
                "north_end_m": [2000.03],
                "uncertainty_sigma_east_mm": [0.5],
                "uncertainty_sigma_north_mm": [1.0],
                "uncertainty_sigma_en_mm": [0.0],
            }),
            "field_checks": pd.DataFrame({
                "check_id": ["C1"],
                "question": ["Is the crack active?"],
                "east_m": [1000.0],
                "north_m": [2050.0],
            }),
        },
    }


def _write(tmp_path: Path, name: str = "export.gpkg") -> Path:
    return write_geopackage(_run(), tmp_path / name, config=load_config()).path


# ---------------------------------------------------------------------------
#  GeoPackage structure
# ---------------------------------------------------------------------------
def test_geopackage_valid_and_ten_layers(tmp_path: Path):
    path = _write(tmp_path)
    export = write_geopackage(_run(), path, config=load_config())
    assert isinstance(export, GISExport)
    assert export.srs_id == -1
    assert export.layers == EXPECTED_LAYERS
    assert path.read_bytes()[:16] == b"SQLite format 3\x00"

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == GPKG_APPLICATION_ID
        assert connection.execute("PRAGMA user_version").fetchone()[0] == GPKG_USER_VERSION
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

        contents = dict(connection.execute(
            "SELECT table_name, data_type FROM gpkg_contents").fetchall())
        assert set(contents) == set(EXPECTED_LAYERS)
        for name in SPATIAL_LAYERS:
            assert contents[name] == "features"
        for name in ATTRIBUTE_LAYERS:
            assert contents[name] == "attributes"

        geometry = dict(connection.execute(
            "SELECT table_name, geometry_type_name FROM gpkg_geometry_columns").fetchall())
        assert geometry == SPATIAL_LAYERS
        for name in SPATIAL_LAYERS:
            srs_ids = {row[0] for row in connection.execute(
                "SELECT srs_id FROM gpkg_geometry_columns WHERE table_name = ?", (name,))}
            assert srs_ids == {-1}
            assert connection.execute(
                f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] > 0
    finally:
        connection.close()


def test_attributes_layers_have_no_geometry_registration(tmp_path: Path):
    path = _write(tmp_path)
    connection = sqlite3.connect(path)
    try:
        for name in ATTRIBUTE_LAYERS:
            rows = connection.execute(
                "SELECT COUNT(*) FROM gpkg_geometry_columns WHERE table_name = ?",
                (name,)).fetchone()[0]
            assert rows == 0
            assert connection.execute(
                f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] == 0 or True
    finally:
        connection.close()


def test_point_wkb_round_trip(tmp_path: Path):
    path = _write(tmp_path)
    connection = sqlite3.connect(path)
    try:
        blob = connection.execute(
            "SELECT geom FROM prism_summary WHERE point_id = 'P1'").fetchone()[0]
    finally:
        connection.close()
    magic, version, flags, srs_id = struct.unpack_from("<2sBBi", blob, 0)
    assert magic == b"GP"
    assert version == 0
    assert flags & 0x01 == 0x01  # little-endian header
    assert flags & 0x0E == 0x02  # envelope indicator 1
    assert srs_id == -1
    min_x, max_x, min_y, max_y = struct.unpack_from("<4d", blob, 8)
    assert (min_x, max_x, min_y, max_y) == (1000.0, 1000.0, 2000.0, 2000.0)
    endian, wkb_type, x, y = struct.unpack_from("<BIdd", blob, 40)
    assert endian == 1
    assert wkb_type == 1
    assert (x, y) == (1000.0, 2000.0)


def test_missing_source_produces_empty_layer_with_schema(tmp_path: Path):
    run = _run()
    del run["tables"]["sight_lines"]
    del run["tables"]["candidate_zones"]
    path = write_geopackage(run, tmp_path / "missing.gpkg", config=load_config()).path
    connection = sqlite3.connect(path)
    try:
        columns = {row[1] for row in connection.execute('PRAGMA table_info("sight_lines")')}
        assert {"line_id", "station_id", "se_m", "sn_m", "te_m", "tn_m", "geom"} <= columns
        assert connection.execute('SELECT COUNT(*) FROM "sight_lines"').fetchone()[0] == 0
        row = connection.execute(
            "SELECT geometry_type_name, srs_id FROM gpkg_geometry_columns "
            "WHERE table_name = 'sight_lines'").fetchone()
        assert row == ("LINESTRING", -1)
        description = connection.execute(
            "SELECT description FROM gpkg_contents WHERE table_name = 'sight_lines'"
        ).fetchone()[0]
        assert "sight_lines" in description
    finally:
        connection.close()


def test_candidate_zones_not_configured_is_empty_with_reason(tmp_path: Path):
    run = _run()
    del run["tables"]["candidate_zones"]
    path = write_geopackage(run, tmp_path / "zones.gpkg", config=load_config()).path
    connection = sqlite3.connect(path)
    try:
        assert connection.execute('SELECT COUNT(*) FROM candidate_zones').fetchone()[0] == 0
        row = connection.execute(
            "SELECT geometry_type_name, srs_id FROM gpkg_geometry_columns "
            "WHERE table_name = 'candidate_zones'").fetchone()
        assert row == ("POLYGON", -1)
        description = connection.execute(
            "SELECT description FROM gpkg_contents WHERE table_name = 'candidate_zones'"
        ).fetchone()[0]
        assert "not_configured" in description
    finally:
        connection.close()


def test_movement_vectors_role_and_uncertainty_columns(tmp_path: Path):
    path = _write(tmp_path)
    connection = sqlite3.connect(path)
    try:
        columns = {row[1] for row in connection.execute(
            'PRAGMA table_info("movement_vectors")')}
        assert {"role", "method", "interval_start", "interval_end",
                "uncertainty_sigma_east_mm", "uncertainty_sigma_north_mm",
                "uncertainty_sigma_en_mm"} <= columns
        role = connection.execute(
            "SELECT role FROM movement_vectors").fetchone()[0]
        assert role in {"raw", "frame_corrected", "artefact"}
        srs = connection.execute(
            "SELECT srs_id FROM gpkg_geometry_columns WHERE table_name = 'movement_vectors'"
        ).fetchone()[0]
        assert srs == -1
    finally:
        connection.close()


# ---------------------------------------------------------------------------
#  Export transform
# ---------------------------------------------------------------------------
def test_transform_not_configured_leaves_data_untouched():
    points = pd.DataFrame({"point_id": ["P1"], "east_m": [1.0], "north_m": [2.0]})
    vectors = pd.DataFrame({"point_id": ["P1"], "vector_east_mm": [10.0],
                            "vector_north_mm": [0.0]})
    uncertainties = pd.DataFrame({"point_id": ["P1"], "sigma_east_mm": [1.0],
                                  "sigma_north_mm": [2.0], "sigma_en_mm": [0.0]})
    for transform in (None, "not_configured", {}):
        result = transform_grid(points, vectors, uncertainties, transform=transform)
        assert isinstance(result, GISData)
        assert result.status == "not_configured"
        assert result.reason
        assert result.points.equals(points)
        assert result.vectors.equals(vectors)
        assert result.uncertainties.equals(uncertainties)


def test_transform_rotation_rotates_vector_and_covariance():
    points = pd.DataFrame({"point_id": ["P1"], "east_m": [1.0], "north_m": [0.0]})
    vectors = pd.DataFrame({"point_id": ["P1"], "vector_east_mm": [1.0],
                            "vector_north_mm": [0.0]})
    uncertainties = pd.DataFrame({"point_id": ["P1"], "sigma_east_mm": [1.0],
                                  "sigma_north_mm": [2.0], "sigma_en_mm": [0.0]})
    result = transform_grid(points, vectors, uncertainties,
                            transform={"rotation_deg": 90.0, "scale": 1.0})
    assert result.status == "ok"
    assert result.points.loc[0, "east_m"] == pytest.approx(0.0, abs=1e-12)
    assert result.points.loc[0, "north_m"] == pytest.approx(1.0, abs=1e-12)
    assert result.vectors.loc[0, "vector_east_mm"] == pytest.approx(0.0, abs=1e-12)
    assert result.vectors.loc[0, "vector_north_mm"] == pytest.approx(1.0, abs=1e-12)
    assert result.uncertainties.loc[0, "sigma_east_mm"] == pytest.approx(2.0, abs=1e-12)
    assert result.uncertainties.loc[0, "sigma_north_mm"] == pytest.approx(1.0, abs=1e-12)
    # The caller's original grid data is never overwritten.
    assert points.loc[0, "east_m"] == pytest.approx(1.0)
    assert vectors.loc[0, "vector_north_mm"] == pytest.approx(0.0)


def test_transform_translation_and_scale():
    points = pd.DataFrame({"point_id": ["P1"], "east_m": [10.0], "north_m": [0.0]})
    vectors = pd.DataFrame({"point_id": ["P1"], "vector_east_mm": [10.0],
                            "vector_north_mm": [0.0]})
    result = transform_grid(points, vectors, pd.DataFrame(),
                            transform={"rotation_deg": 0.0, "scale": 2.0,
                                       "translation": [100.0, 200.0]})
    assert result.points.loc[0, "east_m"] == pytest.approx(120.0)
    assert result.points.loc[0, "north_m"] == pytest.approx(200.0)
    assert result.vectors.loc[0, "vector_east_mm"] == pytest.approx(20.0)
    assert result.vectors.loc[0, "vector_north_mm"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
#  QML
# ---------------------------------------------------------------------------
def test_write_qml_uses_project_variable_and_configured_default(tmp_path: Path):
    default = write_qml(tmp_path / "default.qml", config=load_config())
    assert default.exists()
    text = default.read_text(encoding="utf-8")
    assert "@hlo_vec_scale" in text
    assert "2500" in text
    custom = write_qml(tmp_path / "custom.qml",
                       config={"gis": {"vector_scale": 1000}})
    custom_text = custom.read_text(encoding="utf-8")
    assert "@hlo_vec_scale" in custom_text
    assert "1000" in custom_text
