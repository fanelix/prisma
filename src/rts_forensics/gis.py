"""
gis.py — ten-layer GeoPackage export, export-only grid transform and QML style.

The exporter deliberately uses only the standard library (``sqlite3`` and
``struct``). It writes a valid GeoPackage:

* ``application_id`` ``0x47504B47`` and ``user_version`` ``10300``;
* ``gpkg_spatial_ref_sys``, ``gpkg_contents`` and ``gpkg_geometry_columns``;
* geometry blobs with the GeoPackage header ("GP", little-endian, explicit
  envelope) followed by little-endian WKB.

Layer names are exactly the ten documented in ``docs/design.md``::

    prism_summary          POINT
    station_cycles         POINT
    frame_cycles           attributes
    timeseries_24h         attributes
    event_register         attributes
    investigation_register attributes
    field_checks           POINT
    sight_lines            LINESTRING
    candidate_zones        POLYGON
    movement_vectors       LINESTRING

A missing or empty source table produces an empty layer with the fixed schema
and a reason recorded in ``gpkg_contents.description`` — never an error. The
default spatial reference is ``srs_id = -1`` (undefined Cartesian); another
identifier is used only when the user configured it explicitly. No EPSG code
is ever inferred.

Key columns per layer (full schema in ``_SCHEMAS``)::

    prism_summary          point_id (concern/reliability/raw+corrected values)
    station_cycles         station_id + cycle_id
    frame_cycles           station_id + segment_id + cycle_id
    timeseries_24h         point_id + segment_id + block_start
    event_register         event_id
    investigation_register investigation_id
    field_checks           check_id
    sight_lines            line_id (station -> target)
    candidate_zones        zone_id
    movement_vectors       point_id + segment_id + role
                           (raw / frame_corrected / artefact)

:func:`transform_grid` is export-only: the original grid data is never
overwritten. When no transform is configured the input frames are returned
unchanged with status ``not_configured``. When an affine transform is
configured, point coordinates, vector components and covariance/uncertainty
geometry are transformed together.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import is_not_configured
from .models import NOT_CONFIGURED, STATUS_OK

GPKG_APPLICATION_ID = 0x47504B47
GPKG_USER_VERSION = 10300
GPKG_HEADER_MAGIC = b"GP"
GPKG_HEADER_VERSION = 0
#: little-endian header, envelope indicator 1 ([minx, maxx, miny, maxy]).
GPKG_HEADER_FLAGS = 0x03

VECTOR_ROLES = ("raw", "frame_corrected", "artefact")

#: The ten layers in the documented order: (name, geometry type or None).
GIS_LAYERS: tuple[tuple[str, str | None], ...] = (
    ("prism_summary", "POINT"),
    ("station_cycles", "POINT"),
    ("frame_cycles", None),
    ("timeseries_24h", None),
    ("event_register", None),
    ("investigation_register", None),
    ("field_checks", "POINT"),
    ("sight_lines", "LINESTRING"),
    ("candidate_zones", "POLYGON"),
    ("movement_vectors", "LINESTRING"),
)

#: Fixed column schemas. Key columns are documented here and in the module
#: docstring; no geometry column is invented beyond ``geom``.
_SCHEMAS: dict[str, tuple[tuple[str, str], ...]] = {
    "prism_summary": (
        ("point_id", "TEXT"), ("station_id", "TEXT"), ("group_id", "TEXT"),
        ("movement_concern", "TEXT"), ("concern_status", "TEXT"), ("concern_reason", "TEXT"),
        ("reliability_grade", "TEXT"), ("reliability_status", "TEXT"),
        ("reliability_reason", "TEXT"), ("reliability_flags", "TEXT"),
        ("los_raw_mm", "DOUBLE"), ("los_fc_mm", "DOUBLE"),
        ("ver_raw_mm", "DOUBLE"), ("ver_fc_mm", "DOUBLE"),
        ("d_east_raw_mm", "DOUBLE"), ("d_north_raw_mm", "DOUBLE"),
        ("net_mm", "DOUBLE"), ("sigma_mm", "DOUBLE"),
        ("coverage", "DOUBLE"), ("n_prism_cycles", "INTEGER"),
        ("status", "TEXT"), ("reason", "TEXT"),
    ),
    "station_cycles": (
        ("station_id", "TEXT"), ("cycle_id", "INTEGER"),
        ("cycle_start", "TEXT"), ("cycle_end", "TEXT"),
        ("se_m", "DOUBLE"), ("sn_m", "DOUBLE"), ("sh_m", "DOUBLE"),
        ("orientation_deg", "DOUBLE"), ("scale_ppm", "DOUBLE"),
        ("n_obs", "INTEGER"), ("status", "TEXT"), ("reason", "TEXT"),
    ),
    "frame_cycles": (
        ("station_id", "TEXT"), ("segment_id", "TEXT"), ("cycle_id", "INTEGER"),
        ("rotation_arcsec", "DOUBLE"), ("rotation_se", "DOUBLE"),
        ("tE_mm", "DOUBLE"), ("tE_se", "DOUBLE"),
        ("tN_mm", "DOUBLE"), ("tN_se", "DOUBLE"),
        ("scale_ppm", "DOUBLE"), ("scale_se", "DOUBLE"),
        ("vert_offset_mm", "DOUBLE"), ("vert_offset_se", "DOUBLE"),
        ("vert_index_mm_per_km", "DOUBLE"), ("vert_index_se", "DOUBLE"),
        ("tilt_sin", "DOUBLE"), ("tilt_sin_se", "DOUBLE"),
        ("tilt_cos", "DOUBLE"), ("tilt_cos_se", "DOUBLE"),
        ("rank", "INTEGER"), ("condition_number", "DOUBLE"),
        ("rms_hz", "DOUBLE"), ("rms_D", "DOUBLE"), ("rms_v", "DOUBLE"),
        ("n_frame_members", "INTEGER"), ("n_used", "INTEGER"),
        ("status", "TEXT"), ("reason", "TEXT"),
    ),
    "timeseries_24h": (
        ("point_id", "TEXT"), ("segment_id", "TEXT"),
        ("block_start", "TEXT"), ("block_end", "TEXT"),
        ("los_raw_mm", "DOUBLE"), ("los_fc_mm", "DOUBLE"),
        ("ver_raw_mm", "DOUBLE"), ("ver_fc_mm", "DOUBLE"),
        ("tan_raw_mm", "DOUBLE"), ("tan_fc_mm", "DOUBLE"),
        ("d_east_mm", "DOUBLE"), ("d_north_mm", "DOUBLE"), ("d_up_mm", "DOUBLE"),
        ("n_obs", "INTEGER"), ("n_night", "INTEGER"),
        ("status", "TEXT"), ("reason", "TEXT"),
    ),
    "event_register": (
        ("event_id", "TEXT"), ("event_class", "TEXT"), ("prisms", "TEXT"),
        ("start", "TEXT"), ("end", "TEXT"), ("effect", "TEXT"),
        ("evidence", "TEXT"), ("alternatives", "TEXT"),
        ("confidence", "TEXT"), ("follow_up", "TEXT"),
        ("status", "TEXT"), ("reason", "TEXT"),
    ),
    "investigation_register": (
        ("investigation_id", "TEXT"), ("opened", "TEXT"), ("topic", "TEXT"),
        ("candidate_explanation", "TEXT"), ("competing_explanation", "TEXT"),
        ("evidence", "TEXT"), ("status", "TEXT"), ("reason", "TEXT"),
    ),
    "field_checks": (
        ("check_id", "TEXT"), ("question", "TEXT"), ("supports_if", "TEXT"),
        ("alternative_if", "TEXT"), ("status", "TEXT"), ("outcome", "TEXT"),
        ("notes", "TEXT"), ("photo", "TEXT"),
        ("east_m", "DOUBLE"), ("north_m", "DOUBLE"),
    ),
    "sight_lines": (
        ("line_id", "TEXT"), ("station_id", "TEXT"), ("point_id", "TEXT"),
        ("sight_status", "TEXT"),
        ("se_m", "DOUBLE"), ("sn_m", "DOUBLE"),
        ("te_m", "DOUBLE"), ("tn_m", "DOUBLE"),
        ("sd_m", "DOUBLE"), ("reason", "TEXT"),
    ),
    "candidate_zones": (
        ("zone_id", "TEXT"), ("method", "TEXT"), ("rule", "TEXT"),
        ("n_members", "INTEGER"), ("status", "TEXT"), ("reason", "TEXT"),
    ),
    "movement_vectors": (
        ("point_id", "TEXT"), ("segment_id", "TEXT"), ("role", "TEXT"), ("method", "TEXT"),
        ("interval_start", "TEXT"), ("interval_end", "TEXT"),
        ("east_start_m", "DOUBLE"), ("north_start_m", "DOUBLE"),
        ("east_end_m", "DOUBLE"), ("north_end_m", "DOUBLE"),
        ("vector_east_mm", "DOUBLE"), ("vector_north_mm", "DOUBLE"),
        ("uncertainty_sigma_east_mm", "DOUBLE"),
        ("uncertainty_sigma_north_mm", "DOUBLE"),
        ("uncertainty_sigma_en_mm", "DOUBLE"),
        ("net_mm", "DOUBLE"), ("status", "TEXT"), ("reason", "TEXT"),
    ),
}

_MISSING = object()


@dataclass
class GISExport:
    """Result of a GeoPackage export."""

    path: Path
    layers: list[str]
    srs_id: int
    status: str
    reason: str | None = None


@dataclass
class GISData:
    """Points, vectors and uncertainties ready for export.

    ``points``/``vectors``/``uncertainties`` are copies; the caller's frames
    are never modified. When no transform is configured they equal the inputs
    and ``status`` is ``not_configured``. When a transform is applied the
    coordinate/vector/uncertainty columns hold export-ready values while the
    original grid values remain available on the caller's frames.
    """

    points: pd.DataFrame | None
    vectors: pd.DataFrame | None
    uncertainties: pd.DataFrame | None
    status: str = NOT_CONFIGURED
    reason: str | None = None
    transform: Mapping[str, Any] | None = field(default=None)


# ---------------------------------------------------------------------------
#  run / mapping and config access helpers
# ---------------------------------------------------------------------------
def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _dig(obj: Any, path: str, default: Any = None) -> Any:
    current = obj
    for part in path.split("."):
        current = _get(current, part, _MISSING)
        if current is _MISSING or current is None:
            return default
    return current


def _tables(run: Any) -> dict[str, pd.DataFrame]:
    tables = _get(run, "tables", {}) or {}
    if not isinstance(tables, Mapping):
        return {}
    return {str(k): v for k, v in tables.items() if isinstance(v, pd.DataFrame)}


def _lookup(run: Any, names: tuple[str, ...], paths: tuple[str, ...] = ()) -> pd.DataFrame | None:
    tables = _tables(run)
    for name in names:
        frame = tables.get(name)
        if frame is not None:
            return frame
    for path in paths:
        value = _dig(run, path)
        if isinstance(value, pd.DataFrame):
            return value
    return None


def _pick(frame: pd.DataFrame | None, options: tuple[str, ...]) -> str | None:
    if frame is None:
        return None
    for name in options:
        if name in frame.columns:
            return name
    return None


def _project(frame: pd.DataFrame, aliases: Mapping[str, tuple[str, ...]]) -> pd.DataFrame:
    """Copy only the aliased columns under their canonical layer names."""
    out = pd.DataFrame(index=frame.index)
    for target, options in aliases.items():
        column = _pick(frame, options)
        if column is not None:
            out[target] = frame[column]
    return out


def _gis_config(config: Any) -> Any:
    return _get(config, "gis", {}) or {}


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    return _get(_gis_config(config), key, default)


# ---------------------------------------------------------------------------
#  Source resolvers: each returns (frame or None, reason or None)
# ---------------------------------------------------------------------------
_PRISM_SUMMARY_ALIASES: dict[str, tuple[str, ...]] = {
    "point_id": ("point_id",),
    "station_id": ("station_id", "station"),
    "group_id": ("group_id", "group"),
    "movement_concern": ("movement_concern", "concern_class", "concern", "final_class"),
    "concern_status": ("concern_status",),
    "concern_reason": ("concern_reason",),
    "reliability_grade": ("reliability_grade", "grade"),
    "reliability_status": ("reliability_status",),
    "reliability_reason": ("reliability_reason",),
    "reliability_flags": ("reliability_flags", "flags"),
    "los_raw_mm": ("los_raw_mm", "d_rad_mm"),
    "los_fc_mm": ("los_fc_mm",),
    "ver_raw_mm": ("ver_raw_mm", "d_vert_mm"),
    "ver_fc_mm": ("ver_fc_mm",),
    "d_east_raw_mm": ("d_east_raw_mm", "d_east_mm"),
    "d_north_raw_mm": ("d_north_raw_mm", "d_north_mm"),
    "net_mm": ("net_mm", "net_change_mm"),
    "sigma_mm": ("sigma_mm", "noise_mm"),
    "coverage": ("coverage", "coverage_pct"),
    "n_prism_cycles": ("n_prism_cycles",),
    "status": ("status",),
    "reason": ("reason",),
    # geometry pass-through (not part of the fixed layer schema)
    "east_m": ("east_m", "east_out_m", "te_m", "east", "x"),
    "north_m": ("north_m", "north_out_m", "tn_m", "north", "y"),
}

_STATION_CYCLE_ALIASES: dict[str, tuple[str, ...]] = {
    "station_id": ("station_id", "station"),
    "cycle_id": ("cycle_id",),
    "cycle_start": ("cycle_start", "start", "ts", "time"),
    "cycle_end": ("cycle_end", "end"),
    "se_m": ("se_m", "station_east_m", "stE_m", "east_m"),
    "sn_m": ("sn_m", "station_north_m", "stN_m", "north_m"),
    "sh_m": ("sh_m", "station_height_m", "stH_m", "height_m"),
    "orientation_deg": ("orientation_deg", "orientation"),
    "scale_ppm": ("scale_ppm", "scale"),
    "n_obs": ("n_obs", "n"),
    "status": ("status",),
    "reason": ("reason",),
}

_TIMESERIES_ALIASES: dict[str, tuple[str, ...]] = {
    "point_id": ("point_id",),
    "segment_id": ("segment_id",),
    "block_start": ("block_start", "start", "block_start_ts"),
    "block_end": ("block_end", "end"),
    "los_raw_mm": ("los_raw_mm", "d_rad_mm"),
    "los_fc_mm": ("los_fc_mm",),
    "ver_raw_mm": ("ver_raw_mm", "d_vert_mm"),
    "ver_fc_mm": ("ver_fc_mm",),
    "tan_raw_mm": ("tan_raw_mm", "d_tan_mm"),
    "tan_fc_mm": ("tan_fc_mm",),
    "d_east_mm": ("d_east_mm",),
    "d_north_mm": ("d_north_mm",),
    "d_up_mm": ("d_up_mm",),
    "n_obs": ("n_obs", "n"),
    "n_night": ("n_night",),
    "status": ("status",),
    "reason": ("reason",),
}

_EVENT_ALIASES: dict[str, tuple[str, ...]] = {
    "event_id": ("event_id", "id"),
    "event_class": ("event_class", "class"),
    "prisms": ("prisms", "prism_ids"),
    "start": ("start",),
    "end": ("end",),
    "effect": ("effect",),
    "evidence": ("evidence",),
    "alternatives": ("alternatives", "competing_explanation"),
    "confidence": ("confidence",),
    "follow_up": ("follow_up", "followup"),
    "status": ("status",),
    "reason": ("reason",),
}

_INVESTIGATION_ALIASES: dict[str, tuple[str, ...]] = {
    "investigation_id": ("investigation_id", "id"),
    "opened": ("opened",),
    "topic": ("topic", "hypothesis"),
    "candidate_explanation": ("candidate_explanation",),
    "competing_explanation": ("competing_explanation", "alternative_explanation"),
    "evidence": ("evidence",),
    "status": ("status",),
    "reason": ("reason",),
}

_FIELD_CHECK_ALIASES: dict[str, tuple[str, ...]] = {
    "check_id": ("check_id", "id"),
    "question": ("question",),
    "supports_if": ("supports_if",),
    "alternative_if": ("alternative_if",),
    "status": ("status",),
    "outcome": ("outcome",),
    "notes": ("notes",),
    "photo": ("photo",),
    "east_m": ("east_m", "east", "x"),
    "north_m": ("north_m", "north", "y"),
}

_SIGHT_LINE_ALIASES: dict[str, tuple[str, ...]] = {
    "line_id": ("line_id", "id"),
    "station_id": ("station_id", "station"),
    "point_id": ("point_id",),
    "sight_status": ("sight_status", "status"),
    "se_m": ("se_m", "station_east_m", "east_start_m", "start_east_m"),
    "sn_m": ("sn_m", "station_north_m", "north_start_m", "start_north_m"),
    "te_m": ("te_m", "target_east_m", "east_end_m", "end_east_m"),
    "tn_m": ("tn_m", "target_north_m", "north_end_m", "end_north_m"),
    "sd_m": ("sd_m", "distance_m"),
    "reason": ("reason",),
}

_CANDIDATE_ZONE_ALIASES: dict[str, tuple[str, ...]] = {
    "zone_id": ("zone_id", "id"),
    "method": ("method",),
    "rule": ("rule",),
    "n_members": ("n_members",),
    "status": ("status",),
    "reason": ("reason",),
    # geometry pass-through (list of (x, y) rings)
    "ring_coords": ("ring_coords", "geometry", "ring"),
}

_MOVEMENT_VECTOR_ALIASES: dict[str, tuple[str, ...]] = {
    "point_id": ("point_id",),
    "segment_id": ("segment_id",),
    "role": ("role",),
    "method": ("method",),
    "interval_start": ("interval_start", "start", "t_start"),
    "interval_end": ("interval_end", "end", "t_end"),
    "east_start_m": ("east_start_m", "start_east_m", "east_m", "te_m"),
    "north_start_m": ("north_start_m", "start_north_m", "north_m", "tn_m"),
    "east_end_m": ("east_end_m", "end_east_m", "ee_m"),
    "north_end_m": ("north_end_m", "end_north_m", "en_m"),
    "vector_east_mm": ("vector_east_mm", "ve_mm", "d_east_mm"),
    "vector_north_mm": ("vector_north_mm", "vn_mm", "d_north_mm"),
    "uncertainty_sigma_east_mm": (
        "uncertainty_sigma_east_mm", "sigma_east_mm", "sd_east_mm"),
    "uncertainty_sigma_north_mm": (
        "uncertainty_sigma_north_mm", "sigma_north_mm", "sd_north_mm"),
    "uncertainty_sigma_en_mm": ("uncertainty_sigma_en_mm", "sigma_en_mm", "cov_en_mm"),
    "net_mm": ("net_mm", "net_change_mm"),
    "status": ("status",),
    "reason": ("reason",),
}


def _prism_summary_source(run: Any) -> tuple[pd.DataFrame | None, str | None]:
    frame = _lookup(run, ("prism_summary", "summary", "ringkasan"), ())
    if frame is not None:
        return frame, None
    concern = _lookup(run, ("concern",), ("classification.concern",))
    reliability = _lookup(run, ("reliability",), ("classification.reliability",))
    if concern is None and reliability is None:
        return None, "no prism_summary/summary source table or classification registered"
    parts = [part.copy() for part in (concern, reliability) if part is not None]
    if len(parts) == 2:
        keys = [k for k in ("point_id", "segment_id")
                if k in parts[0].columns and k in parts[1].columns] or ["point_id"]
        merged = parts[0].merge(parts[1], on=keys, how="outer", suffixes=("", "_rel"))
        return merged, "assembled from concern and reliability (no summary table)"
    return parts[0], "assembled from a single classification table (no summary table)"


def _station_cycles_source(run: Any) -> tuple[pd.DataFrame | None, str | None]:
    frame = _lookup(run, ("station_cycles", "cycle_table"), ("cycles.cycles",))
    if frame is None:
        return None, "no station_cycles/cycle_table source registered"
    return frame, None


def _frame_cycles_source(run: Any) -> tuple[pd.DataFrame | None, str | None]:
    frame = _lookup(run, ("frame_cycles", "frame_coefficients"), ("frames.coefficients",))
    if frame is None:
        return None, "no frame coefficient table registered"
    return frame, None


def _timeseries_source(run: Any) -> tuple[pd.DataFrame | None, str | None]:
    frame = _lookup(run, ("timeseries_24h", "timeseries", "block_series"), ())
    if frame is None:
        return None, "no timeseries_24h/block source registered"
    return frame, None


def _event_register_source(run: Any) -> tuple[pd.DataFrame | None, str | None]:
    frame = _lookup(run, ("event_register", "events"), ("cycles.candidate_events",))
    if frame is None:
        return None, "no event_register source registered"
    return frame, None


def _investigation_source(run: Any) -> tuple[pd.DataFrame | None, str | None]:
    frame = _lookup(
        run, ("investigation_register", "investigations"), ("forensics.investigations",))
    if frame is None:
        return None, "no investigation_register source registered"
    return frame, None


def _field_checks_source(run: Any) -> tuple[pd.DataFrame | None, str | None]:
    frame = _lookup(run, ("field_checks",), ())
    if frame is None:
        return None, "no field_checks source registered"
    return frame, None


def _sight_lines_source(run: Any) -> tuple[pd.DataFrame | None, str | None]:
    frame = _lookup(run, ("sight_lines", "los_lines"), ())
    if frame is None:
        return None, "no sight_lines source registered"
    return frame, None


def _candidate_zones_source(run: Any, config: Any) -> tuple[pd.DataFrame | None, str | None]:
    frame = _lookup(run, ("candidate_zones", "zones"), ())
    rule = _config_value(config, "candidate_zone_rule", None)
    if frame is not None:
        if frame.empty and is_not_configured(rule):
            return frame, "candidate_zone_rule not_configured; layer is empty"
        return frame, None
    if is_not_configured(rule):
        return None, "candidate_zone_rule not_configured; layer is empty"
    return None, "candidate_zone_rule configured but no candidate_zones table was produced"


def _movement_vectors_source(run: Any) -> tuple[pd.DataFrame | None, str | None]:
    frame = _lookup(run, ("movement_vectors", "vectors"), ())
    if frame is None:
        return None, "no movement_vectors source registered"
    return frame, None


# ---------------------------------------------------------------------------
#  SQLite / GeoPackage primitives
# ---------------------------------------------------------------------------
def _sql_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat(sep=" ")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, dict, set)):
        return str(value)
    return value


def _envelope(coords: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    return min(xs), max(xs), min(ys), max(ys)


def _encode_geometry(geom_type: str, coords: Any, srs_id: int) -> tuple[bytes, tuple]:
    """GeoPackage header + little-endian WKB for POINT/LINESTRING/POLYGON."""
    if geom_type == "POINT":
        x, y = float(coords[0]), float(coords[1])
        envelope = (x, x, y, y)
        wkb = struct.pack("<BI", 1, 1) + struct.pack("<dd", x, y)
    elif geom_type == "LINESTRING":
        ring = [(float(x), float(y)) for x, y in coords]
        envelope = _envelope(ring)
        wkb = struct.pack("<BII", 1, 2, len(ring))
        wkb += b"".join(struct.pack("<dd", x, y) for x, y in ring)
    elif geom_type == "POLYGON":
        ring = [(float(x), float(y)) for x, y in coords]
        envelope = _envelope(ring)
        wkb = struct.pack("<BII", 1, 3, 1)
        wkb += struct.pack("<I", len(ring))
        wkb += b"".join(struct.pack("<dd", x, y) for x, y in ring)
    else:  # pragma: no cover - guarded by GIS_LAYERS
        raise ValueError(f"unsupported geometry type: {geom_type}")
    header = struct.pack(
        "<2sBBi4d", GPKG_HEADER_MAGIC, GPKG_HEADER_VERSION, GPKG_HEADER_FLAGS,
        int(srs_id), envelope[0], envelope[1], envelope[2], envelope[3])
    return header + wkb, envelope


def _create_core_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE gpkg_spatial_ref_sys (
            srs_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL PRIMARY KEY,
            organization TEXT NOT NULL,
            organization_coordsys_id INTEGER NOT NULL,
            definition TEXT NOT NULL,
            description TEXT
        );
        CREATE TABLE gpkg_contents (
            table_name TEXT NOT NULL PRIMARY KEY,
            data_type TEXT NOT NULL,
            identifier TEXT UNIQUE,
            description TEXT DEFAULT '',
            last_change DATETIME NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
            srs_id INTEGER,
            CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id)
                REFERENCES gpkg_spatial_ref_sys(srs_id)
        );
        CREATE TABLE gpkg_geometry_columns (
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL,
            m TINYINT NOT NULL,
            CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
            CONSTRAINT fk_gc_tn FOREIGN KEY (table_name)
                REFERENCES gpkg_contents(table_name),
            CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id)
                REFERENCES gpkg_spatial_ref_sys(srs_id)
        );
        """
    )


_WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
    '298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4326"]]'
)


def _insert_srs(connection: sqlite3.Connection, config: Any, srs_id: int) -> None:
    # The -1, 0 and 4326 rows are the mandatory GeoPackage metadata rows; the
    # 4326 definition is never applied to a layer. No CRS is inferred for the
    # data: layers use `srs_id` (default -1, undefined Cartesian).
    rows = [
        ("Undefined Cartesian SRS", -1, "NONE", -1, "undefined",
         "undefined Cartesian coordinate reference system"),
        ("Undefined Geographic SRS", 0, "NONE", 0, "undefined",
         "undefined geographic coordinate reference system"),
        ("WGS 84 geodetic", 4326, "EPSG", 4326, _WGS84_WKT,
         "mandatory GeoPackage metadata row; not applied to any layer"),
    ]
    if srs_id not in (-1, 0):
        definition = _config_value(config, "srs_wkt", None) or "undefined"
        rows.append((f"User-configured SRS {srs_id}", int(srs_id), "NONE", int(srs_id),
                     str(definition), "configured by the user; no EPSG code inferred"))
    connection.executemany(
        "INSERT INTO gpkg_spatial_ref_sys "
        "(srs_name, srs_id, organization, organization_coordsys_id, definition, description) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def _create_layer(connection: sqlite3.Connection, table: str, schema: Any,
                  geom_type: str | None) -> None:
    columns = ", ".join(f'"{name}" {kind}' for name, kind in schema)
    if geom_type:
        connection.execute(
            f'CREATE TABLE "{table}" ("fid" INTEGER PRIMARY KEY AUTOINCREMENT, '
            f'"geom" BLOB, {columns})')
    else:
        connection.execute(
            f'CREATE TABLE "{table}" ("fid" INTEGER PRIMARY KEY AUTOINCREMENT, {columns})')


def _write_layer(connection: sqlite3.Connection, table: str, frame: pd.DataFrame | None,
                 schema: Any, geom_type: str | None, srs_id: int,
                 geometry_for: Any) -> tuple | None:
    names = [name for name, _ in schema]
    columns_sql = ", ".join(f'"{name}"' for name in names)
    placeholders = ", ".join("?" for _ in names)
    if geom_type:
        insert_sql = (
            f'INSERT INTO "{table}" ("geom", {columns_sql}) '
            f"VALUES (?, {placeholders})")
    else:
        insert_sql = f'INSERT INTO "{table}" ({columns_sql}) VALUES ({placeholders})'
    rows: list[list[Any]] = []
    envelope: list[float] | None = None
    if frame is not None and not frame.empty:
        for _, record in frame.iterrows():
            blob = None
            if geom_type and geometry_for is not None:
                geometry = geometry_for(record)
                if geometry is not None:
                    blob, bounds = _encode_geometry(geom_type, geometry, srs_id)
                    if envelope is None:
                        envelope = list(bounds)
                    else:
                        envelope = [
                            min(envelope[0], bounds[0]), max(envelope[1], bounds[1]),
                            min(envelope[2], bounds[2]), max(envelope[3], bounds[3]),
                        ]
            values = [blob] if geom_type else []
            values += [_sql_value(record.get(name)) if name in record.index else None
                       for name in names]
            rows.append(values)
    if rows:
        connection.executemany(insert_sql, rows)
    return tuple(envelope) if envelope is not None else None


# ---------------------------------------------------------------------------
#  Layer frame assembly
# ---------------------------------------------------------------------------
def _project_or_none(frame: pd.DataFrame | None,
                     aliases: Mapping[str, tuple[str, ...]]) -> pd.DataFrame | None:
    if frame is None:
        return None
    return _project(frame, aliases)


def _assemble_layers(run: Any, config: Any) -> dict[str, tuple[pd.DataFrame | None, str | None]]:
    raw: dict[str, tuple[pd.DataFrame | None, str | None]] = {
        "prism_summary": _prism_summary_source(run),
        "station_cycles": _station_cycles_source(run),
        "frame_cycles": _frame_cycles_source(run),
        "timeseries_24h": _timeseries_source(run),
        "event_register": _event_register_source(run),
        "investigation_register": _investigation_source(run),
        "field_checks": _field_checks_source(run),
        "sight_lines": _sight_lines_source(run),
        "candidate_zones": _candidate_zones_source(run, config),
        "movement_vectors": _movement_vectors_source(run),
    }
    assembled: dict[str, tuple[pd.DataFrame | None, str | None]] = {}
    for name, (frame, reason) in raw.items():
        if name == "prism_summary":
            assembled[name] = (_project_or_none(frame, _PRISM_SUMMARY_ALIASES), reason)
        elif name == "station_cycles":
            assembled[name] = (_project_or_none(frame, _STATION_CYCLE_ALIASES), reason)
        elif name == "frame_cycles":
            if frame is not None:
                aliases = {column: (column,) for column, _ in _SCHEMAS[name]}
                assembled[name] = (_project(frame, aliases), reason)
            else:
                assembled[name] = (None, reason)
        elif name == "timeseries_24h":
            assembled[name] = (_project_or_none(frame, _TIMESERIES_ALIASES), reason)
        elif name == "event_register":
            assembled[name] = (_project_or_none(frame, _EVENT_ALIASES), reason)
        elif name == "investigation_register":
            assembled[name] = (_project_or_none(frame, _INVESTIGATION_ALIASES), reason)
        elif name == "field_checks":
            assembled[name] = (_project_or_none(frame, _FIELD_CHECK_ALIASES), reason)
        elif name == "sight_lines":
            assembled[name] = (_project_or_none(frame, _SIGHT_LINE_ALIASES), reason)
        elif name == "candidate_zones":
            assembled[name] = (_project_or_none(frame, _CANDIDATE_ZONE_ALIASES), reason)
        elif name == "movement_vectors":
            assembled[name] = (_project_or_none(frame, _MOVEMENT_VECTOR_ALIASES), reason)
    return assembled


def _to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
#  Geometry functions per layer
# ---------------------------------------------------------------------------
def _point_geometry(record: pd.Series, east_col: str | None,
                    north_col: str | None) -> tuple[float, float] | None:
    if east_col is None or north_col is None:
        return None
    east = _to_float(record.get(east_col))
    north = _to_float(record.get(north_col))
    if east is None or north is None:
        return None
    return east, north


def _line_geometry(record: pd.Series, start: tuple[str | None, str | None],
                   end: tuple[str | None, str | None]) -> list[tuple[float, float]] | None:
    if None in start or None in end:
        return None
    start_east = _to_float(record.get(start[0]))
    start_north = _to_float(record.get(start[1]))
    end_east = _to_float(record.get(end[0]))
    end_north = _to_float(record.get(end[1]))
    if None in (start_east, start_north, end_east, end_north):
        return None
    return [(start_east, start_north), (end_east, end_north)]


def _geometry_function(name: str, frame: pd.DataFrame | None) -> Any:
    if frame is None or name not in {geom for geom, kind in GIS_LAYERS if kind}:
        return None
    if name in ("prism_summary", "field_checks"):
        east = _pick(frame, ("east_m", "east", "x"))
        north = _pick(frame, ("north_m", "north", "y"))
        return lambda record: _point_geometry(record, east, north)
    if name == "station_cycles":
        east = _pick(frame, ("se_m", "station_east_m", "stE_m", "east_m"))
        north = _pick(frame, ("sn_m", "station_north_m", "stN_m", "north_m"))
        return lambda record: _point_geometry(record, east, north)
    if name == "sight_lines":
        start = (_pick(frame, ("se_m", "station_east_m", "east_start_m", "start_east_m")),
                 _pick(frame, ("sn_m", "station_north_m", "north_start_m", "start_north_m")))
        end = (_pick(frame, ("te_m", "target_east_m", "east_end_m", "end_east_m")),
               _pick(frame, ("tn_m", "target_north_m", "north_end_m", "end_north_m")))
        return lambda record: _line_geometry(record, start, end)
    if name == "candidate_zones":
        ring_col = _pick(frame, ("ring_coords", "geometry", "ring"))
        if ring_col is None:
            return None
        return lambda record: record.get(ring_col)
    if name == "movement_vectors":
        start = (_pick(frame, ("east_start_m", "start_east_m", "east_m", "te_m")),
                 _pick(frame, ("north_start_m", "start_north_m", "north_m", "tn_m")))
        end = (_pick(frame, ("east_end_m", "end_east_m", "ee_m")),
               _pick(frame, ("north_end_m", "end_north_m", "en_m")))
        vector = (_pick(frame, ("vector_east_mm", "ve_mm", "d_east_mm")),
                  _pick(frame, ("vector_north_mm", "vn_mm", "d_north_mm")))

        def get(record: pd.Series) -> list[tuple[float, float]] | None:
            line = _line_geometry(record, start, end)
            if line is not None:
                return line
            if None in start or None in vector:
                return None
            base_east = _to_float(record.get(start[0]))
            base_north = _to_float(record.get(start[1]))
            vec_east = _to_float(record.get(vector[0]))
            vec_north = _to_float(record.get(vector[1]))
            if None in (base_east, base_north, vec_east, vec_north):
                return None
            return [(base_east, base_north),
                    (base_east + vec_east / 1000.0, base_north + vec_north / 1000.0)]

        return get
    return None


# ---------------------------------------------------------------------------
#  Public GeoPackage writer
# ---------------------------------------------------------------------------
def _srs_id(config: Any) -> int:
    value = _config_value(config, "srs_id", -1)
    if is_not_configured(value) or value == "":
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def write_geopackage(run: Any, path: str | Path, *, config: Any) -> GISExport:
    """Write the ten documented layers; missing sources become empty layers."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    srs_id = _srs_id(config)
    assembled = _assemble_layers(run, config)
    last_change = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    layers: list[str] = []
    empty_layers: list[str] = []
    connection = sqlite3.connect(str(target))
    try:
        connection.execute(f"PRAGMA application_id = {GPKG_APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version = {GPKG_USER_VERSION}")
        _create_core_tables(connection)
        _insert_srs(connection, config, srs_id)
        for name, geom_type in GIS_LAYERS:
            frame, reason = assembled.get(name, (None, "no source resolver"))
            if frame is not None and frame.empty and reason is None:
                reason = "source table is empty"
            if reason is not None:
                empty_layers.append(name)
            _create_layer(connection, name, _SCHEMAS[name], geom_type)
            geometry_for = _geometry_function(name, frame)
            envelope = _write_layer(connection, name, frame, _SCHEMAS[name],
                                    geom_type, srs_id, geometry_for)
            description = "" if reason is None else f"empty layer: {reason}"
            if geom_type:
                connection.execute(
                    "INSERT INTO gpkg_contents (table_name, data_type, identifier, "
                    "description, last_change, min_x, min_y, max_x, max_y, srs_id) "
                    "VALUES (?, 'features', ?, ?, ?, ?, ?, ?, ?, ?)",
                    (name, name, description, last_change,
                     envelope[0] if envelope else None,
                     envelope[2] if envelope else None,
                     envelope[1] if envelope else None,
                     envelope[3] if envelope else None,
                     srs_id))
                connection.execute(
                    "INSERT INTO gpkg_geometry_columns (table_name, column_name, "
                    "geometry_type_name, srs_id, z, m) VALUES (?, 'geom', ?, ?, 0, 0)",
                    (name, geom_type, srs_id))
            else:
                connection.execute(
                    "INSERT INTO gpkg_contents (table_name, data_type, identifier, "
                    "description, last_change, min_x, min_y, max_x, max_y, srs_id) "
                    "VALUES (?, 'attributes', ?, ?, ?, NULL, NULL, NULL, NULL, NULL)",
                    (name, name, description, last_change))
            layers.append(name)
        connection.commit()
    finally:
        connection.close()
    reason = None
    if empty_layers:
        reason = "empty layers: " + ", ".join(empty_layers)
    return GISExport(path=target, layers=layers, srs_id=srs_id,
                     status=STATUS_OK, reason=reason)


# ---------------------------------------------------------------------------
#  Export-only transform
# ---------------------------------------------------------------------------
_EAST_COLS = ("east_out_m", "east_m", "te_m", "east", "x")
_NORTH_COLS = ("north_out_m", "north_m", "tn_m", "north", "y")
_VECTOR_EAST_COLS = ("vector_east_out_mm", "vector_east_mm", "ve_mm", "d_east_mm",
                     "east_mm", "de_mm")
_VECTOR_NORTH_COLS = ("vector_north_out_mm", "vector_north_mm", "vn_mm", "d_north_mm",
                      "north_mm", "dn_mm")
_SIGMA_EAST_COLS = ("sigma_east_out_mm", "sigma_east_mm", "sd_east_mm")
_SIGMA_NORTH_COLS = ("sigma_north_out_mm", "sigma_north_mm", "sd_north_mm")
_SIGMA_EN_COLS = ("sigma_en_out_mm", "sigma_en_mm", "cov_en_mm")


def _copy_frame(frame: Any) -> pd.DataFrame | None:
    return frame.copy() if isinstance(frame, pd.DataFrame) else None


def _transform_configured(transform: Any) -> bool:
    if transform is None:
        return False
    if isinstance(transform, str):
        return transform.strip().lower() not in ("", NOT_CONFIGURED)
    if isinstance(transform, Mapping):
        return bool(transform)
    return True


def _transform_terms(transform: Mapping[str, Any]) -> tuple[np.ndarray, float, float, float]:
    """Return (rotation matrix [[c, -s], [s, c]], scale, translation east, north)."""
    rotation_deg = float(
        transform.get("rotation_deg", transform.get("rotation",
                      transform.get("theta_deg", 0.0))))
    scale = float(transform.get("scale", transform.get("scale_factor", 1.0)))
    translation = transform.get("translation", transform.get("translation_m"))
    if translation is not None:
        translation_east = float(translation[0])
        translation_north = float(translation[1])
    else:
        translation_east = float(
            transform.get("translation_east_m", transform.get("tE_m", 0.0)))
        translation_north = float(
            transform.get("translation_north_m", transform.get("tN_m", 0.0)))
    angle = math.radians(rotation_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    return rotation, scale, translation_east, translation_north


def _coordinate_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    return _pick(frame, _EAST_COLS), _pick(frame, _NORTH_COLS)


def _vector_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    return _pick(frame, _VECTOR_EAST_COLS), _pick(frame, _VECTOR_NORTH_COLS)


def _uncertainty_columns(frame: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    return (_pick(frame, _SIGMA_EAST_COLS), _pick(frame, _SIGMA_NORTH_COLS),
            _pick(frame, _SIGMA_EN_COLS))


def transform_grid(points: Any, vectors: Any, uncertainties: Any, *,
                   transform: Any) -> GISData:
    """Apply a user-configured affine transform on export, preserving the input.

    With no configured transform the input frames are returned unchanged (as
    copies) with status ``not_configured``. A configured transform
    (``rotation_deg``/``rotation``, ``scale``, ``translation[_east_m|_north_m]``)
    rotates and scales point coordinates and vector components, and transforms
    covariance/uncertainty geometry with the same rotation and scale. The
    original grid data is never overwritten in the caller's frames.
    """
    points_out = _copy_frame(points)
    vectors_out = _copy_frame(vectors)
    uncertainties_out = _copy_frame(uncertainties)

    if not _transform_configured(transform):
        return GISData(
            points=points_out, vectors=vectors_out, uncertainties=uncertainties_out,
            status=NOT_CONFIGURED,
            reason="site-grid transform not configured; original grid data preserved",
            transform=None,
        )

    if not isinstance(transform, Mapping):
        raise ValueError("transform must be a mapping with rotation/translation/scale")

    rotation, scale, translation_east, translation_north = _transform_terms(transform)

    if points_out is not None:
        east_col, north_col = _coordinate_columns(points_out)
        if east_col is not None and north_col is not None:
            east = points_out[east_col].astype(float).to_numpy()
            north = points_out[north_col].astype(float).to_numpy()
            rotated = scale * (east * rotation[0, 0] + north * rotation[0, 1])
            along = scale * (east * rotation[1, 0] + north * rotation[1, 1])
            points_out[east_col] = rotated + translation_east
            points_out[north_col] = along + translation_north

    if vectors_out is not None:
        east_col, north_col = _vector_columns(vectors_out)
        if east_col is not None and north_col is not None:
            east = vectors_out[east_col].astype(float).to_numpy()
            north = vectors_out[north_col].astype(float).to_numpy()
            vectors_out[east_col] = scale * (east * rotation[0, 0] + north * rotation[0, 1])
            vectors_out[north_col] = scale * (east * rotation[1, 0] + north * rotation[1, 1])

    if uncertainties_out is not None:
        east_col, north_col, covariance_col = _uncertainty_columns(uncertainties_out)
        if east_col is not None and north_col is not None:
            var_east = (uncertainties_out[east_col].astype(float).to_numpy()) ** 2
            var_north = (uncertainties_out[north_col].astype(float).to_numpy()) ** 2
            covariance = (
                uncertainties_out[covariance_col].astype(float).to_numpy()
                if covariance_col is not None else np.zeros_like(var_east))
            c00, c01 = rotation[0, 0], rotation[0, 1]
            c10, c11 = rotation[1, 0], rotation[1, 1]
            new_var_east = scale ** 2 * (
                c00 ** 2 * var_east + 2 * c00 * c01 * covariance + c01 ** 2 * var_north)
            new_var_north = scale ** 2 * (
                c10 ** 2 * var_east + 2 * c10 * c11 * covariance + c11 ** 2 * var_north)
            new_covariance = scale ** 2 * (
                c00 * c10 * var_east
                + (c00 * c11 + c01 * c10) * covariance
                + c01 * c11 * var_north)
            uncertainties_out[east_col] = np.sqrt(np.clip(new_var_east, 0.0, None))
            uncertainties_out[north_col] = np.sqrt(np.clip(new_var_north, 0.0, None))
            if covariance_col is not None:
                uncertainties_out[covariance_col] = new_covariance

    return GISData(
        points=points_out, vectors=vectors_out, uncertainties=uncertainties_out,
        status=STATUS_OK,
        reason="affine transform applied on export; original grid data preserved",
        transform=dict(transform),
    )


# ---------------------------------------------------------------------------
#  QML style
# ---------------------------------------------------------------------------
_QML_TEMPLATE = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0-Prizren" styleCategories="Symbology">
  <!-- Arrow geometry-generator style for the movement_vectors layer.
       The displayed length is scaled by the project variable @hlo_vec_scale.
       Documented default from config.gis.vector_scale: __VECTOR_SCALE__. -->
  <renderer-v2 type="singleSymbol" forceraster="0" symbollevels="0"
               enableorderby="0" referencescale="-1">
    <symbols>
      <symbol type="line" name="0" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="GeometryGenerator" enabled="1" pass="0" locked="0" id="arrow">
          <prop k="SymbolType" v="Line"/>
          <prop k="geometryModifier" v="make_line(start_point($geometry), start_point($geometry) + coalesce(@hlo_vec_scale, __VECTOR_SCALE__) * (end_point($geometry) - start_point($geometry)))"/>
          <Option type="Map">
            <Option name="SymbolType" type="QString" value="Line"/>
            <Option name="geometryModifier" type="QString" value="make_line(start_point($geometry), start_point($geometry) + coalesce(@hlo_vec_scale, __VECTOR_SCALE__) * (end_point($geometry) - start_point($geometry)))"/>
          </Option>
          <symbol type="line" name="arrow_line" alpha="1" clip_to_extent="1" force_rhr="0">
            <layer class="SimpleLine" enabled="1" pass="0" locked="0" id="shaft">
              <prop k="line_color" v="35,35,35,255"/>
              <prop k="line_width" v="0.4"/>
            </layer>
            <layer class="MarkerLine" enabled="1" pass="0" locked="0" id="head">
              <prop k="placement" v="lastvertex"/>
              <prop k="symbol" v="arrow_head"/>
            </layer>
          </symbol>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <layerGeometryType>1</layerGeometryType>
</qgis>
"""


def write_qml(path: str | Path, *, config: Any, layer: str = "movement_vectors") -> Path:
    """Write a minimal QML arrow style using ``@hlo_vec_scale`` as default.

    ``config.gis.vector_scale`` is written into the file as the documented
    default for the project variable; change the QGIS project variable to
    override it without editing the layer style.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scale = _config_value(config, "vector_scale", 2500.0)
    try:
        scale_text = f"{float(scale):g}"
    except (TypeError, ValueError):
        scale_text = "2500"
    text = _QML_TEMPLATE.replace("__VECTOR_SCALE__", scale_text)
    text = text.replace("movement_vectors layer", f"{layer} layer")
    target.write_text(text, encoding="utf-8")
    return target
