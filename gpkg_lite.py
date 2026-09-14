"""
gpkg_lite.py — penulis GeoPackage tanpa GDAL / geopandas / fiona
================================================================
GeoPackage pada dasarnya hanyalah basis data SQLite dengan tabel metadata
tertentu. Modul ini menulisnya langsung memakai `sqlite3` dan `struct` dari
pustaka standar Python, sehingga aplikasi tidak perlu memasang GDAL
(±200 MB) hanya untuk mengekspor peta.

Mendukung: Point, LineString, Polygon (cincin luar saja).
SRS      : srs_id -1 = "Undefined cartesian" -> tepat untuk grid tambang lokal.
           Isi `srs_wkt` + `srs_id` bila definisi CRS resmi sudah tersedia.

Diuji terhadap QGIS 3.x dan pyogrio/GDAL.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

GeomType = Literal["POINT", "LINESTRING", "POLYGON"]

APPLICATION_ID = 0x47504B47  # 'GPKG'
USER_VERSION = 10300         # GeoPackage 1.3


# --------------------------------------------------------------------------
#  WKB
# --------------------------------------------------------------------------
def _wkb_point(xy: Sequence[float]) -> bytes:
    return struct.pack("<BIdd", 1, 1, float(xy[0]), float(xy[1]))


def _wkb_linestring(coords: Sequence[Sequence[float]]) -> bytes:
    out = [struct.pack("<BII", 1, 2, len(coords))]
    out += [struct.pack("<dd", float(x), float(y)) for x, y in coords]
    return b"".join(out)


def _wkb_polygon(ring: Sequence[Sequence[float]]) -> bytes:
    r = list(ring)
    if r[0] != r[-1]:
        r.append(r[0])
    out = [struct.pack("<BII", 1, 3, 1), struct.pack("<I", len(r))]
    out += [struct.pack("<dd", float(x), float(y)) for x, y in r]
    return b"".join(out)


def _gpkg_blob(wkb: bytes, srs_id: int, env: tuple[float, float, float, float] | None) -> bytes:
    """Header GeoPackage binary + WKB.

    flags bit0 = 1 (header little-endian); bit1..3 = indikator envelope
    (0 = tanpa envelope, 1 = envelope XY).
    """
    if env is None:
        flags = 0b00000001
        head = struct.pack("<ccBBi", b"G", b"P", 0, flags, srs_id)
        return head + wkb
    flags = 0b00000011
    head = struct.pack("<ccBBi", b"G", b"P", 0, flags, srs_id)
    head += struct.pack("<dddd", env[0], env[1], env[2], env[3])
    return head + wkb


def _envelope(geom_type: GeomType, coords: Any) -> tuple[float, float, float, float]:
    if geom_type == "POINT":
        x, y = float(coords[0]), float(coords[1])
        return (x, x, y, y)
    pts = list(coords)
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return (min(xs), max(xs), min(ys), max(ys))


def encode_geometry(geom_type: GeomType, coords: Any, srs_id: int = -1,
                    with_envelope: bool = True) -> bytes:
    """Ubah koordinat Python menjadi blob geometri GeoPackage."""
    if geom_type == "POINT":
        wkb = _wkb_point(coords)
    elif geom_type == "LINESTRING":
        wkb = _wkb_linestring(coords)
    elif geom_type == "POLYGON":
        wkb = _wkb_polygon(coords)
    else:
        raise ValueError(f"tipe geometri tidak didukung: {geom_type}")
    env = _envelope(geom_type, coords) if with_envelope else None
    return _gpkg_blob(wkb, srs_id, env)


# --------------------------------------------------------------------------
#  Pemetaan tipe kolom
# --------------------------------------------------------------------------
def _sql_type(value: Any) -> str:
    import datetime as _dt
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "DOUBLE"
    if isinstance(value, (_dt.date, _dt.datetime)):
        return "TEXT"
    return "TEXT"


def _infer_schema(rows: Sequence[dict], skip: set[str]) -> dict[str, str]:
    """Tentukan tipe SQL tiap kolom dari nilai non-null pertama.

    Penting: kolom numerik HARUS menjadi DOUBLE/INTEGER, bukan TEXT, agar
    simbologi bergradasi dan pengurutan numerik di QGIS berfungsi.
    """
    schema: dict[str, str] = {}
    for row in rows:
        for k, v in row.items():
            if k in skip or v is None:
                continue
            if k not in schema:
                schema[k] = _sql_type(v)
            elif schema[k] == "INTEGER" and isinstance(v, float):
                schema[k] = "DOUBLE"
    for row in rows:
        for k in row:
            if k not in skip:
                schema.setdefault(k, "TEXT")
    return schema


# --------------------------------------------------------------------------
#  Penulis
# --------------------------------------------------------------------------
class GpkgWriter:
    """Penulis GeoPackage minimal.

    Contoh
    ------
    >>> with GpkgWriter("hasil.gpkg") as w:
    ...     w.add_layer("prisma", "POINT",
    ...                 [{"point_id": "CND_210_3", "v": 1.146,
    ...                   "geom": (175571.98, 9045968.46)}])
    """

    def __init__(self, path: str | Path, srs_id: int = -1,
                 srs_wkt: str | None = None, srs_name: str = "Grid tambang lokal"):
        self.path = Path(path)
        self.srs_id = srs_id
        self.srs_wkt = srs_wkt
        self.srs_name = srs_name
        if self.path.exists():
            self.path.unlink()
        self.con = sqlite3.connect(self.path)
        self._init_gpkg()

    # -- protokol context manager -----------------------------------------
    def __enter__(self) -> "GpkgWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.con.commit()
        self.con.close()

    # -- kerangka --------------------------------------------------------
    def _init_gpkg(self) -> None:
        c = self.con
        c.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        c.execute(f"PRAGMA user_version = {USER_VERSION}")
        c.executescript(
            """
            CREATE TABLE gpkg_spatial_ref_sys (
              srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,
              organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL,
              definition TEXT NOT NULL, description TEXT);

            CREATE TABLE gpkg_contents (
              table_name TEXT PRIMARY KEY, data_type TEXT NOT NULL,
              identifier TEXT UNIQUE, description TEXT DEFAULT '',
              last_change DATETIME NOT NULL DEFAULT
                (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
              min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
              srs_id INTEGER,
              CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id)
                REFERENCES gpkg_spatial_ref_sys(srs_id));

            CREATE TABLE gpkg_geometry_columns (
              table_name TEXT NOT NULL, column_name TEXT NOT NULL,
              geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL,
              z TINYINT NOT NULL, m TINYINT NOT NULL,
              CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
              CONSTRAINT fk_gc_tn FOREIGN KEY (table_name)
                REFERENCES gpkg_contents(table_name));
            """
        )
        srs = [
            ("Undefined cartesian SRS", -1, "NONE", -1, "undefined",
             "sistem koordinat kartesian tak terdefinisi"),
            ("Undefined geographic SRS", 0, "NONE", 0, "undefined",
             "sistem koordinat geografis tak terdefinisi"),
            ("WGS 84 geodetic", 4326, "EPSG", 4326,
             'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
             'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]', "WGS 84"),
        ]
        c.executemany("INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)", srs)
        if self.srs_wkt and self.srs_id not in (-1, 0, 4326):
            c.execute(
                "INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)",
                (self.srs_name, self.srs_id, "CUSTOM", self.srs_id, self.srs_wkt,
                 "grid tambang"),
            )
        c.commit()

    # -- layer -----------------------------------------------------------
    def add_layer(self, name: str, geom_type: GeomType, rows: Iterable[dict],
                  geom_key: str = "geom", description: str = "") -> int:
        """Tambahkan satu layer. `rows` = iterable dict; satu kunci berisi geometri.

        Geometri: POINT -> (x, y); LINESTRING/POLYGON -> [(x, y), ...]
        Nilai None ditulis sebagai NULL.
        """
        rows = [r for r in rows if r.get(geom_key) is not None]
        if not rows:
            return 0

        schema = _infer_schema(rows, skip={geom_key})
        cols = list(schema)
        ddl = ", ".join(f'"{k}" {schema[k]}' for k in cols)
        self.con.execute(
            f'CREATE TABLE "{name}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, '
            f'{ddl}, "{geom_key}" BLOB)'
        )

        placeholders = ",".join("?" * (len(cols) + 1))
        quoted = ", ".join(f'"{k}"' for k in cols)
        sql = f'INSERT INTO "{name}" ({quoted}, "{geom_key}") VALUES ({placeholders})'

        minx = miny = float("inf")
        maxx = maxy = float("-inf")
        batch = []
        for r in rows:
            g = r[geom_key]
            blob = encode_geometry(geom_type, g, self.srs_id)
            e = _envelope(geom_type, g)
            minx, maxx = min(minx, e[0]), max(maxx, e[1])
            miny, maxy = min(miny, e[2]), max(maxy, e[3])
            vals = []
            for k in cols:
                v = r.get(k)
                if v is not None and schema[k] == "TEXT" and not isinstance(v, str):
                    v = str(v)
                if isinstance(v, float) and v != v:      # NaN -> NULL
                    v = None
                vals.append(v)
            batch.append((*vals, blob))
        self.con.executemany(sql, batch)

        self.con.execute(
            "INSERT INTO gpkg_contents "
            "(table_name, data_type, identifier, description, min_x, min_y, max_x, max_y, srs_id) "
            "VALUES (?,'features',?,?,?,?,?,?,?)",
            (name, name, description, minx, miny, maxx, maxy, self.srs_id),
        )
        self.con.execute(
            "INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,0,0)",
            (name, geom_key, geom_type, self.srs_id),
        )
        self.con.commit()
        return len(batch)


def df_to_rows(df, geom_type: GeomType, geom_builder, kolom: Sequence[str] | None = None
               ) -> list[dict]:
    """Bantu ubah DataFrame pandas menjadi list dict siap `add_layer`.

    `geom_builder(baris) -> koordinat`. Nilai NaN diubah menjadi None.
    """
    import math
    cols = list(kolom) if kolom else [c for c in df.columns]
    out = []
    for _, r in df.iterrows():
        d = {}
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                d[c] = None
            elif hasattr(v, "item"):
                d[c] = v.item()
            else:
                d[c] = v
        d["geom"] = geom_builder(r)
        out.append(d)
    return out
