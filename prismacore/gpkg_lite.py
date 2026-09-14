"""
gpkg_lite.py — penulis GeoPackage tanpa GDAL / geopandas / fiona
================================================================
GeoPackage pada dasarnya hanyalah basis data SQLite dengan tabel metadata
tertentu. Modul ini menulisnya langsung, sehingga aplikasi tidak perlu
memasang GDAL (±200 MB) hanya untuk mengekspor peta.

Dua jalur penulisan, dipilih otomatis:

* ``sqlite3``  — modul pustaka standar, dipakai bila ada (CPython biasa).
* ``murni``    — :mod:`prismacore.sqlite_tulis`, perakit berkas SQLite murni
  Python. Dipakai di Pyodide/stlite (aplikasi di GitHub Pages), yang
  membangun CPython **tanpa** ekstensi ``_sqlite3``; di sana ``import
  sqlite3`` gagal dengan ``ModuleNotFoundError``. Impor modul ini karena
  itu tidak boleh pernah bergantung pada ``sqlite3``.

Mendukung: Point, LineString, Polygon (cincin luar saja).
SRS      : srs_id -1 = "Undefined cartesian" -> tepat untuk grid tambang lokal.
           Isi `srs_wkt` + `srs_id` bila definisi CRS resmi sudah tersedia.

Diuji terhadap QGIS 3.x dan pyogrio/GDAL.
"""

from __future__ import annotations

import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from . import sqlite_tulis

try:                                    # tidak tersedia di Pyodide/stlite
    import sqlite3
except ModuleNotFoundError:             # pragma: no cover - bergantung platform
    sqlite3 = None                      # type: ignore[assignment]

SQLITE3_TERSEDIA = sqlite3 is not None

GeomType = Literal["POINT", "LINESTRING", "POLYGON"]
Backend = Literal["auto", "sqlite3", "murni"]

APPLICATION_ID = 0x47504B47  # 'GPKG'
USER_VERSION = 10300         # GeoPackage 1.3


def pilih_backend(backend: Backend = "auto") -> str:
    """Tentukan jalur penulisan; "auto" memakai sqlite3 bila tersedia."""
    if backend == "auto":
        return "sqlite3" if SQLITE3_TERSEDIA else "murni"
    if backend == "sqlite3" and not SQLITE3_TERSEDIA:
        raise RuntimeError(
            "modul sqlite3 tidak tersedia pada runtime ini (lazim di "
            'Pyodide/stlite); pakai backend="murni" atau "auto"')
    if backend not in ("sqlite3", "murni"):
        raise ValueError(f"backend tidak dikenal: {backend}")
    return backend


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
#  Kerangka metadata GeoPackage
# --------------------------------------------------------------------------
DDL_SRS = (
    "CREATE TABLE gpkg_spatial_ref_sys (\n"
    "  srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,\n"
    "  organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL,\n"
    "  definition TEXT NOT NULL, description TEXT)"
)
KOLOM_SRS = ["srs_name", "srs_id", "organization", "organization_coordsys_id",
             "definition", "description"]
TIPE_SRS = ["TEXT", "INTEGER", "TEXT", "INTEGER", "TEXT", "TEXT"]

DDL_CONTENTS = (
    "CREATE TABLE gpkg_contents (\n"
    "  table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,\n"
    "  identifier TEXT UNIQUE, description TEXT DEFAULT '',\n"
    "  last_change DATETIME NOT NULL DEFAULT\n"
    "    (strftime('%Y-%m-%dT%H:%M:%fZ','now')),\n"
    "  min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,\n"
    "  srs_id INTEGER,\n"
    "  CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id)\n"
    "    REFERENCES gpkg_spatial_ref_sys(srs_id))"
)
KOLOM_CONTENTS = ["table_name", "data_type", "identifier", "description",
                  "last_change", "min_x", "min_y", "max_x", "max_y", "srs_id"]
TIPE_CONTENTS = ["TEXT", "TEXT", "TEXT", "TEXT", "DATETIME",
                 "DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE", "INTEGER"]
INDEKS_CONTENTS = [("sqlite_autoindex_gpkg_contents_1", ["table_name"]),
                   ("sqlite_autoindex_gpkg_contents_2", ["identifier"])]

DDL_GEOM_COLS = (
    "CREATE TABLE gpkg_geometry_columns (\n"
    "  table_name TEXT NOT NULL, column_name TEXT NOT NULL,\n"
    "  geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL,\n"
    "  z TINYINT NOT NULL, m TINYINT NOT NULL,\n"
    "  CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),\n"
    "  CONSTRAINT fk_gc_tn FOREIGN KEY (table_name)\n"
    "    REFERENCES gpkg_contents(table_name))"
)
KOLOM_GEOM_COLS = ["table_name", "column_name", "geometry_type_name", "srs_id",
                   "z", "m"]
TIPE_GEOM_COLS = ["TEXT", "TEXT", "TEXT", "INTEGER", "TINYINT", "TINYINT"]
INDEKS_GEOM_COLS = [("sqlite_autoindex_gpkg_geometry_columns_1",
                     ["table_name", "column_name"])]

SRS_BAWAAN = [
    ("Undefined cartesian SRS", -1, "NONE", -1, "undefined",
     "sistem koordinat kartesian tak terdefinisi"),
    ("Undefined geographic SRS", 0, "NONE", 0, "undefined",
     "sistem koordinat geografis tak terdefinisi"),
    ("WGS 84 geodetic", 4326, "EPSG", 4326,
     'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
     'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]', "WGS 84"),
]


def _stempel_waktu() -> str:
    """Format `last_change` sesuai GeoPackage: ISO-8601 UTC, milidetik."""
    t = datetime.now(timezone.utc)
    return f"{t:%Y-%m-%dT%H:%M:%S}.{t.microsecond // 1000:03d}Z"


# --------------------------------------------------------------------------
#  Penulis
# --------------------------------------------------------------------------
class GpkgWriter:
    """Penulis GeoPackage minimal.

    Layer dikumpulkan di memori lalu berkas ditulis sekaligus saat `close()`
    (atau saat blok `with` berakhir), sehingga kedua backend menghasilkan
    berkas dengan isi yang sama.

    Contoh
    ------
    >>> with GpkgWriter("hasil.gpkg") as w:
    ...     w.add_layer("prisma", "POINT",
    ...                 [{"point_id": "CND_210_3", "v": 1.146,
    ...                   "geom": (175571.98, 9045968.46)}])
    """

    def __init__(self, path: str | Path, srs_id: int = -1,
                 srs_wkt: str | None = None, srs_name: str = "Grid tambang lokal",
                 backend: Backend = "auto"):
        self.path = Path(path)
        self.srs_id = srs_id
        self.srs_wkt = srs_wkt
        self.srs_name = srs_name
        self.backend = pilih_backend(backend)
        self._layer: list[dict] = []
        self._nama_terpakai: set[str] = set()
        self._tertutup = False
        if self.path.exists():
            self.path.unlink()

    # -- protokol context manager -----------------------------------------
    def __enter__(self) -> "GpkgWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- layer -----------------------------------------------------------
    def add_layer(self, name: str, geom_type: GeomType, rows: Iterable[dict],
                  geom_key: str = "geom", description: str = "") -> int:
        """Tambahkan satu layer. `rows` = iterable dict; satu kunci berisi geometri.

        Geometri: POINT -> (x, y); LINESTRING/POLYGON -> [(x, y), ...]
        Nilai None ditulis sebagai NULL.
        """
        if self._tertutup:
            raise RuntimeError("GpkgWriter sudah ditutup")
        if name in self._nama_terpakai:
            raise ValueError(f"layer sudah ada: {name}")
        rows = [r for r in rows if r.get(geom_key) is not None]
        if not rows:
            return 0

        schema = _infer_schema(rows, skip={geom_key})
        cols = list(schema)

        minx = miny = float("inf")
        maxx = maxy = float("-inf")
        data = []
        for r in rows:
            g = r[geom_key]
            blob = encode_geometry(geom_type, g, self.srs_id)
            e = _envelope(geom_type, g)
            minx, maxx = min(minx, e[0]), max(maxx, e[1])
            miny, maxy = min(miny, e[2]), max(maxy, e[3])
            vals: list[Any] = [None]                 # fid: ditetapkan otomatis
            for k in cols:
                v = r.get(k)
                if v is not None and schema[k] == "TEXT" and not isinstance(v, str):
                    v = str(v)
                if isinstance(v, float) and v != v:      # NaN -> NULL
                    v = None
                vals.append(v)
            vals.append(blob)
            data.append(tuple(vals))

        self._layer.append({
            "nama": name, "geom_key": geom_key, "geom_type": geom_type,
            "kolom": cols, "tipe": [schema[k] for k in cols],
            "baris": data, "deskripsi": description,
            "env": (minx, miny, maxx, maxy),
        })
        self._nama_terpakai.add(name)
        return len(data)

    # -- perakitan -------------------------------------------------------
    def _srs_baris(self) -> list[tuple]:
        srs = list(SRS_BAWAAN)
        if self.srs_wkt and self.srs_id not in (-1, 0, 4326):
            srs.append((self.srs_name, self.srs_id, "CUSTOM", self.srs_id,
                        self.srs_wkt, "grid tambang"))
        return srs

    def _skema(self) -> list[dict]:
        """Definisi seluruh tabel, netral terhadap backend."""
        stempel = _stempel_waktu()
        kontens, geomcols, tabel = [], [], []
        for L in self._layer:
            minx, miny, maxx, maxy = L["env"]
            kontens.append((L["nama"], "features", L["nama"], L["deskripsi"],
                            stempel, minx, miny, maxx, maxy, self.srs_id))
            geomcols.append((L["nama"], L["geom_key"], L["geom_type"],
                             self.srs_id, 0, 0))
            ddl = (f'CREATE TABLE "{L["nama"]}" '
                   f'(fid INTEGER PRIMARY KEY AUTOINCREMENT, '
                   + ", ".join(f'"{k}" {t}' for k, t in zip(L["kolom"], L["tipe"]))
                   + f', "{L["geom_key"]}" BLOB)')
            tabel.append({
                "nama": L["nama"], "ddl": ddl,
                "kolom": ["fid"] + L["kolom"] + [L["geom_key"]],
                "tipe": ["INTEGER"] + L["tipe"] + ["BLOB"],
                "rowid_kolom": 0, "autoincrement": True,
                "indeks": [], "baris": L["baris"],
            })

        kerangka = [
            {"nama": "gpkg_spatial_ref_sys", "ddl": DDL_SRS, "kolom": KOLOM_SRS,
             "tipe": TIPE_SRS, "rowid_kolom": 1, "autoincrement": False,
             "indeks": [], "baris": self._srs_baris()},
            {"nama": "gpkg_contents", "ddl": DDL_CONTENTS, "kolom": KOLOM_CONTENTS,
             "tipe": TIPE_CONTENTS, "rowid_kolom": None, "autoincrement": False,
             "indeks": INDEKS_CONTENTS, "baris": kontens},
            {"nama": "gpkg_geometry_columns", "ddl": DDL_GEOM_COLS,
             "kolom": KOLOM_GEOM_COLS, "tipe": TIPE_GEOM_COLS,
             "rowid_kolom": None, "autoincrement": False,
             "indeks": INDEKS_GEOM_COLS, "baris": geomcols},
        ]
        return kerangka + tabel

    def close(self) -> Path:
        """Tulis berkas GeoPackage. Aman dipanggil lebih dari sekali."""
        if self._tertutup:
            return self.path
        skema = self._skema()
        if self.backend == "sqlite3":
            self._tulis_sqlite3(skema)
        else:
            self._tulis_murni(skema)
        self._tertutup = True
        return self.path

    def _tulis_sqlite3(self, skema: list[dict]) -> None:
        con = sqlite3.connect(self.path)
        try:
            con.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            con.execute(f"PRAGMA user_version = {USER_VERSION}")
            for t in skema:
                con.execute(t["ddl"])
                if not t["baris"]:
                    continue
                kolom = ", ".join(f'"{k}"' for k in t["kolom"])
                tanya = ",".join("?" * len(t["kolom"]))
                con.executemany(
                    f'INSERT INTO "{t["nama"]}" ({kolom}) VALUES ({tanya})',
                    t["baris"])
            con.commit()
        finally:
            con.close()

    def _tulis_murni(self, skema: list[dict]) -> None:
        db = sqlite_tulis.BasisData(application_id=APPLICATION_ID,
                                    user_version=USER_VERSION)
        for t in skema:
            tab = db.tabel(t["nama"], t["ddl"], t["kolom"], t["tipe"],
                           rowid_kolom=t["rowid_kolom"],
                           autoincrement=t["autoincrement"])
            tab.sisip(t["baris"])
            for nama_ix, kolom_ix in t["indeks"]:
                tab.tambah_indeks(nama_ix, kolom_ix)
        db.tulis(self.path)


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
