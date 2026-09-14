"""
test_gpkg.py — regresi tipe kolom GeoPackage dan kelengkapan CSV ekspor.

Bug yang pernah terjadi: seluruh kolom numerik tertulis sebagai TEXT sehingga
simbologi bergradasi QGIS gagal dan pengurutan menjadi leksikografis
('80.1' < '9.2'). Uji ini menulis GPKG nyata lalu membacanya kembali dengan
`sqlite3` untuk memastikan tipe kolom tetap DOUBLE/INTEGER.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

import prismacore as pc
from prismacore.export import tulis_csv, tulis_gpkg

AKAR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AKAR / "tools"))

import buat_data_uji  # noqa: E402


@pytest.fixture(scope="module")
def keluaran(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("keluaran")
    df = buat_data_uji.buat(hari=6, per_hari=12, seed=1)
    csv = tmp / "uji.csv"
    csv.write_bytes(df.to_csv(sep=";", index=False, lineterminator="\r\n")
                     .encode("ISO-8859-1"))
    hasil = pc.jalankan([str(csv)])
    csvs = tulis_csv(hasil, tmp, "uji")
    gpkg = tulis_gpkg(hasil, tmp / "prisma_uji.gpkg", hasil["config"])
    return tmp, csvs, gpkg


def tipe_kolom(gpkg: Path, tabel: str) -> dict[str, str]:
    with sqlite3.connect(gpkg) as con:
        baris = con.execute(f'PRAGMA table_info("{tabel}")').fetchall()
    return {r[1]: (r[2] or "").upper() for r in baris}


def test_tipe_kolom_numerik_double(keluaran):
    _, _, gpkg = keluaran
    tipe = tipe_kolom(gpkg, "prisma_ringkasan")
    assert tipe["v_turunlereng_mmhari"] == "DOUBLE"
    assert tipe["d_turunlereng_mm"] == "DOUBLE"
    assert tipe["d_total_3d_mm"] == "DOUBLE"
    assert tipe["skor_prioritas"] == "DOUBLE"
    assert tipe["rasio_HV"] == "DOUBLE"
    assert tipe["seg"] == "INTEGER"
    assert tipe["n_obs"] == "INTEGER"
    assert tipe["point_id"] == "TEXT"

    tipe_geo = tipe_kolom(gpkg, "vektor_turunlereng")
    assert tipe_geo["v_turunlereng_mmhari"] == "DOUBLE"

    with sqlite3.connect(gpkg) as con:
        geom_col = con.execute(
            "SELECT geometry_type_name FROM gpkg_geometry_columns "
            "WHERE table_name = 'prisma_ringkasan'").fetchone()
        n_baris = con.execute('SELECT COUNT(*) FROM "prisma_ringkasan"').fetchone()[0]
    assert geom_col == ("POINT",)
    assert n_baris > 0


def test_csv_mentah_dan_terkoreksi_berdampingan(keluaran):
    tmp, csvs, _ = keluaran
    perepoch = [p for p in csvs if p.name.startswith("deret_perepoch")][0]
    d = pd.read_csv(perepoch)
    for kolom in ("d_rad", "d_tan", "dZ", "d_rad_c", "d_tan_c", "dZ_c"):
        assert kolom in d.columns, f"kolom {kolom} hilang dari deret_perepoch"
