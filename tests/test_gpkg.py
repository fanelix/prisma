"""
test_gpkg.py — regresi tipe kolom GeoPackage dan kelengkapan CSV ekspor.

Bug yang pernah terjadi: seluruh kolom numerik tertulis sebagai TEXT sehingga
simbologi bergradasi QGIS gagal dan pengurutan menjadi leksikografis
('80.1' < '9.2'). Uji ini menulis GPKG nyata lalu membacanya kembali dengan
`sqlite3` untuk memastikan tipe kolom tetap DOUBLE/INTEGER.

Seluruh uji dijalankan untuk KEDUA backend penulisan:
  * "sqlite3" — modul pustaka standar (CPython biasa);
  * "murni"   — perakit SQLite murni Python, satu-satunya jalan di Pyodide/
    stlite, yang tidak memuat ekstensi `_sqlite3`.
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

BACKEND = ["sqlite3", "murni"]


@pytest.fixture(scope="module")
def hasil_uji(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("masukan")
    df = buat_data_uji.buat(hari=6, per_hari=12, seed=1)
    csv = tmp / "uji.csv"
    csv.write_bytes(df.to_csv(sep=";", index=False, lineterminator="\r\n")
                     .encode("ISO-8859-1"))
    return pc.jalankan([str(csv)])


@pytest.fixture(scope="module")
def keluaran(hasil_uji, tmp_path_factory):
    """CSV + satu GPKG per backend."""
    tmp = tmp_path_factory.mktemp("keluaran")
    csvs = tulis_csv(hasil_uji, tmp, "uji")
    gpkg = {b: tulis_gpkg(hasil_uji, tmp / f"prisma_uji_{b}.gpkg",
                          hasil_uji["config"], backend=b) for b in BACKEND}
    return tmp, csvs, gpkg


def tipe_kolom(gpkg: Path, tabel: str) -> dict[str, str]:
    with sqlite3.connect(gpkg) as con:
        baris = con.execute(f'PRAGMA table_info("{tabel}")').fetchall()
    return {r[1]: (r[2] or "").upper() for r in baris}


@pytest.mark.parametrize("backend", BACKEND)
def test_tipe_kolom_numerik_double(keluaran, backend):
    _, _, gpkg = keluaran
    tipe = tipe_kolom(gpkg[backend], "prisma_ringkasan")
    assert tipe["v_turunlereng_mmhari"] == "DOUBLE"
    assert tipe["d_turunlereng_mm"] == "DOUBLE"
    assert tipe["d_total_3d_mm"] == "DOUBLE"
    assert tipe["skor_prioritas"] == "DOUBLE"
    assert tipe["rasio_HV"] == "DOUBLE"
    assert tipe["seg"] == "INTEGER"
    assert tipe["n_obs"] == "INTEGER"
    assert tipe["point_id"] == "TEXT"

    tipe_geo = tipe_kolom(gpkg[backend], "vektor_turunlereng")
    assert tipe_geo["v_turunlereng_mmhari"] == "DOUBLE"

    with sqlite3.connect(gpkg[backend]) as con:
        geom_col = con.execute(
            "SELECT geometry_type_name FROM gpkg_geometry_columns "
            "WHERE table_name = 'prisma_ringkasan'").fetchone()
        n_baris = con.execute('SELECT COUNT(*) FROM "prisma_ringkasan"').fetchone()[0]
    assert geom_col == ("POINT",)
    assert n_baris > 0


@pytest.mark.parametrize("backend", BACKEND)
def test_berkas_gpkg_sah(keluaran, backend):
    """Berkas harus lolos pemeriksaan SQLite sendiri, bukan sekadar terbaca."""
    _, _, gpkg = keluaran
    p = gpkg[backend]
    assert p.read_bytes()[:16] == b"SQLite format 3\x00"
    con = sqlite3.connect(p)
    try:
        assert con.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        assert con.execute("PRAGMA application_id").fetchone()[0] == 0x47504B47
        assert con.execute("PRAGMA user_version").fetchone()[0] == 10300
        layer = {r[0] for r in con.execute("SELECT table_name FROM gpkg_contents")}
        assert {"prisma_ringkasan", "vektor_turunlereng", "garis_los",
                "stasiun_rts"} <= layer
        # tiap layer terdaftar juga di gpkg_geometry_columns dan berisi geometri
        for nama in layer:
            assert con.execute("SELECT COUNT(*) FROM gpkg_geometry_columns "
                               "WHERE table_name = ?", (nama,)).fetchone()[0] == 1
            assert con.execute(f'SELECT COUNT(*) FROM "{nama}" '
                               "WHERE geom IS NOT NULL").fetchone()[0] > 0
    finally:
        con.close()


def test_kedua_backend_menghasilkan_isi_sama(keluaran):
    """Backend murni bukan jalur kelas dua: isinya harus identik."""
    _, _, gpkg = keluaran
    a, b = sqlite3.connect(gpkg["sqlite3"]), sqlite3.connect(gpkg["murni"])
    try:
        skema = "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY name"
        assert a.execute(skema).fetchall() == b.execute(skema).fetchall()
        tabel = [r[0] for r in a.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name <> 'sqlite_sequence'")]
        for t in tabel:
            kol = [r[1] for r in a.execute(f'PRAGMA table_info("{t}")')]
            # last_change memang cap waktu penulisan, jadi dikecualikan
            pilih = ", ".join(f'"{k}"' for k in kol if k != "last_change")
            q = f'SELECT {pilih} FROM "{t}" ORDER BY rowid'
            assert a.execute(q).fetchall() == b.execute(q).fetchall(), t
    finally:
        a.close()
        b.close()


def test_csv_mentah_dan_terkoreksi_berdampingan(keluaran):
    tmp, csvs, _ = keluaran
    perepoch = [p for p in csvs if p.name.startswith("deret_perepoch")][0]
    d = pd.read_csv(perepoch)
    for kolom in ("d_rad", "d_tan", "dZ", "d_rad_c", "d_tan_c", "dZ_c"):
        assert kolom in d.columns, f"kolom {kolom} hilang dari deret_perepoch"
