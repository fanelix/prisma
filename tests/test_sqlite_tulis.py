"""
test_sqlite_tulis.py — perakit SQLite murni Python harus menghasilkan berkas
yang sah menurut SQLite sendiri.
=============================================================================
Modul `prismacore.sqlite_tulis` ada karena Pyodide (stlite, aplikasi di GitHub
Pages) membangun CPython tanpa ekstensi `_sqlite3`; di sana `import sqlite3`
gagal dan seluruh paket ikut gagal diimpor. Uji ini memakai modul `sqlite3`
CPython sebagai wasit: berkas yang ditulis tanpa SQLite harus lolos
`PRAGMA integrity_check`, dibaca kembali persis, dan tetap bisa ditulisi.
"""

import random
import sqlite3
import string

import pytest

from prismacore import sqlite_tulis as sw


# --------------------------------------------------------------------------
#  Primitif
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n, harap", [
    (0, b"\x00"), (1, b"\x01"), (127, b"\x7f"),
    (128, b"\x81\x00"), (200, b"\x81\x48"), (16383, b"\xff\x7f"),
    (16384, b"\x81\x80\x00"),
])
def test_varint_nilai_kunci(n, harap):
    assert sw.varint(n) == harap


@pytest.mark.parametrize("n", [0, 1, 127, 128, 300, 2 ** 20, 2 ** 48,
                               2 ** 56 - 1, 2 ** 56, 2 ** 63 - 1, -1, -(2 ** 63)])
def test_varint_panjang_wajar(n):
    b = sw.varint(n)
    assert 1 <= len(b) <= 9
    if 0 <= n <= 127:
        assert len(b) == 1


@pytest.mark.parametrize("nilai, tipe", [
    (None, 0), (0, 8), (1, 9), (2, 1), (-129, 2), (10 ** 12, 5),
    (2 ** 62, 6), (1.5, 7), (b"ab", 16), ("ab", 17),
])
def test_tipe_serial(nilai, tipe):
    assert sw.nilai_serial(nilai)[0] == tipe


def test_nan_menjadi_null_seperti_sqlite():
    assert sw.nilai_serial(float("nan")) == (0, b"")


@pytest.mark.parametrize("deklarasi, harap", [
    ("INTEGER", "INTEGER"), ("TINYINT", "INTEGER"), ("DOUBLE", "REAL"),
    ("FLOAT", "REAL"), ("TEXT", "TEXT"), ("VARCHAR(9)", "TEXT"),
    ("BLOB", "BLOB"), ("", "BLOB"), ("DATETIME", "NUMERIC"), ("BOOLEAN", "NUMERIC"),
])
def test_afinitas_sesuai_aturan_sqlite(deklarasi, harap):
    assert sw.afinitas(deklarasi) == harap


def test_afinitas_tidak_merusak_cap_waktu():
    """Kolom DATETIME berafinitas NUMERIC; teks ISO harus tetap teks."""
    assert sw.terapkan_afinitas("2026-09-14T02:33:19.123Z", "NUMERIC") == \
        "2026-09-14T02:33:19.123Z"
    assert sw.terapkan_afinitas(3, "REAL") == 3.0
    assert sw.terapkan_afinitas(3.0, "INTEGER") == 3


# --------------------------------------------------------------------------
#  Berkas utuh
# --------------------------------------------------------------------------
def _periksa(path, tabel_harap: dict):
    con = sqlite3.connect(path)
    try:
        assert con.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        for nama, baris in tabel_harap.items():
            assert con.execute(f'SELECT * FROM "{nama}"').fetchall() == baris
    finally:
        con.close()


def test_tipe_nilai_dan_pragma(tmp_path):
    db = sw.BasisData(application_id=0x47504B47, user_version=10300)
    t = db.tabel("uji", "CREATE TABLE uji (a TEXT, b INTEGER, c DOUBLE, d BLOB)",
                 ["a", "b", "c", "d"], ["TEXT", "INTEGER", "DOUBLE", "BLOB"])
    baris = [("halo", 1, 1.5, b"\x00\x01"), ("dunia", -9, 2, None),
             (None, 2 ** 40, -3.25, b"x")]
    t.sisip(baris)
    p = db.tulis(tmp_path / "uji.db")

    _periksa(p, {"uji": [("halo", 1, 1.5, b"\x00\x01"), ("dunia", -9, 2.0, None),
                         (None, 2 ** 40, -3.25, b"x")]})
    con = sqlite3.connect(p)
    assert con.execute("PRAGMA application_id").fetchone()[0] == 0x47504B47
    assert con.execute("PRAGMA user_version").fetchone()[0] == 10300
    assert [r[2] for r in con.execute('PRAGMA table_info("uji")')] == \
        ["TEXT", "INTEGER", "DOUBLE", "BLOB"]
    # afinitas REAL: bilangan bulat pun tersimpan sebagai real, seperti sqlite3
    assert [r[0] for r in con.execute("SELECT typeof(c) FROM uji")] == ["real"] * 3
    con.close()


def test_rowid_alias_termasuk_negatif(tmp_path):
    db = sw.BasisData()
    t = db.tabel("srs", "CREATE TABLE srs (nama TEXT NOT NULL, srs_id INTEGER PRIMARY KEY)",
                 ["nama", "srs_id"], ["TEXT", "INTEGER"], rowid_kolom=1)
    t.sisip([("kartesian", -1), ("geografis", 0), ("WGS 84", 4326)])
    p = db.tulis(tmp_path / "srs.db")
    con = sqlite3.connect(p)
    assert con.execute("SELECT rowid, srs_id FROM srs ORDER BY rowid").fetchall() == \
        [(-1, -1), (0, 0), (4326, 4326)]
    con.close()


def test_btree_bertingkat_dan_autoincrement(tmp_path):
    """5.000 baris memaksa banyak halaman daun + halaman dalam."""
    db = sw.BasisData()
    t = db.tabel("fitur",
                 "CREATE TABLE fitur (fid INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "nama TEXT, v DOUBLE, geom BLOB)",
                 ["fid", "nama", "v", "geom"], ["INTEGER", "TEXT", "DOUBLE", "BLOB"],
                 rowid_kolom=0, autoincrement=True)
    baris = [(None, f"prisma_{i:05d}", i * 0.25, bytes([i % 251]) * 40)
             for i in range(5000)]
    t.sisip(baris)
    p = db.tulis(tmp_path / "fitur.gpkg")
    _periksa(p, {})

    con = sqlite3.connect(p)
    assert con.execute("SELECT COUNT(*) FROM fitur").fetchone()[0] == 5000
    assert con.execute("SELECT fid, nama FROM fitur WHERE fid = 4321").fetchone() == \
        (4321, "prisma_04320")
    # sqlite_sequence wajib ada, kalau tidak AUTOINCREMENT lanjutan akan salah
    assert con.execute("SELECT seq FROM sqlite_sequence WHERE name='fitur'"
                       ).fetchone()[0] == 5000
    con.execute("INSERT INTO fitur (nama, v, geom) VALUES ('baru', 1.0, NULL)")
    con.commit()
    assert con.execute("SELECT MAX(fid) FROM fitur").fetchone()[0] == 5001
    con.close()


def test_halaman_overflow(tmp_path):
    db = sw.BasisData()
    t = db.tabel("besar", "CREATE TABLE besar (fid INTEGER PRIMARY KEY, isi BLOB)",
                 ["fid", "isi"], ["INTEGER", "BLOB"], rowid_kolom=0)
    baris = [(1, bytes(range(256)) * 300),      # jauh melebihi satu halaman
             (2, b"kecil"), (3, bytes(4020)), (4, bytes(4100))]
    t.sisip(baris)
    _periksa(db.tulis(tmp_path / "besar.db"), {"besar": baris})


def test_indeks_implisit_ditegakkan(tmp_path):
    db = sw.BasisData()
    t = db.tabel("isi",
                 "CREATE TABLE isi (table_name TEXT NOT NULL PRIMARY KEY, "
                 "identifier TEXT UNIQUE, n INTEGER)",
                 ["table_name", "identifier", "n"], ["TEXT", "TEXT", "INTEGER"])
    t.sisip([("zona", "Zona", 3), ("prisma", "Prisma", 1), ("los", "LOS", 2)])
    t.tambah_indeks("sqlite_autoindex_isi_1", ["table_name"])
    t.tambah_indeks("sqlite_autoindex_isi_2", ["identifier"])
    p = db.tulis(tmp_path / "isi.db")
    _periksa(p, {})

    con = sqlite3.connect(p)
    rencana = con.execute(
        "EXPLAIN QUERY PLAN SELECT n FROM isi WHERE table_name='prisma'").fetchall()
    assert any("sqlite_autoindex_isi_1" in str(r) for r in rencana), rencana
    assert con.execute("SELECT n FROM isi WHERE table_name='prisma'").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO isi VALUES ('prisma', 'X', 9)")
    con.close()


def test_btree_indeks_bertingkat(tmp_path):
    """Halaman 512 byte + 1.200 kunci memaksa halaman dalam pada indeks."""
    db = sw.BasisData(ukuran_halaman=512)
    t = db.tabel("kamus", "CREATE TABLE kamus (kunci TEXT PRIMARY KEY, nilai INTEGER)",
                 ["kunci", "nilai"], ["TEXT", "INTEGER"])
    rnd = random.Random(7)
    kata = sorted({"".join(rnd.choices(string.ascii_lowercase, k=12))
                   for _ in range(1200)})
    data = [(k, i) for i, k in enumerate(kata)]
    t.sisip(data)
    t.tambah_indeks("sqlite_autoindex_kamus_1", ["kunci"])
    p = db.tulis(tmp_path / "kamus.db")
    _periksa(p, {"kamus": data})

    con = sqlite3.connect(p)
    for k, i in rnd.sample(data, 40):
        assert con.execute("SELECT nilai FROM kamus WHERE kunci = ?", (k,)
                           ).fetchone()[0] == i
    con.close()


def test_tabel_kosong_dan_skema_panjang(tmp_path):
    db = sw.BasisData()
    db.tabel("kosong", "CREATE TABLE kosong (a TEXT)", ["a"], ["TEXT"])
    for j in range(40):                          # sqlite_master > 1 halaman
        kol = [f"k{c:02d}" for c in range(30)]
        ddl = (f"CREATE TABLE t{j} (fid INTEGER PRIMARY KEY AUTOINCREMENT, "
               + ", ".join(f'"{c}" DOUBLE' for c in kol) + ")")
        t = db.tabel(f"t{j}", ddl, ["fid"] + kol, ["INTEGER"] + ["DOUBLE"] * 30,
                     rowid_kolom=0, autoincrement=True)
        t.sisip([tuple([None] + [i * 1.0 + c for c in range(30)]) for i in range(20)])
    p = db.tulis(tmp_path / "banyak.db")
    _periksa(p, {"kosong": []})

    con = sqlite3.connect(p)
    assert con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                       ).fetchone()[0] == 42          # 40 + kosong + sqlite_sequence
    assert con.execute("SELECT COUNT(*) FROM t39").fetchone()[0] == 20
    con.close()


def test_skalar_numpy_tidak_menjadi_teks(tmp_path):
    np = pytest.importorskip("numpy")
    db = sw.BasisData()
    t = db.tabel("np", "CREATE TABLE np (i INTEGER, f DOUBLE, b BOOLEAN)",
                 ["i", "f", "b"], ["INTEGER", "DOUBLE", "BOOLEAN"])
    t.sisip([(np.int64(7), np.float64(1.25), np.bool_(True))])
    p = db.tulis(tmp_path / "np.db")
    con = sqlite3.connect(p)
    assert con.execute("SELECT i, f, b FROM np").fetchone() == (7, 1.25, 1)
    assert con.execute("SELECT typeof(i), typeof(f) FROM np").fetchone() == \
        ("integer", "real")
    con.close()


# --------------------------------------------------------------------------
#  Penjaga indeks implisit
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ddl, perlu", [
    ("CREATE TABLE a (x INTEGER PRIMARY KEY, y TEXT)", 0),
    ("CREATE TABLE a (x INTEGER PRIMARY KEY AUTOINCREMENT, y TEXT)", 0),
    ("CREATE TABLE a (x TEXT PRIMARY KEY, y TEXT)", 1),
    ("CREATE TABLE a (x TEXT PRIMARY KEY, y TEXT UNIQUE)", 2),
    ("CREATE TABLE a (x TEXT, y TEXT, CONSTRAINT pk PRIMARY KEY (x, y))", 1),
    ("CREATE TABLE a (x TEXT, y TEXT)", 0),
])
def test_hitungan_indeks_implisit(ddl, perlu):
    assert sw.indeks_implisit_diperlukan(ddl) == perlu


def test_menolak_menulis_indeks_implisit_yang_hilang(tmp_path):
    """SQLite menolak berkas yang kehilangan sqlite_autoindex; gagal lebih awal."""
    db = sw.BasisData()
    t = db.tabel("a", "CREATE TABLE a (k TEXT UNIQUE, v INTEGER)", ["k", "v"],
                 ["TEXT", "INTEGER"])
    t.sisip([("x", 1)])
    with pytest.raises(ValueError, match="indeks implisit"):
        db.tulis(tmp_path / "rusak.db")

    t.tambah_indeks("sqlite_autoindex_a_1", ["k"])
    _periksa(db.tulis(tmp_path / "baik.db"), {"a": [("x", 1)]})


def test_tabel_kosong_dengan_indeks_implisit(tmp_path):
    """Tabel kosong pun harus membawa indeksnya, termasuk di halaman kecil."""
    db = sw.BasisData(ukuran_halaman=512)
    t = db.tabel("a", "CREATE TABLE a (k TEXT UNIQUE, v INTEGER)", ["k", "v"],
                 ["TEXT", "INTEGER"])
    t.tambah_indeks("sqlite_autoindex_a_1", ["k"])
    p = db.tulis(tmp_path / "kosong.db")
    _periksa(p, {"a": []})

    con = sqlite3.connect(p)
    try:
        con.execute("INSERT INTO a VALUES ('x', 1)")
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("INSERT INTO a VALUES ('x', 2)")
        con.commit()
    finally:
        con.close()
