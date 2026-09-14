"""
test_tanpa_sqlite3.py — regresi galat yang dilaporkan dari aplikasi stlite:

    File "/home/pyodide/prismacore/gpkg_lite.py", line 18, in <module>
        import sqlite3
    ModuleNotFoundError: No module named 'sqlite3'

Pyodide membangun CPython tanpa ekstensi `_sqlite3`, sehingga satu `import
sqlite3` di tingkat modul cukup untuk menjatuhkan seluruh aplikasi sebelum
sebaris data pun dibaca. Uji ini menjalankan proses anak yang MEMBLOKIR modul
sqlite3, lalu memastikan paket tetap dapat diimpor dan seluruh keluaran —
termasuk GeoPackage — tetap dihasilkan.

Uji kedua menjaga bug pendamping: `streamlit run app/app.py` hanya menaruh
folder `app/` pada sys.path, bukan direktori kerja, sehingga `import
prismacore` gagal sebelum app.py menambahkan akar repo sendiri.
"""

import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

AKAR = Path(__file__).resolve().parents[1]
DATA = AKAR / "data" / "Candrian_Sep_w1_w2_2026.csv"

BLOKIR = '''
import sys

class _Blokir:
    """Tiru Pyodide: sqlite3 sama sekali tidak ada."""
    def find_spec(self, nama, path=None, target=None):
        if nama.split(".")[0] in ("sqlite3", "_sqlite3"):
            raise ModuleNotFoundError("No module named " + repr(nama), name=nama)
        return None

for _m in [m for m in sys.modules if m.split(".")[0] in ("sqlite3", "_sqlite3")]:
    del sys.modules[_m]
sys.meta_path.insert(0, _Blokir())
sys.path.insert(0, AKAR_REPO)
'''


def _jalankan(kode: str, tanpa_sqlite3: bool = True) -> str:
    """Jalankan `kode` di proses anak, secara bawaan tanpa modul sqlite3."""
    awalan = f"AKAR_REPO = {str(AKAR)!r}\n" + BLOKIR if tanpa_sqlite3 else ""
    hasil = subprocess.run([sys.executable, "-c", awalan + textwrap.dedent(kode)],
                           capture_output=True, text=True,
                           cwd=str(AKAR), timeout=900)
    assert hasil.returncode == 0, hasil.stdout + hasil.stderr
    return hasil.stdout


def test_simulasi_blokir_memang_menutup_sqlite3():
    """Kalau blokirnya tidak bekerja, dua uji di bawah jadi tidak berarti."""
    keluar = _jalankan('''
        try:
            import sqlite3
            print("MASIH ADA")
        except ModuleNotFoundError:
            print("TERBLOKIR")
    ''')
    assert "TERBLOKIR" in keluar


def test_impor_paket_tanpa_sqlite3():
    keluar = _jalankan('''
        import prismacore as pc
        print("backend:", pc.gpkg_lite.pilih_backend())
        print("tersedia:", pc.gpkg_lite.SQLITE3_TERSEDIA)
    ''')
    assert "backend: murni" in keluar
    assert "tersedia: False" in keluar


@pytest.mark.skipif(not DATA.exists(), reason="data acuan tidak tersedia")
def test_alur_penuh_tanpa_sqlite3(tmp_path):
    """Pipeline + CSV + GeoPackage + skrip reproduksi, semuanya tanpa sqlite3."""
    gpkg = tmp_path / "prisma_uji.gpkg"
    keluar = _jalankan(f'''
        from pathlib import Path
        import prismacore as pc
        from prismacore.export import tulis_csv, tulis_gpkg, tulis_skrip

        cfg = pc.muat_konfigurasi({str(AKAR / "konfigurasi" / "candrian.json")!r})
        hasil = pc.jalankan([{str(DATA)!r}], cfg)
        keluaran = Path({str(tmp_path)!r})
        tulis_csv(hasil, keluaran, "uji")
        tulis_gpkg(hasil, Path({str(gpkg)!r}), cfg)
        tulis_skrip(hasil, keluaran / "analisis_uji.py", ["uji.csv"], "uji")
        print("prisma:", hasil["info"]["n_prisma"])
    ''')
    assert "prisma: 45" in keluar
    for nama in ("ringkasan_prisma_uji.csv", "deret_harian_uji.csv",
                 "deret_perepoch_uji.csv", "log_qc_uji.csv", "analisis_uji.py"):
        assert (tmp_path / nama).is_file(), nama

    # berkas hasil tulisan murni Python harus sah menurut SQLite sungguhan
    assert gpkg.read_bytes()[:16] == b"SQLite format 3\x00"
    con = sqlite3.connect(gpkg)
    try:
        assert con.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert con.execute("PRAGMA application_id").fetchone()[0] == 0x47504B47
        layer = {r[0] for r in con.execute("SELECT table_name FROM gpkg_contents")}
        assert {"prisma_ringkasan", "vektor_turunlereng", "trajektori_harian",
                "garis_los", "stasiun_rts"} <= layer
        assert con.execute("SELECT COUNT(*) FROM prisma_ringkasan").fetchone()[0] == 47
    finally:
        con.close()


def test_app_menemukan_prismacore_dari_folder_app(tmp_path):
    """`streamlit run app/app.py` hanya menyisipkan folder app/ ke sys.path."""
    kepala = (AKAR / "app" / "app.py").read_text(encoding="utf-8")
    kepala = kepala[:kepala.index("import numpy as np")]
    skrip = tmp_path / "kepala_app.py"
    skrip.write_text(kepala + "\nimport prismacore\nprint('OK', prismacore.VERSI)\n",
                     encoding="utf-8")

    hasil = subprocess.run(
        [sys.executable, "-P", "-c",
         f"import sys; sys.path.insert(0, {str(AKAR / 'app')!r}); "
         f"src = open({str(skrip)!r}, encoding='utf-8').read(); "
         f"exec(compile(src, {str(AKAR / 'app' / 'app.py')!r}, 'exec'), "
         f"{{'__file__': {str(AKAR / 'app' / 'app.py')!r}}})"],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=300)
    assert hasil.returncode == 0, hasil.stdout + hasil.stderr
    assert "OK" in hasil.stdout
