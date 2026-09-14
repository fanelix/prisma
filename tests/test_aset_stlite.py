"""
test_aset_stlite.py — `app/index.html` harus mendaftarkan setiap modul paket.

stlite menyalin berkas ke sistem berkas virtual Pyodide satu per satu dari
peta `files` di index.html. Modul yang ada di repo tetapi tidak terdaftar di
sana akan hilang di browser, dan aplikasi gagal dengan ModuleNotFoundError
yang hanya muncul di GitHub Pages — tidak pernah saat diuji lokal. Bug itulah
yang membuat uji ini ada.
"""

import re
from pathlib import Path

AKAR = Path(__file__).resolve().parents[1]
INDEX = AKAR / "app" / "index.html"
PAKET = AKAR / "prismacore"


def _berkas_terdaftar() -> set[str]:
    teks = INDEX.read_text(encoding="utf-8")
    blok = teks[teks.index("files:"):]
    return set(re.findall(r'"([^"]+\.(?:py|json))":\s*\{', blok))


def test_seluruh_modul_paket_terdaftar():
    perlu = {f"prismacore/{p.name}" for p in PAKET.iterdir()
             if p.suffix in (".py", ".json")}
    kurang = perlu - _berkas_terdaftar()
    assert not kurang, (
        "modul berikut tidak akan ikut ke browser: " + ", ".join(sorted(kurang)))


def test_entrypoint_dan_konfigurasi_terdaftar():
    daftar = _berkas_terdaftar()
    assert "app.py" in daftar
    assert "konfigurasi/candrian.json" in daftar
    teks = INDEX.read_text(encoding="utf-8")
    assert 'entrypoint: "app.py"' in teks


def test_berkas_yang_didaftarkan_memang_ada():
    """Setiap URL menunjuk berkas nyata relatif terhadap susunan _site/."""
    for nama in _berkas_terdaftar():
        asal = AKAR / nama if nama != "app.py" else AKAR / "app" / "app.py"
        assert asal.is_file(), f"terdaftar tetapi tidak ada di repo: {nama}"
