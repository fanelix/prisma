"""
sqlite_tulis.py — perakit basis data SQLite sekali-tulis, murni Python.
=======================================================================
Modul `sqlite3` pustaka standar TIDAK selalu ada. Build CPython yang dipakai
Pyodide/stlite (aplikasi di GitHub Pages) tidak memuat ekstensi `_sqlite3`,
sehingga `import sqlite3` di sana gagal:

    ModuleNotFoundError: No module named 'sqlite3'

Karena GeoPackage pada dasarnya hanyalah berkas SQLite, modul ini merakit
berkas tersebut langsung dari byte — tanpa mesin SQL, tanpa ekstensi C.
Cukup untuk pola pakai `gpkg_lite`: buat tabel, isi baris, tutup berkas.

Yang didukung
-------------
* tabel dengan `INTEGER PRIMARY KEY` (alias rowid) maupun tanpa,
* indeks implisit `sqlite_autoindex_*` untuk batasan PRIMARY KEY/UNIQUE
  pada kolom non-integer (QGIS/GDAL menolak skema yang kehilangan indeks ini),
* `sqlite_sequence` untuk tabel AUTOINCREMENT,
* b-tree bertingkat (halaman dalam) dan rantai halaman overflow,
* afinitas tipe kolom SQLite, sehingga nilai tersimpan dengan tipe yang sama
  seperti bila ditulis lewat modul `sqlite3`.

Yang TIDAK didukung (tidak diperlukan di sini): pembaruan/penghapusan baris,
transaksi, WAL, indeks eksplisit `CREATE INDEX`, tabel WITHOUT ROWID, dan
kolase selain BINARY.

Acuan format: https://www.sqlite.org/fileformat2.html
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

MAGIC = b"SQLite format 3\x00"
UKURAN_HALAMAN_BAWAAN = 4096
VERSI_SQLITE = 3045000           # angka versi yang ditulis ke header berkas

# jenis halaman b-tree
INDEKS_DALAM = 0x02
TABEL_DALAM = 0x05
INDEKS_DAUN = 0x0A
TABEL_DAUN = 0x0D

SQL_SQLITE_SEQUENCE = "CREATE TABLE sqlite_sequence(name,seq)"


# ==========================================================================
#  Primitif: varint, nilai serial, rekaman
# ==========================================================================
def varint(n: int) -> bytes:
    """Varint SQLite: big-endian, 7 bit per byte, maksimum 9 byte."""
    if n < 0:
        n += 1 << 64
    if not 0 <= n < (1 << 64):
        raise ValueError(f"di luar jangkauan 64-bit: {n}")
    if n <= 0x7F:
        return bytes([n])
    if n >= (1 << 56):                       # bentuk 9 byte: 8x7 bit + 8 bit
        out = bytearray(9)
        out[8] = n & 0xFF
        n >>= 8
        for i in range(7, -1, -1):
            out[i] = (n & 0x7F) | 0x80
            n >>= 7
        return bytes(out)
    bagian = []
    while n:
        bagian.append(n & 0x7F)
        n >>= 7
    bagian.reverse()
    return bytes(p | 0x80 for p in bagian[:-1]) + bytes([bagian[-1]])


_LEBAR_INT = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (6, 8))


def asli(v: Any) -> Any:
    """Ubah skalar numpy/pandas menjadi tipe Python bawaan bila perlu.

    `np.int64` bukan turunan `int`, sehingga tanpa langkah ini ia akan
    tersimpan sebagai teks. `np.float64` sudah turunan `float`.
    """
    if v is None or isinstance(v, (bool, int, float, str, bytes, bytearray,
                                   memoryview)):
        return v
    ambil = getattr(v, "item", None)
    if callable(ambil):
        try:
            return ambil()
        except Exception:                    # noqa: BLE001 - tipe tak terduga
            return v
    return v


def nilai_serial(v: Any) -> tuple[int, bytes]:
    """(tipe serial, byte badan) untuk satu nilai kolom."""
    v = asli(v)
    if v is None:
        return 0, b""
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, int):
        if v == 0:
            return 8, b""
        if v == 1:
            return 9, b""
        for tipe, lebar in _LEBAR_INT:
            batas = 1 << (lebar * 8 - 1)
            if -batas <= v < batas:
                return tipe, v.to_bytes(lebar, "big", signed=True)
        raise ValueError(f"bilangan bulat melebihi 64 bit: {v}")
    if isinstance(v, float):
        if v != v:                           # NaN disimpan SQLite sebagai NULL
            return 0, b""
        return 7, struct.pack(">d", v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        return 12 + 2 * len(b), b
    s = str(v).encode("utf-8")
    return 13 + 2 * len(s), s


def rekaman(nilai: Sequence[Any]) -> bytes:
    """Rekaman SQLite: header tipe serial + badan nilai."""
    tipe, badan = [], []
    for v in nilai:
        t, b = nilai_serial(v)
        tipe.append(varint(t))
        badan.append(b)
    isi_tipe = b"".join(tipe)
    n = len(isi_tipe) + 1                    # panjang header termasuk dirinya
    while len(varint(n)) + len(isi_tipe) != n:
        n = len(varint(n)) + len(isi_tipe)
    return varint(n) + isi_tipe + b"".join(badan)


# ==========================================================================
#  Afinitas tipe kolom (aturan SQLite 3 bagian "Type Affinity")
# ==========================================================================
def afinitas(tipe_deklarasi: str | None) -> str:
    t = (tipe_deklarasi or "").upper()
    if "INT" in t:
        return "INTEGER"
    if "CHAR" in t or "CLOB" in t or "TEXT" in t:
        return "TEXT"
    if "BLOB" in t or not t:
        return "BLOB"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "NUMERIC"


def _teks_sqlite(v: int | float) -> str:
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    return "%.15g" % v


def _angka_dari_teks(s: str):
    """Kembalikan int/float bila teks adalah angka utuh; selain itu None."""
    t = s.strip()
    if not t:
        return None
    try:
        return int(t)
    except ValueError:
        pass
    try:
        f = float(t)
    except ValueError:
        return None
    return None if f != f or f in (float("inf"), float("-inf")) else f


def terapkan_afinitas(v: Any, aff: str) -> Any:
    """Tiru konversi yang dilakukan SQLite saat nilai disimpan ke kolom."""
    v = asli(v)
    if v is None or aff == "BLOB":
        return v
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return v
    if aff == "TEXT":
        return _teks_sqlite(v) if isinstance(v, (int, float)) else v
    if aff == "REAL":
        if isinstance(v, str):
            a = _angka_dari_teks(v)
            return float(a) if a is not None else v
        return float(v)
    # INTEGER dan NUMERIC: pakai bilangan bulat bila representasinya persis
    if isinstance(v, str):
        a = _angka_dari_teks(v)
        if a is None:
            return v
        v = a
    if isinstance(v, float) and v == v and -(2 ** 63) <= v < 2 ** 63 and float(int(v)) == v:
        return int(v)
    return v


def kunci_urut(v: Any):
    """Kunci pengurutan indeks: NULL < angka < teks < blob (kolase BINARY)."""
    v = asli(v)
    if v is None:
        return (0, b"")
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, (int, float)):
        return (1, float(v))
    if isinstance(v, (bytes, bytearray, memoryview)):
        return (3, bytes(v))
    return (2, str(v).encode("utf-8"))


# ==========================================================================
#  Definisi skema
# ==========================================================================
class Indeks:
    """Indeks implisit `sqlite_autoindex_<tabel>_<n>` atas beberapa kolom."""

    def __init__(self, nama: str, tabel: "Tabel", kolom: Sequence[str]):
        self.nama = nama
        self.tabel = tabel
        self.kolom = list(kolom)


# Batasan PRIMARY KEY/UNIQUE pada kolom non-integer membuat SQLite membentuk
# indeks implisit. Bila DDL menyebutnya tetapi indeksnya tidak ikut ditulis,
# SQLite menolak seluruh berkas saat dibuka ("orphan index" / "database disk
# image is malformed") — kegagalan yang baru terlihat jauh dari penyebabnya.
_RE_UNIQUE = re.compile(r"\bUNIQUE\b", re.I)
_RE_PK = re.compile(r"(?<!INTEGER )\bPRIMARY\s+KEY\b", re.I)


def indeks_implisit_diperlukan(ddl: str) -> int:
    """Perkiraan jumlah `sqlite_autoindex_*` yang dituntut oleh sebuah DDL."""
    return len(_RE_UNIQUE.findall(ddl)) + len(_RE_PK.findall(ddl))


class Tabel:
    def __init__(self, nama: str, ddl: str, kolom: Sequence[str],
                 tipe: Sequence[str] | None = None,
                 rowid_kolom: int | None = None, autoincrement: bool = False):
        if tipe is not None and len(tipe) != len(kolom):
            raise ValueError("jumlah tipe kolom tidak sama dengan jumlah kolom")
        self.nama = nama
        self.ddl = ddl
        self.kolom = list(kolom)
        self.tipe = list(tipe) if tipe is not None else [""] * len(kolom)
        self.rowid_kolom = rowid_kolom
        self.autoincrement = autoincrement
        self.baris: list[tuple] = []
        self.indeks: list[Indeks] = []
        self._aff = [afinitas(t) for t in self.tipe]

    # -- isi -------------------------------------------------------------
    def sisip(self, baris: Iterable[Sequence[Any]]) -> int:
        n = 0
        for b in baris:
            b = tuple(b)
            if len(b) != len(self.kolom):
                raise ValueError(
                    f"{self.nama}: {len(b)} nilai untuk {len(self.kolom)} kolom")
            self.baris.append(tuple(terapkan_afinitas(v, a)
                                    for v, a in zip(b, self._aff)))
            n += 1
        return n

    def tambah_indeks(self, nama: str, kolom: Sequence[str]) -> Indeks:
        ix = Indeks(nama, self, kolom)
        self.indeks.append(ix)
        return ix

    # -- pembentukan entri b-tree ----------------------------------------
    def _rowid(self, i: int, baris: Sequence[Any]) -> int:
        """rowid baris ke-i: kolom alias bila ada, selain itu nomor urut."""
        if self.rowid_kolom is None or baris[self.rowid_kolom] is None:
            return i + 1
        return int(baris[self.rowid_kolom])

    def _entri_tabel(self) -> list[tuple[int, bytes]]:
        """(rowid, rekaman) per baris; kolom alias rowid disimpan sebagai NULL."""
        out = []
        for i, b in enumerate(self.baris):
            nilai = list(b)
            if self.rowid_kolom is not None:
                nilai[self.rowid_kolom] = None
            out.append((self._rowid(i, b), rekaman(nilai)))
        out.sort(key=lambda r: r[0])
        return out

    def _entri_indeks(self, ix: Indeks) -> list[bytes]:
        """Rekaman indeks (kolom kunci + rowid), terurut kolase BINARY."""
        pos = [self.kolom.index(k) for k in ix.kolom]
        entri = []
        for i, b in enumerate(self.baris):
            rowid = self._rowid(i, b)
            kunci = [b[p] for p in pos]
            entri.append((tuple(kunci_urut(v) for v in kunci) + (rowid,),
                          rekaman(kunci + [rowid])))
        entri.sort(key=lambda e: e[0])
        return [r for _, r in entri]


# ==========================================================================
#  Perakit halaman
# ==========================================================================
class _Alokator:
    def __init__(self, ukuran_halaman: int):
        self.ukuran = ukuran_halaman
        self.berikutnya = 1
        self.halaman: dict[int, bytes] = {}

    def ambil(self) -> int:
        no = self.berikutnya
        self.berikutnya += 1
        return no

    def taruh(self, no: int, data: bytes) -> None:
        if len(data) != self.ukuran:
            raise AssertionError(f"halaman {no}: {len(data)} != {self.ukuran} byte")
        self.halaman[no] = data


def _pecah_payload(payload: bytes, U: int, alok: _Alokator,
                   indeks: bool) -> tuple[bytes, int]:
    """Bagi payload menjadi bagian lokal + rantai halaman overflow."""
    X = (((U - 12) * 64 // 255) - 23) if indeks else (U - 35)
    P = len(payload)
    if P <= X:
        return payload, 0
    M = ((U - 12) * 32 // 255) - 23
    K = M + ((P - M) % (U - 4))
    lokal = K if K <= X else M
    sisa = payload[lokal:]
    potong = [sisa[i:i + U - 4] for i in range(0, len(sisa), U - 4)]
    nomor = [alok.ambil() for _ in potong]
    for i, p in enumerate(potong):
        lanjut = nomor[i + 1] if i + 1 < len(nomor) else 0
        alok.taruh(nomor[i], struct.pack(">I", lanjut) + p
                   + bytes(alok.ukuran - 4 - len(p)))
    return payload[:lokal], nomor[0]


def _rakit_halaman(jenis: int, sel: Sequence[bytes], kanan: int | None,
                   awal: int, ukuran: int) -> bytes:
    """Susun satu halaman b-tree. `awal` = 100 pada halaman 1, 0 selainnya."""
    panjang_hdr = 12 if jenis in (TABEL_DALAM, INDEKS_DALAM) else 8
    buf = bytearray(ukuran)
    akhir = ukuran
    penunjuk = []
    for c in sel:
        akhir -= len(c)
        buf[akhir:akhir + len(c)] = c
        penunjuk.append(akhir)
    batas = awal + panjang_hdr + 2 * len(sel)
    if batas > akhir:
        raise AssertionError("sel melebihi kapasitas halaman")
    buf[awal] = jenis
    struct.pack_into(">H", buf, awal + 1, 0)                 # freeblock
    struct.pack_into(">H", buf, awal + 3, len(sel))
    struct.pack_into(">H", buf, awal + 5, akhir if akhir < 65536 else 0)
    buf[awal + 7] = 0                                        # byte terfragmentasi
    if panjang_hdr == 12:
        struct.pack_into(">I", buf, awal + 8, int(kanan))
    for i, p in enumerate(penunjuk):
        struct.pack_into(">H", buf, awal + panjang_hdr + 2 * i, p)
    return bytes(buf)


def _kelompokkan(item: Sequence[Any], biaya, kapasitas: int) -> list[list]:
    """Kemas item berurutan ke dalam kelompok selama muat satu halaman."""
    grup: list[list] = []
    kini: list = []
    pakai = 0
    for it in item:
        b = biaya(it)
        if b > kapasitas:
            raise AssertionError(f"satu sel {b} byte tidak muat di halaman")
        if kini and pakai + b > kapasitas:
            grup.append(kini)
            kini, pakai = [], 0
        kini.append(it)
        pakai += b
    grup.append(kini)
    # Halaman dalam dengan satu anak (nol sel) tidak pernah dibuat SQLite sendiri;
    # pinjam satu item dari kelompok sebelumnya — tetapi hanya bila tetap muat,
    # sebab sel besar bisa membuat dua item sekaligus melewati kapasitas.
    if len(grup) > 1 and len(grup[-1]) == 1 and len(grup[-2]) > 1:
        pinjam = grup[-2][-1]
        if biaya(pinjam) + biaya(grup[-1][0]) <= kapasitas:
            grup[-1].insert(0, grup[-2].pop())
    return grup


def _pohon_tabel(entri: Sequence[tuple[int, bytes]], U: int, alok: _Alokator,
                 cadangan: int = 0, akar_tetap: int | None = None) -> int:
    """Bangun b-tree tabel; kembalikan nomor halaman akar."""
    sel = []
    for rowid, payload in entri:
        lokal, ov = _pecah_payload(payload, U, alok, indeks=False)
        c = varint(len(payload)) + varint(rowid) + lokal
        if ov:
            c += struct.pack(">I", ov)
        sel.append((rowid, c))

    simpul: list[dict] = []
    tingkat: list[tuple[int, int]] = []          # (indeks simpul, rowid terbesar)
    for g in _kelompokkan(sel, lambda s: len(s[1]) + 2, U - 8 - cadangan):
        simpul.append({"jenis": TABEL_DAUN, "sel": [c for _, c in g],
                       "dalam": None, "kanan": None})
        tingkat.append((len(simpul) - 1, g[-1][0] if g else 0))

    while len(tingkat) > 1:
        baru = []
        for g in _kelompokkan(tingkat, lambda a: 4 + 9 + 2, U - 12 - cadangan):
            *kiri, kanan = g
            simpul.append({"jenis": TABEL_DALAM, "sel": None,
                           "dalam": list(kiri), "kanan": kanan[0]})
            baru.append((len(simpul) - 1, kanan[1]))
        tingkat = baru

    return _beri_nomor(simpul, alok, U, akar_tetap)


def _sel_indeks(bagian: tuple[int, bytes, int], anak: int | None = None) -> bytes:
    """Sel indeks: [penunjuk anak] varint(P) payload_lokal [halaman overflow]."""
    P, lokal, ov = bagian
    depan = struct.pack(">I", anak) if anak is not None else b""
    return depan + varint(P) + lokal + (struct.pack(">I", ov) if ov else b"")


def _pohon_indeks(entri: Sequence[bytes], U: int, alok: _Alokator,
                  cadangan: int = 0) -> int:
    """Bangun b-tree indeks; kembalikan nomor halaman akar.

    Berbeda dari b-tree tabel, halaman dalam indeks IKUT menyimpan kunci
    pemisah: kunci itu DIPINDAHKAN dari daun ke induknya, bukan disalin.
    """
    bagian = [_pecah_payload(p, U, alok, indeks=True) for p in entri]
    bagian = [(len(p), lokal, ov) for p, (lokal, ov) in zip(entri, bagian)]

    kap_daun = U - 8 - cadangan
    kap_dalam = U - 12 - cadangan
    simpul: list[dict] = []

    # --- tingkat daun; sel pertama tiap kelompok lanjutan naik jadi pemisah
    anak: list[int] = []
    pemisah: list[tuple[int, bytes, int]] = []
    for i, g in enumerate(_kelompokkan(bagian, lambda b: len(_sel_indeks(b)) + 2,
                                       kap_daun)):
        if i and g:
            pemisah.append(g.pop(0))
        simpul.append({"jenis": INDEKS_DAUN, "sel": [_sel_indeks(b) for b in g],
                       "dalam": None, "kanan": None})
        anak.append(len(simpul) - 1)

    # --- tingkat dalam, satu per satu sampai tersisa satu akar
    while len(anak) > 1:
        # Batas kelompok ditentukan lebih dulu supaya kelompok terakhir tidak
        # berisi satu anak saja (halaman dalam tanpa sel).
        batas: list[tuple[int, int]] = []
        i = 0
        while i < len(anak):
            pakai = 0
            j = i + 1
            while j < len(anak):
                b = len(_sel_indeks(pemisah[j - 1], 0)) + 2
                if b > kap_dalam:
                    raise AssertionError("kunci indeks terlalu besar untuk halaman")
                if pakai + b > kap_dalam:
                    break
                pakai += b
                j += 1
            batas.append((i, j))
            i = j
        if len(batas) > 1 and batas[-1][1] - batas[-1][0] == 1:
            a0, a1 = batas[-2]
            if a1 - a0 > 1:
                batas[-2] = (a0, a1 - 1)
                batas[-1] = (a1 - 1, batas[-1][1])

        anak_baru: list[int] = []
        pemisah_baru: list[tuple[int, bytes, int]] = []
        for a0, a1 in batas:
            kunci = [pemisah[t] for t in range(a0, a1 - 1)]
            simpul.append({"jenis": INDEKS_DALAM, "sel": None,
                           "dalam": list(zip(anak[a0:a1 - 1], kunci)),
                           "kanan": anak[a1 - 1]})
            anak_baru.append(len(simpul) - 1)
            if a1 < len(anak):
                pemisah_baru.append(pemisah[a1 - 1])   # naik ke tingkat berikutnya
        anak, pemisah = anak_baru, pemisah_baru

    return _beri_nomor(simpul, alok, U, None)


def _beri_nomor(simpul: list[dict], alok: _Alokator, U: int,
                akar_tetap: int | None) -> int:
    akar = len(simpul) - 1
    nomor = [0] * len(simpul)
    nomor[akar] = akar_tetap if akar_tetap else alok.ambil()
    for i in range(len(simpul)):
        if i != akar:
            nomor[i] = alok.ambil()
    for i, s in enumerate(simpul):
        awal = 100 if nomor[i] == 1 else 0
        if s["jenis"] in (TABEL_DAUN, INDEKS_DAUN):
            alok.taruh(nomor[i], _rakit_halaman(s["jenis"], s["sel"], None, awal, U))
        elif s["jenis"] == TABEL_DALAM:
            sel = [struct.pack(">I", nomor[a]) + varint(k) for a, k in s["dalam"]]
            alok.taruh(nomor[i], _rakit_halaman(s["jenis"], sel, nomor[s["kanan"]],
                                                awal, U))
        else:
            sel = [_sel_indeks(b, nomor[a]) for a, b in s["dalam"]]
            alok.taruh(nomor[i], _rakit_halaman(s["jenis"], sel, nomor[s["kanan"]],
                                                awal, U))
    return nomor[akar]


# ==========================================================================
#  Basis data
# ==========================================================================
class BasisData:
    """Rakit berkas SQLite dari definisi tabel + baris, lalu tulis sekaligus."""

    def __init__(self, application_id: int = 0, user_version: int = 0,
                 ukuran_halaman: int = UKURAN_HALAMAN_BAWAAN):
        if ukuran_halaman < 512 or ukuran_halaman & (ukuran_halaman - 1):
            raise ValueError("ukuran halaman harus pangkat dua >= 512")
        self.application_id = application_id
        self.user_version = user_version
        self.ukuran_halaman = ukuran_halaman
        self._tabel: list[Tabel] = []

    def tabel(self, nama: str, ddl: str, kolom: Sequence[str],
              tipe: Sequence[str] | None = None, rowid_kolom: int | None = None,
              autoincrement: bool = False) -> Tabel:
        t = Tabel(nama, ddl, kolom, tipe, rowid_kolom, autoincrement)
        self._tabel.append(t)
        return t

    # -- perakitan --------------------------------------------------------
    def _daftar_tabel(self) -> list[Tabel]:
        """Sisipkan sqlite_sequence tepat sebelum tabel AUTOINCREMENT pertama."""
        daftar = list(self._tabel)
        if not any(t.autoincrement for t in daftar):
            return daftar
        if any(t.nama == "sqlite_sequence" for t in daftar):
            return daftar
        seq = Tabel("sqlite_sequence", SQL_SQLITE_SEQUENCE, ["name", "seq"])
        seq.sisip([(t.nama, max(t._rowid(i, b) for i, b in enumerate(t.baris)))
                   for t in daftar if t.autoincrement and t.baris])
        pertama = next(i for i, t in enumerate(daftar) if t.autoincrement)
        daftar.insert(pertama, seq)
        return daftar

    def ke_bytes(self) -> bytes:
        U = self.ukuran_halaman
        alok = _Alokator(U)
        alok.ambil()                                   # halaman 1 = sqlite_master

        skema: list[tuple] = []
        for t in self._daftar_tabel():
            perlu = indeks_implisit_diperlukan(t.ddl)
            if perlu > len(t.indeks):
                raise ValueError(
                    f"{t.nama}: DDL menyebut {perlu} batasan PRIMARY KEY/UNIQUE "
                    f"non-integer tetapi hanya {len(t.indeks)} indeks implisit "
                    "didaftarkan. Tanpa indeks itu SQLite menolak berkasnya. "
                    "Panggil tambah_indeks('sqlite_autoindex_<tabel>_<n>', [...]) "
                    "untuk tiap batasan, atau hapus batasannya dari DDL.")
            akar = _pohon_tabel(t._entri_tabel(), U, alok)
            skema.append(("table", t.nama, t.nama, akar, t.ddl))
            for ix in t.indeks:
                akar_ix = _pohon_indeks(t._entri_indeks(ix), U, alok)
                skema.append(("index", ix.nama, t.nama, akar_ix, None))

        entri_skema = [(i + 1, rekaman(baris)) for i, baris in enumerate(skema)]
        _pohon_tabel(entri_skema, U, alok, cadangan=100, akar_tetap=1)

        n_halaman = alok.berikutnya - 1
        kosong = [no for no in range(1, n_halaman + 1) if no not in alok.halaman]
        if kosong:
            raise AssertionError(f"halaman tidak tertulis: {kosong}")

        berkas = bytearray()
        for no in range(1, n_halaman + 1):
            berkas += alok.halaman[no]
        berkas[0:100] = self._header(n_halaman, len(skema))
        return bytes(berkas)

    def _header(self, n_halaman: int, n_objek_skema: int) -> bytes:
        h = bytearray(100)
        h[0:16] = MAGIC
        struct.pack_into(">H", h, 16, 1 if self.ukuran_halaman == 65536
                         else self.ukuran_halaman)
        h[18], h[19] = 1, 1                      # versi tulis/baca: legacy
        h[20] = 0                                # byte cadangan tiap halaman
        h[21], h[22], h[23] = 64, 32, 32         # fraksi payload
        struct.pack_into(">I", h, 24, 1)         # pencacah perubahan berkas
        struct.pack_into(">I", h, 28, n_halaman)
        struct.pack_into(">I", h, 32, 0)         # freelist: kosong
        struct.pack_into(">I", h, 36, 0)
        struct.pack_into(">I", h, 40, 1 if n_objek_skema else 0)   # schema cookie
        struct.pack_into(">I", h, 44, 4)         # format skema 4
        struct.pack_into(">I", h, 48, 0)         # ukuran cache bawaan
        struct.pack_into(">I", h, 52, 0)         # auto-vacuum mati
        struct.pack_into(">I", h, 56, 1)         # enkode teks: UTF-8
        struct.pack_into(">I", h, 60, self.user_version & 0xFFFFFFFF)
        struct.pack_into(">I", h, 64, 0)         # incremental vacuum mati
        struct.pack_into(">I", h, 68, self.application_id & 0xFFFFFFFF)
        struct.pack_into(">I", h, 92, 1)         # version-valid-for = pencacah
        struct.pack_into(">I", h, 96, VERSI_SQLITE)
        return bytes(h)

    def tulis(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_bytes(self.ke_bytes())
        return p
