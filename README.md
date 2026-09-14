# Analisis Prisma RTS

Aplikasi analisis pemantauan prisma RTS (Robotic Total Station): mengubah ekspor
CSV mentah menjadi CSV ringkasan, GeoPackage siap QGIS, dan skrip Python yang
mereproduksi hasilnya. **Tanpa GDAL, tanpa scipy, tanpa server.**

Data hidup di repo, aplikasi membacanya:

```
        git push / drag-drop web UI
                    │
                    ▼
          data/*.csv  ──────────────┐
                    │               │  (raw.githubusercontent)
      (GitHub Actions)              ▼
      jalankan prismacore    app stlite di GitHub Pages
                    │        (Pyodide, 100% di browser)
                    ▼               │
          hasil/<tag>/              ▼
          ├── ringkasan_*.csv   unduh CSV / GPKG / skrip .py
          ├── deret_harian_*.csv
          ├── deret_perepoch_*.csv
          ├── prisma_*.gpkg
          ├── log_qc_*.csv
          └── analisis_*.py
```

| Lapis | Isi | Dependensi |
|---|---|---|
| **Inti** | `prismacore/` — mesin analisis | pandas, numpy |
| **Batch** | GitHub Actions + `tools/jalankan_batch.py` | inti + runner ubuntu |
| **Interaktif** | stlite di GitHub Pages + `app/` | inti + streamlit (Pyodide) |

---

## 1. Cara pakai

**Tanpa instalasi (browser):** buka `https://fanelix.github.io/prisma/`.
Muat pertama 30–60 detik (unduh Pyodide + pandas ±20 MB, setelahnya di-cache
browser). Pilih berkas dari folder `data/` repo, atau unggah CSV dari komputer.
Seluruh proses berjalan di browser; data tidak dikirim ke mana pun.

**Lokal (Streamlit):** Windows klik ganda `jalankan.bat`; Linux/macOS `bash jalankan.sh`.
Manual:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

**CLI / batch (tanpa streamlit):**

```bash
pip install -r requirements-core.txt
python tools/jalankan_batch.py --config konfigurasi/candrian.json \
    --data data/ --keluar hasil/
python -c "import prismacore; h=prismacore.jalankan(['data/Candrian_Sep_w1_w2_2026.csv']); print(h['info'])"
```

Aplikasi **tidak pernah menyatakan lereng aman atau tidak aman**, dan tidak
pernah mengusulkan ambang TARP. Alur kerja lengkap: unggah/pilih CSV → pilih
prisma referensi (panel kiri) → tinjau log QC → unduh CSV/GPKG/skrip.

---

## 2. Struktur repo

```
├── prismacore/            paket inti (pandas + numpy)
│   ├── core.py            mesin analisis (jalankan)
│   ├── robust.py          Theil–Sen + MAD tanpa scipy
│   ├── gpkg_lite.py       penulis GeoPackage via sqlite3 bawaan
│   ├── export.py          CSV, GPKG, generator skrip reproduksi
│   └── konfigurasi_bawaan.json
├── app/
│   ├── app.py             antarmuka Streamlit (lokal & stlite)
│   └── index.html         shell stlite untuk GitHub Pages
├── data/                  CSV mentah + manifest.json
├── hasil/                 keluaran batch (ditulis Actions)
├── konfigurasi/           override JSON per situs (mis. candrian.json)
├── tools/
│   ├── jalankan_batch.py  CLI untuk Actions
│   └── buat_data_uji.py   pembangkit data sintetis 3 bulan
├── tests/                 test_golden, test_gpkg, test_robust
└── .github/workflows/     analisis.yml, pages.yml, tests.yml
```

### Konfigurasi per situs

`konfigurasi/candrian.json` cukup memuat kunci yang ingin diubah; kunci lain
mengikuti bawaan `prismacore/konfigurasi_bawaan.json`:

```json
{
  "datum": {"referensi": ["CND_30002", "CND_30003", "CND_30004"]}
}
```

---

## 3. Otomatisasi

| Workflow | Pemicu | Yang dilakukan |
|---|---|---|
| `analisis.yml` | push ke `data/**`, manual | jalankan pipeline → `hasil/<tag>/`, perbarui `data/manifest.json`, commit balik dengan `GITHUB_TOKEN`, unggah artifact 90 hari, tulis ringkasan ke tab Actions |
| `tests.yml` | push ke `prismacore/**`, `tests/**`, `data/**` | jalankan `pytest` |
| `pages.yml` | push ke `app/**`, `prismacore/**`, `konfigurasi/**` | rakit `_site/` → deploy GitHub Pages |

Aktifkan sekali di **Settings → Pages → Source: GitHub Actions**. Tidak perlu
PAT: workflow memakai `GITHUB_TOKEN` bawaan dengan `permissions: contents: write`.

---

## 4. Keluaran

| Berkas | Isi |
|---|---|
| `ringkasan_prisma_*.csv` | Metrik per prisma-segmen: perpindahan, kecepatan + CI95, H/V, signifikansi, kelas prioritas, catatan QC |
| `deret_harian_*.csv` | Deret harian radial/tangensial/vertikal terkoreksi |
| `deret_perepoch_*.csv` | Per epoch, **mentah dan terkoreksi berdampingan** — untuk audit |
| `log_qc_*.csv` | Tiap temuan QC: prisma, waktu, jenis, besaran, tindakan |
| `prisma_*.gpkg` | GeoPackage 6 layer: `prisma_ringkasan`, `vektor_turunlereng`, `trajektori_harian`, `garis_los`, `stasiun_rts`, `zona_prioritas` |
| `analisis_*.py` | Skrip mandiri yang mereproduksi run dan memverifikasi dirinya sendiri |

Kolom numerik GeoPackage ditulis sebagai `DOUBLE`/`INTEGER` (bukan teks) agar
simbologi bergradasi dan pengurutan numerik QGIS berfungsi. CRS bawaan
`srs_id = -1` (*Undefined Cartesian*); **jangan menebak EPSG** — Easting grid
Candrian tidak sesuai zona UTM manapun.

---

## 5. Verifikasi pada data nyata

`tests/test_golden.py` mengunci hasil pipeline pada `data/Candrian_Sep_w1_w2_2026.csv`
(4.546 baris, 46 prisma, 13 hari) dengan referensi `[CND_30002, CND_30003, CND_30004]`:

| Besaran | Nilai |
|---|---|
| Baris / prisma / segmen / siklus | 4.533 / 45 / 47 / 142 |
| Konstanta orientasi | 17,49″ (0,004858°), sd antar-epoch 0,18″ |
| Sisa rekonstruksi | 0,44 mm horizontal, 0,29 mm vertikal |
| Hanyutan datum | −0,293 ppm/hari; −0,248 mm/hari |
| CND_210_3 | 13,42 mm, 1,146 mm/hari, t = 24,1 |
| CND_210_4 | 12,06 mm, 1,028 mm/hari, t = 37,5 |
| CND_200_5 | vertikal −13,14 mm |
| Temuan QC | 8 (2 tukar target, 3 duplikat, 1 gap, 2 segmen pendek) |

Catatan penyimpangan dari dokumen rencana awal:
- `sisa_rekon_h/z` memakai kolom **`Horz Distance [m]`** hasil hitung instrumen
  (rencana menulis 0,31 mm vertikal; angka itu hanya tercapai bila
  `k_refraksi = 0,12`, sedangkan konfigurasi memakai 0,13).
- Jumlah temuan 8, bukan 9: spike transien CND_1505 (43,9 mm pada 12 Sep 10:02)
  tidak dilaporkan "loncatan" karena pergeseran median 24 jam-nya hanya 12,8 mm
  — di bawah ambang 20 mm. Jendela 24 jam sengaja dipakai untuk menekan
  loncatan palsu akibat siklus diurnal.

---

## 6. Status kapasitas 3 bulan

Volume dan kinerja bukan masalah: 20.422 baris / 91 hari selesai ±6 detik
(pandas+NumPy), 100.000 baris masih nyaman. Stlite 3–5× lebih lambat
(September ±5 detik), memori tab ±2–4 GB — aman untuk 100.000 baris.

**Segmentasi otomatis kini tervalidasi pada fixture sintetis:**
`tools/buat_data_uji.py` membangkitkan 20 prisma 91 hari dengan satu pemasangan
ulang 200 mm di hari ke-45. Dengan `qc.segmen_dari_loncatan = true` (bawaan),
hanya prisma tersebut yang terpecah **tepat 2 segmen**, 19 prisma lain tetap
1 segmen; pada data September nyata jumlah segmen tetap 47. Segmentasi berbasis
gap (>72 jam) dan tukar target (>1000 mm) juga tetap aktif.

Dua bug generator yang ditemukan dan diperbaiki:
1. Hanyutan skala jarak tertulis `-0.30 * hari / 1000` = −300 ppm/hari
   (seribu kali lipat). Kini `-0.30e-6 * hari` (−0,30 ppm/hari). Sebelum
   diperbaiki, prisma stabil tampak bergerak 77–266 mm/hari.
2. Setelah perbaikan, kecepatan terpulihkan prisma stabil/referensi
   ≤ 0,003 mm/hari (ambang 0,05) dan hanyutan datum terpulihkan
   −0,3018 ppm/hari (acuan −0,30).

---

## 7. Yang tidak boleh berubah

Keputusan desain, bukan detail implementasi:

1. Aplikasi **tidak pernah menyatakan lereng aman atau tidak aman**.
2. Kelas **P1–P4 adalah peringkat relatif di dalam dataset, bukan TARP situs**.
   Kolom `tarp` tetap `belum_dikonfigurasi` sampai pengguna mengisi ambang dari
   dokumen kendali lereng.
3. **Koreksi datum tidak diterapkan sampai prisma referensi dipilih manual.**
   Aplikasi hanya memberi saran kandidat; pemilihan otomatis pernah memilih
   prisma di dalam timbunan dan menghasilkan hanyutan vertikal yang salah
   (−0,434 vs −0,248 mm/hari).
4. **Inverse velocity dipagari di kode**: menolak menghasilkan tanggal prediksi
   bila rezimnya bukan progresif.
5. **Kolom mentah dipertahankan berdampingan dengan kolom terkoreksi** di
   `deret_perepoch_*.csv`.
6. **CRS bawaan `srs_id = -1`.** Jangan menebak EPSG.
7. **Rasio H/V dikosongkan** bila komponen vertikal di bawah derau harian.

Batasan yang melekat pada data (bukan pada aplikasi): stasiun tunggal tanpa
reseksi/backsight; datum vertikal tidak terkunci lebih baik dari ±3 mm; komponen
melintang garis pandang ±4× lebih berderau daripada komponen jarak sehingga
azimut pergerakan per prisma tidak terkendali untuk perpindahan < 15 mm.
