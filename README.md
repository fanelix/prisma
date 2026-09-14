# Aplikasi Analisis Prisma RTS

Aplikasi web ringan untuk mengolah raw data prisma RTS menjadi **CSV**, **GeoPackage**,
dan **skrip Python reproduksi**. Berjalan lokal, gratis seluruhnya, tanpa GDAL.

---

## 1. Cara menjalankan

**Windows** — klik ganda `jalankan.bat`
**Linux / macOS** — `bash jalankan.sh`

Manual:
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Browser terbuka di `http://localhost:8501`. Unggah satu atau beberapa CSV ekspor RTS.

**Data tidak pernah keluar dari komputer Anda.** Aplikasi berjalan sepenuhnya lokal.
Jangan menaruhnya di Streamlit Community Cloud atau Hugging Face Spaces bila data
pemantauan lereng bersifat rahasia — layanan gratis itu meng-hosting berkas di server pihak
ketiga. Untuk akses beberapa orang, jalankan di satu PC/VM di jaringan site:
`streamlit run app.py --server.address 0.0.0.0`.

---

## 2. Tumpukan teknologi

| Komponen | Pilihan | Alasan |
|---|---|---|
| Antarmuka | **Streamlit** | Paling sedikit kode untuk UI web yang fungsional |
| Analisis | **pandas + numpy** | Cukup untuk 100.000+ baris |
| Regresi robust | `robust.py` (numpy) | Menggantikan **scipy** (−114 MB); diverifikasi identik dengan `scipy.stats.theilslopes` (selisih 0,000e+00 pada 270 deret uji) |
| GeoPackage | `gpkg_lite.py` (`sqlite3` bawaan) | Menggantikan **GDAL/geopandas/shapely/fiona** (−200 MB); keluaran diuji terbaca oleh GDAL dan QGIS |
| Grafik | Bawaan Streamlit | Tidak perlu matplotlib/plotly |

Total unduhan ±250 MB, seluruhnya wheel pip biasa. Tidak ada pustaka sistem, Docker,
atau basis data yang perlu dipasang.

---

## 3. Berkas

| Berkas | Isi |
|---|---|
| `app.py` | Antarmuka Streamlit |
| `prisma_core.py` | Mesin analisis — hanya butuh pandas + numpy |
| `robust.py` | Theil–Sen + MAD tanpa scipy |
| `gpkg_lite.py` | Penulis GeoPackage tanpa GDAL |
| `export.py` | CSV, GPKG, generator skrip reproduksi |
| `buat_data_uji.py` | Pembangkit data sintetis (lihat §7 — masih ada cacat) |

---

## 4. Keluaran

**CSV (4 berkas)**

| Berkas | Isi |
|---|---|
| `ringkasan_prisma_*.csv` | 46 kolom per prisma-segmen: perpindahan turun-lereng & vertikal, H/V, plunge, kecepatan + CI95, akselerasi, rezim, signifikansi, koherensi, derau, catatan QC |
| `deret_harian_*.csv` | Deret harian radial/tangensial/vertikal terkoreksi |
| `deret_perepoch_*.csv` | Per epoch, **mentah dan terkoreksi** berdampingan — untuk audit |
| `log_qc_*.csv` | Tiap temuan QC: prisma, waktu, jenis, besaran, tindakan |

**GeoPackage (6 layer)** — `prisma_ringkasan`, `vektor_turunlereng`, `trajektori_harian`,
`garis_los`, `stasiun_rts`, `zona_prioritas`.
Kolom numerik ditulis sebagai `DOUBLE`/`INTEGER`, bukan teks, sehingga simbologi
bergradasi QGIS berfungsi.
CRS bawaan = `srs_id -1` (*Undefined Cartesian*), tepat untuk grid tambang lokal.
**Jangan menebak EPSG** — Easting grid Candrian tidak sesuai zona UTM manapun untuk
Banyuwangi; menetapkan EPSG:32749 akan menggeser data ±325 km.

**Skrip Python** — `analisis_<tag>.py` berisi konfigurasi lengkap yang ditanam,
SHA-256 tiap berkas masukan, dan blok `assert` yang memverifikasi hasilnya sendiri.
ZIP unduhan menyertakan `prisma_core.py`, `robust.py`, `gpkg_lite.py`, `export.py`
sehingga skrip langsung jalan setelah diekstrak, tanpa memasang aplikasi ini.

---

## 5. Alur kerja di aplikasi

1. **Unggah** satu atau beberapa CSV (baris ganda antar berkas dibuang otomatis).
2. **Pilih prisma referensi** di panel kiri. Tab *QC & Datum* menampilkan kandidat
   terurut dari yang paling tenang — **periksa lokasinya dulu**: prisma tenang yang
   berada di dalam timbunan bukan referensi yang sah. Selama belum dipilih, koreksi
   datum tidak diterapkan dan perpindahan bersifat relatif terhadap stasiun.
3. **Tinjau log QC** — duplikat target, tukar target, loncatan, gap, setup ulang stasiun.
4. **Atur bobot** skor prioritas bila perlu; peringkat berubah langsung.
5. **Unduh** CSV / GPKG / skrip.

---

## 6. Hasil verifikasi pada data nyata

Dijalankan pada `Candrian_Sep_w1_w2_2026.csv` (4.546 baris, 46 prisma, 13 hari),
hasilnya sama dengan analisis manual yang sudah divalidasi:

| Besaran | Nilai |
|---|---|
| Waktu pipeline | 1,2 detik |
| Konstanta orientasi | 17,49″, sd antar-epoch 0,18″ |
| Sisa rekonstruksi koordinat | 0,44 mm horizontal, 0,31 mm vertikal |
| Hanyutan datum | −0,293 ppm/hari; −0,248 mm/hari |
| CND_210_3 | 13,42 mm, 1,146 mm/hari, t = 24,1 |
| CND_210_4 | 12,06 mm, 1,028 mm/hari, t = 37,5 |
| CND_200_5 | vertikal −13,14 mm, t = −6,5 |
| Temuan QC | 9 (2 tukar target, 3 duplikat, 1 loncatan, 1 gap, 2 segmen pendek) |

Ketiga cacat yang saya temukan manual ditemukan ulang secara otomatis:
CND_1602/1603 salah target (lompatan 51,3 m dan 41,9 m pada 2 Sep 10:03),
CND_200_3 ≡ CND_200_8 (1,1 mm), dan loncatan CND_1505 pada 11→12 Sep.

---

## 7. Status kapasitas 3 bulan — baca ini

Volume dan kinerja **tidak** menjadi masalah: 20.422 baris / 91 hari selesai dalam
9 detik; 100.000 baris masih nyaman.

Yang **belum tervalidasi** adalah segmentasi otomatis untuk arsip panjang:

- Segmentasi berbasis **gap** (>72 jam) dan **tukar target** (>1000 mm) bekerja dan aktif.
- Segmentasi berbasis **pergeseran level** (dugaan pemasangan ulang) saya **matikan
  secara default** (`qc.segmen_dari_loncatan = False`). Pada pengujian, fitur ini
  memecah satu prisma menjadi puluhan segmen. Saya belum dapat memastikan apakah
  penyebabnya ambang yang terlalu ketat atau cacat pada pembangkit data uji saya
  (`buat_data_uji.py` masih menghasilkan derau sudut yang jauh lebih besar dari
  spesifikasi — lihat catatan di berkasnya). **Sampai itu tuntas, jangan aktifkan
  fitur tersebut tanpa meninjau hasilnya satu per satu.**
- Pergeseran level tetap **dicatat di log QC** sebagai temuan, jadi tidak ada informasi
  yang hilang — hanya baseline yang tidak direset otomatis.

**Implikasi praktis:** untuk arsip 3 bulan, jalankan aplikasi, lalu periksa log QC.
Bila ada prisma yang benar-benar dipasang ulang, pisahkan berkas masukannya secara
manual (sebelum dan sesudah pemasangan ulang) dan jalankan dua kali. Itu cara yang
aman sampai segmentasi otomatis tervalidasi.

---

## 8. Batasan yang melekat pada data, bukan pada aplikasi

1. Stasiun tunggal, orientasi tetap, **tanpa reseksi/backsight** — stabilitas stasiun
   tidak dapat diverifikasi dari berkas.
2. Datum vertikal tidak terkunci lebih baik dari ±3 mm.
3. Komponen melintang garis pandang 4× lebih berderau daripada komponen jarak, sehingga
   **azimut pergerakan per prisma tidak terkendali baik** untuk perpindahan < 15 mm.
   Aplikasi melaporkan turun-lereng + vertikal, bukan azimut.
4. Kelas **P1–P4 adalah peringkat relatif di dalam dataset, BUKAN TARP situs.**
   Kolom `tarp` sengaja dibiarkan `belum_dikonfigurasi`; aplikasi tidak akan pernah
   mengusulkan ambang sendiri.
5. **Aplikasi tidak menyatakan lereng aman atau tidak aman.** Data prisma saja tidak cukup.
