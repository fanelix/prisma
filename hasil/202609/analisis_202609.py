#!/usr/bin/env python3
# =============================================================================
#  Skrip reproduksi otomatis - 202609
#  Dihasilkan : 2026-09-14 01:28
#  prismacore : v1.0.0
#  Berkas masukan:
#    - Candrian_Sep_w1_w2_2026.csv  sha256=20c9aa654a8a4302…  (4546 baris, 2026-09-01 s/d 2026-09-13)
#
#  Cara pakai:
#    1. pip install "pandas>=2.0" "numpy>=1.24"
#    2. pastikan folder paket `prismacore/` berada di salah satu lokasi:
#       di samping skrip ini, di akar repo, atau di direktori kerja
#    3. letakkan berkas CSV mentah seperti pada daftar di atas
#       (di samping skrip, relatif terhadap akar repo, atau di direktori kerja)
#    4. python analisis_202609.py
#
#  Skrip akan menghasilkan CSV + GPKG yang identik dengan hasil aplikasi,
#  lalu memverifikasi sendiri melalui blok assert di bagian akhir.
# =============================================================================
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _dasar in (HERE, *HERE.parents, Path.cwd(), *Path.cwd().parents):
    if (_dasar / "prismacore" / "__init__.py").is_file():
        sys.path.insert(0, str(_dasar))
        break

import prismacore as pc
from prismacore.export import tulis_csv, tulis_gpkg

BERKAS = ['data/Candrian_Sep_w1_w2_2026.csv']
KELUARAN = Path("keluaran_202609")

# --- konfigurasi yang dipakai pada run ini (ditanam, bukan dari berkas luar) --
KONFIG = {   'proyek': {   'nama': 'Candrian — Pemantauan Prisma RTS',
                  'crs_id': -1,
                  'crs_wkt': None},
    'masukan': {   'pemisah': ';',
                   'enkode': ['ISO-8859-1', 'utf-8', 'cp1252'],
                   'format_waktu': '%d/%m/%Y %H:%M',
                   'cycle_gap_menit': 20},
    'geometri': {   'k_refraksi': 0.13,
                    'radius_bumi_m': 6371000.0,
                    'orientasi_lompatan_arcsec': 2.0},
    'qc': {   'dup_jarak_m': 0.5,
              'swap_jarak_m': 0.1,
              'step_mm': 20.0,
              'step_sigma': 6.0,
              'step_jendela_jam': 24,
              'reinstall_mm': 50.0,
              'reinstall_sigma': 5.0,
              'step_ambang_maks_mm': 60.0,
              'segmen_dari_loncatan': True},
    'segmentasi': {   'gap_segmen_jam': 72,
                      'baseline_n_epoch': 6,
                      'cakupan_min': 0.6,
                      'min_epoch_segmen': 10},
    'dekomposisi': {'n_tetangga': 7},
    'datum': {   'referensi': ['CND_30002', 'CND_30003', 'CND_30004'],
                 'auto_referensi': False,
                 'auto_jumlah': 3,
                 'koreksi_orientasi': True,
                 'ref_batas_mm_hari': 0.15},
    'agregasi': {'min_bacaan_harian': 4},
    'tren': {   'jendela_bergulir_hari': 7,
                'min_hari_tren': 4,
                'changepoint_min_hari': 5,
                'changepoint_perbaikan': 0.25,
                'iv_wajib_akselerasi': True,
                'iv_v_min': 0.5},
    'skor': {   'bobot': {   'mag': 0.24,
                             'vel': 0.24,
                             'acc': 0.18,
                             'sig': 0.14,
                             'vert': 0.12,
                             'coh': 0.08},
                'bagi': {   'mag': 16.0,
                            'vel': 1.3,
                            'acc': 0.35,
                            'sig': 8.0,
                            'vert': 10.0,
                            'coh': 1.0},
                'ambang': {'P1': 52, 'P2': 35, 'P3': 18},
                'aktif_jam': 12,
                'terputus_jam': 48},
    'ekspor': {'skala_vektor': 1000, 'buffer_zona_m': 25}}

# --- nilai acuan untuk verifikasi mandiri ------------------------------------
ACUAN = {   'n_baris': 4533,
    'n_prisma': 45,
    'orientasi_deg': 0.0048575,
    'v_turunlereng': {   'CND_210_5#0': 0.7238,
                         'CND_210_3#0': 1.1456,
                         'CND_210_4#0': 1.0277,
                         'CND_210_6#0': 0.5094,
                         'CND_200_8#0': 0.9853}}


def selesaikan_berkas(berkas):
    """Cari berkas masukan: apa adanya, di samping skrip, di akar repo,
    atau di subfolder `data/` pada lokasi-lokasi tersebut."""
    for dasar in (Path("."), HERE, *HERE.parents, Path.cwd()):
        ditemukan = []
        for b in berkas:
            p = Path(b)
            kandidat = [dasar / p]
            if not p.is_absolute():
                kandidat.append(dasar / "data" / p.name)
            pilih = next((c for c in kandidat if c.is_file()), None)
            if pilih is None:
                break
            ditemukan.append(pilih)
        else:
            return [str(p) for p in ditemukan]
    return [str(Path(b)) for b in berkas]


def main() -> None:
    hasil = pc.jalankan(selesaikan_berkas(BERKAS), KONFIG)

    KELUARAN.mkdir(exist_ok=True)
    csv_paths = tulis_csv(hasil, KELUARAN, "202609")
    gpkg = tulis_gpkg(hasil, KELUARAN / "prisma_202609.gpkg", KONFIG)

    print("CSV :", *[p.name for p in csv_paths], sep="\n  ")
    print("GPKG:", gpkg.name)

    # ---------------- verifikasi mandiri ------------------------------------
    info = hasil["info"]
    assert info["n_baris"] == ACUAN["n_baris"], \
        f"jumlah baris berbeda: {info['n_baris']} != {ACUAN['n_baris']}"
    assert info["n_prisma"] == ACUAN["n_prisma"], "jumlah prisma berbeda"
    assert abs(info["orientasi_deg"] - ACUAN["orientasi_deg"]) < 1e-6, \
        "konstanta orientasi berbeda"

    S = hasil["ringkasan"]
    for kunci, v_acuan in ACUAN["v_turunlereng"].items():
        v = float(S.loc[kunci, "v_turunlereng_mmhari"])
        assert abs(v - v_acuan) < 0.01, f"{kunci}: {v:.3f} != {v_acuan:.3f}"

    print("\nVerifikasi mandiri LULUS - hasil identik dengan run aplikasi.")


if __name__ == "__main__":
    main()
