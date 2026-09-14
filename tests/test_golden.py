"""
test_golden.py — jaring pengaman perilaku pipeline pada data nyata Candrian.
============================================================================
Data: 4.546 baris ekspor RTS September 2026 (grup Candrian), referensi datum
[CND_30002, CND_30003, CND_30004]. Setiap angka di GOLDEN sudah diverifikasi
manual; toleransi 1% untuk perpindahan/kecepatan, 2% untuk statistik derau.

Catatan penyimpangan terdokumentasi dari dokumen rencana:
- `sisa_rekon_h_mm_sd`/`sisa_rekon_z_mm_sd` memakai kolom "Horz Distance [m]"
  (acuan terukur 0,438 / 0,293 mm; rencana menulis 0,44 / 0,31 — angka 0,31
  hanya tercapai bila k_refraksi = 0,12, sedangkan konfigurasi memakai 0,13).
- `n_temuan_qc` = 8, bukan 9: spike transien CND_1505 (43,9 mm pada
  12 Sep 10:02) tidak dilaporkan sebagai "loncatan" karena pergeseran median
  24 jam-nya hanya 12,8 mm, di bawah ambang 20 mm. Jendela 24 jam sengaja
  dipakai untuk menekan loncatan palsu akibat siklus diurnal.
"""

from pathlib import Path

import numpy as np
import pytest

import prismacore as pc

AKAR = Path(__file__).resolve().parents[1]
DATA = AKAR / "data" / "Candrian_Sep_w1_w2_2026.csv"
REFERENSI = ["CND_30002", "CND_30003", "CND_30004"]

GOLDEN = {
    "n_baris": 4533,
    "n_prisma": 45,
    "n_segmen": 47,
    "n_siklus": 142,
    "orientasi_deg": 0.004858,
    "orientasi_sd_arcsec": 0.18,
    "sisa_rekon_h_mm_sd": 0.438,
    "sisa_rekon_z_mm_sd": 0.293,
    "datum_ppm_per_hari": -0.293,
    "datum_vert_mm_per_hari": -0.248,
    "CND_210_3#0": {"v": 1.146, "d": 13.42, "t": 24.09},
    "CND_210_4#0": {"v": 1.028, "d": 12.06, "t": 37.49},
    "CND_200_5#0": {"d_vertikal": -13.14},
    "n_temuan_qc": 8,
    "swap": {"CND_1602": 51254, "CND_1603": 41944},
    "duplikat": ("CND_200_3", "CND_200_8", 1.1),
}


@pytest.fixture(scope="module")
def hasil():
    assert DATA.exists(), f"data golden tidak ditemukan: {DATA}"
    cfg = pc.muat_konfigurasi(timpa={"datum": {"referensi": REFERENSI}})
    return pc.jalankan([str(DATA)], cfg)


def dekat(nilai, acuan, tol=0.01):
    """Pemeriksaan relatif; aman untuk nilai negatif."""
    assert abs(nilai - acuan) <= tol * abs(acuan), f"{nilai} != {acuan} ({tol:.0%})"


def test_info(hasil):
    info = hasil["info"]
    assert info["n_baris"] == GOLDEN["n_baris"]
    assert info["n_prisma"] == GOLDEN["n_prisma"]
    assert info["n_segmen"] == GOLDEN["n_segmen"]
    assert info["n_siklus"] == GOLDEN["n_siklus"]
    assert abs(info["orientasi_deg"] - GOLDEN["orientasi_deg"]) < 1e-6
    dekat(info["orientasi_sd_arcsec"], GOLDEN["orientasi_sd_arcsec"], 0.02)
    dekat(info["sisa_rekon_h_mm_sd"], GOLDEN["sisa_rekon_h_mm_sd"], 0.02)
    dekat(info["sisa_rekon_z_mm_sd"], GOLDEN["sisa_rekon_z_mm_sd"], 0.02)


def test_datum(hasil):
    dat = hasil["datum"]
    assert dat["koreksi_diterapkan"] is True
    assert dat["ref_ditolak"] == []
    dekat(dat["ppm_per_hari"], GOLDEN["datum_ppm_per_hari"])
    dekat(dat["vert_mm_per_hari"], GOLDEN["datum_vert_mm_per_hari"])


def test_prisma_acuan(hasil):
    S = hasil["ringkasan"]
    for kunci in ("CND_210_3#0", "CND_210_4#0"):
        r = S.loc[kunci]
        dekat(r["v_turunlereng_mmhari"], GOLDEN[kunci]["v"])
        dekat(r["d_turunlereng_mm"], GOLDEN[kunci]["d"])
        dekat(r["t_turunlereng"], GOLDEN[kunci]["t"])
    r = S.loc["CND_200_5#0"]
    dekat(r["d_vertikal_mm"], GOLDEN["CND_200_5#0"]["d_vertikal"])


def test_temuan_qc(hasil):
    log = hasil["log"]
    assert len(log) == GOLDEN["n_temuan_qc"]
    assert set(log.jenis) == {"swap_target", "duplikat_target", "gap", "segmen_pendek"}
    assert (log.tindakan.isin(["segmen_baru", "ditandai", "dibuang"])).all()

    swap = log[log.jenis == "swap_target"].set_index("point_id").besaran
    for pid, acuan in GOLDEN["swap"].items():
        dekat(float(swap[pid]), acuan)

    a, b, jarak = GOLDEN["duplikat"]
    dup = log[(log.jenis == "duplikat_target")
              & log.point_id.str.contains(a, regex=False)
              & log.point_id.str.contains(b, regex=False)]
    assert len(dup) == 1
    dekat(float(dup.besaran.iloc[0]), jarak, 0.02)


def test_invarian_keputusan(hasil):
    """Keputusan desain yang tidak boleh berubah (lihat README §Yang tidak boleh berubah)."""
    S = hasil["ringkasan"]
    assert (S.tarp == "belum_dikonfigurasi").all()
    nilai_iv = set(S.inverse_velocity.dropna().unique())
    assert nilai_iv <= {"tidak_berlaku", "dapat_dihitung_tinjau_manual"}
    # rasio H/V dikosongkan bila komponen vertikal di bawah derau harian
    kosong = S.rasio_HV.isna()
    assert kosong.any(), "harus ada prisma dengan rasio H/V dikosongkan"
