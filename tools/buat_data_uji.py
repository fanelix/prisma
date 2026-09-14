"""
buat_data_uji.py — pembangkit data uji sintetis 3 bulan
=======================================================
Menghasilkan berkas CSV dengan format identik ekspor RTS, berisi cacat yang
SUDAH DIKETAHUI, sehingga pipeline dapat diuji apakah menemukannya:

  1. tren turun-lereng linier pada klaster aktif
  2. tren yang MEMPERCEPAT pada satu prisma
  3. PEMASANGAN ULANG (offset 200 mm) pada hari ke-45
  4. prisma REFERENSI yang mulai bergerak pada hari ke-30
  5. atrisi target (beberapa prisma berhenti terukur)
  6. siklus diurnal pada sudut, bukan pada jarak
  7. hanyutan datum skala jarak

Pemakaian:
    python buat_data_uji.py                 -> data_uji_3bulan.csv
    python buat_data_uji.py --hari 180      -> periode lain
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SE, SN, SH = 176089.028, 9045633.904, 78.939
ORIENT = 0.0048577          # konstanta orientasi (derajat)
K, R = 0.13, 6371000.0

# (point_id, E, N, Z, kelompok)
PRISMA = [
    ("CND_30002", 175417.870, 9045868.292, 210.932, "referensi"),
    ("CND_30003", 175469.928, 9045870.833, 201.271, "referensi"),
    ("CND_30004", 175504.699, 9045863.297, 190.974, "referensi_drift"),
    ("CND_210_3", 175571.980, 9045968.460, 209.152, "aktif_cepat"),
    ("CND_210_4", 175648.856, 9046021.445, 209.726, "aktif_cepat"),
    ("CND_210_5", 175727.490, 9046077.900, 210.920, "aktif_akselerasi"),
    ("CND_210_6", 175793.460, 9046117.790, 210.820, "aktif_lambat"),
    ("CND_200_3", 175600.000, 9045956.630, 200.680, "aktif_cepat"),
    ("CND_200_4", 175653.390, 9045993.940, 199.410, "reinstall"),
    ("CND_200_5", 175708.350, 9046029.650, 199.790, "aktif_turun"),
    ("CND_200_6", 175781.120, 9046077.850, 199.570, "aktif_lambat"),
    ("CND_190_3", 175576.403, 9045913.166, 192.556, "aktif_sedang"),
    ("CND_190_4", 175629.386, 9045948.631, 191.401, "aktif_sedang"),
    ("CND_190_5", 175695.920, 9045990.480, 190.020, "aktif_sedang"),
    ("CND_180_2", 175612.638, 9045898.025, 179.183, "aktif_lambat"),
    ("CND_170_1", 175649.375, 9045873.114, 170.147, "stabil"),
    ("CND_22", 175979.080, 9045865.720, 120.780, "stabil"),
    ("CND_25", 175918.280, 9045795.340, 110.250, "atrisi"),
    ("CND_30", 175938.990, 9045838.840, 120.600, "atrisi"),
    ("CND_34", 175936.000, 9045905.150, 130.990, "stabil"),
]

# kecepatan turun-lereng (mm/hari), penurunan (mm/hari)
GERAK = {
    "referensi": (0.0, 0.0), "referensi_drift": (0.0, 0.0),
    "stabil": (0.0, 0.0), "atrisi": (0.0, 0.0),
    "aktif_cepat": (1.10, -0.55), "aktif_sedang": (0.70, -0.35),
    "aktif_lambat": (0.35, -0.55), "aktif_turun": (0.45, -1.10),
    "aktif_akselerasi": (0.30, -0.60), "reinstall": (0.90, -0.50),
}
DIPDIR = 146.0


def _dms(deg: float) -> str:
    deg = deg % 360
    d = int(deg)
    m_ = (deg - d) * 60
    m = int(m_)
    s = (m_ - m) * 60
    return f' {d:3d}° {m:02d}\' {s:08.5f}"'


def polar(e, n, z):
    dE, dN, dZ = e - SE, n - SN, z - SH
    hd = np.hypot(dE, dN)
    az = np.degrees(np.arctan2(dE, dN)) % 360
    d = np.hypot(hd, dZ)
    for _ in range(3):
        c = d ** 2 * (1 - K) / (2 * R)
        d = np.hypot(hd, dZ - c)
    c = d ** 2 * (1 - K) / (2 * R)
    v = np.degrees(np.arctan2(hd, dZ - c))
    return (az - ORIENT) % 360, v, d, hd


def buat(hari: int = 91, per_hari: int = 12, seed: int = 42,
         rotasi_arcsec: float = 4.0, v_term_arcsec: float = 2.0,
         ppm_per_hari: float = -0.30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-07-01 00:00")
    rows = []

    for hh in range(hari):
        for ep in range(per_hari):
            t = t0 + pd.Timedelta(hours=hh * 24 + ep * (24 / per_hari))
            hari_f = hh + ep / per_hari
            jam = t.hour
            # sistematik: rotasi termal pada sudut, hanyutan skala pada jarak
            rot = np.radians(rotasi_arcsec * np.sin((jam - 8) / 24 * 2 * np.pi) / 3600)
            v_term = v_term_arcsec * np.sin((jam - 9) / 24 * 2 * np.pi) / 3600
            ppm = ppm_per_hari * 1e-6 * hari_f

            for pid, e0, n0, z0, grp in PRISMA:
                if grp == "atrisi" and hari_f > hari * 0.45:
                    continue
                if pid == "CND_210_6" and hari_f > hari * 0.8:
                    continue

                vh, vz = GERAK[grp]
                if grp == "aktif_akselerasi":
                    vh = 0.20 + 0.016 * hari_f          # 0,20 -> ~1,7 mm/hari
                    vz = -0.30 - 0.010 * hari_f
                dh = vh * hari_f
                dz = vz * hari_f
                if grp == "referensi_drift" and hari_f > 30:
                    dh += 0.40 * (hari_f - 30)          # referensi mulai bergerak
                if grp == "reinstall" and hari_f > 45:
                    dh = vh * (hari_f - 45) + 200.0     # dipasang ulang, offset 200 mm

                a = np.radians(DIPDIR)
                e = e0 + (dh * np.sin(a)) / 1000
                n = n0 + (dh * np.cos(a)) / 1000
                z = z0 + dz / 1000

                hz, vv, d, hd = polar(e, n, z)
                d *= (1 + ppm)
                sig = 0.0006 + hd * 1.3e-6
                d += rng.normal(0, sig)
                hz += np.degrees(rot) + rng.normal(0, 0.8 / 3600)
                vv += v_term + rng.normal(0, 1.1 / 3600)

                dd = round(d, 3)
                hdc = dd * np.sin(np.radians(vv))
                az = (hz + ORIENT) % 360
                rows.append({
                    "Point ID": pid,
                    "Time": t.strftime("%d/%m/%Y %H:%M"),
                    "Hz [dms]": _dms(hz), "V [dms]": _dms(vv), "D [m]": f"{dd:.3f}",
                    "PPM Type": "Atmos PPM", "PPM": "0", "Pressure [mBar]": "1013.25",
                    "Av Temp [°C]": "11.1012",
                    "Target Easting [m]": f"{SE + hdc*np.sin(np.radians(az)):.3f}",
                    "Target Northing [m]": f"{SN + hdc*np.cos(np.radians(az)):.3f}",
                    "Target Elevation [m]":
                        f"{SH + dd*np.cos(np.radians(vv)) + dd**2*(1-K)/(2*R):.3f}",
                    "Station Easting [m]": f"{SE:.3f}", "Station Northing [m]": f"{SN:.3f}",
                    "Station Height [m]": f"{SH:.3f}",
                    "Null Measurement [m]": f"{hd:.3f}", "Horz Distance [m]": f"{hdc:.3f}",
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hari", type=int, default=91)
    ap.add_argument("--per-hari", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rotasi-arcsec", type=float, default=4.0,
                    help="Amplitudo rotasi termal sudut (0 = matikan)")
    ap.add_argument("--v-term-arcsec", type=float, default=2.0,
                    help="Amplitudo termal vertikal (0 = matikan)")
    ap.add_argument("--ppm-per-hari", type=float, default=-0.30,
                    help="Hanyutan skala jarak jarak-jauh (0 = matikan)")
    ap.add_argument("--keluaran", default="data_uji_3bulan.csv")
    a = ap.parse_args()
    df = buat(a.hari, a.per_hari, a.seed,
              rotasi_arcsec=a.rotasi_arcsec, v_term_arcsec=a.v_term_arcsec,
              ppm_per_hari=a.ppm_per_hari)
    Path(a.keluaran).write_bytes(
        df.to_csv(sep=";", index=False, lineterminator="\r\n").encode("ISO-8859-1"))
    print(f"{a.keluaran}: {len(df):,} baris, {df['Point ID'].nunique()} prisma, "
          f"{a.hari} hari")
