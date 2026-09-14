"""
core.py — mesin analisis pemantauan prisma RTS
==============================================
Dependensi: pandas, numpy (robust, gpkg_lite internal). Tanpa scipy, tanpa GDAL.

Dirancang untuk arsip hingga 3 bulan: segmentasi otomatis per prisma,
deteksi setup ulang stasiun, validasi prisma referensi, dan estimator
kecepatan yang sadar-gap.

Pemanggilan tunggal:
    import prismacore
    hasil = prismacore.jalankan(["data/Candrian_Sep_w1_w2_2026.csv"])
    hasil = prismacore.jalankan(berkas, prismacore.muat_konfigurasi("konfigurasi/candrian.json"))
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import robust as rb

VERSI = "1.0.0"


# =============================================================================
#  KONFIGURASI
# =============================================================================
def _gabung_dalam(dasar: dict, tambahan: dict) -> dict:
    """Gabung dua dict secara rekursif; nilai di `tambahan` menang."""
    for k, v in tambahan.items():
        if isinstance(v, dict) and isinstance(dasar.get(k), dict):
            _gabung_dalam(dasar[k], v)
        else:
            dasar[k] = v
    return dasar


def _muat_bawaan() -> dict[str, Any]:
    with (resources.files(__package__) / "konfigurasi_bawaan.json").open(
            "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG: dict[str, Any] = _muat_bawaan()


def muat_konfigurasi(berkas: str | Path | None = None,
                     timpa: dict[str, Any] | None = None) -> dict[str, Any]:
    """Konfigurasi bawaan, digabung (rekursif) dengan berkas JSON dan/atau dict.

    Berkas `konfigurasi/*.json` cukup memuat kunci yang ingin diubah — kunci
    lain tetap mengikuti bawaan. Contoh isi `konfigurasi/candrian.json`:
        {"datum": {"referensi": ["CND_30002", "CND_30003", "CND_30004"]}}
    """
    cfg = _muat_bawaan()
    if berkas is not None:
        with open(berkas, "r", encoding="utf-8") as f:
            _gabung_dalam(cfg, json.load(f))
    if timpa:
        _gabung_dalam(cfg, timpa)
    return cfg

KOLOM_WAJIB = ["Point ID", "Time", "Hz [dms]", "V [dms]", "D [m]",
               "Target Easting [m]", "Target Northing [m]", "Target Elevation [m]",
               "Station Easting [m]", "Station Northing [m]", "Station Height [m]"]

_DMS = re.compile(r"(-?\d+)\D+(\d+)\D+([\d.]+)")


def dms(s: Any) -> float:
    if not isinstance(s, str):
        return np.nan
    m = _DMS.search(s.strip())
    if not m:
        return np.nan
    d, mi, se = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return (1 if d >= 0 else -1) * (abs(d) + mi / 60 + se / 3600)


def _log(catatan: list, pid, ts, jenis, besaran, satuan, tindakan, teks=""):
    catatan.append({"point_id": pid, "waktu": ts, "jenis": jenis,
                    "besaran": None if besaran is None else round(float(besaran), 3),
                    "satuan": satuan, "tindakan": tindakan, "catatan": teks})


# =============================================================================
#  T1 — PEMBACAAN
# =============================================================================
def baca(berkas: Sequence[Any], cfg: dict) -> tuple[pd.DataFrame, list, list]:
    """berkas: list path, atau list (nama, bytes). Mengembalikan (df, log, info_berkas)."""
    cat: list = []
    info: list = []
    bagian = []

    for b in berkas:
        if isinstance(b, tuple):
            nama, data = b
            raw_bytes = data if isinstance(data, bytes) else Path(data).read_bytes()
        else:
            nama, raw_bytes = Path(b).name, Path(b).read_bytes()
        sha = hashlib.sha256(raw_bytes).hexdigest()

        df0 = None
        for enc in cfg["masukan"]["enkode"]:
            try:
                df0 = pd.read_csv(io.BytesIO(raw_bytes), sep=cfg["masukan"]["pemisah"],
                                  encoding=enc, dtype=str)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if df0 is None:
            raise ValueError(f"{nama}: enkode tidak dikenali")

        kurang = [c for c in KOLOM_WAJIB if c not in df0.columns]
        if kurang:
            raise ValueError(f"{nama}: kolom wajib hilang -> {kurang}")

        d = pd.DataFrame({
            "point_id": df0["Point ID"].str.strip(),
            "ts": pd.to_datetime(df0["Time"], format=cfg["masukan"]["format_waktu"],
                                 errors="coerce"),
            "hz": df0["Hz [dms]"].map(dms), "v": df0["V [dms]"].map(dms),
            "d": pd.to_numeric(df0["D [m]"], errors="coerce"),
            "e": pd.to_numeric(df0["Target Easting [m]"], errors="coerce"),
            "n": pd.to_numeric(df0["Target Northing [m]"], errors="coerce"),
            "z": pd.to_numeric(df0["Target Elevation [m]"], errors="coerce"),
            "st_e": pd.to_numeric(df0["Station Easting [m]"], errors="coerce"),
            "st_n": pd.to_numeric(df0["Station Northing [m]"], errors="coerce"),
            "st_h": pd.to_numeric(df0["Station Height [m]"], errors="coerce"),
        })
        for src, dst in [("Null Measurement [m]", "null_meas"),
                         ("Horz Distance [m]", "horz_dist"),
                         ("Av Temp [°C]", "temp"), ("PPM", "ppm")]:
            d[dst] = pd.to_numeric(df0[src], errors="coerce") if src in df0.columns else np.nan
        d["berkas"] = nama

        gagal = d.ts.isna().mean()
        if gagal > 0.001:
            raise ValueError(f"{nama}: {gagal:.1%} baris gagal parse waktu")
        d = d.dropna(subset=["ts"])
        info.append({"berkas": nama, "sha256": sha, "n_baris": len(d),
                     "t_awal": d.ts.min(), "t_akhir": d.ts.max()})
        bagian.append(d)

    df = pd.concat(bagian, ignore_index=True).sort_values(["ts", "point_id"])
    sebelum = len(df)
    df = df.drop_duplicates(subset=["point_id", "ts"], keep="last").reset_index(drop=True)
    if sebelum != len(df):
        _log(cat, None, None, "duplikat_baris", sebelum - len(df), "baris",
             "dibuang", "baris ganda antar berkas")

    ts = np.sort(df.ts.unique())
    gap = np.concatenate([[0], np.diff(ts).astype("timedelta64[m]").astype(float)])
    df["cycle"] = df.ts.map(dict(zip(pd.to_datetime(ts),
                                     np.cumsum((gap > cfg["masukan"]["cycle_gap_menit"]).astype(int)))))
    df["hari"] = df.ts.dt.floor("D")
    return df, cat, info


# =============================================================================
#  T2 — GEOMETRI & STASIUN
# =============================================================================
def geometri(df: pd.DataFrame, cfg: dict, cat: list) -> tuple[pd.DataFrame, dict]:
    st = df[["st_e", "st_n", "st_h"]].median()
    if df.st_e.nunique() > 1 or df.st_n.nunique() > 1:
        _log(cat, None, None, "station_reset", df.st_e.nunique(), "posisi",
             "ditandai", "koordinat stasiun tidak tunggal dalam berkas")

    dE, dN = df.e - st.st_e, df.n - st.st_n
    az = np.degrees(np.arctan2(dE, dN)) % 360
    df["range_h"] = np.hypot(dE, dN)
    df["az_los"] = az
    off = (az - df.hz) % 360
    med = off.median()

    per = off.groupby(df.cycle).median()
    lompat = (per.diff().abs() * 3600) > cfg["geometri"]["orientasi_lompatan_arcsec"]
    for c in per.index[lompat.fillna(False)]:
        t = df.loc[df.cycle == c, "ts"].min()
        _log(cat, None, t, "station_reset", (per[c] - med) * 3600, "arcsec",
             "ditandai", f"lompatan konstanta orientasi pada siklus {c}")

    k, R = cfg["geometri"]["k_refraksi"], cfg["geometri"]["radius_bumi_m"]
    # Kolom "Horz Distance [m]" adalah jarak horizontal hasil hitung instrumen
    # (sudah termasuk koreksi sudut vertikal), jadi ia acuan yang benar untuk
    # sisa rekonstruksi. Bila kolom tidak ada, jatuh kembali ke d·sin(V).
    horiz = (df.horz_dist if df.horz_dist.notna().all()
             else df.d * np.sin(np.radians(df.v)))
    z_rec = st.st_h + df.d * np.cos(np.radians(df.v)) + (horiz ** 2) * (1 - k) / (2 * R)
    sisa_z = (df.z - z_rec) * 1000
    sisa_h = (np.hypot(dE, dN) - horiz) * 1000

    info = {"stasiun_e": float(st.st_e), "stasiun_n": float(st.st_n),
            "stasiun_h": float(st.st_h),
            "orientasi_deg": float(med), "orientasi_sd_arcsec": float(off.std() * 3600),
            "sisa_rekon_z_mm_sd": float(sisa_z.std()),
            "sisa_rekon_h_mm_sd": float(sisa_h.std()),
            "n_station_reset": int(lompat.sum())}
    return df, info


# =============================================================================
#  T3/T4 — QC TARGET, SEGMENTASI, BASELINE
# =============================================================================
def segmentasi(df: pd.DataFrame, cfg: dict, cat: list) -> pd.DataFrame:
    """Tetapkan kolom `seg` per prisma. Deteksi swap, loncatan, gap, reinstall."""
    df = df.sort_values(["point_id", "ts"]).reset_index(drop=True)
    nb = cfg["segmentasi"]["baseline_n_epoch"]

    awal = (df.groupby("point_id").head(nb)
              .groupby("point_id")[["e", "n", "z"]].median())

    # --- swap target: posisi awal satu ID cocok dengan posisi awal ID lain
    ids = awal.index.tolist()
    P = awal.values
    swap_pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            jarak = float(np.sqrt(((P[i] - P[j]) ** 2).sum()))
            if jarak < cfg["qc"]["swap_jarak_m"]:
                swap_pairs.append((ids[i], ids[j], jarak))

    df["d3d_step"] = (np.sqrt(df.groupby("point_id").e.diff() ** 2
                              + df.groupby("point_id").n.diff() ** 2
                              + df.groupby("point_id").z.diff() ** 2) * 1000)
    df["gap_jam"] = df.groupby("point_id").ts.diff().dt.total_seconds() / 3600

    seg = np.zeros(len(df), dtype=int)
    for pid, g in df.groupby("point_id", sort=False):
        idx = g.index.to_numpy()
        st_ = g.d3d_step.dropna()
        mad = float(np.median(np.abs(st_ - np.median(st_))) * 1.4826) if len(st_) > 8 else 0.0
        ambang_step = float(np.clip(cfg["qc"]["step_sigma"] * mad,
                                    cfg["qc"]["step_mm"],
                                    cfg["qc"]["step_ambang_maks_mm"]))
        # sebaran skala-harian: dasar ambang pemasangan ulang, bebas dari diurnal
        hh = g.set_index("ts")[["e", "n", "z"]].resample("D").median().dropna()
        dh_ = np.sqrt((hh.diff() ** 2).sum(axis=1)).dropna() * 1000
        mad_hari = (float(np.median(np.abs(dh_ - np.median(dh_))) * 1.4826)
                    if len(dh_) > 5 else 0.0)
        s = 0
        for k in range(len(idx)):
            i = idx[k]
            step = df.at[i, "d3d_step"]
            gap = df.at[i, "gap_jam"]
            pecah = False
            if pd.notna(gap) and gap > cfg["segmentasi"]["gap_segmen_jam"]:
                pecah = True
                _log(cat, pid, df.at[i, "ts"], "gap", gap, "jam", "segmen_baru", "")
            if pd.notna(step) and step > ambang_step:
                if step > 1000:
                    pecah = True
                    _log(cat, pid, df.at[i, "ts"], "swap_target", step, "mm",
                         "segmen_baru", "lompatan besar: dugaan salah round list")
                else:
                    # Uji pergeseran LEVEL pada jendela 24 jam di kedua sisi.
                    # Jendela penuh satu hari wajib: jendela pendek akan
                    # menangkap siklus diurnal sudut (beberapa arcsec) sebagai
                    # "pergeseran", menghasilkan segmentasi berlebihan.
                    tt = df.at[i, "ts"]
                    jw = pd.Timedelta(hours=cfg["qc"]["step_jendela_jam"])
                    m_sb = (g.ts >= tt - jw) & (g.ts < tt)
                    m_ss = (g.ts >= tt) & (g.ts < tt + jw)
                    if m_sb.sum() >= 3 and m_ss.sum() >= 3:
                        sblm = g.loc[m_sb, ["e", "n", "z"]].median()
                        ssdh = g.loc[m_ss, ["e", "n", "z"]].median()
                        shift = float(np.sqrt(((ssdh - sblm) ** 2).sum()) * 1000)
                        ambang_ri = max(cfg["qc"]["reinstall_mm"],
                                        cfg["qc"]["reinstall_sigma"] * mad_hari)
                        if shift > ambang_ri and cfg["qc"]["segmen_dari_loncatan"]:
                            pecah = True
                            _log(cat, pid, tt, "reinstall", shift, "mm", "segmen_baru",
                                 f"pergeseran level 24 jam {shift:.0f} mm melebihi "
                                 f"{ambang_ri:.0f} mm - dugaan pemasangan ulang / "
                                 "target diganti")
                        elif shift > ambang_ri:
                            _log(cat, pid, tt, "reinstall", shift, "mm", "ditandai",
                                 f"pergeseran level 24 jam {shift:.0f} mm melebihi "
                                 f"{ambang_ri:.0f} mm - dugaan pemasangan ulang. "
                                 "Aktifkan qc.segmen_dari_loncatan bila ingin "
                                 "baseline direset otomatis (TINJAU DULU)")
                        elif shift > ambang_step:
                            _log(cat, pid, tt, "loncatan", shift, "mm", "ditandai",
                                 f"pergeseran level 24 jam {shift:.0f} mm (ambang "
                                 f"{ambang_step:.0f}) - verifikasi lapangan")
            if pecah:
                s += 1
            seg[i] = s
    df["seg"] = seg

    for a, b, j in swap_pairs:
        _log(cat, f"{a} / {b}", None, "duplikat_target", j * 1000, "mm", "ditandai",
             "posisi baseline identik: target ganda atau salah round list")

    # buang segmen yang terlalu pendek untuk dianalisis
    n = df.groupby(["point_id", "seg"]).ts.transform("size")
    kecil = n < cfg["segmentasi"]["min_epoch_segmen"]
    for (pid, s), g in df[kecil].groupby(["point_id", "seg"]):
        _log(cat, pid, g.ts.min(), "segmen_pendek", len(g), "epoch", "dibuang",
             f"segmen {s} < {cfg['segmentasi']['min_epoch_segmen']} epoch")
    df = df[~kecil].copy()

    base = (df.groupby(["point_id", "seg"]).head(nb)
              .groupby(["point_id", "seg"])[["e", "n", "z"]].median()
              .rename(columns={"e": "e0", "n": "n0", "z": "z0"}))
    df = df.join(base, on=["point_id", "seg"])
    df["kunci"] = df.point_id + "#" + df.seg.astype(str)
    return df


# =============================================================================
#  T5 — DEKOMPOSISI
# =============================================================================
def dekomposisi(df: pd.DataFrame, info: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sE, sN = info["stasiun_e"], info["stasiun_n"]
    df["dE"] = (df.e - df.e0) * 1000
    df["dN"] = (df.n - df.n0) * 1000
    df["dZ"] = (df.z - df.z0) * 1000

    L = np.hypot(df.e0 - sE, df.n0 - sN)
    uxr, uyr = (df.e0 - sE) / L, (df.n0 - sN) / L
    df["rng0"] = L
    df["az0"] = np.degrees(np.arctan2(df.e0 - sE, df.n0 - sN)) % 360
    df["d_rad"] = df.dE * uxr + df.dN * uyr
    df["d_tan"] = df.dE * (-uyr) + df.dN * uxr

    # bidang lokal per kunci segmen
    b = df.groupby("kunci")[["e0", "n0", "z0", "az0", "rng0"]].first()
    pts = b[["e0", "n0", "z0"]].values
    nb = min(cfg["dekomposisi"]["n_tetangga"], len(b))
    dip, dipdir = [], []
    for i in range(len(b)):
        jj = np.argsort(np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1]))[:nb]
        Pp = pts[jj]
        A = np.c_[Pp[:, 0] - Pp[:, 0].mean(), Pp[:, 1] - Pp[:, 1].mean(), np.ones(len(Pp))]
        try:
            c = np.linalg.lstsq(A, Pp[:, 2], rcond=None)[0]
        except np.linalg.LinAlgError:
            c = np.array([0.0, 0.0, 0.0])
        dip.append(np.degrees(np.arctan(np.hypot(c[0], c[1]))))
        dipdir.append(np.degrees(np.arctan2(-c[0], -c[1])) % 360)
    b["dip_lokal"] = dip
    b["dipdir_lokal"] = dipdir
    b["backaz"] = (b.az0 + 180) % 360
    b["dev_los"] = ((b.backaz - b.dipdir_lokal + 180) % 360) - 180
    b["proj_los"] = np.cos(np.radians(b.dev_los))
    return df, b


def derau(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for kunci, g in df.groupby("kunci"):
        g = g.sort_values("ts")
        t = (g.ts - g.ts.min()).dt.total_seconds().values / 86400
        r = {"kunci": kunci}
        for c, nm in [("d_rad", "rad"), ("d_tan", "tan"), ("dZ", "vert")]:
            if len(g) >= 6:
                sl, ic = rb.theilslopes(g[c].values, t)[:2]
                res = g[c].values - (ic + sl * t)
                r["sig_" + nm] = float(np.median(np.abs(res - np.median(res))) * 1.4826)
            else:
                r["sig_" + nm] = np.nan
        out.append(r)
    return pd.DataFrame(out).set_index("kunci")


# =============================================================================
#  T6 — DATUM
# =============================================================================
def datum(df: pd.DataFrame, geo: pd.DataFrame, cfg: dict, cat: list) -> tuple[pd.DataFrame, dict]:
    t0 = df.ts.min()
    df["td"] = (df.ts - t0).dt.total_seconds() / 86400

    # --- tabel saran referensi (untuk ditampilkan di UI; BUKAN pilihan otomatis)
    kand = []
    for kunci, g in df.groupby("kunci"):
        if len(g) < 20:
            continue
        vr = abs(rb.theilslopes(g.d_rad.values, g.td.values)[0])
        vz = abs(rb.theilslopes(g.dZ.values, g.td.values)[0])
        kand.append({"kunci": kunci, "v_radial": vr, "v_vertikal": vz,
                     "v_gabungan": vr + vz, "jarak_m": float(g.rng0.iloc[0]),
                     "rl": float(g.z0.iloc[0]),
                     "cakupan": len(g) / max(1, df.cycle.nunique())})
    saran = (pd.DataFrame(kand).query("cakupan > 0.8")
             .sort_values("v_gabungan").head(8).round(3)
             if kand else pd.DataFrame())

    refs = [r for r in cfg["datum"]["referensi"] if r]
    # terima ID tanpa nomor segmen: "CND_30002" -> "CND_30002#0"
    tersedia = set(df.kunci.unique())
    refs = [r if r in tersedia else f"{r}#0" for r in refs]
    refs = [r for r in refs if r in tersedia]

    if not refs and cfg["datum"]["auto_referensi"] and len(saran):
        refs = saran.kunci.head(cfg["datum"]["auto_jumlah"]).tolist()
        _log(cat, None, None, "referensi_auto", len(refs), "prisma", "ditandai",
             "referensi dipilih otomatis (TINJAU MANUAL): " + ", ".join(refs))
    elif not refs:
        _log(cat, None, None, "referensi_kosong", 0, "prisma", "ditandai",
             "TIDAK ADA prisma referensi ditetapkan - koreksi datum TIDAK diterapkan. "
             "Perpindahan bersifat relatif terhadap stasiun, bukan terhadap kerangka stabil.")

    hasil = {"referensi": refs, "ppm_per_hari": 0.0, "vert_mm_per_hari": 0.0,
             "ppm_ci": (0.0, 0.0), "vert_ci": (0.0, 0.0), "ref_ditolak": [],
             "saran_referensi": saran, "koreksi_diterapkan": bool(refs)}

    if refs:
        r = df[df.kunci.isin(refs)].copy()
        r["ppm"] = r.d_rad / r.rng0 * 1000
        agg = r.groupby("cycle").agg(td=("td", "min"), ppm=("ppm", "median"),
                                     dZ=("dZ", "median")).dropna()
        if len(agg) >= 8:
            sp, _, lo_p, hi_p = rb.theilslopes(agg.ppm.values, agg.td.values, 0.95)
            sz, _, lo_z, hi_z = rb.theilslopes(agg.dZ.values, agg.td.values, 0.95)
            hasil.update(ppm_per_hari=float(sp), vert_mm_per_hari=float(sz),
                         ppm_ci=(float(lo_p), float(hi_p)), vert_ci=(float(lo_z), float(hi_z)))
        # validasi: referensi yang bergerak sendiri
        for kunci in refs:
            g = df[df.kunci == kunci]
            v = rb.theilslopes(g.d_rad.values, g.td.values)[0]
            v_rel = abs(v - hasil["ppm_per_hari"] * g.rng0.iloc[0] / 1000)
            if v_rel > cfg["datum"]["ref_batas_mm_hari"]:
                hasil["ref_ditolak"].append(kunci)
                _log(cat, kunci, None, "referensi_bergerak", v_rel, "mm/hari",
                     "ditolak_sebagai_referensi", "melebihi ref_batas_mm_hari")

    df["d_rad_c"] = df.d_rad - hasil["ppm_per_hari"] * df.td * df.rng0 / 1000
    df["dZ_c"] = df.dZ - hasil["vert_mm_per_hari"] * df.td

    if cfg["datum"]["koreksi_orientasi"]:
        rot = df.groupby("cycle").apply(
            lambda g: float(np.median(g.d_tan / g.rng0)), include_groups=False)
        df["d_tan_c"] = df.d_tan - df.cycle.map(rot) * df.rng0
        hasil["orientasi_median_arcsec"] = float(np.degrees(rot.median() / 1000) * 3600)
    else:
        df["d_tan_c"] = df.d_tan
        hasil["orientasi_median_arcsec"] = 0.0
    return df, hasil


# =============================================================================
#  T7 — AGREGASI HARIAN
# =============================================================================
def harian(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    g = (df.groupby(["point_id", "seg", "kunci", "hari"])
           .agg(n_bacaan=("d_rad_c", "size"), d_rad=("d_rad_c", "median"),
                d_tan=("d_tan_c", "median"), d_vert=("dZ_c", "median"))
           .reset_index())
    g = g[g.n_bacaan >= cfg["agregasi"]["min_bacaan_harian"]].copy()
    g["td"] = (g.hari - g.hari.min()).dt.total_seconds() / 86400
    return g.sort_values(["kunci", "hari"])


def uji_diurnal(df: pd.DataFrame) -> pd.DataFrame:
    acc = []
    for kunci, g in df.groupby("kunci"):
        if len(g) < 40:
            continue
        g = g.sort_values("td")
        row = {"jam": g.ts.dt.hour.values, "rng": g.rng0.values}
        for c in ["d_rad", "d_tan", "dZ"]:
            sl, ic = rb.theilslopes(g[c].values, g.td.values)[:2]
            row["r_" + c] = g[c].values - (ic + sl * g.td.values)
        acc.append(pd.DataFrame(row))
    if not acc:
        return pd.DataFrame()
    R = pd.concat(acc)
    t = R.groupby("jam").agg(radial_mm=("r_d_rad", "median"),
                             tangensial_mm=("r_d_tan", "median"),
                             vertikal_mm=("r_dZ", "median"))
    t["tangensial_arcsec"] = np.degrees(t.tangensial_mm / 1000 / R.groupby("jam").rng.median()) * 3600
    t["vertikal_arcsec"] = np.degrees(t.vertikal_mm / 1000 / R.groupby("jam").rng.median()) * 3600
    return t.round(2)


# =============================================================================
#  T8 — TREN
# =============================================================================
def _ts(y, x, ci=0.95):
    r = rb.theilslopes(y, x, ci)
    return float(r[0]), float(r[1]), float(r[2]), float(r[3])


def _changepoint(x, y, cfg):
    """Binary segmentation sederhana: satu titik potong terbaik bila membaik cukup."""
    n = len(x)
    mmin = cfg["tren"]["changepoint_min_hari"]
    if n < 2 * mmin + 2:
        return None, 0.0
    sl, ic = rb.theilslopes(y, x)[:2]
    base = np.abs(y - (ic + sl * x)).sum()
    if base <= 0:
        return None, 0.0
    terbaik, skor = None, 0.0
    for k in range(mmin, n - mmin):
        s1, i1 = rb.theilslopes(y[:k], x[:k])[:2]
        s2, i2 = rb.theilslopes(y[k:], x[k:])[:2]
        e = (np.abs(y[:k] - (i1 + s1 * x[:k])).sum()
             + np.abs(y[k:] - (i2 + s2 * x[k:])).sum())
        perbaikan = 1 - e / base
        if perbaikan > skor:
            skor, terbaik = perbaikan, k
    if skor < cfg["tren"]["changepoint_perbaikan"]:
        return None, skor
    return terbaik, skor


def tren(H: pd.DataFrame, geo: pd.DataFrame, sig: pd.DataFrame, cfg: dict,
         n_cycle_total: int) -> pd.DataFrame:
    out = []
    for kunci, g in H.groupby("kunci"):
        g = g.sort_values("td")
        pid, seg = kunci.split("#")
        gg = geo.loc[kunci] if kunci in geo.index else None
        cosdev = np.cos(np.radians(gg.dev_los)) if gg is not None else 1.0
        r: dict[str, Any] = {"kunci": kunci, "point_id": pid, "seg": int(seg),
                             "n_hari": len(g), "hari_awal": g.hari.min(),
                             "hari_akhir": g.hari.max()}
        if len(g) < cfg["tren"]["min_hari_tren"]:
            out.append(r)
            continue

        x = g.td.values
        span = x.max() - x.min() + 1
        r["cakupan"] = len(g) / span

        v_rad, _, lo, hi = _ts(g.d_rad.values, x)
        v_z = _ts(g.d_vert.values, x)[0]
        sd_rad = float(np.std(g.d_rad.values - np.polyval(np.polyfit(x, g.d_rad.values, 1), x), ddof=1))
        sd_z = float(np.std(g.d_vert.values - np.polyval(np.polyfit(x, g.d_vert.values, 1), x), ddof=1))

        r["d_turunlereng_mm"] = -(g.d_rad.iloc[-1] - g.d_rad.iloc[0]) / cosdev
        r["d_vertikal_mm"] = g.d_vert.iloc[-1] - g.d_vert.iloc[0]
        r["d_total_3d_mm"] = float(np.hypot(r["d_turunlereng_mm"], r["d_vertikal_mm"]))
        r["v_turunlereng_mmhari"] = -v_rad / cosdev
        r["v_vertikal_mmhari"] = v_z
        r["v_ci95_mmhari"] = abs(hi - lo) / 2 / abs(cosdev)
        r["t_turunlereng"] = r["d_turunlereng_mm"] / (sd_rad / abs(cosdev) * np.sqrt(2)) if sd_rad > 0 else np.nan
        r["t_vertikal"] = r["d_vertikal_mm"] / (sd_z * np.sqrt(2)) if sd_z > 0 else np.nan
        r["sd_harian_rad_mm"], r["sd_harian_vert_mm"] = sd_rad, sd_z
        # H/V hanya bermakna bila komponen vertikal di atas derau harian
        r["rasio_HV"] = (r["d_turunlereng_mm"] / abs(r["d_vertikal_mm"])
                         if abs(r["d_vertikal_mm"]) > max(2.0, 2 * sd_z) else np.nan)
        r["plunge_deg"] = float(np.degrees(np.arctan2(r["d_vertikal_mm"],
                                                      abs(r["d_turunlereng_mm"]))))

        h = len(g) // 2
        if h >= 3:
            v1 = -_ts(g.d_rad.values[:h], x[:h])[0] / cosdev
            v2 = -_ts(g.d_rad.values[h:], x[h:])[0] / cosdev
            r["v_paruh1_mmhari"], r["v_paruh2_mmhari"] = v1, v2
            r["delta_v_mmhari"] = v2 - v1

        w = cfg["tren"]["jendela_bergulir_hari"]
        ek = g[g.td > x.max() - w]
        if len(ek) >= 3:
            r[f"v_{w}hari_mmhari"] = -_ts(ek.d_rad.values, ek.td.values)[0] / cosdev

        k, skor = _changepoint(x, g.d_rad.values, cfg)
        r["changepoint"] = g.hari.iloc[k] if k else pd.NaT
        r["changepoint_skor"] = round(skor, 2)

        dv = r.get("delta_v_mmhari", 0.0) or 0.0
        v = r["v_turunlereng_mmhari"]
        if abs(v) < 0.1:
            r["rezim"] = "stabil"
        elif dv > 0.15 * max(abs(v), 0.1):
            r["rezim"] = "progresif"
        elif dv < -0.15 * max(abs(v), 0.1):
            r["rezim"] = "regresif"
        else:
            r["rezim"] = "linier"

        # inverse velocity dengan pagar pengaman
        if (cfg["tren"]["iv_wajib_akselerasi"] and r["rezim"] != "progresif") or \
           v < cfg["tren"]["iv_v_min"]:
            r["inverse_velocity"] = "tidak_berlaku"
        else:
            r["inverse_velocity"] = "dapat_dihitung_tinjau_manual"

        out.append(r)

    T = pd.DataFrame(out).set_index("kunci")
    return T


def koherensi(H: pd.DataFrame, geo: pd.DataFrame, n_tetangga: int = 4) -> pd.Series:
    piv = H.pivot_table(index="hari", columns="kunci", values="d_rad")
    hasil = {}
    for kunci in geo.index:
        if kunci not in piv.columns:
            continue
        d = np.hypot(geo.e0 - geo.e0[kunci], geo.n0 - geo.n0[kunci])
        d[kunci] = np.inf
        nb = [k for k in d.nsmallest(n_tetangga).index if k in piv.columns]
        if not nb:
            continue
        ref = piv[nb].mean(axis=1)
        ok = piv[kunci].notna() & ref.notna()
        hasil[kunci] = (float(np.corrcoef(piv[kunci][ok], ref[ok])[0, 1])
                        if ok.sum() >= 5 else np.nan)
    return pd.Series(hasil, name="koherensi_tetangga")


# =============================================================================
#  T9 — RINGKASAN & SKOR
# =============================================================================
def ringkas(df, geo, sig, T, koh, cfg, t_akhir) -> pd.DataFrame:
    meta = (df.groupby("kunci")
              .agg(n_obs=("ts", "size"), tgl_awal=("ts", "min"), tgl_akhir=("ts", "max"),
                   easting=("e0", "first"), northing=("n0", "first"), rl=("z0", "first"),
                   null_meas=("null_meas", "first"), horz_akhir=("horz_dist", "last")))
    S = meta.join(geo[["rng0", "az0", "dip_lokal", "dipdir_lokal", "dev_los", "proj_los"]])
    S = S.join(sig).join(T.drop(columns=["point_id", "seg"], errors="ignore")).join(koh)
    S["point_id"] = [k.split("#")[0] for k in S.index]
    S["seg"] = [int(k.split("#")[1]) for k in S.index]
    S = S.rename(columns={"rng0": "jarak_stasiun_m", "az0": "az_los_deg",
                          "dip_lokal": "dip_lokal_deg", "dipdir_lokal": "dipdir_lokal_deg",
                          "sig_rad": "sigma_radial_mm", "sig_tan": "sigma_tangensial_mm",
                          "sig_vert": "sigma_vertikal_mm"})
    S["gerak_sejak_pasang_mm"] = (S.horz_akhir - S.null_meas) * 1000
    S["jam_sejak_akhir"] = (t_akhir - S.tgl_akhir).dt.total_seconds() / 3600
    a, b = cfg["skor"]["aktif_jam"], cfg["skor"]["terputus_jam"]
    S["status_data"] = np.where(S.jam_sejak_akhir <= a, "AKTIF",
                                np.where(S.jam_sejak_akhir <= b, "TERPUTUS_BARU", "HILANG"))
    S["segmen_terakhir"] = S.seg == S.groupby("point_id").seg.transform("max")

    w, dv = cfg["skor"]["bobot"], cfg["skor"]["bagi"]
    cl = lambda s, d: np.clip(pd.to_numeric(s, errors="coerce").fillna(0) / d, 0, 1)
    vkol = [c for c in S.columns if c.startswith("v_") and c.endswith("hari_mmhari")]
    vter = S[vkol[0]] if vkol else S.get("v_turunlereng_mmhari")
    S["skor_prioritas"] = 100 * (
        w["mag"] * cl(S.get("d_total_3d_mm"), dv["mag"])
        + w["vel"] * cl(vter, dv["vel"])
        + w["acc"] * cl(S.get("delta_v_mmhari"), dv["acc"])
        + w["sig"] * cl(S.get("t_turunlereng"), dv["sig"])
        + w["vert"] * cl(-pd.to_numeric(S.get("d_vertikal_mm"), errors="coerce"), dv["vert"])
        + w["coh"] * cl(S.get("koherensi_tetangga"), dv["coh"]))
    t = cfg["skor"]["ambang"]
    S["kelas"] = np.select(
        [S.skor_prioritas >= t["P1"], S.skor_prioritas >= t["P2"], S.skor_prioritas >= t["P3"]],
        ["P1-PRIORITAS", "P2-PERHATIAN", "P3-PANTAU RUTIN"], "P4-STABIL")
    S["tarp"] = "belum_dikonfigurasi"
    return S.sort_values("skor_prioritas", ascending=False)


def catatan_prisma(S: pd.DataFrame, log: pd.DataFrame) -> pd.Series:
    txt = {}
    for kunci, r in S.iterrows():
        c = []
        if r.status_data == "HILANG":
            c.append("berhenti terukur - perlu pemulihan target")
        elif r.status_data == "TERPUTUS_BARU":
            c.append("tidak terukur 12-48 jam terakhir")
        if r.seg > 0:
            c.append(f"segmen ke-{r.seg + 1}: baseline direset, perpindahan tidak disambung")
        if pd.notna(r.get("cakupan")) and r.get("cakupan", 1) < 0.6:
            c.append("cakupan harian rendah - kecepatan kurang andal")
        sub = log[log.point_id.astype(str).str.contains(r.point_id, regex=False, na=False)]
        for _, t in sub.iterrows():
            if t.jenis in ("loncatan", "duplikat_target", "swap_target", "referensi_bergerak"):
                c.append(f"{t.jenis}: {t.besaran} {t.satuan}")
        txt[kunci] = "; ".join(dict.fromkeys(c))
    return pd.Series(txt, name="catatan")


# =============================================================================
#  ORKESTRATOR
# =============================================================================
def jalankan(berkas: Sequence[Any], cfg: dict | None = None) -> dict:
    cfg = json.loads(json.dumps(cfg or CONFIG))
    df, cat, info_berkas = baca(berkas, cfg)
    df, info = geometri(df, cfg, cat)
    df = segmentasi(df, cfg, cat)
    df, geo = dekomposisi(df, info, cfg)
    sig = derau(df)
    df, dat = datum(df, geo, cfg, cat)
    H = harian(df, cfg)
    T = tren(H, geo, sig, cfg, df.cycle.nunique())
    koh = koherensi(H, geo)
    log = pd.DataFrame(cat) if cat else pd.DataFrame(
        columns=["point_id", "waktu", "jenis", "besaran", "satuan", "tindakan", "catatan"])
    S = ringkas(df, geo, sig, T, koh, cfg, df.ts.max())
    S["catatan"] = catatan_prisma(S, log)
    info.update(n_baris=len(df), n_prisma=df.point_id.nunique(),
                n_segmen=df.kunci.nunique(), n_siklus=df.cycle.nunique(),
                t_awal=df.ts.min(), t_akhir=df.ts.max(),
                durasi_hari=float((df.ts.max() - df.ts.min()).total_seconds() / 86400))
    return {"df": df, "harian": H, "geo": geo, "ringkasan": S, "log": log,
            "info": info, "datum": dat, "diurnal": uji_diurnal(df),
            "berkas": info_berkas, "config": cfg, "versi": VERSI}
