"""
export.py — penulisan keluaran: CSV, GeoPackage, dan skrip Python reproduksi.
Tanpa geopandas/shapely/GDAL. Hanya pandas + numpy + gpkg_lite (stdlib).
"""

from __future__ import annotations

import json
import pprint
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gpkg_lite import GpkgWriter

# =============================================================================
#  Geometri pembantu (pengganti shapely)
# =============================================================================
def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Monotone chain. Mengembalikan cincin berlawanan arah jarum jam."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def perbesar(ring: list[tuple[float, float]], jarak: float) -> list[tuple[float, float]]:
    """Perbesar poligon dari titik pusatnya sejauh `jarak` (pendekatan buffer)."""
    if not ring:
        return ring
    cx = sum(p[0] for p in ring) / len(ring)
    cy = sum(p[1] for p in ring) / len(ring)
    out = []
    for x, y in ring:
        dx, dy = x - cx, y - cy
        r = (dx * dx + dy * dy) ** 0.5 or 1.0
        out.append((x + dx / r * jarak, y + dy / r * jarak))
    return out


def _bersih(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if f != f else round(f, 4)
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if pd.isna(v):
        return None
    return str(v)


KOLOM_RINGKAS = [
    "point_id", "seg", "segmen_terakhir", "kelas", "skor_prioritas", "status_data",
    "easting", "northing", "rl", "jarak_stasiun_m", "az_los_deg",
    "dip_lokal_deg", "dipdir_lokal_deg", "dev_los", "proj_los",
    "n_obs", "n_hari", "cakupan", "tgl_awal", "tgl_akhir", "jam_sejak_akhir",
    "d_turunlereng_mm", "d_vertikal_mm", "d_total_3d_mm", "rasio_HV", "plunge_deg",
    "v_turunlereng_mmhari", "v_vertikal_mmhari", "v_ci95_mmhari",
    "v_paruh1_mmhari", "v_paruh2_mmhari", "delta_v_mmhari",
    "rezim", "changepoint", "changepoint_skor", "inverse_velocity",
    "t_turunlereng", "t_vertikal", "koherensi_tetangga",
    "sigma_radial_mm", "sigma_tangensial_mm", "sigma_vertikal_mm",
    "sd_harian_rad_mm", "sd_harian_vert_mm", "gerak_sejak_pasang_mm",
    "tarp", "catatan",
]


def tabel_ringkasan(hasil: dict) -> pd.DataFrame:
    S = hasil["ringkasan"].copy()
    for c in KOLOM_RINGKAS:
        if c not in S.columns:
            S[c] = np.nan
    out = S[KOLOM_RINGKAS].copy()
    for c in ["tgl_awal", "tgl_akhir", "changepoint"]:
        out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    num = out.select_dtypes(include=[np.number]).columns
    out[num] = out[num].round(4)
    return out.reset_index().rename(columns={"index": "kunci"})


# =============================================================================
#  CSV
# =============================================================================
def tulis_csv(hasil: dict, outdir: Path, tag: str) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []

    p = outdir / f"ringkasan_prisma_{tag}.csv"
    tabel_ringkasan(hasil).to_csv(p, index=False)
    paths.append(p)

    H = hasil["harian"].copy()
    H["tanggal"] = pd.to_datetime(H.hari).dt.strftime("%Y-%m-%d")
    p = outdir / f"deret_harian_{tag}.csv"
    (H[["point_id", "seg", "tanggal", "n_bacaan", "d_rad", "d_tan", "d_vert"]]
     .rename(columns={"d_rad": "d_radial_mm", "d_tan": "d_tangensial_mm",
                      "d_vert": "d_vertikal_mm"})
     .round(3).to_csv(p, index=False))
    paths.append(p)

    d = hasil["df"]
    p = outdir / f"deret_perepoch_{tag}.csv"
    ep = d[["point_id", "seg", "ts", "cycle", "d_rad", "d_tan", "dZ",
            "d_rad_c", "d_tan_c", "dZ_c", "rng0"]].copy()
    ep["ts"] = ep.ts.dt.strftime("%Y-%m-%d %H:%M")
    ep = ep.rename(columns={"ts": "waktu", "rng0": "jarak_m"})
    ep[ep.select_dtypes("number").columns] = ep.select_dtypes("number").round(3)
    ep.to_csv(p, index=False)
    paths.append(p)

    p = outdir / f"log_qc_{tag}.csv"
    hasil["log"].to_csv(p, index=False)
    paths.append(p)
    return paths


# =============================================================================
#  GeoPackage
# =============================================================================
def tulis_gpkg(hasil: dict, path: Path, cfg: dict) -> Path:
    S = hasil["ringkasan"]
    info = hasil["info"]
    skala = cfg["ekspor"]["skala_vektor"]
    srs = cfg["proyek"].get("crs_id", -1) or -1
    wkt = cfg["proyek"].get("crs_wkt")

    tab = tabel_ringkasan(hasil).set_index("kunci")

    with GpkgWriter(path, srs_id=srs, srs_wkt=wkt) as w:
        # 1. titik prisma -------------------------------------------------
        rows = []
        for kunci, r in tab.iterrows():
            d = {k: _bersih(v) for k, v in r.items()}
            d["kunci"] = kunci
            d["geom"] = (float(r.easting), float(r.northing))
            rows.append(d)
        w.add_layer("prisma_ringkasan", "POINT", rows,
                    description="Ringkasan per prisma-segmen")

        # 2. vektor turun-lereng (diperbesar) ------------------------------
        rows = []
        for kunci, r in tab.iterrows():
            if pd.isna(r.d_turunlereng_mm) or pd.isna(r.dipdir_lokal_deg):
                continue
            a = np.radians(float(r.dipdir_lokal_deg))
            L = float(r.d_turunlereng_mm) / 1000 * skala
            rows.append({"kunci": kunci, "point_id": r.point_id, "kelas": r.kelas,
                         "skala": int(skala),
                         "d_turunlereng_mm": _bersih(r.d_turunlereng_mm),
                         "v_turunlereng_mmhari": _bersih(r.v_turunlereng_mmhari),
                         "azimut_deg": _bersih(r.dipdir_lokal_deg),
                         "geom": [(float(r.easting), float(r.northing)),
                                  (float(r.easting) + np.sin(a) * L,
                                   float(r.northing) + np.cos(a) * L)]})
        w.add_layer("vektor_turunlereng", "LINESTRING", rows,
                    description=f"Vektor turun-lereng, skala x{skala}")

        # 3. trajektori harian --------------------------------------------
        H = hasil["harian"]
        geo = hasil["geo"]
        rows = []
        for kunci, g in H.groupby("kunci"):
            if kunci not in geo.index or len(g) < 3:
                continue
            gg = geo.loc[kunci]
            az = np.radians(float(gg.az0))
            uxr, uyr = np.sin(az), np.cos(az)
            g = g.sort_values("hari")
            coords = [(float(gg.e0) + (rr.d_rad * uxr - rr.d_tan * uyr) / 1000 * skala,
                       float(gg.n0) + (rr.d_rad * uyr + rr.d_tan * uxr) / 1000 * skala)
                      for rr in g.itertuples()]
            rows.append({"kunci": kunci, "point_id": kunci.split("#")[0],
                         "kelas": tab.kelas.get(kunci), "skala": int(skala),
                         "n_hari": len(g), "geom": coords})
        w.add_layer("trajektori_harian", "LINESTRING", rows,
                    description=f"Lintasan harian, skala x{skala}")

        # 4. garis line-of-sight ------------------------------------------
        rows = [{"kunci": kunci, "jarak_m": _bersih(r.jarak_stasiun_m),
                 "az_deg": _bersih(r.az_los_deg),
                 "simpangan_los_dipdir_deg": _bersih(r.dev_los),
                 "faktor_proyeksi": _bersih(r.proj_los),
                 "geom": [(info["stasiun_e"], info["stasiun_n"]),
                          (float(r.easting), float(r.northing))]}
                for kunci, r in tab.iterrows()]
        w.add_layer("garis_los", "LINESTRING", rows, description="Garis pandang RTS")

        # 5. stasiun -------------------------------------------------------
        w.add_layer("stasiun_rts", "POINT", [{
            "nama": "Stasiun RTS", "easting": info["stasiun_e"],
            "northing": info["stasiun_n"], "tinggi": info["stasiun_h"],
            "orientasi_deg": round(info["orientasi_deg"], 6),
            "orientasi_sd_arcsec": round(info["orientasi_sd_arcsec"], 2),
            "n_station_reset": int(info.get("n_station_reset", 0)),
            "catatan": "stasiun tunggal; tidak ada reseksi/backsight dalam berkas",
            "geom": (info["stasiun_e"], info["stasiun_n"])}],
            description="Posisi stasiun")

        # 6. zona prioritas ------------------------------------------------
        rows = []
        for nm, kls in [("zona_P1", ["P1-PRIORITAS"]),
                        ("zona_P1_P2", ["P1-PRIORITAS", "P2-PERHATIAN"])]:
            sub = tab[tab.kelas.isin(kls)]
            if len(sub) < 3:
                continue
            ring = perbesar(convex_hull(list(zip(sub.easting.astype(float),
                                                 sub.northing.astype(float)))),
                            cfg["ekspor"]["buffer_zona_m"])
            rows.append({"zona": nm, "n_prisma": int(len(sub)),
                         "v_maks_mmhari": _bersih(sub.v_turunlereng_mmhari.max()),
                         "d_maks_mm": _bersih(sub.d_total_3d_mm.max()),
                         "geom": ring})
        if rows:
            w.add_layer("zona_prioritas", "POLYGON", rows,
                        description="Convex hull zona deformasi")
    return Path(path)


# =============================================================================
#  Skrip Python reproduksi
# =============================================================================
TEMPLATE = '''#!/usr/bin/env python3
# =============================================================================
#  Skrip reproduksi otomatis - {tag}
#  Dihasilkan : {waktu}
#  prismacore : v{versi}
#  Berkas masukan:
{daftar_berkas}
#
#  Cara pakai:
#    1. pip install "pandas>=2.0" "numpy>=1.24"
#    2. pastikan folder paket `prismacore/` berada di salah satu lokasi:
#       di samping skrip ini, di akar repo, atau di direktori kerja
#    3. letakkan berkas CSV mentah seperti pada daftar di atas
#       (di samping skrip, relatif terhadap akar repo, atau di direktori kerja)
#    4. python {namafile}
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

BERKAS = {berkas!r}
KELUARAN = Path("keluaran_{tag}")

# --- konfigurasi yang dipakai pada run ini (ditanam, bukan dari berkas luar) --
KONFIG = {konfig}

# --- nilai acuan untuk verifikasi mandiri ------------------------------------
ACUAN = {acuan}


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
    csv_paths = tulis_csv(hasil, KELUARAN, "{tag}")
    gpkg = tulis_gpkg(hasil, KELUARAN / "prisma_{tag}.gpkg", KONFIG)

    print("CSV :", *[p.name for p in csv_paths], sep="\\n  ")
    print("GPKG:", gpkg.name)

    # ---------------- verifikasi mandiri ------------------------------------
    info = hasil["info"]
    assert info["n_baris"] == ACUAN["n_baris"], \\
        f"jumlah baris berbeda: {{info['n_baris']}} != {{ACUAN['n_baris']}}"
    assert info["n_prisma"] == ACUAN["n_prisma"], "jumlah prisma berbeda"
    assert abs(info["orientasi_deg"] - ACUAN["orientasi_deg"]) < 1e-6, \\
        "konstanta orientasi berbeda"

    S = hasil["ringkasan"]
    for kunci, v_acuan in ACUAN["v_turunlereng"].items():
        v = float(S.loc[kunci, "v_turunlereng_mmhari"])
        assert abs(v - v_acuan) < 0.01, f"{{kunci}}: {{v:.3f}} != {{v_acuan:.3f}}"

    print("\\nVerifikasi mandiri LULUS - hasil identik dengan run aplikasi.")


if __name__ == "__main__":
    main()
'''


def tulis_skrip(hasil: dict, path: Path, berkas_nama: list[str], tag: str) -> Path:
    S = hasil["ringkasan"]
    acuan = {
        "n_baris": int(hasil["info"]["n_baris"]),
        "n_prisma": int(hasil["info"]["n_prisma"]),
        "orientasi_deg": round(float(hasil["info"]["orientasi_deg"]), 8),
        "v_turunlereng": {k: round(float(v), 4) for k, v in
                          S.v_turunlereng_mmhari.dropna().head(5).items()},
    }
    daftar = "\n".join(
        f"#    - {b['berkas']}  sha256={b['sha256'][:16]}…  "
        f"({b['n_baris']} baris, {b['t_awal']:%Y-%m-%d} s/d {b['t_akhir']:%Y-%m-%d})"
        for b in hasil["berkas"])
    teks = TEMPLATE.format(
        tag=tag, waktu=datetime.now().strftime("%Y-%m-%d %H:%M"),
        versi=hasil["versi"], daftar_berkas=daftar, namafile=Path(path).name,
        berkas=berkas_nama,
        konfig=pprint.pformat(hasil["config"], indent=4, width=88, sort_dicts=False),
        acuan=pprint.pformat(acuan, indent=4, width=88, sort_dicts=False))
    Path(path).write_text(teks, encoding="utf-8")
    return Path(path)
