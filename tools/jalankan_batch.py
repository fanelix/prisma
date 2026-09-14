#!/usr/bin/env python3
"""
jalankan_batch.py — jalankan pipeline pada seluruh CSV di folder data/.
=======================================================================
Dipakai GitHub Actions (`.github/workflows/analisis.yml`) atau manual:

    python tools/jalankan_batch.py \
        --config konfigurasi/candrian.json \
        --data   data/ \
        --keluar hasil/

Keluaran per run: `hasil/<tag>/` berisi CSV, GeoPackage, dan skrip reproduksi.
`data/manifest.json` ikut diperbarui (nama berkas, periode, jumlah baris) agar
aplikasi stlite di GitHub Pages dapat menampilkan daftar berkas tanpa memanggil
API GitHub. Ringkasan singkat dicetak pula ke `$GITHUB_STEP_SUMMARY`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# agar `prismacore` dapat diimpor walau skrip dipanggil dari direktori lain
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import prismacore as pc  # noqa: E402
from prismacore.export import tulis_csv, tulis_gpkg, tulis_skrip  # noqa: E402


def perbarui_manifest(datadir: Path, hasil: dict) -> dict:
    entri = [
        {"nama": b["berkas"],
         "periode": f'{b["t_awal"]:%Y-%m-%d} s/d {b["t_akhir"]:%Y-%m-%d}',
         "n_baris": int(b["n_baris"])}
        for b in sorted(hasil["berkas"], key=lambda x: x["berkas"])
    ]
    man = {"diperbarui": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "berkas": entri}
    (datadir / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return man


def ringkasan_markdown(hasil: dict, tag: str, berkas: list[Path]) -> str:
    info, S, log = hasil["info"], hasil["ringkasan"], hasil["log"]
    hitung = S.kelas.value_counts()
    v_max = S.v_turunlereng_mmhari.max()
    baris = [
        f"## Analisis prisma — {tag}",
        "",
        f"- Data: {len(berkas)} berkas · {info['n_baris']:,} baris · "
        f"{info['n_prisma']} prisma · {info['n_segmen']} segmen · "
        f"{info['n_siklus']} siklus",
        f"- Periode: {info['t_awal']:%Y-%m-%d} s/d {info['t_akhir']:%Y-%m-%d} "
        f"({info['durasi_hari']:.1f} hari)",
        "- Kelas: " + " · ".join(f"{k}: {v}" for k, v in hitung.items()),
        f"- Temuan QC: {len(log)}"
        + (f" — {', '.join(f'{j} ({n})' for j, n in log.jenis.value_counts().items())}"
           if len(log) else ""),
    ]
    if len(S):
        teratas = S.nlargest(3, "v_turunlereng_mmhari")
        baris.append(f"- Kecepatan tertinggi: "
                     + " · ".join(f"{r.point_id} {r.v_turunlereng_mmhari:.2f} mm/hari"
                                  for r in teratas.itertuples()))
        baris.append(f"- Kecepatan maksimum: {v_max:.2f} mm/hari")
    baris += ["",
              "> Kelas P1–P4 adalah peringkat relatif di dalam dataset, bukan TARP situs. "
              "Aplikasi tidak menyatakan lereng aman atau tidak aman."]
    return "\n".join(baris) + "\n"


def utama() -> None:
    ap = argparse.ArgumentParser(description="Batch analisis prisma RTS")
    ap.add_argument("--config", default=None,
                    help="JSON konfigurasi situs (mis. konfigurasi/candrian.json)")
    ap.add_argument("--data", default="data", help="Folder berisi CSV mentah")
    ap.add_argument("--keluar", default="hasil", help="Folder dasar keluaran")
    ap.add_argument("--tag", default=None, help="Label keluaran (bawaan: YYYYMM)")
    a = ap.parse_args()

    datadir = Path(a.data)
    berkas = sorted(p for p in datadir.glob("*.csv"))
    if not berkas:
        sys.exit(f"Tidak ada berkas .csv di {datadir.resolve()}")

    cfg = pc.muat_konfigurasi(a.config) if a.config else pc.muat_konfigurasi()
    hasil = pc.jalankan([str(p) for p in berkas], cfg)

    tag = a.tag or pd.Timestamp(hasil["info"]["t_akhir"]).strftime("%Y%m")
    outdir = Path(a.keluar) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    csvs = tulis_csv(hasil, outdir, tag)
    gpkg = tulis_gpkg(hasil, outdir / f"prisma_{tag}.gpkg", cfg)
    skrip = tulis_skrip(hasil, outdir / f"analisis_{tag}.py",
                        [str(p) for p in berkas], tag)
    man = perbarui_manifest(datadir, hasil)

    laporan = ringkasan_markdown(hasil, tag, berkas)
    print(laporan)
    print("Keluaran:")
    for p in (*csvs, gpkg, skrip):
        print(f"  {p}")
    print(f"  {datadir / 'manifest.json'}  ({len(man['berkas'])} berkas terdaftar)")

    tujuan = os.environ.get("GITHUB_STEP_SUMMARY")
    if tujuan:
        with open(tujuan, "a", encoding="utf-8") as f:
            f.write(laporan)


if __name__ == "__main__":
    utama()
