"""
app.py — antarmuka Streamlit / stlite untuk analisis prisma RTS.

Dua cara menjalankan:
  1. Lokal :  streamlit run app/app.py
  2. GitHub Pages (stlite, 100% di browser): app/index.html

Dependensi: streamlit (hanya lokal), pandas, numpy. Tanpa GDAL.
Data dapat diunggah dari komputer atau diambil dari folder `data/` di repo
(dibaca lewat raw.githubusercontent, tanpa token dan tanpa API GitHub).
"""

from __future__ import annotations

import copy
import io
import json
import os
import shutil
import tempfile
import zipfile
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import prismacore as pc
from prismacore.export import tabel_ringkasan, tulis_csv, tulis_gpkg, tulis_skrip

# --- alamat data repo (ubah bila repo/akun berbeda; dapat di-override env) ---
REPO = "fanelix/prisma"
URL_REPO = os.environ.get(
    "PRISMA_URL_DATA", f"https://raw.githubusercontent.com/{REPO}/main/data/")
KELUARAN = Path(tempfile.gettempdir()) / "keluaran"

st.set_page_config(page_title="Analisis Prisma RTS", layout="wide",
                   initial_sidebar_state="expanded")

WARNA = {"P1-PRIORITAS": "#c0272d", "P2-PERHATIAN": "#e08214",
         "P3-PANTAU RUTIN": "#3b7dd8", "P4-STABIL": "#7f8c8d"}


@st.cache_data(show_spinner="Menjalankan pipeline…", max_entries=4)
def _jalankan(payload: list[tuple[str, bytes]], cfg_json: str) -> dict:
    return pc.jalankan(payload, json.loads(cfg_json))


@st.cache_data(show_spinner=False, ttl=600)
def _daftar_repo() -> list[dict]:
    """Daftar berkas dari data/manifest.json (tanpa API GitHub, tanpa token)."""
    import urllib.request
    try:
        with urllib.request.urlopen(URL_REPO + "manifest.json", timeout=20) as r:
            return json.loads(r.read().decode("utf-8")).get("berkas", [])
    except Exception:  # noqa: BLE001 - offline / repo belum punya manifest
        return []


@st.cache_data(show_spinner=False, ttl=600)
def _ambil_repo(nama: str) -> bytes:
    import urllib.request
    with urllib.request.urlopen(URL_REPO + nama, timeout=120) as r:
        return r.read()


def _konfigurasi_awal() -> dict:
    """Konfigurasi bawaan, ditimpa konfigurasi situs bila berkasnya tersedia."""
    for jalur in ("konfigurasi/candrian.json", "./konfigurasi/candrian.json"):
        try:
            return pc.muat_konfigurasi(jalur)
        except (FileNotFoundError, OSError):
            continue
    return copy.deepcopy(pc.CONFIG)


# =============================================================================
#  SIDEBAR — SUMBER DATA
# =============================================================================
st.sidebar.title("Analisis Prisma RTS")
st.sidebar.caption(f"prismacore v{pc.VERSI} · tanpa GDAL · tanpa server")

sumber = st.sidebar.radio("Sumber data", ["Unggah berkas", "Ambil dari repo"],
                          horizontal=True)

payload: list[tuple[str, bytes]] = []
if sumber == "Unggah berkas":
    unggah = st.sidebar.file_uploader(
        "Berkas CSV ekspor RTS", type=["csv"], accept_multiple_files=True,
        help="Boleh lebih dari satu berkas (mis. ekspor per bulan). "
             "Baris ganda antar berkas dibuang otomatis. "
             "Berkas sangat besar (>50 MB) menguras memori tab browser.")
    payload = [(f.name, f.getvalue()) for f in unggah]
else:
    daftar = _daftar_repo()
    if not daftar:
        st.sidebar.warning(
            "Tidak dapat membaca `data/manifest.json` dari repo. "
            "Gunakan mode unggah berkas, atau periksa koneksi.")
    else:
        label = {f'{d["nama"]}  ({d["periode"]})': d["nama"] for d in daftar}
        pilih = st.sidebar.multiselect("Berkas di repo", list(label),
                                       default=list(label)[:1])
        for teks in pilih:
            with st.sidebar:
                with st.spinner(f"Mengunduh {label[teks]}…"):
                    payload.append((label[teks], _ambil_repo(label[teks])))

if not payload:
    st.title("Analisis Pemantauan Prisma RTS")
    st.markdown(
        """
Pilih sumber data di panel kiri untuk memulai: unggah CSV ekspor RTS, atau
ambil langsung dari folder `data/` di repo GitHub.

**Keluaran yang dihasilkan**

| Berkas | Isi |
|---|---|
| `ringkasan_prisma_*.csv` | Metrik per prisma-segmen: perpindahan, kecepatan, H/V, signifikansi, kelas prioritas |
| `deret_harian_*.csv` | Deret waktu harian terkoreksi |
| `deret_perepoch_*.csv` | Data per epoch, mentah **dan** terkoreksi, untuk audit |
| `log_qc_*.csv` | Setiap temuan QC beserta tindakannya |
| `prisma_*.gpkg` | GeoPackage 6 layer siap dibuka di QGIS |
| `analisis_*.py` | Skrip Python mandiri yang mereproduksi run ini dan memverifikasi dirinya sendiri |

**Catatan penting**

- Aplikasi ini **tidak menyatakan lereng aman atau tidak aman**. Kelas P1–P4 bersifat
  relatif terhadap dataset, bukan TARP situs.
- Koreksi datum **tidak diterapkan** sampai Anda memilih prisma referensi.
  Aplikasi hanya memberi saran; keputusan tetap di tangan Anda.
- Muat awal stlite 30–60 detik (unduh Pyodide + pandas, setelah itu di-cache
  browser). Seluruh proses berjalan di browser; data tidak dikirim ke mana pun.
        """)
    st.stop()

# ---- konfigurasi ------------------------------------------------------------
cfg = _konfigurasi_awal()

with st.sidebar.expander("Segmentasi & QC", expanded=False):
    cfg["segmentasi"]["gap_segmen_jam"] = st.number_input(
        "Gap pemicu segmen baru (jam)", 12, 720, int(cfg["segmentasi"]["gap_segmen_jam"]), 12,
        help="Gap lebih panjang dari ini memulai segmen baru dengan baseline sendiri. "
             "Wajib untuk arsip panjang: prisma yang dipasang ulang tidak boleh "
             "disambung dengan baseline lama.")
    cfg["qc"]["step_mm"] = st.number_input("Ambang dasar loncatan (mm)", 5.0, 200.0,
                                           float(cfg["qc"]["step_mm"]), 5.0)
    cfg["qc"]["step_sigma"] = st.number_input("Pengali sigma loncatan", 3.0, 12.0,
                                              float(cfg["qc"]["step_sigma"]), 0.5)
    cfg["segmentasi"]["baseline_n_epoch"] = st.number_input(
        "Epoch baseline", 3, 30, int(cfg["segmentasi"]["baseline_n_epoch"]), 1)
    cfg["agregasi"]["min_bacaan_harian"] = st.number_input(
        "Min bacaan per hari", 1, 24, int(cfg["agregasi"]["min_bacaan_harian"]), 1)
    cfg["qc"]["segmen_dari_loncatan"] = st.checkbox(
        "Reset baseline otomatis saat dugaan pemasangan ulang",
        value=bool(cfg["qc"]["segmen_dari_loncatan"]),
        help="Bila aktif, pergeseran level 24 jam di atas ambang memecah segmen. "
             "Tinjau log QC setelahnya — keputusan tetap milik pengguna.")

with st.sidebar.expander("Ekspor & CRS", expanded=False):
    cfg["ekspor"]["skala_vektor"] = st.number_input(
        "Skala vektor peta", 100, 20000, int(cfg["ekspor"]["skala_vektor"]), 100,
        help="1 mm perpindahan digambar sepanjang nilai ini dalam meter/1000.")
    pakai_crs = st.checkbox("Tetapkan CRS (grid tambang)", value=False)
    if pakai_crs:
        cfg["proyek"]["crs_id"] = st.number_input("srs_id", -1, 999999, 32749)
        cfg["proyek"]["crs_wkt"] = st.text_area("WKT definisi CRS", height=80) or None
        st.caption("Kosongkan bila memakai grid tambang lokal. Jangan menebak EPSG: "
                   "Easting grid Candrian tidak sesuai zona UTM manapun.")

# ---- lintasan pertama: dapatkan saran referensi ------------------------------
try:
    pra = _jalankan(payload, json.dumps(cfg))
except Exception as e:  # noqa: BLE001
    st.error(f"Gagal membaca berkas: {e}")
    st.exception(e)
    st.stop()

saran = pra["datum"].get("saran_referensi", pd.DataFrame())
opsi = sorted(pra["df"].point_id.unique())
default_ref = (saran.kunci.str.split("#").str[0].head(3).tolist()
               if len(saran) else [])

st.sidebar.markdown("### Prisma referensi (datum)")
refs = st.sidebar.multiselect(
    "Pilih prisma yang dianggap stabil", opsi,
    default=[r for r in cfg["datum"]["referensi"] if r in opsi],
    help="Koreksi hanyutan datum hanya diterapkan bila referensi dipilih. "
         "Prisma referensi idealnya berada DI LUAR zona timbunan yang bergerak.")
if not refs:
    st.sidebar.warning(
        "Belum ada referensi. Koreksi datum tidak diterapkan — perpindahan bersifat "
        "relatif terhadap stasiun, bukan terhadap kerangka stabil.")

with st.sidebar.expander("Bobot skor prioritas", expanded=False):
    b = cfg["skor"]["bobot"]
    for k, label in [("mag", "Magnitudo 3D"), ("vel", "Kecepatan"),
                     ("acc", "Akselerasi"), ("sig", "Signifikansi"),
                     ("vert", "Komponen vertikal"), ("coh", "Koherensi")]:
        b[k] = st.slider(label, 0.0, 0.5, float(b[k]), 0.02)
    tot = sum(b.values()) or 1.0
    for k in b:
        b[k] /= tot

cfg["datum"]["referensi"] = refs
try:
    hasil = _jalankan(payload, json.dumps(cfg))
except Exception as e:  # noqa: BLE001
    st.error(f"Pipeline gagal: {e}")
    st.exception(e)
    st.stop()

S = hasil["ringkasan"]
info = hasil["info"]
dat = hasil["datum"]
tab_ringkas = tabel_ringkasan(hasil)

# =============================================================================
#  HALAMAN
# =============================================================================
st.title("Analisis Pemantauan Prisma RTS")

c = st.columns(6)
c[0].metric("Prisma", info["n_prisma"])
c[1].metric("Segmen", info["n_segmen"])
c[2].metric("Baris", f"{info['n_baris']:,}")
c[3].metric("Durasi", f"{info['durasi_hari']:.1f} hari")
c[4].metric("P1 Prioritas", int((S.kelas == "P1-PRIORITAS").sum()))
c[5].metric("Temuan QC", len(hasil["log"]))

t1, t2, t3, t4, t5 = st.tabs(
    ["Peringkat", "QC & Datum", "Deret waktu", "Peta", "Unduh"])

# ---------------------------------------------------------------- Peringkat
with t1:
    f1, f2, f3 = st.columns([2, 2, 3])
    kelas_pilih = f1.multiselect("Kelas", list(WARNA), default=list(WARNA)[:2])
    status_pilih = f2.multiselect("Status data", sorted(S.status_data.unique()),
                                  default=sorted(S.status_data.unique()))
    hanya_akhir = f3.checkbox("Hanya segmen terakhir tiap prisma", value=True)

    V = tab_ringkas.copy()
    if kelas_pilih:
        V = V[V.kelas.isin(kelas_pilih)]
    if status_pilih:
        V = V[V.status_data.isin(status_pilih)]
    if hanya_akhir:
        V = V[V.segmen_terakhir == True]  # noqa: E712

    kolom = ["point_id", "seg", "kelas", "skor_prioritas", "status_data", "rl",
             "d_turunlereng_mm", "d_vertikal_mm", "d_total_3d_mm", "rasio_HV",
             "plunge_deg", "v_turunlereng_mmhari", "delta_v_mmhari", "rezim",
             "t_turunlereng", "koherensi_tetangga", "n_hari", "catatan"]
    st.dataframe(V[kolom], use_container_width=True, height=460, hide_index=True)

    st.caption(
        "**Kelas P1–P4 adalah peringkat relatif di dalam dataset ini, bukan TARP situs.** "
        "Ambang TARP harus berasal dari dokumen kendali lereng dan dimasukkan terpisah. "
        "`rasio_HV` dikosongkan bila komponen vertikal berada di bawah derau harian.")

    if len(V):
        st.bar_chart(V.set_index("point_id")[["v_turunlereng_mmhari"]].head(25),
                     height=280)

# ------------------------------------------------------------- QC & Datum
with t2:
    k1, k2 = st.columns(2)
    with k1:
        st.subheader("Konsistensi geometri")
        st.dataframe(pd.DataFrame({
            "besaran": ["Konstanta orientasi", "Sebaran orientasi antar-epoch",
                        "Sisa rekonstruksi horizontal", "Sisa rekonstruksi vertikal",
                        "Dugaan setup ulang stasiun", "Siklus terdeteksi"],
            "nilai": [f"{info['orientasi_deg']:.6f}°  ({info['orientasi_deg']*3600:.2f}″)",
                      f"{info['orientasi_sd_arcsec']:.2f}″",
                      f"{info['sisa_rekon_h_mm_sd']:.2f} mm",
                      f"{info['sisa_rekon_z_mm_sd']:.2f} mm",
                      str(info.get("n_station_reset", 0)),
                      str(info["n_siklus"])]}),
            hide_index=True, use_container_width=True)

        st.subheader("Koreksi datum")
        if dat.get("koreksi_diterapkan"):
            st.success("Koreksi diterapkan — referensi: " + ", ".join(dat["referensi"]))
            st.dataframe(pd.DataFrame({
                "komponen": ["Hanyutan skala jarak", "Hanyutan vertikal",
                             "Koreksi orientasi (median)"],
                "nilai": [f"{dat['ppm_per_hari']:+.3f} ppm/hari "
                          f"(CI95 {dat['ppm_ci'][0]:+.3f}…{dat['ppm_ci'][1]:+.3f})",
                          f"{dat['vert_mm_per_hari']:+.3f} mm/hari "
                          f"(CI95 {dat['vert_ci'][0]:+.3f}…{dat['vert_ci'][1]:+.3f})",
                          f"{dat['orientasi_median_arcsec']:+.2f}″"]}),
                hide_index=True, use_container_width=True)
            if dat["ref_ditolak"]:
                st.error("Referensi ditolak karena ikut bergerak: "
                         + ", ".join(dat["ref_ditolak"]))
        else:
            st.warning("Koreksi datum TIDAK diterapkan — belum ada prisma referensi.")

    with k2:
        st.subheader("Saran kandidat referensi")
        st.caption("Diurutkan dari yang paling tenang. **Periksa lokasinya** — "
                   "prisma tenang yang berada di dalam timbunan bukan referensi yang sah.")
        if len(saran):
            st.dataframe(saran, hide_index=True, use_container_width=True)

        st.subheader("Siklus diurnal")
        d = hasil["diurnal"]
        if len(d):
            st.line_chart(d[["radial_mm", "tangensial_mm", "vertikal_mm"]], height=220)
            st.caption(
                f"Amplitudo: radial {d.radial_mm.max()-d.radial_mm.min():.2f} mm · "
                f"tangensial {d.tangensial_mm.max()-d.tangensial_mm.min():.2f} mm · "
                f"vertikal {d.vertikal_mm.max()-d.vertikal_mm.min():.2f} mm. "
                "Sinyal pada sudut tetapi tidak pada jarak menunjuk ke rotasi termal "
                "instrumen atau refraksi lateral, bukan atmosfer EDM.")

    st.subheader("Log QC")
    st.dataframe(hasil["log"], use_container_width=True, hide_index=True, height=260)

    st.subheader("Ketersediaan data per hari")
    av = hasil["df"].groupby("hari").point_id.nunique()
    st.bar_chart(av, height=200)

# ------------------------------------------------------------ Deret waktu
with t3:
    H = hasil["harian"]
    default = S.head(6).point_id.tolist()
    pilih = st.multiselect("Prisma", opsi, default=default)
    komp = st.radio("Komponen", ["Turun-lereng", "Vertikal", "Tangensial"],
                    horizontal=True)
    kolom_map = {"Turun-lereng": "d_rad", "Vertikal": "d_vert", "Tangensial": "d_tan"}
    kol = kolom_map[komp]

    sub = H[H.point_id.isin(pilih)].copy()
    if komp == "Turun-lereng":
        cosdev = hasil["geo"].dev_los.map(lambda x: np.cos(np.radians(x)))
        sub["nilai"] = -sub[kol] / sub.kunci.map(cosdev)
    else:
        sub["nilai"] = sub[kol]
    if len(sub):
        piv = sub.pivot_table(index="hari", columns="kunci", values="nilai")
        st.line_chart(piv, height=420)
        st.caption(f"{komp} kumulatif (mm), rerata harian, terkoreksi datum. "
                   "Segmen berbeda dari prisma yang sama tampil sebagai garis terpisah "
                   "— perpindahan antar segmen tidak disambung.")
    else:
        st.info("Pilih minimal satu prisma.")

# ------------------------------------------------------------------- Peta
with t4:
    m = tab_ringkas.dropna(subset=["easting", "northing"]).copy()
    m["ukuran"] = (m.d_total_3d_mm.fillna(0).clip(0, 25) + 4) * 6
    m["warna"] = m.kelas.map(WARNA)
    st.scatter_chart(m, x="easting", y="northing", color="warna",
                     size="ukuran", height=520)
    st.caption("Ukuran titik sebanding dengan perpindahan 3D. "
               "Peta vektor lengkap dengan panah tersedia di berkas GeoPackage.")
    st.dataframe(m[["point_id", "kelas", "easting", "northing", "rl",
                    "d_total_3d_mm", "v_turunlereng_mmhari"]],
                 hide_index=True, use_container_width=True, height=240)

# ------------------------------------------------------------------ Unduh
with t5:
    tag = st.text_input("Label keluaran", value=pd.Timestamp(info["t_akhir"]).strftime("%Y%m"))
    tag = "".join(ch for ch in tag if ch.isalnum() or ch in "_-") or "hasil"

    if st.button("Siapkan berkas keluaran", type="primary"):
        with st.spinner("Menulis CSV, GeoPackage, dan skrip Python…"):
            if KELUARAN.exists():
                shutil.rmtree(KELUARAN)
            KELUARAN.mkdir(parents=True, exist_ok=True)
            csvs = tulis_csv(hasil, KELUARAN, tag)
            gpkg = tulis_gpkg(hasil, KELUARAN / f"prisma_{tag}.gpkg", cfg)
            skrip = tulis_skrip(hasil, KELUARAN / f"analisis_{tag}.py",
                                [n for n, _ in payload], tag)

            # sertakan paket prismacore + requirements agar skrip reproduksi
            # dapat dijalankan setelah ZIP diekstrak, tanpa aplikasi ini
            paket = KELUARAN / "prismacore"
            paket.mkdir(exist_ok=True)
            for nama in ("__init__.py", "core.py", "robust.py", "gpkg_lite.py",
                         "export.py", "konfigurasi_bawaan.json"):
                (paket / nama).write_bytes(
                    resources.files("prismacore").joinpath(nama).read_bytes())
            (KELUARAN / "requirements-core.txt").write_text(
                'pandas>=2.0\nnumpy>=1.24\n', encoding="utf-8")

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for p in sorted(KELUARAN.rglob("*")):
                    if p.is_file():
                        z.write(p, p.relative_to(KELUARAN))
            st.session_state["zip"] = buf.getvalue()
            st.session_state["gpkg"] = gpkg.read_bytes()
            st.session_state["skrip"] = skrip.read_text(encoding="utf-8")
            st.session_state["ringkas_csv"] = csvs[0].read_bytes()
            st.session_state["tag"] = tag

    if "zip" in st.session_state:
        t = st.session_state["tag"]
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Semua (ZIP)", st.session_state["zip"],
                           f"prisma_{t}.zip", "application/zip", use_container_width=True)
        d2.download_button("Ringkasan CSV", st.session_state["ringkas_csv"],
                           f"ringkasan_prisma_{t}.csv", "text/csv", use_container_width=True)
        d3.download_button("GeoPackage", st.session_state["gpkg"],
                           f"prisma_{t}.gpkg", "application/geopackage+sqlite3",
                           use_container_width=True)
        d4.download_button("Skrip Python", st.session_state["skrip"],
                           f"analisis_{t}.py", "text/x-python", use_container_width=True)

        st.success("ZIP berisi seluruh keluaran plus paket `prismacore/` dan "
                   "`requirements-core.txt` — skrip reproduksi dapat dijalankan "
                   "langsung setelah diekstrak, tanpa memasang aplikasi ini.")
        with st.expander("Pratinjau skrip reproduksi"):
            st.code(st.session_state["skrip"][:4000], language="python")
