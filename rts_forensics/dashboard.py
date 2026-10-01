"""Six-page raw-observation dashboard, shared by Streamlit and stlite."""

import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .config import load_config
from .pipeline import run
from .report import write_results


@st.cache_data(max_entries=2, show_spinner=False)
def analyze(payload, config_json):
    return run(payload, load_config(overrides=json.loads(config_json)))


@st.cache_data(max_entries=2, show_spinner=False)
def download_bundle(result):
    with tempfile.TemporaryDirectory() as directory:
        write_results(result, directory)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in Path(directory).iterdir():
                archive.write(path, path.name)
        return buffer.getvalue()


def _table(table):
    st.dataframe(table.drop(columns=["source_refs"], errors="ignore"), hide_index=True)


def _map(result):
    import altair as alt

    s = result["prism_summary"].copy()
    st.caption("Grid input lokal (meter). Panah diperbesar untuk tampilan; koreksi bersifat eksperimental.")
    scale = st.number_input("Skala tampilan vektor", min_value=1.0, value=1000.0, step=100.0)
    s["x2"] = s.e + s.vector_e_mm / 1000 * scale
    s["y2"] = s.n + s.vector_n_mm / 1000 * scale
    point = (
        alt.Chart(s)
        .mark_circle(size=75)
        .encode(
            x=alt.X("e:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("n:Q", scale=alt.Scale(zero=False)),
            color="movement_concern:N",
            tooltip=["pid", "movement_concern", "reliability", "net_los_raw_mm", "net_los_fc_mm", "flags"],
        )
    )
    vector = alt.Chart(s).mark_rule(color="#f29d38").encode(x="e:Q", y="n:Q", x2="x2:Q", y2="y2:Q")
    layers = [point, vector]
    if st.checkbox("Tampilkan vektor artefak rotasi"):
        rows = []
        for r in s.itertuples():
            g = result["series"].query("pid == @r.pid and station == @r.station")
            a = np.radians(r.az_deg)
            effect = g.tan_raw.iloc[-1] - g.tan_fc.iloc[-1]
            rows.append(
                {
                    "e": r.e,
                    "n": r.n,
                    "x2": r.e + effect * np.cos(a) / 1000 * scale,
                    "y2": r.n - effect * np.sin(a) / 1000 * scale,
                }
            )
        layers.append(
            alt.Chart(pd.DataFrame(rows))
            .mark_rule(color="#ab6bb5", strokeDash=[4, 3])
            .encode(x="e:Q", y="n:Q", x2="x2:Q", y2="y2:Q")
        )
    if st.checkbox("Diagnostik SE fit LOS (median 1 SE; bukan ketidakpastian vektor)"):
        rows = []
        for r in s.itertuples():
            if np.isfinite(r.los_fit_se_mm):
                radius = r.los_fit_se_mm / 1000 * scale
                for i, a in enumerate(np.linspace(0, 2 * np.pi, 33)):
                    rows.append(
                        {
                            "pid": r.pid,
                            "order": i,
                            "e": r.e + radius * np.cos(a),
                            "n": r.n + radius * np.sin(a),
                        }
                    )
        if rows:
            layers.append(
                alt.Chart(pd.DataFrame(rows))
                .mark_line(color="#999")
                .encode(x="e:Q", y="n:Q", detail="pid:N", order="order:Q")
            )
    st.altair_chart(alt.layer(*layers).properties(height=520).interactive(), use_container_width=True)
    st.caption(
        "LOS dan tangensial memakai geometri garis pandang yang berbeda. Lingkaran hanya median SE prediksi LOS; gunakan pengukuran kontrol eksternal untuk ketidakpastian absolut."
    )
    _table(s[["pid", "movement_concern", "reliability", "night_success", "lost_final_48h", "flags"]])


def render(list_repo=None, fetch_repo=None):
    st.sidebar.title("RTS Forensics")
    st.sidebar.caption("Raw D/Hz/V · eksperimental · tanpa alarm bawaan")
    source = st.sidebar.radio("Sumber forensik", ["Unggah CSV", "Data repo"])
    payload = []
    if source == "Unggah CSV":
        files = st.sidebar.file_uploader("Ekspor RTS", type=["csv"], accept_multiple_files=True)
        payload = [(f.name, f.getvalue()) for f in files]
    elif list_repo and fetch_repo:
        choices = {d["nama"]: d for d in list_repo()}
        names = st.sidebar.multiselect("Berkas repo", list(choices))
        try:
            payload = [(name, fetch_repo(name)) for name in names]
        except OSError as error:
            st.sidebar.error(str(error))
    cfg_file = st.sidebar.file_uploader("Konfigurasi YAML / JSON (opsional)", type=["yaml", "yml", "json"])
    cfg = load_config()
    try:
        if cfg_file:
            if cfg_file.name.endswith(".json"):
                cfg = load_config(overrides=json.loads(cfg_file.getvalue()))
            else:
                import yaml

                cfg = load_config(overrides=yaml.safe_load(cfg_file.getvalue()))
    except (ValueError, TypeError) as error:
        st.error(f"Konfigurasi: {error}")
        return
    include = st.sidebar.text_input(
        "Frame set eksplisit (ID dipisah koma)", value=",".join(cfg["frame"]["include"])
    )
    cfg["frame"]["include"] = [x.strip() for x in include.split(",") if x.strip()]
    st.sidebar.caption(
        "Tanpa daftar/policy A–D, frame otomatis bersifat provisional. Tidak ada batas TARP yang diasumsikan."
    )
    page = st.sidebar.radio(
        "Halaman",
        ["Upload & audit", "Station frame", "Prism explorer", "Map", "Events & investigations", "Downloads"],
    )
    st.title("RTS observation forensics")
    st.caption(
        "Movement concern dan data reliability ditampilkan terpisah. Nilai raw selalu tersedia bersama koreksi frame."
    )
    identity = hashlib.sha256(
        b"".join(hashlib.sha256(b).digest() for _, b in payload) + json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()
    if st.sidebar.button("Jalankan forensik", disabled=not payload):
        try:
            with st.spinner("Mengaudit observasi dan merekonstruksi frame…"):
                st.session_state["forensic_result"] = analyze(payload, json.dumps(cfg))
                st.session_state["forensic_identity"] = identity
        except (ValueError, OSError) as error:
            st.error(str(error))
            return
    result = st.session_state.get("forensic_result")
    if result is None:
        st.info(
            "Unggah ekspor atau pilih data repo, lalu jalankan forensik. Ulang observasi dipertahankan; tidak ada status alarm tanpa TARP situs."
        )
        return
    if payload and st.session_state.get("forensic_identity") != identity:
        st.warning(
            "Sumber/konfigurasi berubah. Hasil di bawah berasal dari run sebelumnya; jalankan ulang untuk memperbarui."
        )
    with st.expander("Batas interpretasi", expanded=False):
        for warning in result["warnings"]:
            st.write("• " + warning)
    summary = result["prism_summary"]
    if page == "Upload & audit":
        _table(pd.DataFrame(result["inputs"]))
        _table(
            summary[
                [
                    "station",
                    "pid",
                    "movement_concern",
                    "reliability",
                    "coverage",
                    "longest_gap_hours",
                    "n_cycles",
                    "night_success",
                    "flags",
                ]
            ]
        )
        st.subheader("Baris gagal parse (disimpan)")
        _table(result["observations"].loc[result["observations"].parse_error])
        st.subheader("Cycle & repeat diagnostics")
        _table(result["cycle_table"])
        if "repeatability" in result:
            st.caption("Pooled within-cycle repeat SD (repeats kept as real measurements)")
            _table(result["repeatability"])
        _table(result["noise"])
    elif page == "Station frame":
        station = st.selectbox("Station", list(summary.station.unique()))
        f = result["frame_cycles"].query("station == @station").set_index("ts")
        if "vertical_model" in f and len(f):
            st.caption(
                f"Vertical frame model: {f.vertical_model.iloc[0]} ({f.vertical_model_selection.iloc[0]}). "
                f"Full model: {int(f.full_vertical_targets_over_sigma_v.iloc[0])} target(s) with a-priori "
                f"correction SE above sigma_v."
            )
        for columns in [
            ["rotation_arcsec"],
            ["translation_e_mm", "translation_n_mm", "scale_ppm"],
            ["height_mm", "exported_h_mm"],
            ["orientation_shift_arcsec", "applied_minus_fitted_arcsec"],
        ]:
            st.line_chart(f[columns])
        _table(f.reset_index())
        st.subheader("Frame membership dan alasan eksklusi")
        _table(result["frame_members"].query("station == @station"))
    elif page == "Prism explorer":
        options = summary.station + ": " + summary.pid
        choice = st.selectbox("Prism / station", options.tolist())
        station, pid = choice.split(": ", 1)
        g = result["series"].query("station == @station and pid == @pid")
        r = summary.query("station == @station and pid == @pid")
        _table(
            r[
                [
                    "movement_concern",
                    "raw_los_concern",
                    "vertical_concern",
                    "reliability",
                    "tarp_status",
                    "flags",
                ]
            ]
        )
        split = st.radio("Sampling", ["All", "Night only", "Day only"], horizontal=True)
        plot = g if split == "All" else g[g.night if split == "Night only" else ~g.night]
        for columns in [["los_raw", "los_fc"], ["ver_raw", "ver_fc"], ["tan_raw", "tan_fc"]]:
            st.line_chart(plot.set_index("ts")[columns])
        st.caption("As-delivered dE/dN/dZ tidak disambung antar segmen. Spike ditandai dan dipertahankan.")
        for segment, b in g.groupby("coord_segment"):
            st.write(f"Coordinate segment {segment}")
            st.line_chart(b.set_index("ts")[["dE_mm", "dN_mm", "dZ_mm"]])
        _table(g)
    elif page == "Map":
        _map(result)
    elif page == "Events & investigations":
        _table(result["event_register"])
        tests = result["investigation_register"]
        names = st.multiselect("Filter pengujian", tests.test.unique().tolist())
        _table(tests[tests.test.isin(names)] if names else tests)
        st.subheader("Pemeriksaan lapangan")
        _table(result["field_checks"])
    else:
        st.caption(
            "ZIP berisi CSV, report, plot, GeoPackage 10 layer, konfigurasi dan manifest SHA-256. Tidak mengunggah data ke repo."
        )
        with st.spinner("Menyiapkan hasil…"):
            bundle = download_bundle(result)
        st.download_button("Unduh semua hasil", bundle, "rts_forensics_results.zip", "application/zip")
        for name in ["prism_summary", "frame_cycles", "event_register", "investigation_register"]:
            st.download_button(name + ".csv", result[name].to_csv(index=False), name + ".csv", "text/csv")
