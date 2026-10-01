"""Synthetic ground truth: observations generated independently of the fitter."""

import hashlib
import sqlite3

import numpy as np
import pandas as pd
import pytest


def dms_text(degrees):
    degrees %= 360
    d = int(degrees)
    m = int((degrees - d) * 60)
    s = ((degrees - d) * 60 - m) * 60
    return f"{d}° {m}' {s:.8f}\""


def synthetic(days=22):
    rng = np.random.default_rng(482)
    records = []
    for c, time in enumerate(pd.date_range("2024-02-01", periods=days * 6, freq="4h")):
        for i in range(24):
            if i == 23 and c >= days * 6 - 18:
                continue
            az = np.radians(i * 15 + 3)
            z = np.radians(88 + (i % 5))
            distance = 400 + 29 * i
            hd = distance * np.sin(z)
            rotation = -4.5 if c >= 36 else 0
            te, tn, scale = (2.0, -3.0, 1.5) if c >= 54 else (0.0, 0.0, 0.0)
            dhz = rotation + 206.265 * (-te * np.cos(az) + tn * np.sin(az)) / hd
            los = -(te * np.sin(az) + tn * np.cos(az)) * np.sin(z) + scale * distance / 1000
            local = -0.2 * max(0, c / 6 - 3) if i == 0 else 0.0
            bias = -8.0 if i == 22 and 8 <= time.hour < 16 else 0.0
            for repeat in range(2):
                observed_d = distance + (los + local + bias + rng.normal(0, 0.04)) / 1000
                observed_hz = np.degrees(az) - 30 + (dhz + rng.normal(0, 0.02)) / 3600
                records.append(
                    {
                        "Point ID": f"P-{i:02}",
                        "Time": time.strftime("%d/%m/%Y %H:%M"),
                        "Hz [dms]": dms_text(observed_hz),
                        "V [dms]": dms_text(np.degrees(z)),
                        "D [m]": observed_d,
                        "Target Easting [m]": 1000 + hd * np.sin(az),
                        "Target Northing [m]": 2000 + hd * np.cos(az),
                        "Target Elevation [m]": 100 + distance * np.cos(z),
                        "Station Easting [m]": 1000,
                        "Station Northing [m]": 2000,
                        "Station Height [m]": 100 + (0.015 if c >= 96 else 0),
                        "PPM": 0,
                        "Pressure [mBar]": 1013.25,
                        "Av Temp [°C]": 11.1012,
                        "Add Const [m]": 0,
                    }
                )
    return pd.DataFrame(records).to_csv(index=False, sep=";").encode("cp1252")


def config():
    from rts_forensics.config import load_config

    return load_config(overrides={"frame": {"include": [f"P-{i:02}" for i in range(1, 22)]}})


@pytest.fixture(scope="module")
def result():
    from rts_forensics.pipeline import run

    return run(("synthetic.csv", synthetic()), config())


def test_provenance_and_repeats(result):
    raw = result["observations"]
    series = result["series"]
    assert len(raw) == len(series) * 2
    assert raw.src_line.is_unique
    assert raw.src_line.min() == 2
    assert series.n_repeat.eq(2).all()
    assert len(series.iloc[0].src_lines.split(",")) == 2
    assert result["inputs"][0]["sha256"] == hashlib.sha256(synthetic()).hexdigest()


def test_rotation_translation_scale_recovery(result):
    frames = result["frame_cycles"].set_index("cycle")
    assert frames.loc[40, "rotation_arcsec"] == pytest.approx(-4.5, abs=0.08)
    assert frames.loc[70, "translation_e_mm"] == pytest.approx(2, abs=0.15)
    assert frames.loc[70, "translation_n_mm"] == pytest.approx(-3, abs=0.15)
    assert frames.loc[70, "scale_ppm"] == pytest.approx(1.5, abs=0.2)
    s = result["series"].query("pid == 'P-01'")
    assert abs(s.los_fc.tail(6).median()) < 0.2
    assert abs(s.tan_fc.tail(6).median()) < 0.3
    assert abs(s.tan_raw.tail(6).median()) > 5


def test_local_movement_bias_dropout(result):
    s = result["prism_summary"].set_index("pid")
    assert s.loc["P-00", "net_los_fc_mm"] < -3
    assert s.loc["P-23", "lost_final_48h"]
    inv = result["investigation_register"]
    bias = inv[(inv.test == "daytime_bias") & (inv.pid == "P-22")].iloc[0]
    assert bias.effect_mm == pytest.approx(-8, abs=0.3)
    assert s.loc["P-00", "movement_concern"] == "detected (exploratory)"
    assert s.loc["P-00", "tarp_status"] == "not configured"


def test_station_height_change_does_not_enter_raw_vertical(result):
    s = result["series"].query("pid == 'P-01'")
    assert s.coord_segment.nunique() == 2
    assert abs(s.ver_raw.iloc[-1]) < 0.1
    assert abs(s.dZ_mm.iloc[-1]) < 0.1


def test_circular_median_and_explicit_timestamp():
    from rts_forensics.parse import circular_median, read_export

    assert abs((circular_median([359.999, 0.001]) + 180) % 360 - 180) < 0.002
    data = synthetic(1)
    parsed, _ = read_export(("x.csv", data), config())
    assert parsed.ts.iloc[0] == pd.Timestamp("2024-02-01")
    bad = data.replace(b"01/02/2024 00:00", b"99/99/2024 00:00", 1)
    parsed, _ = read_export(("x.csv", bad), config())
    assert len(parsed) == len(read_export(("x.csv", data), config())[0])
    assert parsed.parse_error.sum() == 1


def test_stations_are_partitioned_before_cycles():
    from rts_forensics.cycles import assign_cycles

    rows = pd.DataFrame(
        {
            "pid": ["A", "B", "A", "B"],
            "ts": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 00:40", "2024-01-01 00:45"]
            ),
            "st_e": [100.0, 200.0, 100.0, 200.0],
            "st_n": [0.0] * 4,
            "st_h": [1.0] * 4,
            "parse_error": [False] * 4,
        }
    )
    out = assign_cycles(rows, config())
    assert out.station.nunique() == 2
    assert out.query("pid == 'A'").cycle.nunique() == 2


def test_rates_use_elapsed_time_and_anchored_blocks():
    from rts_forensics.rates import blocks24, slope

    s = pd.DataFrame(
        {"pid": "A", "station": "S1", "ts": pd.date_range("2024-01-01 08:00", periods=60, freq="12h")}
    )
    s["los_raw"] = (s.ts - s.ts.min()).dt.total_seconds() / 86400 * 2
    s["night"] = s.ts.dt.hour.eq(20)
    b = blocks24(s)
    assert b.block_end.max() == s.ts.max()
    assert slope(s, "los_raw", 30)["rate"] == pytest.approx(2)
    assert slope(s.iloc[[0, 1, 10, 11, 20, 30, 40, 50, 59]], "los_raw", 30)["rate"] == pytest.approx(2)


def test_hour_matching_controls_sampling_composition():
    from rts_forensics.investigations import hour_matched_net

    rows = []
    for day in range(10):
        for hour in [0, 12] if day < 5 else [12]:
            rows.append(
                {
                    "ts": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day, hours=hour),
                    "los_raw": 8 if hour == 12 else 0,
                }
            )
    s = pd.DataFrame(rows)
    assert hour_matched_net(s, "los_raw", 48) == 0


def test_export_manifest_and_grid_transform(result, tmp_path):
    from rts_forensics.gis import transform_xy_vectors
    from rts_forensics.report import write_results

    write_results(result, tmp_path)
    assert (tmp_path / "report.html").exists()
    import json

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert (
        manifest["files"]["prism_summary.csv"]
        == hashlib.sha256((tmp_path / "prism_summary.csv").read_bytes()).hexdigest()
    )
    with sqlite3.connect(tmp_path / "forensics.gpkg") as db:
        layers = db.execute("select table_name,srs_id from gpkg_contents").fetchall()
        assert len(layers) == 10
        assert all(srs == -1 for _, srs in layers)
        assert db.execute("select count(*) from layer_styles").fetchone()[0] == 10
    x, y, ve, vn = transform_xy_vectors(
        1.0, 0.0, 2.0, 0.0, {"rotation_deg": 90, "offset_e": 10, "offset_n": 20}
    )
    assert [x, y, ve, vn] == pytest.approx([10, 21, 0, 2])


def test_config_rejects_unknown_keys_and_incomplete_tarp():
    from rts_forensics.config import load_config

    with pytest.raises(ValueError, match="Unknown"):
        load_config(overrides={"baseline_huors": 48})
    with pytest.raises(ValueError, match="TARP"):
        load_config(overrides={"tarp": {"threshold_mm": 4}})


def test_rank_deficient_frame_is_not_applied():
    from rts_forensics.frame import huber_fit

    x = np.ones((10, 3))
    with pytest.raises(ValueError, match="rank"):
        huber_fit(x, np.ones(10), np.ones(10), 1.345)


def test_investigations_cover_sampling_angle_and_loss_diagnostics(result):
    tests = result["investigation_register"]
    assert {
        "pre_loss_trend_rank",
        "observation_success_over_time",
        "daytime_angle_effect",
        "same_geometry_bias_control",
    } <= set(tests.test)
    hz = tests[(tests.test == "reference_inference") & (tests.metric == "dhz_arcsec")]
    assert hz.control_source.eq("orientation_shift_arcsec").all()


def test_vertical_detection_is_visible_in_overall_concern(result):
    from rts_forensics.classify import summarize

    g = result["series"].query("pid == 'P-01'").copy()
    g["los_raw"] = 0.0
    g["los_fc"] = 0.0
    g["ver_raw"] = -0.6 * (g.ts - g.ts.min()).dt.total_seconds() / 86400
    noise = result["noise"].query("pid == 'P-01'").copy()
    noise["sigma_ver_raw"] = 0.1
    s = summarize(g, noise, config())
    assert s.movement_concern.iloc[0].startswith("detected vertical")


def test_pure_python_gis_backend_supports_styles_and_empty_layers(result, tmp_path):
    from rts_forensics.gis import write_gpkg

    path = tmp_path / "browser.gpkg"
    write_gpkg(result, path, backend="murni")
    with sqlite3.connect(path) as db:
        assert db.execute("pragma integrity_check").fetchone()[0] == "ok"
        assert db.execute("select count(*) from layer_styles").fetchone()[0] == 10
        assert db.execute("select count(*) from gpkg_contents").fetchone()[0] == 10


def test_all_dashboard_pages_render_verified_result(result):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/app.py"))
    app.session_state["forensic_result"] = result
    app.run(timeout=30)
    assert not app.exception
    for page in ["Station frame", "Prism explorer", "Map", "Events & investigations", "Downloads"]:
        app.sidebar.radio[2].set_value(page).run(timeout=30)
        assert not app.exception, page
        if page == "Map":
            for checkbox in app.checkbox:
                checkbox.set_value(True)
            app.run(timeout=30)
            assert not app.exception
