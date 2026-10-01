import numpy as np
import pandas as pd
import pytest

from rts_forensics.config import load_config


@pytest.mark.parametrize(
    "extra",
    [
        {"screening": None},
        {"frame": {"include": "P1"}},
        {"detection": {"cluster_radius_m": -1}},
        {"night_hours": [24, 8]},
        {
            "reliability": {
                g: {"coverage_min": 2, "max_gap_hours": 48, "max_los_noise_mm": 1, "max_spike_fraction": 0.1}
                for g in ["A", "B", "C"]
            }
        },
    ],
)
def test_invalid_config_is_rejected(extra):
    with pytest.raises(ValueError):
        load_config(overrides=extra)


@pytest.mark.parametrize(
    "pid,group",
    [("HLO-279", "HLO"), ("HLO-R7", "HLO-R"), ("HLO-W1", "HLO-W"), ("STG4N-10", "STG4N"), ("DAM5_3", "DAM5")],
)
def test_group_names_preserve_prefix_families(pid, group):
    from rts_forensics.classify import group_for_pid

    assert group_for_pid(pid) == group


def test_hourmatching_does_not_move_windows_when_corrections_are_missing():
    from rts_forensics.investigations import hour_matched_net

    s = pd.DataFrame(
        {"ts": pd.date_range("2024-01-01", periods=181, freq="4h"), "value": -np.arange(181) / 6}
    )
    s.loc[s.index >= 151, "value"] = np.nan
    assert np.isnan(hour_matched_net(s, "value", 48))


def test_hourmatching_respects_distinct_baseline_window():
    from rts_forensics.investigations import hour_matched_net

    s = pd.DataFrame(
        {"ts": pd.date_range("2024-01-01", periods=11, freq="24h"), "value": np.arange(11) * 10.0}
    )
    assert hour_matched_net(s, "value", 24, baseline_hours=72) == pytest.approx(90)


def tarp_config():
    return load_config(
        overrides={
            "tarp": {
                "metric": "los_fc",
                "threshold": 1.0,
                "averaging_hours": 48,
                "persistence_hours": 24,
                "max_gap_hours": 6,
                "direction": "above",
                "label": "SITE CONDITION",
            }
        }
    )


def test_tarp_missing_latest_value_is_unavailable():
    from rts_forensics.classify import tarp_status

    s = pd.DataFrame(
        {"ts": pd.to_datetime(["2024-01-01", "2024-01-02"]), "station": "S1", "los_fc": [2.0, np.nan]}
    )
    assert tarp_status(s, tarp_config()) == "unavailable"


def test_tarp_does_not_count_an_unobserved_interval_as_persistence():
    from rts_forensics.classify import tarp_status

    s = pd.DataFrame(
        {"ts": pd.to_datetime(["2024-01-01", "2024-01-05"]), "station": "S1", "los_fc": [2.0, 2.0]}
    )
    assert tarp_status(s, tarp_config()) != "SITE CONDITION"


def test_auto_frame_excludes_vertical_movers():
    from rts_forensics.frame import fit_frames

    series = pd.DataFrame(
        {
            "station": ["S1"],
            "pid": ["MOV"],
            "cycle": [0],
            "ts": pd.to_datetime(["2024-01-01"]),
            "st_e": [0.0],
            "st_n": [0.0],
            "st_h": [0.0],
            "orientation_deg": [0.0],
        }
    )
    summary = pd.DataFrame(
        {
            "station": ["S1"],
            "pid": ["MOV"],
            "n_cycles": [100],
            "movement_concern": ["no credible movement"],
            "vertical_concern": ["detected (exploratory)"],
            "daytime_bias_low_mm": [np.nan],
            "daytime_bias_high_mm": [np.nan],
        }
    )
    _, _, members = fit_frames(series, summary, load_config())
    assert not members.included.iloc[0]
    assert "vertical" in members.reason.iloc[0]


def test_malformed_export_retains_audit_record():
    from rts_forensics.parse import read_export

    header = "Point ID;Time;Hz [dms];V [dms];D [m];Target Easting [m];Target Northing [m];Target Elevation [m];Station Easting [m];Station Northing [m];Station Height [m]\n"
    rows, _ = read_export(("malformed.csv", (header + "P1\n").encode()), load_config())
    assert len(rows) == 1 and rows.parse_error.iloc[0]


def test_repeat_median_does_not_bridge_coordinate_regimes():
    from rts_forensics.cycles import assign_cycles
    from rts_forensics.displacement import prism_cycles

    cfg = load_config(overrides={"processing_changes": {"S1": ["2024-01-01 00:05"]}})
    rows = pd.DataFrame(
        {
            "pid": "P",
            "source": "x.csv",
            "src_line": range(2, 6),
            "ts": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:10", "2024-01-01 01:00", "2024-01-01 02:00"]
            ),
            "st_e": 0.0,
            "st_n": 0.0,
            "st_h": 0.0,
            "e": [100.0, 200.0, 200.0, 200.0],
            "n": 100.0,
            "z": 0.0,
            "d": 200.0,
            "hz": 45.0,
            "v": 90.0,
            "parse_error": False,
        }
    )
    s = prism_cycles(assign_cycles(rows, cfg), cfg)
    assert np.isnan(s.dE_mm.iloc[0])
    assert s.dE_mm.iloc[1:].eq(0).all()


def test_short_single_observation_pipeline_has_no_numeric_warning():
    from rts_forensics.pipeline import run

    header = "Point ID;Time;Hz [dms];V [dms];D [m];Target Easting [m];Target Northing [m];Target Elevation [m];Station Easting [m];Station Northing [m];Station Height [m]\n"
    text = header + "P1;01/01/2024 00:00;0° 0' 0;90° 0' 0;100;0;100;0;0;0;0\n"
    result = run(("single.csv", text.encode("cp1252")))
    assert result["prism_summary"].movement_concern.iloc[0] == "insufficient data"


def test_installed_distribution_does_not_include_field_data():
    import tarfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    distributions = list((root / "dist").glob("*.tar.gz"))
    if not distributions:
        pytest.skip("Run package build for distribution content inspection")
    with tarfile.open(distributions[-1]) as archive:
        names = archive.getnames()
    assert not any("/data/" in n or "/hasil/" in n for n in names)
    assert any("/docs/method.md" in n for n in names)
