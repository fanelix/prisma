"""Real-data regression against the owner-supplied golden values (HLO September 2026).

Fixture: ``tests/fixtures/golden_values.json``, committed verbatim from the owner's
handoff bundle. Input: the export tracked at ``data/HLO Sept 2026.csv``; its MD5 must
equal ``golden["input"]["md5"]``, otherwise the regression is skipped because the
values describe a different file. Override either path with ``RTS_GOLDEN_DATA`` /
``RTS_GOLDEN_VALUES``.

The golden JSON carries values only. Tolerances are stated once in ``TOLERANCES``
with the reason for each; they are proposals for owner review, not site thresholds,
and nothing here is a movement, alarm or slope-safety criterion. Where the pipeline
has no single output for a golden quantity, the value is derived from pipeline tables
in this file and the derivation is named. Golden values that the implementation does
not reproduce are marked ``xfail(strict=True)`` with the reason, so a fix surfaces.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RTS_GOLDEN_DATA") or ROOT / "data" / "HLO Sept 2026.csv")
VALUES = Path(os.environ.get("RTS_GOLDEN_VALUES") or ROOT / "tests" / "fixtures" / "golden_values.json")

# Absolute tolerances in the golden value's own unit.
TOLERANCES = {
    # Golden repeatability is rounded to 0.01 mm / 0.01 arcsec.
    "repeat_sd": 0.01,
    # Median noise; golden LOS rounded to 0.01 mm, vertical to 0.1 mm.
    "cycle_sigma_los_mm": 0.02,
    "cycle_sigma_vert_mm": 0.1,
    # Final rotation; golden rounded to 0.01 arcsec, frame membership is provisional.
    "rotation_end_arcsec": 0.05,
    # Handoff quotes se ~ 0.15 arcsec for the 7 Sep step.
    "rotation_step_arcsec": 0.15,
    # Golden drift rates rounded to 0.01 arcsec/day.
    "drift_arcsec_per_day": 0.02,
    # Mean over ~57 cycles; the provisional frame set differs from the research frame set.
    "applied_minus_fitted_arcsec": 0.15,
    # Golden as-delivered median dE is rounded to 0.5 mm; delivered E is 1 mm resolution.
    "bs_median_de_mm": 0.5,
    # Golden is whole mm "on 21 Sep"; per-cycle network medians that day span 26.0-27.0 mm.
    "network_3d_mm": 1.0,
    # Frame-corrected LOS depends on the frame set (provisional vs research set).
    "g1_net_fc_mm": 0.3,
    # Raw LOS is quantised to 1 mm, so medians move in 0.5 mm steps.
    "g1_net_raw_mm": 0.5,
    # ~0.75 mm over 30 days; below the 1 mm LOS resolution.
    "g1_rate_mm_per_day": 0.025,
    # Placebo group set differs (139 research groups vs groups of targets with >= frame.min_cycles).
    "placebo_min_mm": 0.1,
    # Raw vertical is not frame dependent; golden rounded to 0.01 mm.
    "r7_net_vert_mm": 0.05,
    # Pipeline excludes the initial partial 24-h block (documented); see test for the exact cause.
    "r7_trend_pipeline_mm_30d": 0.6,
    "r7_trend_partial_block_mm_30d": 0.05,
    "r7_rho": 0.05,
}
EVENT_START, EVENT_END = pd.Timestamp("2026-09-07 04:34"), pd.Timestamp("2026-09-07 08:04")
G1_PERIODS = {
    "01-07 Sep": ("2026-09-01", "2026-09-07"),
    "07-14 Sep": ("2026-09-07", "2026-09-14"),
    "14-21 Sep": ("2026-09-14", "2026-09-21"),
    "21 Sep-01 Oct": ("2026-09-21", "2026-10-02"),
}


def _skip_reason():
    if not VALUES.is_file():
        return f"golden values not found: {VALUES}"
    if not DATA.is_file():
        return f"HLO export not found: {DATA}"
    expected = json.loads(VALUES.read_text(encoding="utf-8"))["input"]["md5"]
    if hashlib.md5(DATA.read_bytes()).hexdigest() != expected:
        return f"{DATA.name} is not the export the golden values describe (MD5 differs)"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))


@pytest.fixture(scope="module")
def golden():
    return json.loads(VALUES.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def result():
    from rts_forensics.config import load_config
    from rts_forensics.pipeline import run

    return run(str(DATA), load_config(ROOT / "config.yaml"))


@pytest.fixture(scope="module")
def stations(result):
    """Golden E/W names -> pipeline station IDs, from station easting (no ID is assumed)."""
    easting = result["cycle_table"].groupby("station").station_e.median().sort_values()
    return {"W": easting.index[0], "E": easting.index[-1]}


@pytest.fixture(scope="module")
def summary(result):
    return result["prism_summary"].set_index("pid")


def _east_frames(result, stations):
    f = result["frame_cycles"]
    return f[(f.station == stations["E"]) & (f.status == "fit")].sort_values("ts")


def test_input_matches_golden(result, golden, stations):
    expected = golden["input"]
    assert result["inputs"][0]["rows"] == expected["rows"]
    assert len(result["observations"]) == expected["rows"]
    assert result["inputs"][0]["invalid_rows"] == 0
    assert result["prism_summary"].pid.nunique() == expected["prisms"]
    for name, counts in expected["stations"].items():
        station = stations[name]
        assert (result["prism_summary"].station == station).sum() == counts["prisms"]
        assert (result["cycle_table"].station == station).sum() == counts["cycles"]
    ts = result["observations"].ts
    assert [ts.min().strftime("%Y-%m-%d %H:%M"), ts.max().strftime("%Y-%m-%d %H:%M")] == expected["period"]


def test_repeatability(result, golden):
    expected = golden["repeatability"]
    pooled = result["repeatability"].set_index("station").loc["all"]
    tol = TOLERANCES["repeat_sd"]
    assert pooled.repeat_sd_d_mm == pytest.approx(expected["D_mm"], abs=tol)
    assert pooled.repeat_sd_hz_arcsec == pytest.approx(expected["Hz_arcsec"], abs=tol)
    assert pooled.repeat_sd_v_arcsec == pytest.approx(expected["V_arcsec"], abs=tol)
    noise = result["noise"]
    assert noise.sigma_los_raw.median() == pytest.approx(
        expected["cycle_sigma_los_mm_median"], abs=TOLERANCES["cycle_sigma_los_mm"]
    )
    assert noise.sigma_ver_raw.median() == pytest.approx(
        expected["cycle_sigma_vert_mm_median"], abs=TOLERANCES["cycle_sigma_vert_mm"]
    )


def test_station_e_rotation(result, golden, stations):
    expected = golden["station_E"]
    east = _east_frames(result, stations)
    assert east.rotation_arcsec.iloc[-1] == pytest.approx(
        expected["rotation_end_arcsec"], abs=TOLERANCES["rotation_end_arcsec"]
    )
    # Step: median rotation 24 h after the event window minus 24 h before it.
    day = pd.Timedelta(hours=24)
    before = east[(east.ts < EVENT_START) & (east.ts >= EVENT_START - day)].rotation_arcsec.median()
    after = east[(east.ts > EVENT_END) & (east.ts <= EVENT_END + day)].rotation_arcsec.median()
    assert after - before == pytest.approx(
        expected["rotation_step_7Sep_arcsec"], abs=TOLERANCES["rotation_step_arcsec"]
    )
    change = pd.Timestamp(expected["processing_change"])
    for (start, end), key in [
        ((EVENT_END, change), "drift_7_21Sep_arcsec_per_day"),
        ((change, east.ts.max() + day), "drift_after_21Sep_arcsec_per_day"),
    ]:
        window = east[(east.ts >= start) & (east.ts < end)]
        days = (window.ts - window.ts.min()).dt.total_seconds() / 86400
        rate = np.polyfit(days, window.rotation_arcsec, 1)[0]
        assert rate == pytest.approx(expected[key], abs=TOLERANCES["drift_arcsec_per_day"])
    after_change = east[east.ts >= change]
    assert after_change.applied_minus_fitted_arcsec.mean() == pytest.approx(
        expected["S2_applied_minus_fitted_orientation_mean_arcsec"],
        abs=TOLERANCES["applied_minus_fitted_arcsec"],
    )


def test_station_e_processing_change_and_outlier_cycle(result, golden, stations):
    expected = golden["station_E"]
    rows = result["observations"]
    east = rows[(rows.station == stations["E"]) & ~rows.parse_error]
    # Detected automatically from exported station coordinates; no date is configured.
    assert east[east.coord_segment > 0].ts.min() == pd.Timestamp(expected["processing_change"])
    cycles = result["cycle_table"]
    e158 = cycles[(cycles.station == stations["E"]) & (cycles.cycle == 158)].iloc[0]
    assert e158.start == pd.Timestamp("2026-09-28 00:03")
    assert e158.station_h == pytest.approx(expected["E158_station_height_m"], abs=5e-4)


def test_station_e_as_delivered_artefact(result, golden, stations):
    expected = golden["station_E"]
    series = result["series"]
    segment = series[(series.station == stations["E"]) & (series.coord_segment == 0)]
    backsight = segment[segment.pid == "BS_HL_5"]
    assert backsight.dE_mm.median() == pytest.approx(
        expected["BS_HL_5_S1_median_dE_mm"], abs=TOLERANCES["bs_median_de_mm"]
    )
    # Network median 3D per cycle on the final day of segment E-S1, then median over cycles.
    final = segment[segment.ts.dt.normalize() == segment.ts.max().normalize()]
    three_d = np.sqrt(final.dE_mm**2 + final.dN_mm**2 + final.dZ_mm**2).groupby(final.cycle).median()
    assert three_d.median() == pytest.approx(
        expected["network_median_asdelivered_3D_end_S1_mm"], abs=TOLERANCES["network_3d_mm"]
    )


def test_g1_net_change(summary, golden):
    expected = golden["G1"]
    for pid in expected["prisms"]:
        assert summary.loc[pid, "net_los_fc_mm"] == pytest.approx(
            expected["net_los_frame_corrected_mm"][pid], abs=TOLERANCES["g1_net_fc_mm"]
        ), pid
        assert summary.loc[pid, "net_los_raw_mm"] == pytest.approx(
            expected["net_los_raw_mm"][pid], abs=TOLERANCES["g1_net_raw_mm"]
        ), pid
    assert summary.loc[expected["prisms"], "net_los_fc_mm"].median() == pytest.approx(
        expected["group_median_mm"], abs=TOLERANCES["g1_net_fc_mm"]
    )


def test_g1_period_rates(result, golden):
    from prismacore.robust import theilslopes

    expected = golden["G1"]
    assert set(expected["period_rates_mm_per_day"]) == set(G1_PERIODS)
    series = result["series"]
    group = series[series.pid.isin(expected["prisms"])]
    # Group median of frame-corrected LOS per cycle, with most members present.
    by_cycle = group.groupby("cycle").agg(ts=("ts", "median"), los=("los_fc", "median"), n=("pid", "size"))
    by_cycle = by_cycle[by_cycle.n > len(expected["prisms"]) / 2].dropna()
    for label, (start, end) in G1_PERIODS.items():
        window = by_cycle[(by_cycle.ts >= start) & (by_cycle.ts < end)]
        days = (window.ts - window.ts.min()).dt.total_seconds() / 86400
        rate = theilslopes(window.los.to_numpy(), days.to_numpy(), 0.95)[0]
        assert rate == pytest.approx(
            expected["period_rates_mm_per_day"][label], abs=TOLERANCES["g1_rate_mm_per_day"]
        ), label


def test_g1_spatial_placebo(result, golden, stations):
    expected = golden["G1"]
    s = result["prism_summary"]
    min_cycles = result["config"]["frame"]["min_cycles"]
    s = s[(s.station == stations["E"]) & (s.n_cycles >= min_cycles)].reset_index(drop=True)
    xy = s[["e", "n"]].to_numpy()
    members = set(s.index[s.pid.isin(expected["prisms"])])
    medians = []
    for i in range(len(s)):
        nearest = np.argsort(((xy - xy[i]) ** 2).sum(axis=1))[:7]
        if not members & set(nearest):
            medians.append(np.nanmedian(s.net_los_fc_mm.to_numpy()[nearest]))
    group = s.loc[sorted(members), "net_los_fc_mm"].median()
    assert len(medians) > 100
    assert sum(m <= group for m in medians) == 0
    assert min(medians) == pytest.approx(expected["placebo_null_min_mm"], abs=TOLERANCES["placebo_min_mm"])


def test_hlo_r7(result, summary, golden):
    expected = golden["HLO_R7"]
    r7 = summary.loc["HLO-R7"]
    assert r7.net_vertical_raw_mm == pytest.approx(
        expected["net_vert_raw_mm"], abs=TOLERANCES["r7_net_vert_mm"]
    )
    assert r7.net_los_raw_mm == pytest.approx(expected["net_los_mm"], abs=TOLERANCES["g1_net_raw_mm"])
    assert r7.vertical_raw_30d_rate * 30 == pytest.approx(
        expected["vert_trend_mm_per_30d"], abs=TOLERANCES["r7_trend_pipeline_mm_30d"]
    )
    # The remaining gap is only the initial partial 24-h block, which the pipeline excludes.
    from prismacore.robust import theilslopes

    blocks = result["timeseries_24h"]
    blocks = blocks[blocks.pid == "HLO-R7"].dropna(subset=["ver_raw"])
    days = (blocks.ts - blocks.ts.min()).dt.total_seconds() / 86400
    with_partial = theilslopes(blocks.ver_raw.to_numpy(), days.to_numpy(), 0.95)[0] * 30
    assert with_partial == pytest.approx(
        expected["vert_trend_mm_per_30d"], abs=TOLERANCES["r7_trend_partial_block_mm_30d"]
    )


def _r7_reference_tests(result, stations):
    register = result["investigation_register"]
    return register[
        (register.test == "reference_inference")
        & (register.station == stations["W"])
        & (register.metric == "ver_raw")
        & (register.detrending == "cycle_detrended")
    ].set_index("pid")


def test_hlo_r7_is_the_station_w_height_correlated_target(result, stations):
    tests = _r7_reference_tests(result, stations)
    assert tests.effect_mm.idxmin() == "HLO-R7"
    # Handoff: a real station rise lowers raw vertical, so a reference shows negative rho.
    significant = tests[(tests.p_bonferroni < 0.05) & (tests.effect_mm < 0)]
    assert list(significant.index) == ["HLO-R7"]


@pytest.mark.xfail(
    strict=True,
    reason="Golden rho -0.317 comes from the unsupplied research script; the handoff's "
    "cycle-detrended definition gives -0.20 to -0.23 depending on the detrending median",
)
def test_hlo_r7_rho_value(result, golden, stations):
    rho = _r7_reference_tests(result, stations).loc["HLO-R7", "effect_mm"]
    assert rho == pytest.approx(golden["HLO_R7"]["rho_with_W_station_height"], abs=TOLERANCES["r7_rho"])


def test_lost_final_48h(result, golden):
    s = result["prism_summary"]
    assert set(s[s.lost_final_48h].pid) == set(golden["lost_final_48h"])


def test_sp_daytime_bias(summary, golden):
    expected = set(golden["sp_daytime_bias_prisms"])
    sp = summary[summary.index.str.startswith("SP_")]
    # Without an owner bias floor there is no classification; check ranking and sign only.
    assert set(sp.daytime_bias_mm.nsmallest(len(expected)).index) == expected
    assert (sp.loc[sorted(expected), "daytime_bias_high_mm"] < 0).all()


def test_tangential_only_candidates(summary, golden):
    p90 = np.nanpercentile(summary.net_tangential_fc_mm.abs(), 90)
    for pid in golden["tangential_only_candidates"]:
        r = summary.loc[pid]
        assert abs(r.net_tangential_fc_mm) > p90, pid
        assert not r.movement_concern.startswith("detected"), pid


def test_insufficient_data_count(result, golden):
    s = result["prism_summary"]
    assert (s.movement_concern == "insufficient data").sum() == golden["final_class_counts"][
        "Insufficient data"
    ]


@pytest.mark.xfail(
    strict=True,
    reason="Final classes come from stage-3 synthesis (s3_final.py, unsupplied) with named "
    "clusters and artefact classes; the pipeline reports exploratory screening labels only",
)
def test_final_class_counts(result, golden):
    counts = result["prism_summary"].final_class.value_counts().to_dict()
    assert counts == golden["final_class_counts"]


def test_station_w_vertical_frame_is_identifiable(result, summary, stations):
    """Regression for the narrow-fan W station (not a golden value).

    The handoff's four-term vertical model cannot separate index from along-fan tilt
    across W's 22-degree fan, and its correction at targets outside the fan was
    ill-determined (up to 60 mm). E keeps the full model.
    """
    frames = result["frame_cycles"].groupby("station").first()
    assert frames.loc[stations["E"], "vertical_model"] == "full"
    assert frames.loc[stations["W"], "vertical_model"] != "full"
    assert frames.loc[stations["W"], "full_vertical_targets_over_sigma_v"] > 0
    # The chosen model is the richest whose a-priori correction SE is within sigma_v at
    # every W target (fitted SEs add Huber weighting and variance inflation on top).
    from rts_forensics.frame import select_vertical_model

    members = result["frame_members"]
    members = set(members[(members.station == stations["W"]) & members.included].pid)
    series = result["series"][result["series"].station == stations["W"]]
    chosen, diagnostics = select_vertical_model(
        series,
        members,
        result["config"]["frame"]["sigma_v_mm"],
        frames.loc[stations["W"], "vertical_axis_az_deg"],
    )
    assert chosen == frames.loc[stations["W"], "vertical_model"]
    assert dict((d[0], d[3]) for d in diagnostics)[chosen] == 0
    # HLO-R7's raw subsidence is no longer absorbed by the frame.
    assert summary.loc["HLO-R7", "net_vertical_fc_mm"] < summary.loc["HLO-R7", "net_vertical_raw_mm"] / 2
