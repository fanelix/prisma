"""
test_rates.py — backwards 24 h blocks, Theil-Sen rates and ``compute_rates``.

Series are built with a known slope so the assertions check that elapsed time
(never a row index) drives the fit, that the oldest partial block is flagged
and excluded, that 1 mm quantisation does not destroy the block-median slope
and that insufficient data never fabricates a rate.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from rts_forensics import rates
from rts_forensics.config import default_config
from rts_forensics.models import RATE_COLUMNS, STATUS_INSUFFICIENT, STATUS_OK


def _series(ts, values, *, point_id: str = "P1", segment_id: str = "E-S1",
            metric: str = "d_rad") -> pd.DataFrame:
    return pd.DataFrame({
        "point_id": point_id,
        "segment_id": segment_id,
        "metric": metric,
        "ts": pd.DatetimeIndex(ts),
        "value": np.asarray(values, dtype=float),
    })


def _elapsed_days(ts) -> np.ndarray:
    return (pd.DatetimeIndex(ts) - pd.DatetimeIndex(ts)[0]).total_seconds().to_numpy() / 86400.0


def test_rate_estimate_dataclass_matches_rate_columns():
    assert [field.name for field in fields(rates.RateEstimate)] == list(RATE_COLUMNS)


def test_backward_blocks_anchored_at_final_record():
    config = default_config()
    ts = pd.date_range("2026-08-28 10:00", "2026-09-01 06:00", freq="h")
    ts = ts.append(pd.DatetimeIndex(["2026-09-01 06:37"]))
    series = _series(ts, np.arange(len(ts), dtype=float))
    series["obs_id"] = [f"src:{i}" for i in range(len(ts))]

    blocks = rates.backward_blocks(series, duration=pd.Timedelta(hours=24),
                                   anchor=None, config=config)
    assert list(blocks["block_id"]) == [0, 1, 2, 3]
    assert blocks.loc[0, "end"] == ts.max()
    assert blocks.loc[0, "start"] == ts.max() - pd.Timedelta(hours=24)
    assert list(blocks["partial"]) == [False, False, False, True]

    full = blocks.loc[~blocks["partial"]]
    assert len(full) == 3
    partial = blocks.loc[blocks["partial"]].iloc[0]
    assert partial["start"] == ts.min()
    assert partial["end"] == ts.max() - pd.Timedelta(hours=72)

    assert int(blocks["n_obs"].sum()) == len(ts)
    collected: set[str] = set()
    for ids in blocks["obs_ids"]:
        collected.update(ids)
    assert collected == set(series["obs_id"])

    ordered = full.sort_values("block_id")
    for start, end in zip(ordered["start"].iloc[:-1], ordered["end"].iloc[1:],
                          strict=True):
        assert start == end

    hours = rates.backward_blocks(series, duration=24, anchor=None, config=config)
    pd.testing.assert_frame_equal(blocks, hours)


def test_backward_blocks_explicit_anchor():
    config = default_config()
    ts = pd.date_range("2026-08-30 00:00", "2026-09-01 06:37", freq="2h")
    series = _series(ts, np.zeros(len(ts)))
    anchor = pd.Timestamp("2026-09-01 00:00:00")
    blocks = rates.backward_blocks(series, duration=24, anchor=anchor,
                                   config=config)
    assert blocks.loc[0, "end"] == anchor
    assert blocks.loc[0, "start"] == anchor - pd.Timedelta(hours=24)


def test_backward_blocks_anchor_per_group():
    config = default_config()
    first = pd.date_range("2026-08-30 00:00", "2026-09-01 06:37", freq="4h")
    second = pd.date_range("2026-08-30 00:00", "2026-08-31 12:00", freq="4h")
    series = pd.concat([
        _series(first, np.zeros(len(first)), point_id="A"),
        _series(second, np.zeros(len(second)), point_id="B"),
    ], ignore_index=True)
    blocks = rates.backward_blocks(series, duration=24, anchor=None,
                                   config=config)
    for point_id, group in blocks.groupby("point_id"):
        expected = series.loc[series["point_id"] == point_id, "ts"].max()
        assert group.loc[group["block_id"] == 0, "end"].iloc[0] == expected


def test_backward_blocks_single_series_without_identifiers():
    config = default_config()
    ts = pd.date_range("2026-09-29 00:00", periods=10, freq="h")
    frame = pd.DataFrame({"ts": ts, "value": np.arange(10, dtype=float)})
    blocks = rates.backward_blocks(frame, duration=24, config=config)
    assert list(blocks.columns) == list(rates.BLOCK_COLUMNS)
    assert blocks["partial"].tolist() == [True]
    assert blocks.loc[0, "point_id"] == ""
    assert blocks.loc[0, "metric"] == ""
    assert blocks.loc[0, "end"] == ts.max()


def test_theil_sen_block_medians_recover_slope_with_outliers():
    config = default_config()
    rng = np.random.default_rng(7)
    ts = pd.date_range(end="2026-10-01 06:37", periods=139, freq="4h")
    values = -0.3 * _elapsed_days(ts) + rng.normal(0.0, 0.08, len(ts))
    outliers = rng.random(len(ts)) < 0.05
    values[outliers] += (rng.choice([-1.0, 1.0], int(outliers.sum()))
                         * rng.uniform(3.0, 8.0, int(outliers.sum())))
    series = _series(ts, values)

    estimate = rates.theil_sen_rate(series, metric="d_rad", window="30d",
                                    config=config)
    assert estimate.status == STATUS_OK
    assert estimate.method == "theil_sen"
    assert estimate.rate_mm_day == pytest.approx(-0.3, abs=0.05)
    assert estimate.ci_low < estimate.rate_mm_day < estimate.ci_high
    assert estimate.n_blocks >= 20
    assert estimate.n_obs == len(series) - 1  # oldest partial block excluded
    assert estimate.partial_block is True
    assert estimate.t_end == ts.max()
    assert estimate.t_start < estimate.t_end

    seven = rates.theil_sen_rate(series, metric="d_rad", window="7d",
                                 config=config)
    assert seven.status == STATUS_OK
    assert seven.n_blocks == 7
    assert seven.n_obs > 0
    assert seven.partial_block is False
    assert seven.rate_mm_day == pytest.approx(-0.3, abs=0.15)


def test_theil_sen_72h_uses_cycles_and_notes_autocorrelation():
    config = default_config()
    rng = np.random.default_rng(9)
    ts = pd.date_range(end="2026-10-01 06:37", periods=72, freq="h")
    values = -0.25 * _elapsed_days(ts) + rng.normal(0.0, 0.05, len(ts))
    estimate = rates.theil_sen_rate(_series(ts, values), metric="d_rad",
                                    window="72h", config=config)
    assert estimate.status == STATUS_OK
    assert estimate.method == "theil_sen"
    assert estimate.n_blocks == 72
    assert estimate.n_obs == 72
    assert estimate.partial_block is False
    assert estimate.reason is not None
    assert "autocorrelation" in estimate.reason
    assert estimate.rate_mm_day == pytest.approx(-0.25, abs=0.08)


def test_theil_sen_insufficient_few_points():
    config = default_config()
    ts = pd.date_range("2026-09-30 06:37", periods=4, freq="h")
    estimate = rates.theil_sen_rate(_series(ts, np.zeros(len(ts))),
                                    metric="d_rad", window="72h", config=config)
    assert estimate.status == STATUS_INSUFFICIENT
    assert np.isnan(estimate.rate_mm_day)
    assert np.isnan(estimate.ci_low)
    assert np.isnan(estimate.ci_high)
    assert estimate.reason is not None


def test_theil_sen_insufficient_short_span():
    config = default_config()
    ts = pd.date_range("2026-09-30 00:00", periods=10, freq="h")
    estimate = rates.theil_sen_rate(_series(ts, np.zeros(len(ts))),
                                    metric="d_rad", window="72h", config=config)
    assert estimate.status == STATUS_INSUFFICIENT
    assert np.isnan(estimate.rate_mm_day)
    assert estimate.reason is not None
    assert "span" in estimate.reason


def test_theil_sen_blocked_insufficient_without_six_blocks():
    config = default_config()
    ts = pd.date_range(end="2026-10-01 06:37", periods=18, freq="4h")
    estimate = rates.theil_sen_rate(_series(ts, np.zeros(len(ts))),
                                    metric="d_rad", window="30d", config=config)
    assert estimate.status == STATUS_INSUFFICIENT
    assert estimate.n_blocks == 2
    assert np.isnan(estimate.rate_mm_day)
    assert "required" in estimate.reason


def test_theil_sen_quantised_distance_still_recovers_slope():
    config = default_config()
    rng = np.random.default_rng(3)
    ts = pd.date_range(end="2026-10-01 06:37", periods=181, freq="4h")
    values = np.round(-0.3 * _elapsed_days(ts) + rng.normal(0.0, 0.2, len(ts)))
    estimate = rates.theil_sen_rate(_series(ts, values), metric="d_rad",
                                    window="30d", config=config)
    assert estimate.status == STATUS_OK
    assert estimate.rate_mm_day == pytest.approx(-0.3, abs=0.05)


def test_partial_block_never_enters_the_rate():
    config = default_config()
    rng = np.random.default_rng(4)
    ts = pd.date_range(end="2026-10-01 06:37", periods=63, freq="4h")
    values = -0.3 * _elapsed_days(ts) + rng.normal(0.0, 0.05, len(ts))
    series = _series(ts, values)

    blocks = rates.backward_blocks(series, duration=24, anchor=None,
                                   config=config)
    partial = blocks.loc[blocks["partial"]].iloc[0]
    mask = (series["ts"] >= partial["start"]) & (series["ts"] <= partial["end"])
    series.loc[mask, "value"] = 100.0

    estimate = rates.theil_sen_rate(series, metric="d_rad", window="30d",
                                    config=config)
    assert estimate.partial_block is True
    assert estimate.status == STATUS_OK
    assert estimate.rate_mm_day == pytest.approx(-0.3, abs=0.1)


def test_unknown_window_is_rejected():
    config = default_config()
    ts = pd.date_range("2026-09-01", periods=10, freq="h")
    with pytest.raises(ValueError, match="window"):
        rates.theil_sen_rate(_series(ts, np.zeros(len(ts))), metric="d_rad",
                             window="14x", config=config)


def _wide_series(point_id: str, segment_id: str, ts) -> pd.DataFrame:
    elapsed = _elapsed_days(ts)
    return pd.DataFrame({
        "point_id": point_id,
        "segment_id": segment_id,
        "ts": pd.DatetimeIndex(ts),
        "d_rad_mm": -0.3 * elapsed,
        "d_vert_mm": -0.1 * elapsed,
        "d_tan_mm": 0.05 * elapsed,
        "d_east_mm": -0.3 * elapsed,
        "d_north_mm": -0.1 * elapsed,
        "d_up_mm": -0.2 * elapsed,
    })


def test_compute_rates_covers_every_metric_and_window():
    config = default_config()
    ts = pd.date_range(end="2026-10-01 06:37", periods=12 * 6, freq="4h")
    displacements = pd.concat([
        _wide_series("A", "E-S1", ts),
        _wide_series("B", "W1", ts),
    ], ignore_index=True)
    out = rates.compute_rates(displacements, config=config)

    assert out.columns.tolist() == list(RATE_COLUMNS)
    assert len(out) == 2 * len(rates.RATE_METRICS) * 3
    assert set(out["window"]) == {"30d", "7d", "72h"}
    assert set(out["metric"]) == set(rates.RATE_METRICS)
    assert set(out["point_id"]) == {"A", "B"}
    assert (out["status"] == STATUS_OK).all()


def test_compute_rates_accepts_long_frame():
    config = default_config()
    ts = pd.date_range(end="2026-10-01 06:37", periods=12 * 6, freq="4h")
    long = _series(ts, -0.3 * _elapsed_days(ts), point_id="A",
                   segment_id="E-S1", metric="d_rad")
    out = rates.compute_rates(long, config=config)
    assert len(out) == 3
    assert (out["metric"] == "d_rad").all()
    assert (out["status"] == STATUS_OK).all()


def test_compute_rates_rejects_unknown_columns():
    frame = pd.DataFrame({"point_id": ["A"], "ts": [pd.Timestamp("2026-01-01")],
                          "other_mm": [0.0]})
    with pytest.raises(ValueError, match="metric"):
        rates.compute_rates(frame, config=default_config())
