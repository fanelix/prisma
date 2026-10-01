"""
noise.py — measurement repeatability, robust residual noise and retained spikes.

Supplied research settings (PROJECT_HANDOFF.md section 3, ALGORITHMS.md
section 3) are read from the effective configuration:

* pooled within-cycle repeat SD over ``(point_id, station_id, cycle_id)``
  groups, reported for all groups, groups spanning at most one minute and
  groups spanning at least ten minutes;
* a ``1.4826 x MAD`` robust residual SD from a rolling median of
  ``config.noise.rolling_window_hours`` (centred and with
  ``config.noise.min_observations`` as ``min_periods``);
* a spike flag at ``|robust z| > config.noise.spike_z``.

Flags are retained; no observation is deleted, reordered or overwritten.
Spike floors are the supplied research values d_rad 0.5 mm, d_ver 1.0 mm,
d_tan 1.0 mm and d_up 1.0 mm; floors for other metrics are not supplied and
must be passed explicitly through the ``floors`` argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .models import (
    STATUS_INSUFFICIENT,
    STATUS_OK,
    NoiseResult,
)

SPAN_CLASSES = ("all", "same_minute", "ge_10min")
SAME_MINUTE = pd.Timedelta(minutes=1)
GE_TEN_MINUTES = pd.Timedelta(minutes=10)

#: Supplied research spike floors in millimetres (ALGORITHMS.md section 3).
SPIKE_FLOORS_MM: dict[str, float] = {
    "d_rad": 0.5,
    "d_ver": 1.0,
    "d_tan": 1.0,
    "d_up": 1.0,
}

#: Minimum number of valid residuals before a robust SD is reported.
MIN_RESIDUALS = 5

#: Repeat variables: source column -> (output column, mm/arcsec factor).
REPEAT_VARIABLES: dict[str, tuple[str, float]] = {
    "d_m": ("sd_d_mm", 1000.0),
    "hz_deg": ("sd_hz_arcsec", 3600.0),
    "v_deg": ("sd_v_arcsec", 3600.0),
    "te_m": ("sd_e_mm", 1000.0),
    "tn_m": ("sd_n_mm", 1000.0),
    "tz_m": ("sd_z_mm", 1000.0),
}

POOLED_COLUMNS = (
    "span_class", "n_obs", "n_groups",
    "sd_d_mm", "sd_hz_arcsec", "sd_v_arcsec",
    "sd_e_mm", "sd_n_mm", "sd_z_mm",
)

POOLED_SUMMARY_COLUMNS = ("point_id", "metric", "n", "robust_sd_mm", "status", "reason")

NOISE_SERIES_COLUMNS = (
    "point_id", "segment_id", "ts", "metric", "value",
    "rolling_median_mm", "residual_mm", "robust_sd_mm",
)

FLAG_COLUMNS = (
    "point_id", "segment_id", "ts", "metric", "value",
    "rolling_median_mm", "residual_mm", "robust_sd_mm",
    "robust_z", "spike", "retained",
)


# ---------------------------------------------------------------------------
#  A. Pooled within-cycle repeatability
# ---------------------------------------------------------------------------
def pooled_repeatability(observations: pd.DataFrame, *,
                         cycles=None, config: AnalysisConfig) -> pd.DataFrame:
    """Pooled within-repeat-group SD per span class.

    Groups are ``(point_id, station_id, cycle_id)``; the station component is
    used when the frame provides it. Repeats are deviations from the ordinary
    group mean and the pooled SD is
    ``sqrt(sum(dev^2) / (N_groups_total_obs - n_groups))`` restricted to
    groups with ``n >= 2`` (ALGORITHMS.md section 3).

    ``cycles`` is only consulted when the observation frame lacks
    ``cycle_id``/``station_id``; it may be a cycle table (times bound each
    cycle) or an array-like cycle assignment. The frame is never mutated.
    """
    obs = _prepare_observations(observations, cycles)
    keys = _repeat_group_columns(obs)
    if obs.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in POOLED_COLUMNS})

    obs = obs.copy()
    obs["time"] = pd.to_datetime(obs["time"])
    span = obs.groupby(keys, sort=False, dropna=False)["time"].transform(
        lambda s: s.max() - s.min())
    obs["_span"] = span

    rows: list[dict] = []
    for span_class in SPAN_CLASSES:
        if span_class == "same_minute":
            mask = obs["_span"] <= SAME_MINUTE
        elif span_class == "ge_10min":
            mask = obs["_span"] >= GE_TEN_MINUTES
        else:
            mask = pd.Series(True, index=obs.index)
        sub = obs[mask]
        counts = sub.groupby(keys, sort=False, dropna=False).size()
        counts = counts[counts >= 2]
        row: dict[str, object] = {
            "span_class": span_class,
            "n_obs": int(counts.sum()),
            "n_groups": int(counts.size),
        }
        for column, (out_name, factor) in REPEAT_VARIABLES.items():
            row[out_name] = (_pooled_sd(sub, keys, column, factor)
                             if column in sub.columns else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows, columns=list(POOLED_COLUMNS))


def _repeat_group_columns(observations: pd.DataFrame) -> list[str]:
    keys = ["point_id"]
    if "station_id" in observations.columns:
        keys.append("station_id")
    keys.append("cycle_id")
    return keys


def _pooled_sd(frame: pd.DataFrame, keys: list[str],
               column: str, factor: float) -> float:
    """Pooled SD of one variable over groups with at least two valid values."""
    data = frame[keys].copy()
    data["_v"] = pd.to_numeric(frame[column], errors="coerce").to_numpy()
    data = data.dropna(subset=["_v"])
    if data.empty:
        return float("nan")
    grouped = data.groupby(keys, sort=False, dropna=False)["_v"]
    counts = grouped.transform("size")
    means = grouped.transform("mean")
    keep = counts >= 2
    if not keep.any():
        return float("nan")
    n_total = float(int(keep.sum()))
    n_groups = float(data.loc[keep].groupby(keys, sort=False, dropna=False).ngroups)
    if n_total - n_groups <= 0:
        return float("nan")
    ss = float(((data.loc[keep, "_v"] - means[keep]) ** 2).sum())
    return float(np.sqrt(ss / (n_total - n_groups)) * factor)


def _prepare_observations(observations: pd.DataFrame,
                          cycles) -> pd.DataFrame:
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame")
    obs = observations.copy()
    if "time" not in obs.columns and "ts" in obs.columns:
        obs = obs.rename(columns={"ts": "time"})
    missing = {"point_id", "time"} - set(obs.columns)
    if missing:
        raise ValueError(f"observations missing required columns: {sorted(missing)}")
    if "cycle_id" not in obs.columns:
        obs = _attach_cycle_ids(obs, cycles)
    if "cycle_id" not in obs.columns:
        raise ValueError(
            "cannot group repeat observations: no cycle_id column and the "
            "supplied cycles object could not provide one")
    obs["time"] = pd.to_datetime(obs["time"])
    return obs


def _attach_cycle_ids(obs: pd.DataFrame, cycles) -> pd.DataFrame:
    """Attach ``cycle_id`` from a cycle table or an array-like assignment."""
    if cycles is None:
        return obs

    carried = getattr(cycles, "observations", None)
    if (isinstance(carried, pd.DataFrame) and "obs_id" in obs.columns
            and {"obs_id", "cycle_id"} <= set(carried.columns)):
        mapping = carried[["obs_id", "cycle_id"]]
        return obs.merge(mapping, on="obs_id", how="left",
                         validate="many_to_one")

    table = getattr(cycles, "cycles", cycles)
    if not isinstance(table, pd.DataFrame):
        values = np.asarray(table)
        if values.ndim != 1 or len(values) != len(obs):
            raise ValueError(
                "array-like cycles must be one-dimensional and match the "
                "number of observations")
        out = obs.reset_index(drop=True).copy()
        out["cycle_id"] = values
        return out

    table = table.copy()
    if {"obs_id", "cycle_id"} <= set(table.columns) and "obs_id" in obs.columns:
        return obs.merge(table[["obs_id", "cycle_id"]], on="obs_id",
                         how="left", validate="many_to_one")

    pairs = [("cycle_start", "cycle_end"), ("start", "end"), ("t0", "t1")]
    bounds = next((pair for pair in pairs if set(pair) <= set(table.columns)), None)
    if bounds and {"station_id", "cycle_id"} <= set(table.columns):
        if "station_id" not in obs.columns:
            raise ValueError(
                "a cycles table with time bounds needs observation station_id "
                "to attach cycle_id")
        left = obs.copy()
        left["time"] = pd.to_datetime(left["time"])
        right = table[["station_id", "cycle_id", bounds[0], bounds[1]]].copy()
        right[bounds[0]] = pd.to_datetime(right[bounds[0]])
        right[bounds[1]] = pd.to_datetime(right[bounds[1]])
        left = left.sort_values(["station_id", "time"])
        right = right.sort_values(["station_id", bounds[0]])
        merged = pd.merge_asof(left, right, left_on="time", right_on=bounds[0],
                               by="station_id", direction="backward",
                               allow_exact_matches=True)
        inside = ((merged["time"] >= merged[bounds[0]])
                  & (merged["time"] <= merged[bounds[1]]))
        merged.loc[~inside, "cycle_id"] = np.nan
        return merged.drop(columns=[bounds[0], bounds[1]])
    raise ValueError(
        "cannot attach cycle_id from the supplied cycles object: expected a "
        "cycle table with (station_id, cycle_id, cycle_start, cycle_end) or an "
        "observation-level mapping")


# ---------------------------------------------------------------------------
#  B. Robust residual noise and spikes
# ---------------------------------------------------------------------------
def estimate_noise(series: pd.DataFrame, *, config: AnalysisConfig,
                   floors: dict[str, float] | None = None) -> NoiseResult:
    """Rolling-median residual noise and retained spike flags.

    ``series`` is long with ``point_id``, ``ts``, ``metric`` and ``value``
    (optionally ``segment_id``). For every ``(point_id, metric)`` and every
    observation the residual is the value minus a centred rolling median over
    ``config.noise.rolling_window_hours`` with
    ``min_periods=config.noise.min_observations``; the robust SD is
    ``config.noise.robust_scale x MAD(residual)`` and is only reported once
    :data:`MIN_RESIDUALS` valid residuals exist. A spike is
    ``|residual / max(robust_sd, floor)| > config.noise.spike_z`` and is always
    retained; a metric without a supplied floor cannot be flagged unless a
    floor is passed through ``floors``.
    """
    required = {"point_id", "ts", "metric", "value"}
    missing = required - set(series.columns)
    if missing:
        raise ValueError(f"series missing required columns: {sorted(missing)}")

    cfg = config.noise
    window = pd.Timedelta(hours=float(cfg.rolling_window_hours))
    floor_map = dict(SPIKE_FLOORS_MM)
    if floors:
        floor_map.update({str(k): float(v) for k, v in floors.items()})

    pooled_rows: list[dict] = []
    series_frames: list[pd.DataFrame] = []
    flag_frames: list[pd.DataFrame] = []

    group_keys = ["point_id", "metric"]
    for (point_id, metric), group in series.groupby(group_keys, sort=False,
                                                    dropna=False):
        g = group.sort_values("ts").copy()
        g["ts"] = pd.to_datetime(g["ts"])
        g = g.dropna(subset=["ts"]).reset_index(drop=True)
        if g.empty:
            continue
        rolling = (g.set_index("ts")["value"]
                   .rolling(window=window, center=bool(cfg.centered),
                            min_periods=int(cfg.min_observations))
                   .median()
                   .to_numpy(dtype=float))
        values = pd.to_numeric(g["value"], errors="coerce").to_numpy(dtype=float)
        g["rolling_median_mm"] = rolling
        g["residual_mm"] = values - rolling

        residuals = pd.Series(g["residual_mm"], dtype=float).dropna()
        if len(residuals) >= MIN_RESIDUALS:
            mad = float((residuals - residuals.median()).abs().median())
            robust_sd = float(cfg.robust_scale) * mad
            status = STATUS_OK
            reason = None
        else:
            robust_sd = float("nan")
            status = STATUS_INSUFFICIENT
            reason = (f"only {len(residuals)} residual(s); at least "
                      f"{MIN_RESIDUALS} required")
        g["robust_sd_mm"] = robust_sd
        pooled_rows.append({
            "point_id": point_id,
            "metric": metric,
            "n": int(len(residuals)),
            "robust_sd_mm": robust_sd,
            "status": status,
            "reason": reason,
        })

        floor = floor_map.get(str(metric), float("nan"))
        denominator = np.maximum(np.full(len(g), robust_sd, dtype=float), floor)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = g["residual_mm"].to_numpy(dtype=float) / denominator
        g["robust_z"] = z
        g["spike"] = np.abs(z) > float(cfg.spike_z)
        g["retained"] = True

        if "segment_id" not in g.columns:
            g["segment_id"] = ""
        else:
            g["segment_id"] = g["segment_id"].fillna("")
        series_frames.append(g.reindex(columns=list(NOISE_SERIES_COLUMNS)))
        flag_cols = list(FLAG_COLUMNS)
        if "obs_id" in g.columns:
            flag_cols = ["obs_id", *flag_cols]
        flag_frames.append(g.reindex(columns=flag_cols))

    if not series_frames:
        return NoiseResult(
            pooled=pd.DataFrame({c: pd.Series(dtype="object")
                                 for c in POOLED_SUMMARY_COLUMNS}),
            series=pd.DataFrame({c: pd.Series(dtype="object")
                                 for c in NOISE_SERIES_COLUMNS}),
            flags=pd.DataFrame({c: pd.Series(dtype="object")
                                for c in FLAG_COLUMNS}),
            status=STATUS_INSUFFICIENT,
            reason="series contains no observations",
        )

    pooled = pd.DataFrame(pooled_rows, columns=list(POOLED_SUMMARY_COLUMNS))
    pooled = pooled.sort_values(["point_id", "metric"],
                                kind="stable").reset_index(drop=True)
    series_out = (pd.concat(series_frames, ignore_index=True)
                  .sort_values(["point_id", "metric", "ts"], kind="stable")
                  .reset_index(drop=True))
    flags = (pd.concat(flag_frames, ignore_index=True)
             .sort_values(["point_id", "metric", "ts"], kind="stable")
             .reset_index(drop=True))

    if (pooled["status"] == STATUS_OK).any():
        status = STATUS_OK
        reason = None
    else:
        status = STATUS_INSUFFICIENT
        reason = "no prism/metric had at least five valid residuals"
    return NoiseResult(pooled=pooled, series=series_out, flags=flags,
                       status=status, reason=reason)


# ---------------------------------------------------------------------------
#  C. Spike flags
# ---------------------------------------------------------------------------
def flag_spikes(series: pd.DataFrame, *, noise: NoiseResult,
                config: AnalysisConfig) -> pd.DataFrame:
    """Return the retained spike-flag table, recomputing only when needed."""
    flags = getattr(noise, "flags", None)
    if isinstance(flags, pd.DataFrame) and not flags.empty:
        return flags.copy()
    return estimate_noise(series, config=config).flags
