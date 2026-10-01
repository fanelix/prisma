import re

import numpy as np
import pandas as pd

from .investigations import hour_matched_net, paired_bias
from .rates import slope


def group_for_pid(pid):
    """Keep named subfamilies; strip the final numeric target suffix only."""
    if "-" in pid or "_" in pid:
        return re.sub(r"[-_]\d+$", "", pid) if re.search(r"[-_]\d+$", pid) else re.sub(r"\d+$", "", pid)
    return re.sub(r"\d+$", "", pid)


def screening(net, sigma, rate, n, config, vertical=False):
    if n < config["screening"]["min_cycles"]:
        return "insufficient data"
    if not np.isfinite(net):
        return "unavailable"
    floor = config["screening"]["vertical_floor_mm" if vertical else "los_floor_mm"]
    limit = max(floor, config["screening"]["sigma_multiplier"] * sigma) if np.isfinite(sigma) else floor
    same = (net > 0 and rate["low"] > 0) or (net < 0 and rate["high"] < 0)
    if abs(net) >= limit and same:
        return "detected (exploratory)"
    if abs(net) >= limit or same:
        return "possible (exploratory)"
    return "no credible movement detected within available observations and sensitivity"


def tarp_status(g, config):
    rule = config["tarp"]
    if rule is None:
        return "not configured"
    s = g.sort_values("ts").set_index("ts")[rule["metric"]]
    if s.empty or not np.isfinite(s.iloc[-1]):
        return "unavailable"
    gaps = s.index.to_series().diff().dt.total_seconds().div(3600)
    boundaries = (gaps > rule["max_gap_hours"]) | s.isna() | s.shift().isna()
    regimes = boundaries.cumsum()
    avg = pd.Series(np.nan, index=s.index)
    for _, part in s.groupby(regimes):
        if part.notna().all():
            avg.loc[part.index] = part.rolling(pd.Timedelta(hours=rule["averaging_hours"])).mean()
    threshold = rule["threshold"]
    hit = (
        avg.abs() >= threshold
        if rule["direction"] == "absolute"
        else (avg >= threshold if rule["direction"] == "above" else avg <= threshold)
    )
    # Persistence may not bridge a missing observation interval.
    max_gap = rule["max_gap_hours"]
    start = None
    previous = None
    for time, flag in hit.items():
        if previous is not None and (time - previous).total_seconds() / 3600 > max_gap:
            start = None
        if flag:
            start = time if start is None else start
        else:
            start = None
        previous = time
    if start is not None and (s.index[-1] - start).total_seconds() / 3600 >= rule["persistence_hours"]:
        return str(rule["label"])
    return "configured condition not met in available observations"


def summarize(series, noise, config):
    records = []
    for (station, pid), g in series.groupby(["station", "pid"]):
        g = g.sort_values("ts")
        ns = noise[(noise.station == station) & (noise.pid == pid)].iloc[0]
        all_station = series[series.station == station]
        early = g[g.ts < g.ts.min() + pd.Timedelta(hours=config["baseline_hours"])]
        late = g[g.ts > g.ts.max() - pd.Timedelta(hours=config["end_hours"])]
        end = all_station.ts.max()
        row = {
            "station": station,
            "pid": pid,
            "group": group_for_pid(pid),
            "e": early.e.median(),
            "n": early.n.median(),
            "z": early.z.median(),
            "baseline_d_m": g.baseline_d_m.iloc[0],
            "az_deg": g.baseline_az_deg.iloc[0],
            "n_observations": int(g.n_repeat.sum()),
            "n_cycles": len(g),
            "coverage": len(g) / all_station.cycle.nunique(),
            "longest_gap_hours": g.ts.diff().dt.total_seconds().max() / 3600,
            "observations_final_48h": int(g[g.ts > end - pd.Timedelta(hours=48)].n_repeat.sum()),
            "lost_final_48h": bool(g.ts.max() < end - pd.Timedelta(hours=48)),
            "first_observation": g.ts.min(),
            "last_observation": g.ts.max(),
            "spike_fraction": float(g.spike_los_raw.mean()),
            "sigma_los_mm": ns.sigma_los_raw,
            "sigma_vertical_mm": ns.sigma_ver_raw,
            "reliability": "ungraded (site policy not supplied)",
            "tarp_status": tarp_status(g, config),
            "experimental_frame": True,
        }
        night_cycles = all_station[all_station.night].cycle.nunique()
        row["night_success"] = g[g.night].cycle.nunique() / night_cycles if night_cycles else np.nan
        bias, low, high, _ = paired_bias(g)
        row.update(daytime_bias_mm=bias, daytime_bias_low_mm=low, daytime_bias_high_mm=high)
        for prefix, column in [
            ("los_raw", "los_raw"),
            ("vertical_raw", "ver_raw"),
            ("los_fc", "los_fc"),
            ("vertical_fc", "ver_fc"),
            ("tangential_fc", "tan_fc"),
        ]:
            row["net_" + prefix + "_mm"] = (
                late[column].median() - early[column].median()
                if column in g and late[column].notna().any() and early[column].notna().any()
                else np.nan
            )
            row["net_" + prefix + "_hour_matched_mm"] = (
                hour_matched_net(g, column, config["end_hours"], config["baseline_hours"])
                if column in g
                else np.nan
            )
            for days, label in [(30, "30d"), (7, "7d"), (3, "72h")]:
                rate = (
                    slope(g, column, days) if column in g else {"rate": np.nan, "low": np.nan, "high": np.nan}
                )
                for stat in ["rate", "low", "high"]:
                    row[prefix + "_" + label + "_" + stat] = rate[stat]
        raw_rate = {k: row["los_raw_30d_" + k] for k in ["rate", "low", "high"]}
        row["raw_los_concern"] = screening(row["net_los_raw_mm"], ns.sigma_los_raw, raw_rate, len(g), config)
        use_fc = (
            "los_fc" in g
            and np.isfinite(g.los_fc.iloc[-1])
            and g.los_fc.notna().sum() >= config["screening"]["min_cycles"]
        )
        metric = "los_fc" if use_fc else "los_raw"
        rate = {k: row[metric + "_30d_" + k] for k in ["rate", "low", "high"]}
        row["movement_concern"] = screening(
            row["net_" + metric + "_hour_matched_mm"], ns.sigma_los_raw, rate, len(g), config
        )
        row["concern_basis"] = metric + " (experimental network-relative)" if use_fc else metric
        v_rate = {k: row["vertical_raw_30d_" + k] for k in ["rate", "low", "high"]}
        row["vertical_concern"] = screening(
            row["net_vertical_raw_hour_matched_mm"], ns.sigma_ver_raw, v_rate, len(g), config, True
        )
        row["los_concern"] = row["movement_concern"]
        if row["vertical_concern"].startswith("detected") and not row["movement_concern"].startswith(
            "detected"
        ):
            row["movement_concern"] = "detected vertical (exploratory; independent verification required)"
            row["concern_basis"] = "ver_raw; atmospheric and station-frame checks required"
        row["final_class"] = row["movement_concern"]
        flags = []
        if row["lost_final_48h"]:
            flags.append("no observations in final 48h; no stability inference")
        if row["spike_fraction"] > 0:
            flags.append("spikes retained")
        bias_floor = config["detection"]["bias_floor_mm"]
        if np.isfinite(low) and (low > 0 or high < 0) and (bias_floor is None or abs(bias) >= bias_floor):
            flags.append("day/night bias candidate")
        if g.coord_segment.nunique() > 1:
            flags.append("delivered coordinates segmented")
        if "coord_mixed" in g and g.coord_mixed.any():
            flags.append("mixed-regime repeat cycle: coordinate products unavailable")
        row["flags"] = "; ".join(flags)
        rules = config["reliability"]
        if rules:
            row["reliability"] = "D"
            for grade in ["A", "B", "C"]:
                rule = rules[grade]
                if (
                    row["coverage"] >= rule["coverage_min"]
                    and row["longest_gap_hours"] <= rule["max_gap_hours"]
                    and row["sigma_los_mm"] <= rule["max_los_noise_mm"]
                    and row["spike_fraction"] <= rule["max_spike_fraction"]
                ):
                    row["reliability"] = grade
                    break
        az = np.radians(row["az_deg"])
        # LOS includes a vertical projection; convert it to horizontal radial.
        radial = (
            row["net_los_fc_mm"] - row["net_vertical_fc_mm"] * np.cos(np.radians(g.baseline_v_deg.iloc[0]))
        ) / np.sin(np.radians(g.baseline_v_deg.iloc[0]))
        row["vector_e_mm"] = radial * np.sin(az) + row["net_tangential_fc_mm"] * np.cos(az)
        row["vector_n_mm"] = radial * np.cos(az) - row["net_tangential_fc_mm"] * np.sin(az)
        row["vector_z_mm"] = row["net_vertical_fc_mm"]
        row["los_fit_se_mm"] = (
            g.los_frame_se_mm.median()
            if "los_frame_se_mm" in g and g.los_frame_se_mm.notna().any()
            else np.nan
        )
        row["source_refs"] = ";".join(g.source_refs)
        records.append(row)
    return pd.DataFrame(records)
