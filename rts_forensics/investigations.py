"""Evidence tests compute effects without guessing site decision thresholds."""

import itertools

import numpy as np
import pandas as pd

from .rates import blocks24, slope


def hour_matched_net(series, column, window_hours, baseline_hours=None):
    first, last = series.ts.min(), series.ts.max()
    baseline_hours = window_hours if baseline_hours is None else baseline_hours
    # Narrow first: copying wide provenance columns dominated real-data runtime.
    s = series[["ts", column]].dropna(subset=[column])
    if s.empty:
        return np.nan
    s = s.assign(hour=s.ts.dt.hour)
    early = s[s.ts < first + pd.Timedelta(hours=baseline_hours)].groupby("hour")[column].median()
    late = s[s.ts > last - pd.Timedelta(hours=window_hours)].groupby("hour")[column].median()
    common = early.index.intersection(late.index)
    return float((late[common] - early[common]).median()) if len(common) else np.nan


def paired_bias(g, column="los_raw"):
    """Same-date day-minus-night; 95% CI on date-paired means (not alarms)."""
    g = g[["ts", "night", column]].dropna(subset=[column])
    g = g.assign(date=g.ts.dt.floor("D"))
    paired = g.groupby(["date", "night"])[column].median().unstack("night")
    if False not in paired or True not in paired:
        return np.nan, np.nan, np.nan, 0
    effects = (paired[False] - paired[True]).dropna()
    if effects.empty:
        return np.nan, np.nan, np.nan, 0
    if len(effects) < 2:
        return effects.median(), np.nan, np.nan, len(effects)
    # Descriptive normal approximation; same-date pairing does not remove all trends.
    half = 1.95996398454 * effects.std(ddof=1) / np.sqrt(len(effects))
    return float(effects.median()), float(effects.mean() - half), float(effects.mean() + half), len(effects)


def leave_one_out_median(values):
    """Median of the other finite values in the group, for each finite value."""
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) > 1:
        others = np.tile(values[finite], (len(finite), 1))
        np.fill_diagonal(others, np.nan)
        out[finite] = np.nanmedian(others, axis=1)
    return out


def _record(test, pid="", station="", effect=np.nan, **extra):
    return {
        "test": test,
        "pid": pid,
        "station": station,
        "effect_mm": effect,
        "hypothesis": "measurement artefact or local movement",
        "expected_signature": "effect persists under like-hour comparison",
        "competing_explanation": "atmosphere, geometry, or common ground movement",
        "limitation": "network-relative; external control unavailable",
        "implication": "exploratory evidence; verify independently",
        **extra,
    }


def investigate(series, frames, summary, config):
    from scipy.stats import mannwhitneyu, spearmanr

    records, events = [], []
    stations = {s: f[["ts", "cycle", "night"]] for s, f in series.groupby("station")}
    # Handoff reference inference: cycle-detrended levels. The leave-one-out cycle
    # median removes the common mode without letting a target detrend itself.
    common = series.groupby(["station", "cycle"])
    series = series.assign(
        **{
            column + "_cycle_detrended": series[column] - common[column].transform(leave_one_out_median)
            for column in ["ver_raw", "dhz_arcsec"]
        }
    )
    for (station, pid), g in series.groupby(["station", "pid"]):
        g = g.sort_values("ts")
        station_series = stations[station]
        station_start = station_series.ts.min()
        for period in range(
            int((station_series.ts.max() - station_start).total_seconds() // (7 * 86400)) + 1
        ):
            start = station_start + pd.Timedelta(days=7 * period)
            end = start + pd.Timedelta(days=7)
            expected = station_series[(station_series.ts >= start) & (station_series.ts < end)]
            observed = g[(g.ts >= start) & (g.ts < end)]
            for label, night in [("night", True), ("day", False)]:
                n_expected = expected[expected.night == night].cycle.nunique()
                n_observed = observed[observed.night == night].cycle.nunique()
                records.append(
                    _record(
                        "observation_success_over_time",
                        pid,
                        station,
                        n_observed / n_expected if n_expected else np.nan,
                        period=str(start),
                        sampling=label,
                        n_expected=n_expected,
                        n_observed=n_observed,
                        effect_units="fraction",
                        limitation="relative to network cycles; target-specific observation schedule unknown",
                    )
                )
        angle = g[["ts", "night", "v", "dhz_arcsec"]].copy()
        angle["zenith_arcsec"] = angle.v * 3600
        for column in ["dhz_arcsec", "zenith_arcsec"]:
            effect, low_angle, high_angle, n_angle = paired_bias(angle, column)
            records.append(
                _record(
                    "daytime_angle_effect",
                    pid,
                    station,
                    effect,
                    metric=column,
                    ci_low=low_angle,
                    ci_high=high_angle,
                    n_pairs=n_angle,
                    effect_units="arcsec",
                )
            )
        bias, low, high, n = paired_bias(g)
        records.append(
            _record(
                "daytime_bias",
                pid,
                station,
                bias,
                ci_low_mm=low,
                ci_high_mm=high,
                n_pairs=n,
                limitation="day/night geometry and real trends can confound paired effects",
            )
        )
        weekly = g[["ts", "night", "los_raw"]].assign(
            week=((g.ts - g.ts.min()).dt.total_seconds() // (7 * 86400)).astype(int)
        )
        for week, w in weekly.groupby("week"):
            effect, *_ = paired_bias(w)
            records.append(_record("weekly_daytime_bias", pid, station, effect, period=str(week)))
        for column in ["los_raw", "ver_raw", "los_fc", "ver_fc"]:
            net = hour_matched_net(g, column, config["end_hours"], config["baseline_hours"])
            records.append(_record("hour_matched_net", pid, station, net, metric=column))
        for label, mask in [("night", g.night), ("day", ~g.night)]:
            t = g[mask]
            effect = (
                hour_matched_net(t, "los_raw", config["end_hours"], config["baseline_hours"])
                if len(t)
                else np.nan
            )
            rate = slope(t, "los_raw", 30) if len(t) else {"rate": np.nan}
            records.append(_record(label + "_only", pid, station, effect, rate_mm_day=rate["rate"]))
        block = blocks24(g[[c for c in ["station", "pid", "ts", "night", "los_fc"] if c in g]])
        for half, b in enumerate(np.array_split(np.arange(len(block)), 2)):
            t = block.iloc[b]
            effect = t.los_fc.iloc[-1] - t.los_fc.iloc[0] if len(t) else np.nan
            records.append(_record("temporal_split", pid, station, effect, period=str(half)))
        # Compare fixed slopes, hinge, and three-segment fits; no fit-improvement threshold.
        b = block.dropna(subset=["los_fc"])
        if len(b) >= 6:
            x = (b.ts - b.ts.min()).dt.total_seconds().to_numpy() / 86400
            y = b.los_fc.to_numpy()
            base = np.column_stack([np.ones(len(x)), x])
            base_sse = float(np.sum((y - base @ np.linalg.lstsq(base, y, rcond=None)[0]) ** 2))
            best = (np.inf, None)
            for k in range(2, len(x) - 2):
                design = np.column_stack([base, np.maximum(0, x - x[k])])
                sse = float(np.sum((y - design @ np.linalg.lstsq(design, y, rcond=None)[0]) ** 2))
                if sse < best[0]:
                    best = (sse, b.ts.iloc[k])
            records.append(
                _record(
                    "hinge",
                    pid,
                    station,
                    base_sse - best[0],
                    candidate_time=str(best[1]),
                    limitation="descriptive best fit; selection bias, no significance claim",
                )
            )
            best3 = (np.inf, None, None)
            for k, right in itertools.combinations(range(2, len(x) - 2), 2):
                if right - k < 2:
                    continue
                design = np.column_stack([base, np.maximum(0, x - x[k]), np.maximum(0, x - x[right])])
                sse = float(np.sum((y - design @ np.linalg.lstsq(design, y, rcond=None)[0]) ** 2))
                if sse < best3[0]:
                    best3 = (sse, b.ts.iloc[k], b.ts.iloc[right])
            records.append(
                _record(
                    "three_segment",
                    pid,
                    station,
                    base_sse - best3[0] if np.isfinite(best3[0]) else np.nan,
                    candidate_time=f"{best3[1]};{best3[2]}" if np.isfinite(best3[0]) else "",
                    limitation="descriptive optimization; no inferential change-point threshold",
                )
            )
        f = frames[frames.station == station]
        merged = g.merge(f[["cycle", "exported_h_mm", "orientation_shift_arcsec"]], on="cycle", how="inner")
        for (raw, control), detrending in itertools.product(
            [("ver_raw", "exported_h_mm"), ("dhz_arcsec", "orientation_shift_arcsec")],
            ["cycle_detrended", "time_differenced"],
        ):
            if detrending == "cycle_detrended":
                a = merged[raw + "_cycle_detrended"].to_numpy()
                b = merged[control].to_numpy()
                limitation = (
                    "levels minus leave-one-out cycle median of the station's other targets; "
                    "correlation does not identify a resection reference"
                )
            else:
                a = merged[raw].diff().to_numpy()[1:]
                b = merged[control].diff().to_numpy()[1:]
                limitation = (
                    "differenced robustness check; insensitive to slow co-variation such as "
                    "subsidence; correlation does not identify a resection reference"
                )
            ok = np.isfinite(a) & np.isfinite(b)
            rho, p = (
                spearmanr(a[ok], b[ok])
                if ok.sum() > 2 and np.ptp(a[ok]) > 0 and np.ptp(b[ok]) > 0
                else (np.nan, np.nan)
            )
            records.append(
                _record(
                    "reference_inference",
                    pid,
                    station,
                    rho,
                    metric=raw,
                    control_source=control,
                    detrending=detrending,
                    effect_units="Spearman rho",
                    p_value=p,
                    p_bonferroni=min(1, p * summary[summary.station == station].shape[0] * 2)
                    if np.isfinite(p)
                    else np.nan,
                    limitation=limitation,
                )
            )
        # Source references make every investigation locatable in the export.
        for record in records:
            if record["pid"] == pid and record["station"] == station:
                record["source_refs"] = ";".join(g.source_refs)
    for station, f in frames.groupby("station"):
        f = f.sort_values("ts")
        for control, response, expected in [
            ("exported_h_mm", "height_mm", -1),
            ("exported_e_mm", "translation_e_mm", 1),
            ("exported_n_mm", "translation_n_mm", 1),
        ]:
            b = f[[control, response]].dropna()
            fit = (
                np.polyfit(b[control], b[response], 1)[0]
                if len(b) > 2 and b[control].nunique() > 1
                else np.nan
            )
            records.append(
                _record(
                    "resection_validation",
                    station=station,
                    effect=fit,
                    metric=response,
                    expected_slope=expected,
                    limitation="network fit can share deformation with station motion",
                )
            )
        jump = config["detection"]["orientation_step_arcsec"]
        if jump is not None:
            for pos in np.flatnonzero(f.rotation_arcsec.diff().abs().to_numpy() >= jump):
                row = f.iloc[pos]
                before = f[(f.ts < row.ts) & (f.ts >= row.ts - pd.Timedelta(hours=config["baseline_hours"]))]
                after = f[(f.ts >= row.ts) & (f.ts < row.ts + pd.Timedelta(hours=config["end_hours"]))]
                common_hours = set(before.ts.dt.hour) & set(after.ts.dt.hour)
                effects = [
                    after[after.ts.dt.hour == h].rotation_arcsec.median()
                    - before[before.ts.dt.hour == h].rotation_arcsec.median()
                    for h in common_hours
                ]
                effect = float(np.median(effects)) if effects else np.nan
                records.append(
                    _record(
                        "hour_balanced_frame_step",
                        station=station,
                        effect=effect,
                        candidate_time=str(row.ts),
                        metric="rotation_arcsec",
                    )
                )
                events.append(
                    {
                        "event_id": f"E{len(events) + 1:03}",
                        "class": "frame step candidate",
                        "station": station,
                        "prisms": "network",
                        "start": row.ts,
                        "end": row.ts,
                        "effect": effect,
                        "evidence": "configured threshold and hour-balanced network frame",
                        "alternatives": "common ground movement or fit geometry",
                        "confidence": "exploratory",
                        "follow_up": "station logs and independent survey",
                        "status": "open",
                    }
                )
        stat = summary[summary.station == station].copy()
        ranks = stat.los_raw_7d_rate.rank(pct=True)
        for index, r in stat.iterrows():
            records.append(
                _record(
                    "pre_loss_trend_rank",
                    r.pid,
                    station,
                    r.los_raw_7d_rate,
                    percentile=ranks.loc[index],
                    lost=bool(r.lost_final_48h),
                    effect_units="mm/day",
                    limitation="last available 7-day window; selected missingness, no causal inference",
                )
            )
            # Rank distances in range and circular angle to avoid an invented physical cutoff.
            delta_range = (stat.baseline_d_m - r.baseline_d_m).abs()
            delta_angle = ((stat.az_deg - r.az_deg + 180) % 360 - 180).abs()
            score = delta_range.rank() + delta_angle.rank()
            controls = stat.loc[score.drop(index).nsmallest(7).index]
            values = controls.daytime_bias_mm.dropna()
            control_bias = values.median() if len(values) else np.nan
            records.append(
                _record(
                    "same_geometry_bias_control",
                    r.pid,
                    station,
                    r.daytime_bias_mm - control_bias,
                    control_bias_mm=control_bias,
                    members=",".join(controls.pid),
                    limitation="seven targets nearest in combined range/angle rank; local conditions may differ",
                )
            )
        lost = stat[stat.lost_final_48h]
        retained = stat[~stat.lost_final_48h]
        if len(lost) and len(retained):
            u, p = mannwhitneyu(lost.baseline_d_m, retained.baseline_d_m, alternative="two-sided")
            records.append(
                _record(
                    "lost_vs_retained_distance",
                    station=station,
                    effect=u,
                    p_value=p,
                    limitation="distance association does not establish why targets were lost",
                )
            )
        candidates = stat[stat.movement_concern.str.startswith("detected")]
        locations = stat[["e", "n"]].to_numpy()
        if len(candidates):
            moving = candidates[["e", "n"]].to_numpy()
            distance = np.sqrt(((locations[:, None, :] - moving[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
            stat["distance_to_candidates_m"] = distance
            if len(lost) and len(retained):
                mask = stat.lost_final_48h.to_numpy()
                observed = np.median(distance[mask]) - np.median(distance[~mask])
                rng = np.random.default_rng(0)
                null = []
                for _ in range(999):
                    perm = rng.permutation(mask)
                    null.append(np.median(distance[perm]) - np.median(distance[~perm]))
                p = (1 + np.sum(np.asarray(null) <= observed)) / (len(null) + 1)
                records.append(
                    _record(
                        "lost_distance_permutation",
                        station=station,
                        effect=observed,
                        p_value=p,
                        limitation="moving set selected from same data; association only",
                    )
                )
        # Seven-neighbour placebos are a handoff diagnostic, not automatic clusters.
        nets = stat.net_los_fc_mm.to_numpy()
        for i, r in enumerate(stat.itertuples()):
            nearest = np.argsort(np.sum((locations - locations[i]) ** 2, axis=1))[:7]
            value = np.nanmedian(nets[nearest]) if np.isfinite(nets[nearest]).any() else np.nan
            placebo = []
            for j in range(len(stat)):
                if j in nearest:
                    continue
                other = np.argsort(np.sum((locations - locations[j]) ** 2, axis=1))[:7]
                if np.isfinite(nets[other]).any():
                    placebo.append(np.nanmedian(nets[other]))
            records.append(
                _record(
                    "spatial_placebo",
                    r.pid,
                    station,
                    value,
                    n_placebo=len(placebo),
                    n_more_extreme=sum(abs(v) >= abs(value) for v in placebo),
                    members=",".join(stat.pid.iloc[nearest]),
                    limitation="overlapping seven-neighbour groups; descriptive, not independent p-values",
                )
            )
        if len(candidates) > 1:
            wide = series[series.station == station].pivot(index="cycle", columns="pid", values="los_raw")
            for a, b in itertools.combinations(candidates.pid, 2):
                pair = (wide[a] - wide[b]).dropna()
                records.append(
                    _record(
                        "pairwise_los",
                        a + "," + b,
                        station,
                        pair.iloc[-1] - pair.iloc[0] if len(pair) > 1 else np.nan,
                    )
                )
        radius = config["detection"]["cluster_radius_m"]
        if radius is not None and len(candidates):
            remaining = set(candidates.index)
            while remaining:
                todo = [remaining.pop()]
                members = []
                while todo:
                    idx = todo.pop()
                    members.append(idx)
                    r = candidates.loc[idx]
                    neighbours = [
                        k
                        for k in remaining
                        if np.hypot(candidates.loc[k, "e"] - r.e, candidates.loc[k, "n"] - r.n) <= radius
                        and np.sign(candidates.loc[k, "net_los_fc_mm"]) == np.sign(r.net_los_fc_mm)
                    ]
                    remaining.difference_update(neighbours)
                    todo.extend(neighbours)
                group = candidates.loc[members]
                records.append(
                    _record(
                        "candidate_cluster",
                        station=station,
                        effect=group.net_los_fc_mm.median(),
                        members=",".join(group.pid),
                        limitation="configured spatial radius; concern remains experimental",
                    )
                )
    for r in summary.itertuples():
        if r.movement_concern.startswith("detected") or r.lost_final_48h:
            events.append(
                {
                    "event_id": f"E{len(events) + 1:03}",
                    "class": r.movement_concern if not r.lost_final_48h else "lost target",
                    "station": r.station,
                    "prisms": r.pid,
                    "start": r.first_observation,
                    "end": r.last_observation,
                    "effect": r.net_los_fc_mm,
                    "evidence": "raw/corrected series and coverage",
                    "alternatives": "measurement artefact or network-frame leakage",
                    "confidence": "exploratory",
                    "follow_up": "independent survey, site observations, instrument logs",
                    "status": "open",
                }
            )
    columns = [
        "event_id",
        "class",
        "station",
        "prisms",
        "start",
        "end",
        "effect",
        "evidence",
        "alternatives",
        "confidence",
        "follow_up",
        "status",
    ]
    return pd.DataFrame(records), pd.DataFrame(events, columns=columns)
