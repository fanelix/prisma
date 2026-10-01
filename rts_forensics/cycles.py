"""Station assignment by overlapping station-easting ranges per target.

This has no guessed distance threshold. Ambiguous/shared-station targets need
explicit station_by_pid configuration. Stations are ordered west to east.
"""

import numpy as np
import pandas as pd


def assign_cycles(rows, config):
    rows = rows.copy()
    valid = rows.loc[~rows.parse_error]
    spans = valid.groupby("pid").st_e.agg(["min", "max"]).sort_values("min")
    assignment = {}
    end, group = -np.inf, 0
    for pid, span in spans.iterrows():
        if span["min"] > end:
            group += 1
        end = max(end, span["max"])
        assignment[pid] = f"S{group}"
    assignment.update(config["station_by_pid"])
    rows["station"] = rows.pid.map(assignment)
    rows["cycle"] = pd.Series(pd.NA, index=rows.index, dtype="Int64")
    rows["coord_segment"] = pd.Series(pd.NA, index=rows.index, dtype="Int64")
    for station, g in rows.loc[~rows.parse_error].groupby("station"):
        g = g.sort_values([c for c in ["ts", "src_line"] if c in g])
        gap = config["station_gap_minutes"].get(station, config["cycle_gap_minutes"])
        cycle = (g.ts.diff().dt.total_seconds().div(60).gt(gap)).cumsum()
        rows.loc[g.index, "cycle"] = cycle.astype("Int64")
        start = g.ts.min()
        first = g[g.ts < start + pd.Timedelta(hours=config["baseline_hours"])]
        varying = (first[["st_e", "st_n", "st_h"]].nunique() > 1).any()
        dates = [pd.Timestamp(x) for x in config["processing_changes"].get(station, [])]
        if not varying:
            changed = g[["st_e", "st_n", "st_h"]].ne(g[["st_e", "st_n", "st_h"]].iloc[0]).any(axis=1)
            if changed.any():
                dates.append(g.loc[changed, "ts"].min())
        dates = sorted(set(dates))
        segment = np.zeros(len(g), dtype=int)
        for date in dates:
            segment += (g.ts >= date).astype(int).to_numpy()
        rows.loc[g.index, "coord_segment"] = segment
    return rows


def cycle_table(rows):
    valid = rows.loc[~rows.parse_error]
    return valid.groupby(["station", "cycle"], as_index=False).agg(
        start=("ts", "min"),
        end=("ts", "max"),
        n_observations=("pid", "size"),
        n_prisms=("pid", "nunique"),
        station_e=("st_e", "median"),
        station_n=("st_n", "median"),
        station_h=("st_h", "median"),
    )
