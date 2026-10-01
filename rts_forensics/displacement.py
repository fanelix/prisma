"""Raw polar products in mm; delivered coordinates never bridge a regime change."""

import numpy as np
import pandas as pd

from .parse import circular_median


def prism_cycles(rows, config):
    valid = rows.loc[~rows.parse_error].copy()
    valid["source_refs"] = valid.source.astype(str) + ":" + valid.src_line.astype(str)
    grouped = valid.groupby(["station", "cycle", "pid"], sort=False)
    aggregation = {key: "median" for key in ["ts", "v", "d", "e", "n", "z", "st_e", "st_n", "st_h"]}
    aggregation.update(
        hz=circular_median,
        coord_segment="max",
        source_refs=lambda x: ";".join(x),
        src_line=lambda x: ",".join(x.astype(str)),
    )
    series = grouped.agg(aggregation)
    series["n_repeat"] = grouped.size()
    series["coord_mixed"] = grouped.coord_segment.nunique().gt(1)
    series.loc[series.coord_mixed, "coord_segment"] = -1
    series = series.rename(columns={"src_line": "src_lines"}).reset_index()
    series = series.sort_values(["station", "pid", "ts"]).reset_index(drop=True)
    series["hd"] = series.d * np.sin(np.radians(series.v))
    series["vertical"] = series.d * np.cos(np.radians(series.v))
    series["az_deg"] = np.degrees(np.arctan2(series.e - series.st_e, series.n - series.st_n)) % 360
    series["orientation_deg"] = (series.az_deg - series.hz + 180) % 360 - 180
    coord_distance = np.sqrt(
        (series.e - series.st_e) ** 2 + (series.n - series.st_n) ** 2 + (series.z - series.st_h) ** 2
    )
    series["coordinate_scale_ppm"] = (coord_distance / series.d - 1) * 1e6
    hd_coord = np.hypot(series.e - series.st_e, series.n - series.st_n)
    denom = hd_coord**2
    # Apparent k only: export height reduction can confound this diagnostic.
    series["apparent_refraction_k"] = 1 - 2 * 6371000 * (series.z - series.st_h - series.vertical) / denom
    for _, g in series.groupby(["station", "pid"]):
        baseline = g[g.ts < g.ts.min() + pd.Timedelta(hours=config["baseline_hours"])]
        d0, hd0, v0 = baseline.d.median(), baseline.hd.median(), baseline.vertical.median()
        hz0 = circular_median(baseline.hz)
        idx = g.index
        series.loc[idx, "los_raw"] = (g.d - d0) * 1000
        series.loc[idx, "ver_raw"] = (g.vertical - v0) * 1000
        series.loc[idx, "rad_raw"] = (g.hd - hd0) * 1000
        dhz = (g.hz - hz0 + 180) % 360 - 180
        series.loc[idx, "dhz_arcsec"] = dhz * 3600
        series.loc[idx, "tan_raw"] = g.hd * np.radians(dhz) * 1000
        series.loc[idx, "baseline_d_m"] = d0
        series.loc[idx, "baseline_hd_m"] = hd0
        series.loc[idx, "baseline_az_deg"] = circular_median(baseline.az_deg)
        series.loc[idx, "baseline_v_deg"] = baseline.v.median()
        series.loc[idx, "baseline_start"] = baseline.ts.min()
        series.loc[idx, "baseline_end"] = baseline.ts.max()
    for _, g in series.groupby(["station", "pid", "coord_segment"]):
        baseline = g[g.ts < g.ts.min() + pd.Timedelta(hours=config["baseline_hours"])]
        for column, output in [("e", "dE_mm"), ("n", "dN_mm"), ("z", "dZ_mm")]:
            series.loc[g.index, output] = (g[column] - baseline[column].median()) * 1000
    series.loc[series.coord_mixed, ["dE_mm", "dN_mm", "dZ_mm"]] = np.nan
    start, end = config["night_hours"]
    hour = series.ts.dt.hour
    series["night"] = ((hour >= start) | (hour < end)) if start > end else ((hour >= start) & (hour < end))
    series["hour_class"] = hour // config["hour_class_hours"]
    return series
