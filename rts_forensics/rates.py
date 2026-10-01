"""24-hour blocks counted backwards from the final observation."""

import numpy as np
import pandas as pd

from prismacore.robust import theilslopes

VALUES = ["los_raw", "los_fc", "ver_raw", "ver_fc", "rad_raw", "tan_raw", "tan_fc"]
COORDINATES = ["dE_mm", "dN_mm", "dZ_mm"]


def blocks24(series):
    """One grouped aggregation; a per-block Python loop dominated real-data runtime."""
    grouping = [x for x in ["station", "pid"] if x in series]
    s = series.reset_index(drop=True)
    target = s.groupby(grouping, sort=False).ngroup()
    anchor = s.ts.groupby(target).transform("max")
    first = s.ts.groupby(target).transform("min")
    block = np.floor((anchor - s.ts).dt.total_seconds() / 86400).astype(int)
    grouped = s.groupby([target.rename("_target"), block.rename("_block")], sort=False)
    out = grouped[grouping].first()
    end = grouped.ts.size().index.get_level_values("_block")
    end = anchor.groupby([target, block], sort=False).first().to_numpy() - pd.to_timedelta(end, unit="D")
    out["block_start"] = end - pd.Timedelta(days=1)
    out["block_end"] = end
    out["ts"] = grouped.ts.median()
    out["n"] = grouped.size()
    out["n_night"] = grouped.night.sum().astype(int) if "night" in s else 0
    out["partial"] = (
        out.block_start.to_numpy() < first.groupby([target, block], sort=False).first().to_numpy()
    )
    values = [c for c in VALUES if c in s]
    if values:
        out[values] = grouped[values].median()
    # Never aggregate coordinate displacement across segment boundaries.
    if "coord_segment" in s:
        single = grouped.coord_segment.nunique().eq(1)
        out["coord_segment"] = grouped.coord_segment.nth(0).set_axis(out.index).where(single).astype(float)
        coordinates = [c for c in COORDINATES if c in s]
        if coordinates:
            out[coordinates] = grouped[coordinates].median().where(single, axis=0)
    else:
        out["coord_segment"] = np.nan
    if "source_refs" in s:
        out["source_refs"] = grouped.source_refs.agg(";".join)
    return out.sort_values(grouping + ["ts"]).reset_index(drop=True)


def _complete_block_medians(g, columns):
    """Complete 24-hour block medians of ts and columns, as blocks24 computes them."""
    targets = [c for c in ["station", "pid"] if c in g]
    if targets and len(g[targets].drop_duplicates()) > 1:
        blocks = blocks24(g)
        return blocks[~blocks.partial]
    # Single target: same grouped medians without blocks24's provenance columns.
    anchor = g.ts.max()
    block = np.floor((anchor - g.ts).dt.total_seconds() / 86400).astype(int).to_numpy()
    medians = g[["ts", *columns]].groupby(block).median()
    end = anchor - pd.to_timedelta(medians.index, unit="D")
    medians = medians[np.asarray(end - pd.Timedelta(days=1) >= g.ts.min())]
    return medians.sort_values("ts").reset_index(drop=True)


def slopes(series, columns, days):
    """Theil-Sen rates for several columns; 24-hour blocks are built once per window."""
    keep = [c for c in ["station", "pid", "ts", "night"] if c in series]
    g = series[keep + [c for c in columns if c not in keep]]
    g = g[g.ts >= g.ts.max() - pd.Timedelta(days=days)]
    blocks = None
    rates = {}
    for column in columns:
        count = int(g[column].notna().sum())
        if count < 2:
            rates[column] = {"rate": np.nan, "low": np.nan, "high": np.nan, "n": count}
            continue
        if days > 3:
            if blocks is None:
                blocks = _complete_block_medians(g, columns)
            h = blocks
        else:
            h = g
        h = h.dropna(subset=[column])
        if len(h) < 2:
            rates[column] = {"rate": np.nan, "low": np.nan, "high": np.nan, "n": len(h)}
            continue
        elapsed = (h.ts - h.ts.min()).dt.total_seconds() / 86400
        rate, _, low, high = theilslopes(h[column], elapsed, 0.95)
        rates[column] = {"rate": rate, "low": low, "high": high, "n": len(h)}
    return rates


def slope(series, column, days):
    return slopes(series, [column], days)[column]
