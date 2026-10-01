"""24-hour blocks counted backwards from the final observation."""

import numpy as np
import pandas as pd

from prismacore.robust import theilslopes


def blocks24(series):
    records = []
    grouping = [x for x in ["station", "pid"] if x in series]
    for keys, g in series.groupby(grouping, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        anchor = g.ts.max()
        block_id = np.floor((anchor - g.ts).dt.total_seconds() / 86400).astype(int)
        for block, b in g.groupby(block_id):
            end = anchor - pd.Timedelta(days=int(block))
            row = dict(zip(grouping, keys))
            row.update(
                block_start=end - pd.Timedelta(days=1),
                block_end=end,
                ts=b.ts.median(),
                n=len(b),
                n_night=int(b.night.sum()) if "night" in b else 0,
                partial=bool(end - pd.Timedelta(days=1) < g.ts.min()),
            )
            for column in ["los_raw", "los_fc", "ver_raw", "ver_fc", "rad_raw", "tan_raw", "tan_fc"]:
                if column in b:
                    row[column] = b[column].median() if b[column].notna().any() else np.nan
            # Never aggregate coordinate displacement across segment boundaries.
            row["coord_segment"] = (
                b.coord_segment.iloc[0] if "coord_segment" in b and b.coord_segment.nunique() == 1 else np.nan
            )
            for column in ["dE_mm", "dN_mm", "dZ_mm"]:
                if column in b:
                    row[column] = b[column].median() if b.coord_segment.nunique() == 1 else np.nan
            if "source_refs" in b:
                row["source_refs"] = ";".join(b.source_refs)
            records.append(row)
    return pd.DataFrame(records).sort_values(grouping + ["ts"]).reset_index(drop=True)


def slope(series, column, days):
    g = series[series.ts >= series.ts.max() - pd.Timedelta(days=days)].copy()
    if g[column].notna().sum() < 2:
        return {"rate": np.nan, "low": np.nan, "high": np.nan, "n": int(g[column].notna().sum())}
    if days > 3:
        g = blocks24(g)
        g = g[~g.partial]
    g = g.dropna(subset=[column])
    if len(g) < 2:
        return {"rate": np.nan, "low": np.nan, "high": np.nan, "n": len(g)}
    elapsed = (g.ts - g.ts.min()).dt.total_seconds() / 86400
    rate, _, low, high = theilslopes(g[column], elapsed, 0.95)
    return {"rate": rate, "low": low, "high": high, "n": len(g)}
