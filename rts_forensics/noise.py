import numpy as np
import pandas as pd

from prismacore.robust import mad


def repeat_squares(rows):
    """Within-cycle squared deviations and degrees of freedom per prism/cycle.

    Hz is unwrapped relative to the first repeat so 359.99/0.01 readings stay adjacent.
    """
    valid = rows.loc[~rows.parse_error & rows.cycle.notna()]
    keys = [valid.station, valid.pid, valid.cycle]
    grouped = valid.groupby(keys, sort=False)
    out = pd.DataFrame({"n": grouped.size()})
    for col in ["d", "hz", "v"]:
        values = valid[col]
        if col == "hz":
            values = (values - grouped.hz.transform("first") + 180) % 360 - 180
        deviation = values - values.groupby(keys, sort=False).transform("mean")
        out["ss_" + col] = (deviation**2).groupby(keys, sort=False).sum()
    out = out[out.n > 1]
    out["dof"] = out.n - 1
    return out.reset_index()


def pooled_repeatability(rows):
    """Pooled within-cycle repeat SD per station and overall (handoff stage 1 noise)."""
    squares = repeat_squares(rows)
    records = []
    for station, g in [*squares.groupby("station"), ("all", squares)]:
        dof = g.dof.sum()
        records.append(
            {
                "station": station,
                "n_repeat_cycles": len(g),
                "dof": int(dof),
                "repeat_sd_d_mm": np.sqrt(g.ss_d.sum() / dof) * 1000 if dof else np.nan,
                "repeat_sd_hz_arcsec": np.sqrt(g.ss_hz.sum() / dof) * 3600 if dof else np.nan,
                "repeat_sd_v_arcsec": np.sqrt(g.ss_v.sum() / dof) * 3600 if dof else np.nan,
            }
        )
    return pd.DataFrame(records)


def estimate(series, rows, config):
    series = series.copy()
    squares = repeat_squares(rows).groupby(["station", "pid"])[["ss_d", "ss_hz", "ss_v", "dof"]].sum()
    stats = []
    for (station, pid), g in series.groupby(["station", "pid"]):
        g = g.sort_values("ts")
        row = {"station": station, "pid": pid}
        for column in ["los_raw", "ver_raw", "tan_raw"]:
            rolling = g.set_index("ts")[column].rolling("24h", center=True).median().to_numpy()
            residual = g[column].to_numpy() - rolling
            sigma = mad(residual)
            row["sigma_" + column] = sigma
            # Zero MAD with a nonzero residual: retain an explicit spike flag.
            spike = np.abs(residual) > config["screening"]["spike_z"] * sigma
            series.loc[g.index, "spike_" + column] = spike
        repeat = squares.loc[(station, pid)] if (station, pid) in squares.index else None
        for col, scale, name in [("d", 1000, "mm"), ("hz", 3600, "arcsec"), ("v", 3600, "arcsec")]:
            dof = repeat.dof if repeat is not None else 0
            row[f"repeat_sd_{col}_{name}"] = np.sqrt(repeat["ss_" + col] / dof) * scale if dof else np.nan
        stats.append(row)
    return series, pd.DataFrame(stats)
