import numpy as np
import pandas as pd

from prismacore.robust import mad


def estimate(series, rows, config):
    series = series.copy()
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
        raw = rows[(rows.station == station) & (rows.pid == pid) & ~rows.parse_error]
        for col in ["d", "hz", "v"]:
            squares, n = 0.0, 0
            for _, repeat in raw.groupby("cycle"):
                if len(repeat) > 1:
                    values = repeat[col].to_numpy()
                    if col == "hz":
                        values = (values - values[0] + 180) % 360 - 180
                    squares += np.sum((values - values.mean()) ** 2)
                    n += len(values) - 1
            row["repeat_sd_" + col] = np.sqrt(squares / n) if n else np.nan
        row["repeat_sd_d_mm"] = row.pop("repeat_sd_d") * 1000
        row["repeat_sd_hz_arcsec"] = row.pop("repeat_sd_hz") * 3600
        row["repeat_sd_v_arcsec"] = row.pop("repeat_sd_v") * 3600
        stats.append(row)
    return series, pd.DataFrame(stats)
