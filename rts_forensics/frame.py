"""Huber IRLS joint rotation/translation/scale and vertical network fit."""

import numpy as np
import pandas as pd

PARAMETERS = [
    "rotation_arcsec",
    "translation_e_mm",
    "translation_n_mm",
    "scale_ppm",
    "height_mm",
    "vertical_index_mm_km",
    "tilt_sin_mm_km",
    "tilt_cos_mm_km",
]


def huber_fit(design, observed, sigma, tuning):
    x = np.asarray(design, dtype=float) / np.asarray(sigma)[:, None]
    y = np.asarray(observed, dtype=float) / np.asarray(sigma)
    if len(y) <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("Insufficient geometry or rank-deficient frame")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(50):
        residual = y - x @ beta
        weight = np.minimum(1, tuning / np.maximum(np.abs(residual), np.finfo(float).eps))
        root = np.sqrt(weight)
        new = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
        if np.allclose(new, beta, rtol=1e-9, atol=1e-9):
            beta = new
            break
        beta = new
    residual = y - x @ beta
    weight = np.minimum(1, tuning / np.maximum(np.abs(residual), np.finfo(float).eps))
    covariance = np.linalg.pinv(x.T @ (weight[:, None] * x))
    # Nominal supplied sigmas are lower bounds; never report zero uncertainty.
    covariance *= max(1, float(np.sum(weight * residual**2) / (len(y) - x.shape[1])))
    return beta, np.sqrt(np.diag(covariance)), residual, weight


def design_matrix(g):
    az = np.radians(g.baseline_az_deg.to_numpy())
    v = np.radians(g.baseline_v_deg.to_numpy())
    hd = g.baseline_hd_m.to_numpy()
    km = g.baseline_d_m.to_numpy() / 1000
    n = len(g)
    x = np.zeros((3 * n, 8))
    x[:n, 0] = 1
    x[:n, 1] = -206.265 * np.cos(az) / hd
    x[:n, 2] = 206.265 * np.sin(az) / hd
    x[n : 2 * n, 1] = -np.sin(az) * np.sin(v)
    x[n : 2 * n, 2] = -np.cos(az) * np.sin(v)
    x[n : 2 * n, 3] = km
    x[2 * n :, 4] = 1
    x[2 * n :, 5] = km
    x[2 * n :, 6] = km * np.sin(az)
    x[2 * n :, 7] = km * np.cos(az)
    return x


def fit_frames(series, summary, config):
    s = series.copy()
    for column in ["los_fc", "tan_fc", "ver_fc", "los_frame_se_mm", "tan_frame_se_mm", "ver_frame_se_mm"]:
        s[column] = np.nan
    s["frame_member"] = False
    frames, membership = [], []
    rules = config["frame"]
    explicit = set(rules["include"])
    excluded = set(rules["exclude"]) | set(rules["references_under_test"])
    for station, station_series in s.groupby("station"):
        station_summary = summary[summary.station == station]
        eligible = set()
        for r in station_summary.itertuples():
            reason = "eligible"
            if r.pid in excluded:
                reason = "excluded/reference under test"
            elif explicit:
                if r.pid not in explicit:
                    reason = "not explicitly included"
            elif r.n_cycles < rules["min_cycles"]:
                reason = "too few cycles"
            elif r.movement_concern.startswith("detected"):
                reason = "raw movement candidate"
            elif getattr(r, "raw_los_concern", "").startswith("detected"):
                reason = "raw LOS movement candidate"
            elif getattr(r, "vertical_concern", "").startswith("detected"):
                reason = "raw vertical movement candidate"
            elif (
                config["detection"]["bias_floor_mm"] is not None
                and np.isfinite(r.daytime_bias_low_mm)
                and (r.daytime_bias_low_mm > 0 or r.daytime_bias_high_mm < 0)
                and (abs(r.daytime_bias_mm) >= config["detection"]["bias_floor_mm"])
            ):
                reason = "persistent day/night bias candidate"
            elif config["reliability"] and r.reliability != "A":
                reason = "reliability below configured A"
            if reason == "eligible":
                eligible.add(r.pid)
            membership.append(
                {
                    "station": station,
                    "pid": r.pid,
                    "included": reason == "eligible",
                    "reason": reason,
                    "selection": "explicit"
                    if explicit
                    else (
                        "configured reliability A"
                        if config["reliability"]
                        else "provisional; no grading policy"
                    ),
                }
            )
        s.loc[station_series.index, "frame_member"] = station_series.pid.isin(eligible)
        baseline_coords = station_series[["st_e", "st_n", "st_h"]].iloc[0]
        for cycle, g in station_series.groupby("cycle"):
            f = g[g.pid.isin(eligible)]
            row = {
                "station": station,
                "cycle": int(cycle),
                "ts": g.ts.median(),
                "n_frame": len(f),
                "experimental": True,
                "frame_policy": "explicit" if explicit else "automatic provisional",
                "condition_number": np.nan,
            }
            for field in PARAMETERS:
                row[field] = np.nan
                row[field + "_se"] = np.nan
            for src, out in [("st_e", "exported_e_mm"), ("st_n", "exported_n_mm"), ("st_h", "exported_h_mm")]:
                row[out] = (g[src].median() - baseline_coords[src]) * 1000
            row["orientation_deg"] = g.orientation_deg.median()
            try:
                if len(f) < 4:
                    raise ValueError(
                        "Insufficient geometry or rank-deficient frame (four vertical parameters)"
                    )
                design = design_matrix(f)
                observation = np.concatenate([f.dhz_arcsec, f.los_raw, f.ver_raw])
                sigma = np.repeat(
                    [rules["sigma_hz_arcsec"], rules["sigma_d_mm"], rules["sigma_v_mm"]], len(f)
                )
                beta, se, residual, weights = huber_fit(design, observation, sigma, rules["huber_tuning"])
                row.update(zip(PARAMETERS, beta))
                row.update(zip([p + "_se" for p in PARAMETERS], se))
                row.update(
                    status="fit",
                    condition_number=float(np.linalg.cond(design / sigma[:, None])),
                    rms_normalized=float(np.sqrt(np.mean(residual**2))),
                )
                x = design_matrix(g)
                prediction = x @ beta
                n = len(g)
                s.loc[g.index, "tan_fc"] = (
                    g.tan_raw.to_numpy() - g.hd.to_numpy() * np.radians(prediction[:n] / 3600) * 1000
                )
                s.loc[g.index, "los_fc"] = g.los_raw.to_numpy() - prediction[n : 2 * n]
                s.loc[g.index, "ver_fc"] = g.ver_raw.to_numpy() - prediction[2 * n :]
                # Parameter-diagonal propagation is a diagnostic approximation.
                # Exact covariance is reconstructed from the final IRLS weights.
                fit_x = design / sigma[:, None]
                cov = np.linalg.pinv(fit_x.T @ (weights[:, None] * fit_x))
                cov *= max(1, float(np.sum(weights * residual**2) / (len(residual) - 8)))
                uncertainty = np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", x, cov, x)))
                s.loc[g.index, "tan_frame_se_mm"] = uncertainty[:n] * g.hd.to_numpy() * np.pi / 648000 * 1000
                s.loc[g.index, "los_frame_se_mm"] = uncertainty[n : 2 * n]
                s.loc[g.index, "ver_frame_se_mm"] = uncertainty[2 * n :]
            except ValueError as error:
                row["status"] = str(error)
            frames.append(row)
    frames = pd.DataFrame(frames)
    if len(frames):
        frames["orientation_shift_arcsec"] = frames.groupby("station").orientation_deg.transform(
            lambda x: (np.degrees(np.unwrap(np.radians(x))) - x.iloc[0]) * 3600
        )
        frames["applied_minus_fitted_arcsec"] = -frames.orientation_shift_arcsec - frames.rotation_arcsec
    return s, frames, pd.DataFrame(membership)
