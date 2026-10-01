"""
pipeline.py — deterministic orchestration shared by the CLI and the dashboard.

One pipeline, one audit trail: both interfaces call :func:`run_analysis`, so
there is a single set of algorithms and a single provenance record.

Ordering follows the handoff. Raw products are computed first (parse → stations
and cycles → prism cycles → displacements → noise → rates → exploratory
screening and reliability). The per-cycle frame fit then runs over the chosen
reference set and adds **experimental** frame-corrected columns *beside* the raw
ones. Classification never merges movement concern with measurement
reliability, and no rule that the handoff does not define is invented: such
results carry ``not_configured`` and a reason.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import classify as cl
from . import cycles as cycles_mod
from . import displacement as displacement_mod
from . import frame as frame_mod
from . import gis as gis_mod
from . import noise as noise_mod
from . import rates as rates_mod
from . import report as report_mod
from .config import AnalysisConfig, default_config, validate_config
from .io import read_geomos
from .models import (
    NOT_CONFIGURED,
    STATUS_OK,
    ForensicResult,
    FrameResult,
    RunResult,
    empty_frame,
)
from .provenance import (
    build_manifest,
    hash_outputs,
    runtime_audit,
    write_manifest,
    write_raw_sources,
)
from .tests import run_forensics as run_forensic_tests

# canonical long-series metric -> displacement column
METRIC_COLUMNS: dict[str, str] = {
    "d_rad": "d_rad_mm",
    "d_ver": "d_vert_mm",
    "d_tan": "d_tan_mm",
    "d_east": "d_east_mm",
    "d_north": "d_north_mm",
    "d_up": "d_up_mm",
}
NET_METRICS = ("d_rad", "d_ver", "d_tan")


@dataclass
class OutputManifest:
    out_dir: str
    manifest_path: str
    outputs: dict[str, str] = field(default_factory=dict)
    status: str = STATUS_OK
    reason: str | None = None


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _long_displacement_series(displacements: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    columns = ["point_id", "segment_id", "station_id", "cycle_id", "ts", "obs_ids"]
    for metric, column in METRIC_COLUMNS.items():
        if column not in displacements.columns:
            continue
        part = displacements[[c for c in columns if c in displacements.columns]].copy()
        part["metric"] = metric
        part["value"] = pd.to_numeric(displacements[column], errors="coerce")
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=[*columns, "metric", "value"])
    return pd.concat(parts, ignore_index=True)


def _attach_azimuth(prism_cycles: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable local azimuth per prism from median coordinates."""
    out = prism_cycles.copy()
    if "az_deg" in out.columns and out["az_deg"].notna().any():
        return out
    keys = [c for c in ("point_id", "segment_id") if c in out.columns]
    geometry = (out.groupby(keys, dropna=False)
                   .agg(te=("te_m", "median"), tn=("tn_m", "median"),
                        se=("se_m", "median"), sn=("sn_m", "median"))
                   .reset_index())
    geometry["az_deg"] = np.degrees(np.arctan2(geometry.te - geometry.se,
                                               geometry.tn - geometry.sn)) % 360.0
    return out.merge(geometry[keys + ["az_deg"]], on=keys, how="left")


def _net_table(frame: pd.DataFrame, value_columns: list[str],
               window_hours: float, anchor: pd.Timestamp) -> pd.DataFrame:
    """Final-window minus first-window median per point/segment (mm)."""
    keys = [c for c in ("point_id", "segment_id") if c in frame.columns]
    rows: list[dict[str, Any]] = []
    work = frame.dropna(subset=["ts"]).sort_values("ts")
    for key, group in work.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, key, strict=True))
        t0, t1 = group.ts.min(), group.ts.max()
        first = group[group.ts <= t0 + pd.Timedelta(hours=window_hours)]
        last = group[group.ts >= t1 - pd.Timedelta(hours=window_hours)]
        record["n_first"] = len(first)
        record["n_last"] = len(last)
        record["stale"] = bool((anchor - t1) > pd.Timedelta(hours=window_hours))
        for column in value_columns:
            if column not in group.columns:
                record[f"{column}_net"] = np.nan
                continue
            record[f"{column}_net"] = (last[column].median()
                                       - first[column].median())
        rows.append(record)
    return pd.DataFrame(rows)


def _prism_reliability_inputs(displacements: pd.DataFrame,
                              cycles_res: Any,
                              noise_res: noise_mod.NoiseResult,
                              window_hours: float) -> pd.DataFrame:
    keys = ["point_id", "segment_id"]
    groups = list(displacements.groupby(keys, dropna=False))
    expected = (cycles_res.cycles.groupby("station_id").size().to_dict()
                if cycles_res is not None and len(cycles_res.cycles) else {})
    station_of = (displacements.groupby(keys, dropna=False)["station_id"].first()
                  .to_dict())
    sigma = {}
    if len(noise_res.series):
        tmp = (noise_res.series.groupby(["point_id", "segment_id", "metric"])
               ["robust_sd_mm"].median())
        sigma = tmp.to_dict()
    spikes = {}
    if len(noise_res.flags) and "spike" in noise_res.flags.columns:
        spikes = (noise_res.flags.groupby(["point_id", "segment_id"])["spike"]
                  .sum().to_dict())
    rows = []
    for key, group in groups:
        key_t = key if isinstance(key, tuple) else (key,)
        point_id, segment_id = key_t
        station_id = station_of.get(key)
        n_cycles = len(group)
        n_expected = max(int(expected.get(station_id, n_cycles)), 1)
        ts = pd.to_datetime(group["ts"])
        gap = (ts.sort_values().diff().dt.total_seconds() / 3600).max()
        obs_last = int((ts > ts.max() - pd.Timedelta(hours=window_hours)).sum())
        rows.append({
            "point_id": point_id,
            "segment_id": segment_id,
            "station_id": station_id,
            "n_cycles": n_cycles,
            "coverage_pct": float(min(100.0, 100.0 * n_cycles / n_expected)),
            "max_gap_hours": float(gap) if pd.notna(gap) else np.nan,
            "obs_last48h": obs_last,
            "obs_in_last48h": obs_last,
            "sigma_los_mm": sigma.get((point_id, segment_id, "d_rad"), np.nan),
            "sigma_vert_mm": sigma.get((point_id, segment_id, "d_ver"), np.nan),
            "n_spike_flags": int(spikes.get(key, 0)),
        })
    return pd.DataFrame(rows)


def _summary_metrics(long: pd.DataFrame, noise_res: noise_mod.NoiseResult,
                     rates_df: pd.DataFrame, anchor: pd.Timestamp,
                     window_hours: float) -> pd.DataFrame:
    keys = ["point_id", "segment_id"]
    sigma = {}
    if len(noise_res.series):
        sigma = (noise_res.series.groupby(["point_id", "segment_id", "metric"])
                 ["robust_sd_mm"].median().to_dict())
    ci: dict[tuple, tuple[float, float]] = {}
    if len(rates_df):
        window = "30d"
        rows = rates_df[rates_df.window == window]
        for row in rows.itertuples(index=False):
            ci[(row.point_id, row.segment_id, row.metric)] = (
                float(row.ci_low), float(row.ci_high))
    out = []
    for key, group in long.groupby(keys, dropna=False):
        point_id, segment_id = key
        group = group.sort_values("ts")
        t0, t1 = group.ts.min(), group.ts.max()
        first = group[group.ts <= t0 + pd.Timedelta(hours=window_hours)]
        last = group[group.ts >= t1 - pd.Timedelta(hours=window_hours)]
        for metric, metric_group in group.groupby("metric"):
            first_m = first[first.metric == metric]["value"]
            last_m = last[last.metric == metric]["value"]
            net = (last_m.median() - first_m.median()
                   if len(first_m) and len(last_m) else np.nan)
            low, high = ci.get((point_id, segment_id, metric), (np.nan, np.nan))
            out.append({
                "point_id": point_id,
                "segment_id": segment_id,
                "station_id": metric_group.station_id.iloc[0],
                "metric": metric,
                "net_mm": net,
                "sigma_mm": sigma.get((point_id, segment_id, metric), np.nan),
                "ci_low": low,
                "ci_high": high,
                "n_prism_cycles": len(metric_group),
                "stale": bool((anchor - t1) > pd.Timedelta(hours=window_hours)),
            })
    return pd.DataFrame(out)


def _timeseries_24h(series: pd.DataFrame, anchor: pd.Timestamp,
                    night_hours: set[int], block_hours: float = 24.0) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame()
    work = series.dropna(subset=["ts"]).copy()
    delta = (anchor - work.ts).dt.total_seconds() / (block_hours * 3600.0)
    work["block_index"] = np.floor(delta).astype("Int64")
    work["block_end"] = anchor - pd.to_timedelta(work.block_index, unit="h") * block_hours
    work["block_start"] = work["block_end"] - pd.Timedelta(hours=block_hours)
    work["hour"] = work.ts.dt.hour
    work["is_night"] = work["hour"].isin(night_hours)
    value_columns = [c for c in ("los_raw_mm", "los_fc_mm", "tan_raw_mm",
                                 "tan_fc_mm", "ver_raw_mm", "ver_fc_mm",
                                 "d_east_mm", "d_north_mm", "d_up_mm")
                     if c in work.columns]
    group_keys = ["point_id", "segment_id", "block_start", "block_end"]
    agg = {column: "median" for column in value_columns}
    agg["is_night"] = "sum"
    agg["ts"] = "size"
    out = work.groupby(group_keys, dropna=False).agg(agg).reset_index()
    out = out.rename(columns={"ts": "n_obs", "is_night": "n_night"})
    return out


def _build_frame_series(prism_cycles: pd.DataFrame) -> pd.DataFrame:
    """Build the long frame series from the enriched prism-cycle table."""
    enriched = _attach_azimuth(prism_cycles)
    columns = ["point_id", "station_id", "segment_id", "cycle_id", "cycle_start",
               "d_rad_mm", "d_hz_arcsec", "d_vert_mm", "v_deg", "d_m", "az_deg"]
    series = enriched[[c for c in columns if c in enriched.columns]].copy()
    return series.rename(columns={"cycle_start": "ts", "v_deg": "v", "d_m": "D",
                                  "d_rad_mm": "dD", "d_hz_arcsec": "dHz_as",
                                  "d_vert_mm": "ver_raw"})


def _empty_frame_result(reason: str) -> FrameResult:
    return FrameResult(correction_status=NOT_CONFIGURED, experimental=True,
                       status=NOT_CONFIGURED, reason=reason)


# ---------------------------------------------------------------------------
#  Orchestration
# ---------------------------------------------------------------------------
def run_analysis(source: Any, *, config: AnalysisConfig | None = None) -> RunResult:
    """Run the full audit pipeline on one or more RTS exports."""
    cfg = copy.deepcopy(config) if config is not None else default_config()
    audit = validate_config(cfg)

    parsed = read_geomos(source, config=cfg)
    observations = parsed.observations

    cycles_res = cycles_mod.build_cycles(observations, config=cfg)
    observations = cycles_res.observations

    prism_cycles = displacement_mod.aggregate_prism_cycles(observations, config=cfg)
    displacements = displacement_mod.compute_displacements(
        prism_cycles, segments=cycles_res.segments, config=cfg)

    long = _long_displacement_series(displacements.displacements)
    anchor = (displacements.displacements["ts"].max()
              if len(displacements.displacements) else pd.Timestamp.utcnow().tz_localize(None))
    window_hours = float(cfg.baseline.window_hours or 48.0)

    noise_long = long[long.metric.isin(NET_METRICS)]
    noise_res = noise_mod.estimate_noise(noise_long, config=cfg)
    rates_df = rates_mod.compute_rates(displacements.displacements, config=cfg)

    reliability_inputs = _prism_reliability_inputs(
        displacements.displacements, cycles_res, noise_res, window_hours)
    reliability = cl.grade_reliability(reliability_inputs, config=cfg)

    frame_series = _build_frame_series(displacements.prism_cycles)
    frame_summary = reliability_inputs.copy()
    grade_lookup = {}
    if len(reliability):
        grade_lookup = dict(zip(
            zip(reliability.point_id, reliability.segment_id, strict=False),
            reliability.grade, strict=False))
    frame_summary["reliability"] = [
        grade_lookup.get((p, s), NOT_CONFIGURED)
        for p, s in zip(frame_summary.point_id, frame_summary.segment_id,
                        strict=False)]
    try:
        selection = frame_mod.select_frame_set(frame_summary, config=cfg)
        frames = frame_mod.fit_frames(frame_series, frame_set=selection, config=cfg)
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        selection = None
        frames = _empty_frame_result(f"frame fit unavailable: {exc}")

    corrected = frames.corrected.copy() if len(frames.corrected) else pd.DataFrame()
    if len(corrected) and "ts" not in corrected.columns:
        key = [c for c in ("point_id", "segment_id", "cycle_id")
               if c in corrected.columns]
        timing = displacements.prism_cycles[key + ["cycle_start"]].rename(
            columns={"cycle_start": "ts"})
        corrected = corrected.merge(timing, on=key, how="left")
    net_raw = _net_table(displacements.prism_cycles,
                         ["d_rad_mm", "d_vert_mm", "d_tan_mm"], window_hours, anchor)
    net_fc = (_net_table(corrected, ["los_fc_mm", "ver_fc_mm", "tan_fc_mm"],
                         window_hours, anchor)
              if len(corrected) else pd.DataFrame())

    summary_metrics = _summary_metrics(long, noise_res, rates_df, anchor,
                                       window_hours)
    concern = cl.screen_movement(summary_metrics, config=cfg)
    tarp = cl.apply_tarp(summary_metrics, tarp=cfg.tarp)

    coordinates = (observations.groupby("point_id")
                   .agg(east=("te_m", "median"), north=("tn_m", "median"),
                        v_deg=("v_deg", "median"))
                   .reset_index())
    station_series = pd.DataFrame()
    if len(cycles_res.cycles):
        station_series = cycles_res.cycles.rename(
            columns={"t0": "ts", "sh_m": "height", "orientation_deg": "orientation"}
        )[["station_id", "ts", "height", "orientation"]]
    try:
        forensics = run_forensic_tests(
            long, coordinates=coordinates, station_series=station_series, config=cfg)
    except Exception as exc:  # noqa: BLE001 - recorded explicitly
        forensics = ForensicResult(
            statuses=pd.DataFrame([{"test": "all", "status": "failed",
                                    "reason": str(exc)}]))

    classification = cl.classify(summary_metrics, forensics, config=cfg)
    if len(tarp):
        classification.tarp = tarp

    prism_summary = reliability_inputs.merge(
        net_raw, on=["point_id", "segment_id"], how="left")
    if len(net_fc):
        prism_summary = prism_summary.merge(
            net_fc, on=["point_id", "segment_id"], how="left")
    prism_summary = prism_summary.merge(
        reliability[["point_id", "segment_id", "grade", "status"]].rename(
            columns={"grade": "reliability_grade", "status": "reliability_status"}),
        on=["point_id", "segment_id"], how="left")
    prism_summary["frame_member"] = prism_summary.point_id.isin(
        selection.members if selection is not None else [])

    ts_source = (corrected
                 if len(corrected)
                 else displacements.displacements)
    night_hours = set(cfg.forensics.day_bands.get("night", []))
    timeseries = _timeseries_24h(ts_source, anchor, night_hours)

    tables: dict[str, pd.DataFrame] = {
        "observations": observations,
        "cycle_table": cycles_res.cycles,
        "processing_changes": cycles_res.changes.changes,
        "prism_cycles": prism_cycles,
        "baselines": displacements.baselines,
        "displacements": displacements.displacements,
        "source_links": displacements.source_links,
        "noise_repeatability": noise_res.pooled,
        "noise_flags": noise_res.flags,
        "rates": rates_df,
        "prism_summary": prism_summary,
        "frame_cycles": frames.coefficients,
        "frame_corrected": corrected,
        "timeseries_24h": timeseries,
        "concern": concern,
        "reliability": reliability,
        "tarp": tarp,
        "evidence": forensics.evidence.get("register", empty_frame(
            {"test": "str", "hypothesis": "str", "metric": "str"})),
        "investigation_register": forensics.investigations,
        "event_register": forensics.evidence.get("events", pd.DataFrame()),
        "field_checks": pd.DataFrame(columns=[
            "check_id", "priority", "title", "target", "question",
            "status", "outcome", "notes", "photo"]),
        "sight_lines": observations.assign(
            az_deg=np.degrees(np.arctan2(observations.te_m - observations.se_m,
                                         observations.tn_m - observations.sn_m)) % 360.0
        ).groupby("point_id").agg(
            station_id=("station_id", "first"),
            range_m=("d_m", "median"),
            zenith_deg=("v_deg", "median"),
            az_deg=("az_deg", "median")).reset_index(),
        "candidate_zones": pd.DataFrame(),
    }
    noise_flags = noise_res.flags
    manifest = build_manifest(
        sources={s.name: s.sha256 for s in parsed.sources},
        config_hash=cfg.hash(),
        audit=runtime_audit(),
        outputs={},
        settings={
            "pipeline": "rts_forensics.pipeline.run_analysis",
            "frame_correction": "experimental",
            "tarp": "not_configured" if cfg.tarp is None else "configured",
        },
    )

    result = RunResult(
        observations=observations,
        parsed=parsed,
        cycles=cycles_res,
        displacements=displacements,
        noise=noise_res,
        frames=frames,
        forensics=forensics,
        classification=classification,
        rates=rates_df,
        tables=tables,
        config=cfg,
        audit=audit,
        manifest=manifest,
    )
    result.tables["noise_flags"] = noise_flags
    return result


# ---------------------------------------------------------------------------
#  Export
# ---------------------------------------------------------------------------
def export_run(run: RunResult, out_dir: str | Path) -> OutputManifest:
    """Write every product and a SHA-256 manifest into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg: AnalysisConfig = run.config

    (out / "effective_config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    if cfg.provenance.get("preserve_raw_bytes") and run.parsed.raw_bytes:
        write_raw_sources(run.parsed.raw_bytes, out / "raw")

    paths: list[Path] = []
    paths += list(report_mod.write_tables(run, out))
    paths += list(report_mod.plot_results(run, out))
    report_paths = report_mod.render_report(run, out)
    paths += [Path(report_paths.md), Path(report_paths.html)]

    gpkg = gis_mod.write_geopackage(run, out / "rts_forensics.gpkg", config=cfg)
    if gpkg.status == STATUS_OK:
        paths.append(Path(gpkg.path))
    qml = gis_mod.write_qml(out / "movement_vectors.qml", config=cfg)
    paths.append(Path(qml))

    manifest = dict(run.manifest)
    manifest["outputs"] = hash_outputs(paths, root=out)
    manifest_path = write_manifest(manifest, out / "manifest.json")
    run.manifest = manifest
    run.outputs = manifest["outputs"]
    return OutputManifest(out_dir=str(out), manifest_path=str(manifest_path),
                          outputs=manifest["outputs"])
