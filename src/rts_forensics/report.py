"""
report.py — CSV tables, exploratory figures, Markdown/HTML report and manifest.

The public functions accept either a
:class:`rts_forensics.models.RunResult` or a plain mapping with equivalent
keys (``tables``, ``parsed``, ``config``, ``manifest``, ``observations``,
``frames``, ``classification``, ``displacements``, ``noise``, ``rates``,
``forensics``, ``audit``). The small :func:`_get` helper hides that
difference so every producer shares one code path.

Honesty rules enforced here:

* raw and frame-corrected values are always shown together; the corrected
  values are labelled EXPERIMENTAL and relative to the chosen network;
* movement concern and measurement reliability appear in separate sections
  and are never merged;
* whenever no TARP is configured the report states explicitly that the
  output is exploratory and that no alarm state exists;
* the mandatory limitations from ``PROJECT_HANDOFF.md`` section 8 are always
  printed;
* the manifest hashes every output except itself, so it is not a recursive
  digest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from . import provenance
from .config import is_not_configured
from .models import STATUS_OK, table_dictionary

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

#: Mandatory limitations, verbatim in meaning from PROJECT_HANDOFF.md section 8.
LIMITATIONS: tuple[str, ...] = (
    "No slope stability or factor of safety can be established from RTS data alone.",
    "No alarm status may be issued without a site TARP.",
    "No absolute movement can be established without external control: "
    "the frame is defined by the network.",
    "No causes can be attributed without drivers such as rainfall, blasting, "
    "stacking or geology data.",
    "Resection references are not identifiable from the export.",
    "Unobserved periods are not evidence of stability.",
)

TARP_NOTICE = (
    "TARP is not configured. No alarm state is produced and every label below "
    "is exploratory; it must never be treated as a safety threshold."
)

FRAME_EXPERIMENTAL_NOTE = (
    "Frame correction is EXPERIMENTAL and relative to the chosen network: "
    "corrected values are not absolute ground movement."
)

#: Column candidates used when looking for optional series fields.
_POINT_COLS = ("point_id",)
_TIME_COLS = ("ts", "time", "cycle_start", "block_start", "cycle_id")
_RAW_LOS_COLS = ("d_rad_mm", "los_raw_mm", "los_mm")
_FC_LOS_COLS = ("los_fc_mm", "d_rad_fc_mm")
_RAW_VERT_COLS = ("d_vert_mm", "ver_raw_mm", "d_up_mm", "ver_mm")
_FC_VERT_COLS = ("ver_fc_mm", "d_vert_fc_mm")
_EAST_COLS = ("east_out_m", "east_m", "te_m", "east", "x")
_NORTH_COLS = ("north_out_m", "north_m", "tn_m", "north", "y")
_CONCERN_COLS = ("movement_concern", "concern_class", "concern", "final_class", "class")
_RELIABILITY_COLS = ("reliability_grade", "grade", "reliability")

_MISSING = object()

_MAX_PLOT_PRISMS = 6
_MAX_MAP_LABELS = 30


@dataclass
class ReportPaths:
    """Paths of the report artifacts written by :func:`render_report`."""

    md: Path
    html: Path


# ---------------------------------------------------------------------------
#  RunResult / mapping access helpers
# ---------------------------------------------------------------------------
def _get(run: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping or a result object with attributes."""
    if run is None:
        return default
    if isinstance(run, Mapping):
        return run.get(key, default)
    return getattr(run, key, default)


def _dig(run: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path, treating both missing keys and ``None`` as absent."""
    current = run
    for part in path.split("."):
        current = _get(current, part, _MISSING)
        if current is _MISSING or current is None:
            return default
    return current


def _as_frame(value: Any) -> pd.DataFrame | None:
    """Return ``value`` when it is a DataFrame, else ``None``."""
    return value if isinstance(value, pd.DataFrame) else None


def _tables(run: Any) -> dict[str, pd.DataFrame]:
    """All DataFrame entries of ``run.tables``, ignoring anything else."""
    tables = _get(run, "tables", {}) or {}
    if not isinstance(tables, Mapping):
        return {}
    return {str(k): v for k, v in tables.items() if isinstance(v, pd.DataFrame)}


def _lookup(run: Any, names: tuple[str, ...], paths: tuple[str, ...] = ()) -> pd.DataFrame | None:
    """First registered table matching ``names``, else first nested ``paths``."""
    tables = _tables(run)
    for name in names:
        frame = tables.get(name)
        if frame is not None:
            return frame
    for path in paths:
        frame = _as_frame(_dig(run, path))
        if frame is not None:
            return frame
    return None


def _pick_columns(frame: pd.DataFrame | None, options: tuple[str, ...]) -> str | None:
    if frame is None:
        return None
    for name in options:
        if name in frame.columns:
            return name
    return None


# ---------------------------------------------------------------------------
#  Data resolution
# ---------------------------------------------------------------------------
def _series_frame(run: Any) -> pd.DataFrame | None:
    """Prism-cycle displacement series (raw and frame-corrected side by side)."""
    return _lookup(
        run,
        ("displacements", "prism_series", "corrected", "timeseries"),
        ("displacements.displacements", "frames.corrected"),
    )


def _frame_coefficients(run: Any) -> pd.DataFrame | None:
    return _lookup(
        run,
        ("frame_cycles", "frame_coefficients"),
        ("frames.coefficients",),
    )


def _concern_frame(run: Any) -> pd.DataFrame | None:
    return _lookup(run, ("concern",), ("classification.concern",))


def _reliability_frame(run: Any) -> pd.DataFrame | None:
    return _lookup(run, ("reliability",), ("classification.reliability",))


def _summary_frame(run: Any) -> pd.DataFrame | None:
    frame = _lookup(run, ("prism_summary", "summary", "ringkasan"), ())
    if frame is not None:
        return frame
    concern = _concern_frame(run)
    reliability = _reliability_frame(run)
    if concern is None and reliability is None:
        return None
    if concern is not None and reliability is not None and "point_id" in concern.columns \
            and "point_id" in reliability.columns:
        keys = [c for c in ("point_id", "segment_id") if c in concern.columns and c in reliability.columns]
        return concern.merge(reliability, on=keys or ["point_id"], how="outer")
    return concern if concern is not None else reliability


def _frame_info(run: Any) -> dict[str, Any]:
    frames = _get(run, "frames", {}) or {}
    coefficients = _frame_coefficients(run)
    status = _get(frames, "correction_status", None) or _get(frames, "status", None)
    reason = _get(frames, "reason", None)
    experimental = _get(frames, "experimental", True)
    membership = _get(frames, "membership", None)
    members = _get(membership, "members", None)
    if members is None:
        members = _get(frames, "members", [])
    available = coefficients is not None and not coefficients.empty
    if status is None:
        status = STATUS_OK if available else "not_configured"
    return {
        "status": str(status),
        "reason": reason,
        "experimental": bool(experimental),
        "available": bool(available),
        "members": list(members) if members else [],
        "coefficients": coefficients if coefficients is not None else pd.DataFrame(),
    }


def _source_hashes(run: Any) -> dict[str, str]:
    """Source name -> SHA-256 from ``run.parsed.sources`` or the manifest."""
    sources = _get(_get(run, "parsed", None), "sources", None)
    if sources is None:
        sources = _get(_get(run, "manifest", None), "sources", None)
    result: dict[str, str] = {}
    if isinstance(sources, Mapping):
        for name, value in sources.items():
            sha = value
            if isinstance(value, Mapping):
                sha = value.get("sha256", value.get("hash", ""))
            result[str(name)] = str(sha)
    elif isinstance(sources, (list, tuple)):
        for item in sources:
            if isinstance(item, Mapping):
                name = item.get("name", item.get("source_name"))
                sha = item.get("sha256", "")
            else:
                name = _get(item, "name", None) or _get(item, "source_name", None)
                sha = _get(item, "sha256", "")
            if name is not None:
                result[str(name)] = str(sha or "")
    elif isinstance(sources, pd.DataFrame):
        name_col = _pick_columns(sources, ("name", "source_name"))
        sha_col = _pick_columns(sources, ("sha256", "sha"))
        if name_col is not None:
            for _, row in sources.iterrows():
                result[str(row[name_col])] = str(row[sha_col]) if sha_col else ""
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _config_hash(run: Any) -> str:
    config = _get(run, "config", None)
    if config is not None and hasattr(config, "hash") and callable(config.hash):
        return str(config.hash())
    if isinstance(config, Mapping):
        return hashlib.sha256(_canonical_json(dict(config)).encode("utf-8")).hexdigest()
    explicit = _get(_get(run, "manifest", None), "config_sha256", None)
    if explicit:
        return str(explicit)
    return hashlib.sha256(b"").hexdigest()


def _config_get(config: Any, path: str, default: Any = None) -> Any:
    """Read a dotted config path from an AnalysisConfig or a mapping."""
    current = config
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(part, _MISSING)
        else:
            current = getattr(current, part, _MISSING)
        if current is _MISSING:
            return default
    return current


def _tarp_configured(run: Any) -> bool:
    tarp = _config_get(_get(run, "config", None), "tarp", None)
    return bool(tarp) and not is_not_configured(tarp)


def _summary_counts(run: Any, concern: pd.DataFrame | None,
                    reliability: pd.DataFrame | None) -> list[tuple[str, Any]]:
    observations = _as_frame(_get(run, "observations", None))
    rows: list[tuple[str, Any]] = [
        ("Registered tables", len(_tables(run))),
    ]
    if observations is not None:
        rows.append(("Source observations", int(len(observations))))
        if "point_id" in observations.columns:
            rows.append(("Observed prisms", int(observations["point_id"].nunique())))
    if concern is not None and not concern.empty:
        rows.append(("Concern rows", int(len(concern))))
        col = _pick_columns(concern, _CONCERN_COLS)
        if col is not None:
            counts = concern[col].fillna("(missing)").astype(str).value_counts()
            for label, count in counts.items():
                rows.append((f"Concern: {label}", int(count)))
    if reliability is not None and not reliability.empty:
        rows.append(("Reliability rows", int(len(reliability))))
        col = _pick_columns(reliability, _RELIABILITY_COLS)
        if col is not None:
            counts = reliability[col].fillna("(missing)").astype(str).value_counts()
            for label, count in counts.items():
                rows.append((f"Reliability: {label}", int(count)))
    return rows


# ---------------------------------------------------------------------------
#  1. CSV tables
# ---------------------------------------------------------------------------
def write_tables(run: Any, out_dir: str | Path) -> list[Path]:
    """Write every registered DataFrame as ``<name>.csv`` plus the dictionary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, frame in sorted(_tables(run).items()):
        target = out / f"{name}.csv"
        frame.to_csv(target, index=False)
        written.append(target)
    dictionary_path = out / "table_dictionary.csv"
    table_dictionary().to_csv(dictionary_path, index=False)
    written.append(dictionary_path)
    return written


# ---------------------------------------------------------------------------
#  2. Figures
# ---------------------------------------------------------------------------
def plot_results(run: Any, out_dir: str | Path) -> list[Path]:
    """Write the available exploratory figures; never fail on missing data."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.extend(_plot_prism_timeseries(run, out))
    written.extend(_plot_station_frame(run, out))
    written.extend(_plot_map(run, out))
    return written


def _plot_prism_timeseries(run: Any, out: Path) -> list[Path]:
    frame = _series_frame(run)
    point_col = _pick_columns(frame, _POINT_COLS)
    time_col = _pick_columns(frame, _TIME_COLS)
    raw_los = _pick_columns(frame, _RAW_LOS_COLS)
    fc_los = _pick_columns(frame, _FC_LOS_COLS)
    raw_vert = _pick_columns(frame, _RAW_VERT_COLS)
    fc_vert = _pick_columns(frame, _FC_VERT_COLS)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].set_title("Raw versus frame-corrected displacement (up to 6 prisms)")
    axes[0].set_ylabel("LOS / radial [mm]")
    axes[1].set_ylabel("Vertical [mm]")
    axes[1].set_xlabel("Time / cycle / block")

    plotted = False
    corrected_present = False
    if frame is not None and not frame.empty and point_col is not None:
        prisms = list(dict.fromkeys(frame[point_col].astype(str)))[:_MAX_PLOT_PRISMS]
        palette = plt.get_cmap("tab10")
        for index, pid in enumerate(prisms):
            sub = frame[frame[point_col].astype(str) == pid]
            if time_col is not None:
                sub = sub.sort_values(time_col)
                x = sub[time_col]
            else:
                sub = sub.reset_index(drop=True)
                x = sub.index
            color = palette(index % 10)
            label_raw = "raw" if index == 0 else None
            label_fc = "frame-corrected (experimental)" if index == 0 else None
            if raw_los is not None:
                axes[0].plot(x, sub[raw_los], color=color, linestyle="-",
                             marker=".", markersize=3, linewidth=1.2, label=label_raw)
                plotted = True
            if raw_vert is not None:
                axes[1].plot(x, sub[raw_vert], color=color, linestyle="-",
                             marker=".", markersize=3, linewidth=1.2, label=label_raw)
                plotted = True
            if fc_los is not None:
                axes[0].plot(x, sub[fc_los], color=color, linestyle="--",
                             marker="x", markersize=3, linewidth=1.2, label=label_fc)
                corrected_present = True
                plotted = True
            if fc_vert is not None:
                axes[1].plot(x, sub[fc_vert], color=color, linestyle="--",
                             marker="x", markersize=3, linewidth=1.2, label=label_fc)
                corrected_present = True
                plotted = True

    if not plotted:
        axes[0].text(0.5, 0.5, "No displacement series available",
                     ha="center", va="center", transform=axes[0].transAxes)
    else:
        for axis in axes:
            axis.axhline(0.0, color="grey", linewidth=0.6)
            axis.grid(alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="best", fontsize=8)

    if corrected_present:
        note = FRAME_EXPERIMENTAL_NOTE
    else:
        reason = _frame_info(run).get("reason") or "frame correction is unavailable or not configured"
        note = f"Frame correction unavailable: {reason}. Only raw values are shown."
    fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    target = out / "fig_prism_timeseries.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    return [target]


def _plot_station_frame(run: Any, out: Path) -> list[Path]:
    coefficients = _frame_coefficients(run)
    frame_info = _frame_info(run)
    if coefficients is None or coefficients.empty:
        note = (
            "fig_station_frame.png skipped: frame coefficients are not available.\n"
            f"Status: {frame_info['status']}\n"
            f"Reason: {frame_info['reason'] or 'frame correction not configured'}\n"
        )
        target = out / "fig_station_frame_note.txt"
        target.write_text(note, encoding="utf-8")
        return [target]

    x_col = _pick_columns(coefficients, ("cycle_id", "ts", "time"))
    station_col = _pick_columns(coefficients, ("station_id", "station"))
    stations = (
        list(dict.fromkeys(coefficients[station_col].astype(str)))
        if station_col is not None else ["all"]
    )
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].set_ylabel("Rotation [arcsec]")
    axes[0].set_title("Per-cycle frame terms (EXPERIMENTAL, relative to the chosen network)")
    axes[1].set_ylabel("Translation E/N [mm]")
    axes[2].set_ylabel("Scale [ppm]")
    axes[2].set_xlabel("Cycle / time")
    palette = plt.get_cmap("tab10")
    for index, station in enumerate(stations):
        sub = coefficients
        if station_col is not None:
            sub = coefficients[coefficients[station_col].astype(str) == station]
        x = sub[x_col] if x_col is not None else sub.index
        color = palette(index % 10)
        rotation = _pick_columns(sub, ("rotation_arcsec",))
        if rotation is not None:
            axes[0].plot(x, sub[rotation], color=color, marker=".", markersize=3, label=str(station))
        t_e = _pick_columns(sub, ("tE_mm",))
        t_n = _pick_columns(sub, ("tN_mm",))
        if t_e is not None:
            axes[1].plot(x, sub[t_e], color=color, linestyle="-", marker=".", markersize=3,
                         label=f"{station} tE")
        if t_n is not None:
            axes[1].plot(x, sub[t_n], color=color, linestyle="--", marker="x", markersize=3,
                         label=f"{station} tN")
        scale = _pick_columns(sub, ("scale_ppm",))
        if scale is not None:
            axes[2].plot(x, sub[scale], color=color, marker=".", markersize=3, label=str(station))
    for axis in axes:
        axis.grid(alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="best", fontsize=7)
    figure.text(0.5, 0.005, FRAME_EXPERIMENTAL_NOTE, ha="center", fontsize=8, color="#444444")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    target = out / "fig_station_frame.png"
    figure.savefig(target, dpi=120)
    plt.close(figure)
    return [target]


def _map_positions(run: Any) -> pd.DataFrame | None:
    frame = _summary_frame(run)
    if frame is None:
        frame = _series_frame(run)
    if frame is None:
        return None
    east = _pick_columns(frame, _EAST_COLS)
    north = _pick_columns(frame, _NORTH_COLS)
    if east is None or north is None:
        return None
    out = pd.DataFrame({
        "east_m": pd.to_numeric(frame[east], errors="coerce"),
        "north_m": pd.to_numeric(frame[north], errors="coerce"),
    })
    if "point_id" in frame.columns:
        out["point_id"] = frame["point_id"].astype(str)
    concern = _pick_columns(frame, _CONCERN_COLS)
    if concern is not None:
        out["concern"] = frame[concern].fillna("(missing)").astype(str)
    return out.dropna(subset=["east_m", "north_m"])


def _plot_map(run: Any, out: Path) -> list[Path]:
    positions = _map_positions(run)
    figure, axis = plt.subplots(figsize=(9, 8))
    if positions is None or positions.empty:
        axis.text(0.5, 0.5, "No prism positions available",
                  ha="center", va="center", transform=axis.transAxes)
    else:
        if "concern" in positions.columns:
            groups = sorted(positions["concern"].unique())
            palette = plt.get_cmap("tab10")
            for index, group in enumerate(groups):
                sub = positions[positions["concern"] == group]
                axis.scatter(sub["east_m"], sub["north_m"], s=28,
                             color=palette(index % 10), label=str(group), alpha=0.85)
            axis.legend(title="Movement concern (separate from reliability)",
                        loc="best", fontsize=8)
        else:
            axis.scatter(positions["east_m"], positions["north_m"], s=28, alpha=0.85)
        if "point_id" in positions.columns and len(positions) <= _MAX_MAP_LABELS:
            for _, row in positions.iterrows():
                axis.annotate(row["point_id"], (row["east_m"], row["north_m"]),
                              fontsize=6, xytext=(3, 3), textcoords="offset points")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.3)
    axis.set_xlabel("Easting [m] (input grid, undefined Cartesian)")
    axis.set_ylabel("Northing [m] (input grid, undefined Cartesian)")
    axis.set_title("Prism positions by movement concern")
    figure.text(0.5, 0.005,
                "Drawn in the supplied file grid; no geographic basemap; no CRS inferred.",
                ha="center", fontsize=8, color="#444444")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    target = out / "fig_map.png"
    figure.savefig(target, dpi=120)
    plt.close(figure)
    return [target]


# ---------------------------------------------------------------------------
#  3. Markdown / HTML report
# ---------------------------------------------------------------------------
def _md_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).replace("|", "\\|")


def _df_to_markdown(frame: pd.DataFrame | None, max_rows: int = 50) -> str:
    if frame is None or frame.empty:
        return "_No rows available._\n"
    view = frame.head(max_rows)
    columns = [str(c) for c in view.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join([" --- "] * len(columns)) + "|",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_md_cell(v) for v in row) + " |")
    if len(frame) > max_rows:
        lines.append(f"| ... | {len(frame) - max_rows} more rows |")
    return "\n".join(lines) + "\n"


def _df_to_html(frame: pd.DataFrame | None, max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return "<p><em>No rows available.</em></p>"
    view = frame if max_rows is None else frame.head(max_rows)
    return view.to_html(index=False, border=0, escape=True)


def _build_context(run: Any) -> dict[str, Any]:
    concern = _concern_frame(run)
    reliability = _reliability_frame(run)
    frame = _frame_info(run)
    sources = _source_hashes(run)
    tarp_configured = _tarp_configured(run)
    return {
        "title": "RTS Forensics report",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary_rows": _summary_counts(run, concern, reliability),
        "concern_md": _df_to_markdown(concern),
        "concern_html": _df_to_html(concern),
        "reliability_md": _df_to_markdown(reliability),
        "reliability_html": _df_to_html(reliability),
        "frame": frame,
        "frame_md": _df_to_markdown(frame["coefficients"]),
        "frame_html": _df_to_html(frame["coefficients"]),
        "provenance_rows": sorted(sources.items()),
        "config_sha256": _config_hash(run),
        "limitations": list(LIMITATIONS),
        "tarp_notice": None if tarp_configured else TARP_NOTICE,
        "exploratory": not tarp_configured,
        "tables": sorted(_tables(run).keys()),
    }


def _render_markdown(context: Mapping[str, Any]) -> str:
    frame = context["frame"]
    lines: list[str] = [
        f"# {context['title']}", "",
        f"Generated (UTC): {context['generated_utc']}", "",
        "## Run summary", "",
        "| item | value |", "| --- | --- |",
    ]
    for label, value in context["summary_rows"]:
        lines.append(f"| {_md_cell(label)} | {_md_cell(value)} |")
    lines += ["", "## Movement concern", "",
              "Movement concern is reported separately from measurement reliability "
              "and is never merged with it.",
              "", context["concern_md"], ""]
    lines += ["## Measurement reliability", "",
              "Reliability is graded independently of movement concern.",
              "", context["reliability_md"], ""]
    lines += ["## Frame correction (EXPERIMENTAL)", "",
              f"Status: {frame['status']}", "",
              "Frame correction is EXPERIMENTAL and relative to the chosen network; "
              "it is never absolute ground movement.",
              "",
              f"Frame-set members: {len(frame['members'])}",
              ""]
    if frame["reason"]:
        lines += [f"Reason: {frame['reason']}", ""]
    lines += [context["frame_md"], ""]
    lines += ["## Provenance and source hashes", "",
              "| source | SHA-256 |", "| --- | --- |"]
    if context["provenance_rows"]:
        for name, sha in context["provenance_rows"]:
            lines.append(f"| {_md_cell(name)} | {_md_cell(sha)} |")
    else:
        lines.append("| (none recorded) | |")
    lines += ["", f"Config SHA-256: `{context['config_sha256']}`", ""]
    lines += ["## Limitations (mandatory)", "",
              "These limitations always apply; they are not softened by any label above.", ""]
    for limitation in context["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    if context["tarp_notice"]:
        lines += ["## Exploratory notice", "", context["tarp_notice"], ""]
    return "\n".join(lines) + "\n"


def _render_html(context: Mapping[str, Any]) -> str:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template("report.html.j2")
    return template.render(**context)


def render_report(run: Any, out_dir: str | Path) -> ReportPaths:
    """Write ``report.md`` and ``report.html`` and return both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    context = _build_context(run)
    md_path = out / "report.md"
    md_path.write_text(_render_markdown(context), encoding="utf-8")
    html_path = out / "report.html"
    html_path.write_text(_render_html(context), encoding="utf-8")
    return ReportPaths(md=md_path, html=html_path)


# ---------------------------------------------------------------------------
#  4. Manifest
# ---------------------------------------------------------------------------
def _settings(run: Any) -> dict[str, Any]:
    algorithms = _get(_get(run, "manifest", None), "algorithms", None)
    if isinstance(algorithms, Mapping):
        return dict(algorithms)
    return {}


def write_manifest(run: Any, out_dir: str | Path) -> Path:
    """Hash every output except the manifest itself and write ``manifest.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    files = [
        path for path in sorted(out.rglob("*"))
        if path.is_file() and path.resolve() != manifest_path.resolve()
    ]
    outputs = provenance.hash_outputs(files, root=out)
    manifest = provenance.build_manifest(
        sources=_source_hashes(run),
        config_hash=_config_hash(run),
        audit=provenance.runtime_audit(),
        outputs=outputs,
        settings=_settings(run),
    )
    return provenance.write_manifest(manifest, manifest_path)
