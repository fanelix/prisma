import hashlib
import html
import json
from pathlib import Path

import pandas as pd

from .gis import write_gpkg


def plot_result(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    for station, f in result["frame_cycles"].groupby("station"):
        axes[0].plot(f.ts, f.rotation_arcsec, label=station)
    axes[0].set(ylabel="Rotation (arcsec)", title="Experimental network frame")
    axes[0].legend()
    for station, s in result["series"].groupby("station"):
        c = s.groupby("cycle").agg(ts=("ts", "median"), raw=("los_raw", "median"), fc=("los_fc", "median"))
        axes[1].plot(c.ts, c.raw, label=station + " raw")
        axes[1].plot(c.ts, c.fc, linestyle="--", label=station + " corrected")
    axes[1].set(
        ylabel="LOS (mm)", title="Raw and experimental frame-corrected network medians; composition can vary"
    )
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", rotation=20)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_results(result, directory, figures=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(directory / (key + ".csv"), index=False)
    (directory / "config.json").write_text(json.dumps(result["config"], indent=2), encoding="utf-8")
    write_gpkg(result, directory / "forensics.gpkg")
    if figures:
        plot_result(result, directory / "network.png")
    summary = result["prism_summary"]
    text = [
        "# RTS observation forensics",
        "",
        "Exploratory analysis. Frame corrections are experimental.",
        "",
        f"{len(result['observations'])} source rows; {len(summary)} prism/station series; "
        + f"{result['cycle_table'].station.nunique()} stations.",
        "",
        "## Interpretation limits",
        "",
        *["- " + w for w in result["warnings"]],
        "",
        "## Movement concern and data reliability",
        "",
        summary[
            [
                "station",
                "pid",
                "movement_concern",
                "reliability",
                "net_los_raw_mm",
                "net_los_fc_mm",
                "net_vertical_raw_mm",
                "net_vertical_fc_mm",
                "flags",
            ]
        ].to_csv(index=False),
        "",
        "See prism_summary.csv, frame_members.csv and investigation_register.csv for quantitative evidence.",
        "Source provenance is in observations.csv and series.csv; no rows or spike flags were deleted.",
        "",
        "This report alone does not establish agreement with independent reference results.",
    ]
    (directory / "report.md").write_text("\n".join(text), encoding="utf-8")
    table = summary[
        [
            "station",
            "pid",
            "movement_concern",
            "reliability",
            "net_los_raw_mm",
            "net_los_fc_mm",
            "net_vertical_raw_mm",
            "net_vertical_fc_mm",
            "flags",
        ]
    ].to_html(index=False, escape=True)
    notices = "".join("<li>" + html.escape(w) + "</li>" for w in result["warnings"])
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RTS observation forensics</title><style>body{{font:16px system-ui;margin:2rem;color:#17304a}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border:1px solid #ddd}}img{{max-width:100%}}.scroll{{overflow:auto}}</style>
<h1>RTS observation forensics</h1><p>Exploratory. Raw and experimental corrections are shown together.</p>
<ul>{notices}</ul>{'<img src="network.png" alt="Network frame and raw/corrected LOS">' if figures else ""}
<div class="scroll">{table}</div><p>Read the CSV evidence, frame membership and provenance tables before interpreting movement.</p></html>"""
    (directory / "report.html").write_text(page, encoding="utf-8")
    names = [p for p in directory.iterdir() if p.is_file() and p.name != "manifest.json"]
    manifest = {
        "version": "0.1.0",
        "inputs": result["inputs"],
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(names)},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory
