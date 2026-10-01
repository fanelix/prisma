"""
cli.py — command line interface.

    rts-forensics run data.csv --config config.yaml --out results/

The CLI is a thin wrapper: it loads the configuration, calls the same
:func:`rts_forensics.pipeline.run_analysis` used by the dashboard and exports
the same products. It never applies a TARP, never infers a CRS and never
declares a slope safe or unsafe.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import default_config, load_config
from .pipeline import export_run, run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rts-forensics",
        description="Auditable forensic analysis of RTS prism-monitoring exports.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the full analysis and export results")
    run.add_argument("data", help="raw export CSV (path)")
    run.add_argument("--config", default=None, help="YAML configuration file")
    run.add_argument("--out", required=True, help="output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "run":  # pragma: no cover - argparse enforces
        return 2

    config_path = args.config
    if config_path is not None and not Path(config_path).is_file():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 2
    cfg = load_config(config_path) if config_path else default_config()

    result = run_analysis(args.data, config=cfg)
    manifest = export_run(result, args.out)

    info = result.parsed.sources[0] if result.parsed.sources else None
    print(f"rows          : {info.n_rows if info else 0}")
    print(f"prisms        : {result.observations.point_id.nunique()}")
    print(f"cycles        : {len(result.cycles.cycles)}")
    print(f"segments      : {result.displacements.prism_cycles.segment_id.nunique()}"
          if len(result.displacements.prism_cycles) else "segments      : 0")
    print(f"frame fit     : {result.frames.status}"
          f" ({'experimental' if result.frames.experimental else 'n/a'})")
    if len(result.tables.get("concern", [])):
        flagged = int(result.tables["concern"]["exceeds"].sum())
        print(f"exploratory screen: {flagged} metric rows exceed; no TARP meaning")
    tarp_state = ("not_configured" if cfg.tarp is None else "configured")
    print(f"tarp          : {tarp_state} (a slope is never declared safe or unsafe)")
    print(f"outputs       : {args.out}")
    print(f"manifest      : {manifest.manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
