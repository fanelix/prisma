"""
rts-forensics — auditable forensic analysis of monthly RTS exports.

The core is a pandas/numpy/scipy pipeline with typed configuration and result
contracts (:mod:`rts_forensics.models`). It never imports Streamlit; the CLI
and the dashboard both call :func:`rts_forensics.pipeline.run_analysis`, so
there is exactly one set of algorithms and one audit trail.

Honesty policy: no slope is ever declared safe or unsafe, no CRS is inferred,
no TARP is applied unless supplied, and unobserved periods are never described
as stable. Decisions the handoff does not define are reported as
``not_configured`` instead of being guessed.
"""

from __future__ import annotations

from . import (  # noqa: F401
    classify,
    config,
    cycles,
    displacement,
    frame,
    geometry,
    gis,
    io,
    models,
    noise,
    provenance,
    rates,
)
from .config import AnalysisConfig, ConfigAudit, load_config, validate_config
from .io import parse_dms, read_geomos
from .models import (
    NOT_CONFIGURED,
    STATUS_OK,
    ClassificationResult,
    CycleResult,
    DisplacementResult,
    ForensicResult,
    FrameResult,
    NoiseResult,
    ParsedExport,
    RunResult,
)
from .pipeline import export_run, run_analysis
from .provenance import VERSI

__version__ = VERSI

__all__ = [
    "AnalysisConfig",
    "ConfigAudit",
    "ClassificationResult",
    "CycleResult",
    "DisplacementResult",
    "ForensicResult",
    "FrameResult",
    "NoiseResult",
    "NOT_CONFIGURED",
    "ParsedExport",
    "RunResult",
    "STATUS_OK",
    "classify",
    "config",
    "cycles",
    "displacement",
    "export_run",
    "frame",
    "geometry",
    "gis",
    "io",
    "load_config",
    "models",
    "noise",
    "parse_dms",
    "provenance",
    "rates",
    "read_geomos",
    "run_analysis",
    "validate_config",
]
