"""
config.py — typed configuration, versioned defaults and the not_configured policy.

The handoff defines some numerical values and deliberately leaves others open.
Anything not defined must not be guessed: it is represented as ``None`` in the
dataclass and as ``not_configured`` in the audit, so every consuming module can
report an explicit status instead of inventing a decision.

Units follow :mod:`rts_forensics.models`: geometry metres, displacement
millimetres, internal angles radians, exported rotation arcseconds, rates
mm/day, scale ppm.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import NOT_CONFIGURED


# ---------------------------------------------------------------------------
#  Section dataclasses
# ---------------------------------------------------------------------------
@dataclass
class InputConfig:
    delimiter: str | None = None                     # None = detect
    encodings: list[str] = field(
        default_factory=lambda: ["cp1252", "utf-8", "iso-8859-1"])
    time_formats: list[str] = field(default_factory=lambda: [
        "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"])
    timezone: str = "site-local"
    column_map: dict[str, str] = field(default_factory=dict)


@dataclass
class CyclesConfig:
    gap_minutes: dict[str, int] = field(default_factory=dict)
    default_gap_minutes: int = 30
    processing_break: str | None = None


@dataclass
class StationsConfig:
    association_tolerance_m: float | None = None
    anchors: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BaselineConfig:
    window_hours: float | None = 48.0
    min_observations: int = 6
    anchor: str | None = "first_observations_of_segment"
    end_change_estimator: str | None = "median_final_48h"


@dataclass
class NoiseConfig:
    rolling_window_hours: float = 24.0
    centered: bool = True
    min_observations: int = 3
    robust_scale: float = 1.4826
    spike_z: float = 6.0


@dataclass
class RatesConfig:
    block_hours: float = 24.0
    windows_days: list[int] = field(default_factory=lambda: [30, 7])
    windows_hours: list[int] = field(default_factory=lambda: [72])
    confidence: float = 0.95
    block_anchor: str = "final_record"


@dataclass
class ScreeningConfig:
    sigma_multiplier: float = 3.0
    los_floor_mm: float = 2.0
    vertical_floor_mm: float = 5.0
    possible_sigma_multiplier: float = 2.0
    possible_los_floor_mm: float = 2.0
    possible_vertical_floor_mm: float = 4.0
    min_prism_cycles: int = 20
    require_same_sign_ci: bool = True


@dataclass
class ReliabilityConfig:
    # Supplied research rules (s1_summary.py): exploratory, never alarms.
    boundaries: dict[str, Any] | None = field(default_factory=lambda: {
        "insufficient_below_cycles": 20,
        "coverage_weak_pct": 50,
        "coverage_moderate_pct": 80,
        "max_gap_hours": 72,
        "noisy_los_mm": 2.0,
        "noisy_vertical_mm": 6.0,
        "spike_flags_moderate": 5,
    })
    frame_min_cycles: int = 100


@dataclass
class FrameConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "hz_arcsec": 0.8, "los_mm": 0.8, "vertical_mm": 3.0})
    huber_cutoff: float | None = 1.5
    convergence: dict[str, Any] | None = field(
        default_factory=lambda: {"max_iterations": 30, "scale_floor": 1.0})
    conditioning_limit: float | None = None
    min_prism_cycles_per_fit: int = 20
    frame_set_reliability: str = "A"
    exclusions: list[str] = field(default_factory=list)
    site_exclusions: list[str] = field(default_factory=list)


@dataclass
class ForensicsConfig:
    placebo_neighbors: int | None = 7
    hour_class_hours: int = 4
    day_bands: dict[str, list[int]] = field(
        default_factory=lambda: {"night": [20, 0, 4], "day": [8, 12, 16]})
    cluster: dict[str, Any] = field(default_factory=dict)
    references: dict[str, Any] = field(default_factory=lambda: {
        "alpha": 0.05, "bonferroni": True, "expected_rise_vertical_slope": -1.0})
    missingness: dict[str, Any] = field(default_factory=lambda: {
        "final_window_hours": 48, "permutation_n": 5000, "seed": 1})
    events: dict[str, Any] = field(default_factory=dict)


@dataclass
class GISConfig:
    srs_id: int = -1
    srs_wkt: str | None = None
    transform: dict[str, Any] | None = None
    vector_scale: float = 2500.0
    candidate_zone_rule: dict[str, Any] | None = None


@dataclass
class AnalysisConfig:
    version: int = 2
    provenance: dict[str, Any] = field(default_factory=lambda: {
        "preserve_raw_bytes": True, "store_observation_ids": True})
    input: InputConfig = field(default_factory=InputConfig)
    cycles: CyclesConfig = field(default_factory=CyclesConfig)
    stations: StationsConfig = field(default_factory=StationsConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    rates: RatesConfig = field(default_factory=RatesConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    forensics: ForensicsConfig = field(default_factory=ForensicsConfig)
    tarp: Any = None
    gis: GISConfig = field(default_factory=GISConfig)

    # -- conversion ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AnalysisConfig:
        data = copy.deepcopy(data or {})
        kwargs: dict[str, Any] = {}
        sections = {
            "input": InputConfig,
            "cycles": CyclesConfig,
            "stations": StationsConfig,
            "baseline": BaselineConfig,
            "noise": NoiseConfig,
            "rates": RatesConfig,
            "screening": ScreeningConfig,
            "reliability": ReliabilityConfig,
            "frame": FrameConfig,
            "forensics": ForensicsConfig,
            "gis": GISConfig,
        }
        for name, value in data.items():
            if name == "version":
                kwargs[name] = value
            elif name in sections and isinstance(value, dict):
                kwargs[name] = sections[name](**value)
            else:
                kwargs[name] = value
        return cls(**kwargs)

    def hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
#  Loading / validation
# ---------------------------------------------------------------------------
def default_config() -> AnalysisConfig:
    return AnalysisConfig()


def load_config(path: str | Path | None = None,
                overrides: dict[str, Any] | None = None) -> AnalysisConfig:
    """Load a YAML configuration, falling back to the versioned defaults.

    A partial file is allowed: absent keys keep their documented defaults.
    Unknown top-level keys are rejected to catch typos early.
    """
    data: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        elif path.exists():
            raise ValueError(f"config path is not a file: {path}")
        else:
            raise FileNotFoundError(f"config not found: {path}")

    known = set(AnalysisConfig.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")

    if overrides:
        data = _deep_merge(data, overrides)
    return AnalysisConfig.from_dict(data)


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def is_not_configured(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() == NOT_CONFIGURED
    if isinstance(value, dict):
        return any(is_not_configured(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(is_not_configured(v) for v in value)
    return False


@dataclass
class ConfigAudit:
    configured: list[str] = field(default_factory=list)
    not_configured: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def is_configured(self, path: str) -> bool:
        return path in self.configured

    def requires(self, path: str) -> None:
        """Raise LookupError when a required decision is not configured."""
        if not self.is_configured(path):
            raise LookupError(
                f"required configuration missing: {path} "
                "(the handoff does not define it; supply the reference bundle "
                "or set it explicitly)")

    def summary(self) -> dict[str, Any]:
        return {"n_configured": len(self.configured),
                "n_not_configured": len(self.not_configured),
                "not_configured": sorted(self.not_configured),
                "warnings": self.warnings}


def _walk(prefix: str, value: Any, audit: ConfigAudit, skip: set[str]) -> None:
    if prefix in skip:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            _walk(f"{prefix}.{k}" if prefix else str(k), v, audit, skip)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            audit.not_configured.append(prefix)
        elif all(not isinstance(v, (dict, list)) for v in value) and any(
                is_not_configured(v) for v in value):
            audit.not_configured.append(prefix)
        elif all(not isinstance(v, (dict, list)) for v in value):
            audit.configured.append(prefix)
        return
    if is_not_configured(value):
        audit.not_configured.append(prefix)
    else:
        audit.configured.append(prefix)


_SKIP_AUDIT = {
    "version", "input.column_map", "stations.anchors",
    "frame.exclusions", "frame.site_exclusions",
    "provenance.preserve_raw_bytes", "provenance.store_observation_ids",
    "frame.convergence.max_iterations", "frame.convergence.scale_floor",
}


def validate_config(config: AnalysisConfig) -> ConfigAudit:
    """Collect configured and not_configured decisions with dotted paths."""
    audit = ConfigAudit()
    _walk("", config.to_dict(), audit, _SKIP_AUDIT)
    if is_not_configured(config.stations.association_tolerance_m) and not config.stations.anchors:
        audit.warnings.append(
            "stations: no anchors configured; stations are auto-detected from the "
            "largest gap in station coordinates")
    if is_not_configured(config.tarp):
        audit.warnings.append(
            "tarp: not configured - no alarm state will be produced "
            "(a slope is never declared safe or unsafe)")
    if is_not_configured(config.forensics.cluster.get("radius_m")):
        audit.warnings.append(
            "forensics.cluster.radius_m: not configured - candidate cluster searches "
            "will report not_configured instead of choosing a hidden radius")
    return audit
