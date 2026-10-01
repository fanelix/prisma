"""Explicit configuration. Physical screening constants come from the handoff."""

import json
import math
from copy import deepcopy
from pathlib import Path

DEFAULTS = {
    "input": {"encoding": "cp1252", "separator": ";", "timestamp_format": "%d/%m/%Y %H:%M"},
    "baseline_hours": 48,
    "end_hours": 48,
    "cycle_gap_minutes": 30,
    "station_gap_minutes": {},
    "station_by_pid": {},
    "processing_changes": {},
    "night_hours": [20, 8],
    "hour_class_hours": 4,
    "screening": {
        "min_cycles": 20,
        "sigma_multiplier": 3.0,
        "los_floor_mm": 2.0,
        "vertical_floor_mm": 5.0,
        "spike_z": 6.0,
        # Handoff: slope distance is exported at 1 mm resolution.
        "range_resolution_mm": 1.0,
    },
    "frame": {
        "include": [],
        "exclude": [],
        "references_under_test": [],
        "min_cycles": 100,
        "sigma_hz_arcsec": 0.8,
        "sigma_d_mm": 0.8,
        "sigma_v_mm": 3.0,
        "huber_tuning": 1.345,
        # "auto", a model name, or {station: model}; see frame.VERTICAL_MODELS.
        "vertical_model": "auto",
    },
    "reliability": None,
    "detection": {"orientation_step_arcsec": None, "bias_floor_mm": None, "cluster_radius_m": None},
    "tarp": None,
    "site_transform": None,
}


def _merge(base, extra, prefix=""):
    if not isinstance(extra, dict):
        raise ValueError("Configuration must be a mapping")
    for key, value in extra.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict) and not isinstance(value, dict):
            raise ValueError(f"{prefix}{key} must be a mapping")
        if isinstance(base[key], dict) and isinstance(value, dict):
            if key in {"station_gap_minutes", "station_by_pid", "processing_changes"}:
                base[key] = value
            else:
                _merge(base[key], value, prefix + key + ".")
        else:
            base[key] = value


def load_config(path=None, overrides=None):
    cfg = deepcopy(DEFAULTS)
    if path:
        text = Path(path).read_text(encoding="utf-8")
        if Path(path).suffix == ".json":
            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a mapping")
        _merge(cfg, data)
    if overrides is not None:
        _merge(cfg, overrides)
    for key in ["baseline_hours", "end_hours", "cycle_gap_minutes", "hour_class_hours"]:
        if not isinstance(cfg[key], (float, int)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for block in ["screening", "frame"]:
        for key, value in cfg[block].items():
            if isinstance(DEFAULTS[block][key], (int, float)) and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{block}.{key} must be positive and finite")
    models = {"auto", "full", "merged", "height_index", "height"}
    vertical = cfg["frame"]["vertical_model"]
    if not (
        (isinstance(vertical, str) and vertical in models)
        or (
            isinstance(vertical, dict)
            and all(isinstance(k, str) and isinstance(v, str) and v in models for k, v in vertical.items())
        )
    ):
        raise ValueError(f"frame.vertical_model must be one of {sorted(models)} or a station mapping")
    for key in ["include", "exclude", "references_under_test"]:
        if not isinstance(cfg["frame"][key], list) or any(not isinstance(v, str) for v in cfg["frame"][key]):
            raise ValueError(f"frame.{key} must be a list of target IDs")
    if (
        not isinstance(cfg["night_hours"], list)
        or len(cfg["night_hours"]) != 2
        or any(type(v) is not int or not 0 <= v < 24 for v in cfg["night_hours"])
    ):
        raise ValueError("night_hours must contain two hours in 0..23")
    for key, value in cfg["detection"].items():
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"detection.{key} must be positive and finite, or null")
    for value in cfg["station_gap_minutes"].values():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("station_gap_minutes must be positive and finite")
    tarp = cfg["tarp"]
    if tarp is not None:
        required = {
            "metric",
            "threshold",
            "averaging_hours",
            "persistence_hours",
            "max_gap_hours",
            "direction",
            "label",
        }
        if not isinstance(tarp, dict) or set(tarp) != required:
            raise ValueError(
                "TARP requires metric, threshold, averaging_hours, persistence_hours, max_gap_hours, direction, label"
            )
        if tarp["metric"] not in {"los_raw", "ver_raw", "los_fc", "ver_fc"}:
            raise ValueError("TARP metric must name a displacement column in mm")
        if tarp["direction"] not in {"above", "below", "absolute"}:
            raise ValueError("TARP direction must be above, below, or absolute")
        if any(
            not isinstance(tarp[k], (int, float)) or not math.isfinite(tarp[k])
            for k in ["threshold", "averaging_hours", "persistence_hours", "max_gap_hours"]
        ):
            raise ValueError("TARP numeric values must be finite")
        if tarp["averaging_hours"] <= 0 or tarp["persistence_hours"] < 0 or tarp["max_gap_hours"] <= 0:
            raise ValueError("TARP averaging must be positive and persistence nonnegative")
    if cfg["reliability"] is not None:
        rules = cfg["reliability"]
        keys = {"coverage_min", "max_gap_hours", "max_los_noise_mm", "max_spike_fraction"}
        if not isinstance(rules, dict) or set(rules) != {"A", "B", "C"}:
            raise ValueError("Reliability requires A/B/C rules; D is the remaining class")
        for rule in rules.values():
            if not isinstance(rule, dict) or set(rule) != keys:
                raise ValueError(f"Reliability rule requires {sorted(keys)}")
            for key, value in rule.items():
                if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Reliability {key} must be finite and nonnegative")
                if key in {"coverage_min", "max_spike_fraction"} and value > 1:
                    raise ValueError(f"Reliability {key} must be between 0 and 1")
    transform = cfg["site_transform"]
    if transform is not None:
        keys = {"rotation_deg", "offset_e", "offset_n", "srs_id", "srs_wkt"}
        if not isinstance(transform, dict) or not set(transform) <= keys:
            raise ValueError("Unknown or malformed site_transform")
        for key in ["rotation_deg", "offset_e", "offset_n"]:
            value = transform.get(key, 0)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"site_transform.{key} must be numeric and finite")
        if "srs_id" in transform and type(transform["srs_id"]) is not int:
            raise ValueError("site_transform.srs_id must be an integer")
        if transform.get("srs_id", -1) not in {-1, 0} and not isinstance(transform.get("srs_wkt"), str):
            raise ValueError("site_transform requires an explicit srs_wkt for the configured SRS")
    return cfg
