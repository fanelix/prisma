"""
provenance.py — identity, raw-byte preservation and audit manifests.

Every source file is identified by the SHA-256 of its original bytes; every
observation by ``f"{sha256[:16]}:{src_line}"`` using the physical source line
number (stable under CRLF, quoted delimiters and multiline fields).

The manifest records source hashes, the effective configuration hash, package
and runtime versions, deterministic seeds and output hashes. It deliberately
excludes its own hash to avoid a recursive digest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .models import SOURCE_LINK_COLUMNS

VERSI = "0.1.0"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_observation_id(source_sha256: str, src_line: int) -> str:
    return f"{source_sha256[:16]}:{int(src_line)}"


def write_raw_sources(sources: Mapping[str, bytes], raw_dir: str | Path) -> dict[str, str]:
    """Preserve original input bytes verbatim; returns name -> written path."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, data in sources.items():
        target = raw_dir / Path(name).name
        target.write_bytes(data)
        written[name] = str(target)
    return written


def build_source_links(table_name: str, rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Assemble a normalized source-links frame from row dictionaries.

    Each row must provide ``row_key``, ``role`` and an ``obs_ids`` iterable;
    ``source_sha256``/``src_line`` are decoded from the observation ID.
    """
    records: list[dict[str, Any]] = []
    for row in rows:
        for obs_id in row.get("obs_ids") or ():
            sha, _, line = str(obs_id).partition(":")
            records.append({
                "table_name": table_name,
                "row_key": str(row.get("row_key", "")),
                "role": str(row.get("role", "observation")),
                "obs_id": obs_id,
                "source_sha256": sha,
                "src_line": int(line) if line.isdigit() else -1,
            })
    if not records:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in SOURCE_LINK_COLUMNS})
    return pd.DataFrame(records, columns=list(SOURCE_LINK_COLUMNS))


@dataclass
class RuntimeAudit:
    package_version: str
    python_version: str
    platform: str
    pandas_version: str
    numpy_version: str
    scipy_version: str
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": VERSI,
            "python": self.python_version,
            "platform": self.platform,
            "pandas": self.pandas_version,
            "numpy": self.numpy_version,
            "scipy": self.scipy_version,
            "seed": self.seed,
        }


def runtime_audit(seed: int = 0) -> RuntimeAudit:
    import numpy as np

    try:
        import scipy
        scipy_version = scipy.__version__
    except ModuleNotFoundError:  # pragma: no cover - scipy is a core dependency
        scipy_version = "unavailable"
    return RuntimeAudit(
        package_version=VERSI,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        pandas_version=pd.__version__,
        numpy_version=np.__version__,
        scipy_version=scipy_version,
        seed=seed,
    )


def hash_outputs(paths: Iterable[str | Path], root: str | Path | None = None) -> dict[str, str]:
    """SHA-256 of every produced artifact, keyed by POSIX relative path."""
    root = Path(root) if root is not None else None
    out: dict[str, str] = {}
    for p in sorted(Path(x) for x in paths):
        if not p.is_file():
            continue
        key = p.relative_to(root).as_posix() if root else p.as_posix()
        out[key] = sha256_file(p)
    return out


def build_manifest(*, sources: Mapping[str, str], config_hash: str,
                   audit: RuntimeAudit, outputs: Mapping[str, str],
                   settings: Mapping[str, Any] | None = None,
                   schema_version: int = 1) -> dict[str, Any]:
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": schema_version,
        "sources": dict(sorted(sources.items())),
        "config_sha256": config_hash,
        "runtime": audit.to_dict(),
        "algorithms": dict(settings or {}),
        "outputs": dict(sorted(outputs.items())),
    }


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path
