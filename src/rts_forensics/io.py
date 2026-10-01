"""
io.py — GeoMoS export reading with byte-level and row-level provenance.

The parser reads the documented Leica GeoMoS-style CSV (semicolon separated,
cp1252, CRLF, quoted DMS angles) with :mod:`csv` rather than by splitting lines
itself, so quoted delimiters and embedded multiline fields keep their true
physical source lines. Original bytes are preserved verbatim and identified by
SHA-256; every observation carries an immutable ``obs_id`` of the form
``f"{sha256[:16]}:{src_line}"``.

Nothing is dropped silently: malformed values become NaN and are reported in
``diagnostics``; only records whose field count differs from the header are
reported and skipped. Observation order follows physical source order — no
sorting and no deduplication.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AnalysisConfig, default_config
from .models import OBSERVATION_COLUMNS, FormatSpec, ParsedExport, SourceRecord
from .provenance import build_source_links, make_observation_id, sha256_bytes

_DMS_RE = re.compile(r"^\s*(-?\d+)\D+(\d+)\D+([\d.]+)\D*$")

_DELIMITER_CANDIDATES: tuple[str, ...] = (";", ",", "\t")
_DELIMITER_SCAN_LINES = 5
_DEFAULT_TIME_FORMAT = "%d/%m/%Y %H:%M"
_UNNAMED_SOURCE = "data.csv"

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "point_id", "time", "hz_deg", "v_deg", "d_m",
    "te_m", "tn_m", "tz_m", "se_m", "sn_m", "sh_m",
)

# Normalized header -> canonical observation column. Header normalization
# lowercases, strips bracketed units and collapses non-alphanumeric runs, so
# "Hz [dms]", "Hz" and "HZ" all map to the same canonical name.
_ALIASES: dict[str, str] = {
    "point id": "point_id", "point": "point_id", "pid": "point_id",
    "pointid": "point_id", "point no": "point_id", "target": "point_id",
    "time": "time", "time str": "time", "timestamp": "time",
    "date time": "time", "datetime": "time", "date": "time",
    "hz": "hz_deg", "hz str": "hz_deg", "hz dms": "hz_deg",
    "horizontal angle": "hz_deg",
    "horiz angle": "hz_deg", "horizontal direction": "hz_deg",
    "horiz direction": "hz_deg", "azimuth": "hz_deg",
    "v": "v_deg", "v str": "v_deg", "v dms": "v_deg", "vertical angle": "v_deg",
    "vertical": "v_deg", "zenith": "v_deg", "zenith angle": "v_deg",
    "d": "d_m", "d str": "d_m", "d m": "d_m", "slope distance": "d_m",
    "slope dist": "d_m", "distance": "d_m", "range": "d_m",
    "target easting": "te_m", "target east": "te_m", "easting": "te_m",
    "east": "te_m", "te": "te_m", "e": "te_m",
    "target northing": "tn_m", "target north": "tn_m", "northing": "tn_m",
    "north": "tn_m", "tn": "tn_m", "n": "tn_m",
    "target elevation": "tz_m", "target height": "tz_m", "elevation": "tz_m",
    "height": "tz_m", "tz": "tz_m", "z": "tz_m",
    "station easting": "se_m", "station east": "se_m", "ste": "se_m",
    "station e": "se_m", "st e": "se_m",
    "station northing": "sn_m", "station north": "sn_m", "stn": "sn_m",
    "station n": "sn_m", "st n": "sn_m",
    "station height": "sh_m", "station elevation": "sh_m", "sth": "sh_m",
    "station h": "sh_m", "st h": "sh_m",
    "null measurement": "null_meas_m", "null meas": "null_meas_m",
    "nullm": "null_meas_m", "null": "null_meas_m",
    "horz distance": "horz_dist_m", "horiz distance": "horz_dist_m",
    "horizontal distance": "horz_dist_m", "horz dist": "horz_dist_m",
    "horiz dist": "horz_dist_m", "hd": "horz_dist_m",
    "ppm": "ppm", "atmos ppm": "ppm", "scale ppm": "ppm",
    "av temp": "temp_c", "temp": "temp_c", "temperature": "temp_c",
    "air temperature": "temp_c", "air temp": "temp_c", "av temperature": "temp_c",
}

_UNIT_RE = re.compile(r"\[[^\]]*\]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
def parse_dms(value: Any) -> float:
    """Decode a DMS string such as ``148° 36' 38.36901"`` to decimal degrees.

    The sign is taken from the integer (degree) part and applied to the whole
    angle. Non-strings, whitespace-only strings and values that do not match
    the DMS pattern return NaN.
    """
    if not isinstance(value, str):
        return math.nan
    match = _DMS_RE.match(value)
    if match is None:
        return math.nan
    degrees = float(match.group(1))
    minutes = float(match.group(2))
    seconds = float(match.group(3))
    sign = -1.0 if degrees < 0 else 1.0
    return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)


def detect_format(source: str | Path | bytes | tuple[str, bytes]) -> FormatSpec:
    """Detect encoding, delimiter and canonical column map of one source.

    ``source`` is a path, raw bytes or a ``(name, bytes)`` tuple. Encodings
    from the default configuration are tried in order; the delimiter is the
    configured one or the most frequent of ``;``, ``,`` and tab outside quotes
    in the first non-empty lines. Missing required columns raise ``ValueError``.
    """
    _, data = _single_source(source)
    return _detect_format(data, default_config())


def read_geomos(
    source: str | Path | bytes | tuple[str, bytes]
    | Sequence[str | Path | bytes | tuple[str, bytes]],
    *,
    config: AnalysisConfig | None = None,
) -> ParsedExport:
    """Read one or more GeoMoS exports into canonical observations.

    Original bytes of every source are preserved in ``raw_bytes`` and hashed
    into ``SourceRecord`` entries. Records keep physical order; malformed
    values become NaN with a diagnostic row; wrong field counts are reported
    and skipped. ``source_links`` links every observation to its source line.
    """
    cfg = config if config is not None else default_config()
    items = _normalize_sources(source)
    if not items:
        raise ValueError("no input sources provided")

    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    sources: list[SourceRecord] = []
    raw_bytes: dict[str, bytes] = {}
    export_format: FormatSpec | None = None

    for name, data in items:
        spec = _detect_format(data, cfg)
        if export_format is None:
            export_format = spec
        sha = sha256_bytes(data)
        raw_bytes[name] = data
        file_records, file_diagnostics = _parse_source(name, data, spec, cfg, sha)
        records.extend(file_records)
        diagnostics.extend(file_diagnostics)
        sources.append(SourceRecord(
            name=name, sha256=sha, n_bytes=len(data), n_rows=len(file_records),
            raw_path=None))

    observations = _observations_frame(records)
    source_links = build_source_links(
        "observations",
        ({"row_key": record["obs_id"], "role": "observation",
          "obs_ids": [record["obs_id"]]} for record in records),
    )
    return ParsedExport(
        observations=observations,
        format=export_format,
        sources=sources,
        diagnostics=_diagnostics_frame(diagnostics),
        source_links=source_links,
        raw_bytes=raw_bytes,
    )


# ---------------------------------------------------------------------------
#  Source normalization
# ---------------------------------------------------------------------------
def _is_named_source(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], (str, Path))
        and isinstance(value[1], (bytes, bytearray, str, Path))
    )


def _read_payload(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return Path(value).read_bytes()


def _single_source(source: Any) -> tuple[str, bytes]:
    if isinstance(source, (bytes, bytearray)):
        return _UNNAMED_SOURCE, bytes(source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.name, path.read_bytes()
    if _is_named_source(source):
        return str(source[0]), _read_payload(source[1])
    raise TypeError(f"unsupported source type: {type(source)!r}")


def _normalize_sources(source: Any) -> list[tuple[str, bytes]]:
    if _is_named_source(source) or isinstance(source, (str, Path, bytes, bytearray)):
        return [_single_source(source)]
    if isinstance(source, (list, tuple)):
        items: list[tuple[str, bytes]] = []
        for index, item in enumerate(source):
            if _is_named_source(item):
                items.append((str(item[0]), _read_payload(item[1])))
            elif isinstance(item, (bytes, bytearray)):
                items.append((f"source_{index}.csv", bytes(item)))
            elif isinstance(item, (str, Path)):
                path = Path(item)
                items.append((path.name, path.read_bytes()))
            else:
                raise TypeError(f"unsupported source item: {type(item)!r}")
        return items
    raise TypeError(f"unsupported source type: {type(source)!r}")


# ---------------------------------------------------------------------------
#  Format detection
# ---------------------------------------------------------------------------
def _normalize_header(name: str) -> str:
    text = str(name).strip().lower()
    text = _UNIT_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return " ".join(text.split())


def _detect_encoding(data: bytes, encodings: Sequence[str]) -> str:
    for encoding in encodings:
        try:
            data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        return encoding
    raise ValueError(
        f"none of the configured encodings could decode the input: {list(encodings)}")


def _count_delimiters(line: str, counts: dict[str, int]) -> None:
    in_quotes = False
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif not in_quotes and char in counts:
            counts[char] += 1


def _detect_delimiter(text: str) -> str:
    counts = dict.fromkeys(_DELIMITER_CANDIDATES, 0)
    scanned = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        _count_delimiters(line, counts)
        scanned += 1
        if scanned >= _DELIMITER_SCAN_LINES:
            break
    best = _DELIMITER_CANDIDATES[0]
    for candidate in _DELIMITER_CANDIDATES:
        if counts[candidate] > counts[best]:
            best = candidate
    return best


def _header_fields(text: str, delimiter: str) -> list[str]:
    for fields in csv.reader(io.StringIO(text), delimiter=delimiter):
        if fields:
            return [field.strip() for field in fields]
    return []


def _build_column_map(header: list[str], config: AnalysisConfig) -> dict[str, str]:
    column_map: dict[str, str] = {}
    for name in header:
        canonical = _ALIASES.get(_normalize_header(name))
        if canonical is not None and canonical not in column_map:
            column_map[canonical] = name
    for canonical, source in config.input.column_map.items():
        column_map[str(canonical)] = str(source)
    return column_map


def _detect_format(data: bytes, config: AnalysisConfig) -> FormatSpec:
    encoding = _detect_encoding(data, config.input.encodings)
    text = data.decode(encoding)
    delimiter = config.input.delimiter or _detect_delimiter(text)
    header = _header_fields(text, delimiter)
    column_map = _build_column_map(header, config)
    missing = [name for name in _REQUIRED_COLUMNS if column_map.get(name) not in header]
    if missing:
        raise ValueError(
            f"missing required columns: {', '.join(missing)}; header was {header!r}")
    time_formats = config.input.time_formats or [_DEFAULT_TIME_FORMAT]
    return FormatSpec(
        delimiter=delimiter, encoding=encoding, time_format=time_formats[0],
        column_map=column_map, has_header=True)


def _column_index(header: list[str], column_map: dict[str, str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for canonical, source in column_map.items():
        if source in header:
            index[canonical] = header.index(source)
    return index


# ---------------------------------------------------------------------------
#  Record parsing
# ---------------------------------------------------------------------------
def _parse_time(raw: str | None, time_formats: Sequence[str]) -> datetime | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    for time_format in time_formats:
        try:
            return datetime.strptime(text, time_format)
        except ValueError:
            continue
    return None


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _number(
    raw: str | None, column: str, name: str, line: int,
    diagnostics: list[dict[str, Any]], *, required: bool,
) -> float:
    value = _to_float(raw)
    if value is None and raw is not None and raw.strip():
        diagnostics.append({
            "line": line,
            "severity": "error" if required else "warning",
            "message": f"{name}:{line}: {column}: cannot parse {raw!r}",
        })
    return math.nan if value is None else value


def _angle(
    raw: str | None, column: str, name: str, line: int,
    diagnostics: list[dict[str, Any]], *, required: bool,
) -> float:
    if raw is None or not raw.strip():
        return math.nan
    value = parse_dms(raw)
    if math.isnan(value):
        numeric = _to_float(raw)
        if numeric is None:
            diagnostics.append({
                "line": line,
                "severity": "error" if required else "warning",
                "message": f"{name}:{line}: {column}: cannot parse {raw!r}",
            })
            return math.nan
        value = numeric
    return value


def _parse_record(
    name: str, line: int, fields: list[str], index: dict[str, int],
    config: AnalysisConfig, sha: str, diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    def cell(canonical: str) -> str | None:
        position = index.get(canonical)
        return fields[position] if position is not None else None

    point_raw = cell("point_id")
    point_id = point_raw.strip() if point_raw is not None else ""
    if not point_id:
        diagnostics.append({
            "line": line, "severity": "error",
            "message": f"{name}:{line}: point_id is empty",
        })

    time_raw = cell("time")
    timestamp = _parse_time(time_raw, config.input.time_formats)
    if timestamp is None:
        diagnostics.append({
            "line": line, "severity": "error",
            "message": f"{name}:{line}: time: cannot parse {time_raw!r}",
        })

    return {
        "obs_id": make_observation_id(sha, line),
        "source_sha256": sha,
        "src_line": line,
        "source_name": name,
        "point_id": point_id,
        "time": timestamp if timestamp is not None else pd.NaT,
        "hz_deg": _angle(cell("hz_deg"), "hz_deg", name, line, diagnostics, required=True),
        "v_deg": _angle(cell("v_deg"), "v_deg", name, line, diagnostics, required=True),
        "d_m": _number(cell("d_m"), "d_m", name, line, diagnostics, required=True),
        "te_m": _number(cell("te_m"), "te_m", name, line, diagnostics, required=True),
        "tn_m": _number(cell("tn_m"), "tn_m", name, line, diagnostics, required=True),
        "tz_m": _number(cell("tz_m"), "tz_m", name, line, diagnostics, required=True),
        "se_m": _number(cell("se_m"), "se_m", name, line, diagnostics, required=True),
        "sn_m": _number(cell("sn_m"), "sn_m", name, line, diagnostics, required=True),
        "sh_m": _number(cell("sh_m"), "sh_m", name, line, diagnostics, required=True),
        "temp_c": _number(cell("temp_c"), "temp_c", name, line, diagnostics, required=False),
        "ppm": _number(cell("ppm"), "ppm", name, line, diagnostics, required=False),
        "horz_dist_m": _number(
            cell("horz_dist_m"), "horz_dist_m", name, line, diagnostics, required=False),
        "null_meas_m": _number(
            cell("null_meas_m"), "null_meas_m", name, line, diagnostics, required=False),
    }


def _parse_source(
    name: str, data: bytes, spec: FormatSpec, config: AnalysisConfig, sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reader = csv.reader(io.StringIO(data.decode(spec.encoding)), delimiter=spec.delimiter)
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    header: list[str] | None = None
    index: dict[str, int] = {}
    next_line = 1
    for fields in reader:
        line = next_line
        next_line = reader.line_num + 1
        if not fields:
            continue
        if header is None:
            header = [field.strip() for field in fields]
            index = _column_index(header, spec.column_map)
            continue
        if len(fields) != len(header):
            diagnostics.append({
                "line": line, "severity": "error",
                "message": (f"{name}:{line}: record has {len(fields)} fields, "
                            f"expected {len(header)}"),
            })
            continue
        records.append(_parse_record(name, line, fields, index, config, sha, diagnostics))
    return records, diagnostics


# ---------------------------------------------------------------------------
#  Frames
# ---------------------------------------------------------------------------
_STRING_OBSERVATION_COLUMNS = ("obs_id", "source_sha256", "source_name", "point_id")


def _observations_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records, columns=list(OBSERVATION_COLUMNS))
    dtypes: dict[str, str] = {}
    for column in OBSERVATION_COLUMNS:
        if column in _STRING_OBSERVATION_COLUMNS:
            dtypes[column] = "str"
        elif column == "src_line":
            dtypes[column] = "int64"
        elif column == "time":
            dtypes[column] = "datetime64[ns]"
        else:
            dtypes[column] = "float64"
    return frame.astype(dtypes)


def _diagnostics_frame(diagnostics: list[dict[str, Any]]) -> pd.DataFrame:
    if not diagnostics:
        return pd.DataFrame({
            "line": pd.Series(dtype="int64"),
            "severity": pd.Series(dtype="str"),
            "message": pd.Series(dtype="str"),
        })
    frame = pd.DataFrame(diagnostics, columns=["line", "severity", "message"])
    frame["line"] = frame["line"].astype("int64")
    frame["severity"] = frame["severity"].astype("str")
    frame["message"] = frame["message"].astype("str")
    return frame
