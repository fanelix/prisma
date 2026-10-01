"""Parse once, preserve every source row including invalid measurements."""

import csv
import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = {
    "D [m]": "d",
    "Target Easting [m]": "e",
    "Target Northing [m]": "n",
    "Target Elevation [m]": "z",
    "Station Easting [m]": "st_e",
    "Station Northing [m]": "st_n",
    "Station Height [m]": "st_h",
    "PPM": "ppm",
    "Pressure [mBar]": "pressure",
    "Av Temp [°C]": "temperature",
    "Add Const [m]": "add_const",
    "Null Measurement [m]": "null_measurement",
    "Horz Distance [m]": "horizontal_distance",
}
REQUIRED = ["Point ID", "Time", "Hz [dms]", "V [dms]", *list(NUMERIC)[:7]]
DMS = re.compile(r"^\s*([+-]?\d+)\D+(\d+)\D+(\d+(?:\.\d+)?)\D*$", re.ASCII)


def decode_dms(value):
    match = DMS.match(str(value))
    if not match:
        return np.nan
    degree, minute, second = match.groups()
    if float(minute) >= 60 or float(second) >= 60:
        return np.nan
    sign = -1 if degree.startswith("-") else 1
    return sign * (abs(float(degree)) + float(minute) / 60 + float(second) / 3600)


def circular_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    center = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(values)))))
    return float((center + np.median((values - center + 180) % 360 - 180)) % 360)


def read_export(source, config):
    if isinstance(source, tuple):
        name, content = source
    else:
        path = Path(source)
        name, content = path.name, path.read_bytes()
    if not isinstance(content, bytes):
        raise ValueError("Input must be a path or (name, bytes)")
    reader = csv.reader(
        io.StringIO(content.decode(config["input"]["encoding"])), delimiter=config["input"]["separator"]
    )
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("Empty export") from None
    missing = set(REQUIRED) - set(header)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(set(header)) != len(header):
        raise ValueError("Duplicate column names")
    records = []
    while True:
        start_line = reader.line_num + 1
        try:
            values = next(reader)
        except StopIteration:
            break
        record = dict(zip(header, values))
        record.update(src_line=start_line, source=name, malformed_row=len(values) != len(header))
        records.append(record)
    if not records:
        raise ValueError("Export has no observations")
    rows = pd.DataFrame(records).reindex(columns=header + ["src_line", "source", "malformed_row"])
    rows["pid"] = rows["Point ID"].str.strip()
    rows["ts"] = pd.to_datetime(rows["Time"], format=config["input"]["timestamp_format"], errors="coerce")
    rows["hz"] = rows["Hz [dms]"].map(decode_dms)
    rows["v"] = rows["V [dms]"].map(decode_dms)
    for source_name, dest in NUMERIC.items():
        rows[dest] = pd.to_numeric(rows.get(source_name, np.nan), errors="coerce")
    required = ["ts", "hz", "v", "d", "e", "n", "z", "st_e", "st_n", "st_h"]
    rows["parse_error"] = (
        rows[required].isna().any(axis=1) | rows.malformed_row | rows.pid.isna() | rows.pid.eq("")
    )
    rows["parse_error"] |= ~np.isfinite(rows[required[1:]].astype(float)).all(axis=1)
    rows["parse_error"] |= (rows.d <= 0) | (rows.v <= 0) | (rows.v >= 180)
    info = {
        "source": name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "rows": len(rows),
        "invalid_rows": int(rows.parse_error.sum()),
        "encoding": config["input"]["encoding"],
        "timezone": "unspecified site local",
    }
    return rows, info
