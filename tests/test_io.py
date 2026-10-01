"""
test_io.py — parser contract tests against the independent synthetic model.

Fixtures come from :mod:`synthetic` (cp1252, CRLF, quoted DMS); the parser must
recover row count, physical line provenance, dtypes and exact raw bytes.
Malformed records, a bad timestamp and a multiline quoted field are injected
manually so each failure mode is explicit.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from datetime import datetime
from pathlib import Path

import pytest
import synthetic

from rts_forensics.io import detect_format, parse_dms, read_geomos
from rts_forensics.models import OBSERVATION_COLUMNS, SOURCE_LINK_COLUMNS
from rts_forensics.provenance import make_observation_id


def _write(tmp_path: Path, name: str = "export.csv", **kwargs):
    path, rows = synthetic.write_fixture(tmp_path / name, **kwargs)
    return Path(path), rows, Path(path).read_bytes()


# ---------------------------------------------------------------------------
#  parse_dms
# ---------------------------------------------------------------------------
def test_parse_dms_positive():
    expected = 148 + 36 / 60 + 38.36901 / 3600
    assert parse_dms('148° 36\' 38.36901"') == pytest.approx(expected)


def test_parse_dms_negative():
    expected = -(148 + 36 / 60 + 38.36901 / 3600)
    assert parse_dms('-148° 36\' 38.36901"') == pytest.approx(expected)


def test_parse_dms_wrapped_and_beyond_circle():
    expected = 148 + 36 / 60 + 38.36901 / 3600
    assert parse_dms(' 148° 36\' 38.36901"') == pytest.approx(expected)
    assert parse_dms("380° 00' 00.5\"") == pytest.approx(380 + 0.5 / 3600)


@pytest.mark.parametrize(
    "value",
    [None, 148.5, b"148", "", "   ", "junk", "148° 36'", "° 36' 38\""],
)
def test_parse_dms_junk(value):
    assert math.isnan(parse_dms(value))


# ---------------------------------------------------------------------------
#  Round trip
# ---------------------------------------------------------------------------
def test_synthetic_round_trip(tmp_path):
    path, rows, data = _write(tmp_path, days=1)
    result = read_geomos(path)
    obs = result.observations

    assert len(obs) == len(rows)
    assert list(obs.columns) == list(OBSERVATION_COLUMNS)
    assert obs["src_line"].tolist() == list(range(2, len(rows) + 2))
    assert obs["point_id"].tolist() == [row["Point ID"] for row in rows]

    assert obs["time"].dtype == "datetime64[ns]"
    assert obs["src_line"].dtype == "int64"
    assert obs["d_m"].dtype == "float64"
    assert obs["temp_c"].dtype == "float64"
    assert obs["time"].dt.tz is None

    first = rows[0]
    assert obs["time"].iloc[0] == datetime.strptime(first["Time"], "%d/%m/%Y %H:%M")
    assert obs["d_m"].iloc[0] == pytest.approx(float(first["D [m]"]))
    assert obs["hz_deg"].iloc[0] == pytest.approx(parse_dms(first["Hz [dms]"]))
    assert obs["v_deg"].iloc[0] == pytest.approx(parse_dms(first["V [dms]"]))
    assert obs["se_m"].iloc[0] == pytest.approx(float(first["Station Easting [m]"]))
    assert obs["temp_c"].iloc[0] == pytest.approx(float(first["Av Temp [C]"]))

    record = result.sources[0]
    assert record.name == path.name
    assert record.sha256 == hashlib.sha256(data).hexdigest()
    assert record.n_bytes == len(data)
    assert record.n_rows == len(rows)
    assert result.raw_bytes[path.name] == data
    assert b"\r\n" in data
    assert b"\xb0" in data


def test_repeats_are_retained(tmp_path):
    path, rows, _ = _write(tmp_path, days=1, effects=synthetic.Effects(repeats=3))
    obs = read_geomos(path).observations
    assert len(obs) == len(rows)
    counts = obs.groupby(["point_id", "time"]).size()
    assert counts.max() == 3


def test_optional_columns_absent_become_nan(tmp_path):
    rows = synthetic.build_observations(days=1)
    header = [
        "Point ID", "Time", "Hz [dms]", "V [dms]", "D [m]",
        "Target Easting [m]", "Target Northing [m]", "Target Elevation [m]",
        "Station Easting [m]", "Station Northing [m]", "Station Height [m]",
    ]
    first = rows[0]
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";", lineterminator="\r\n")
    writer.writerow(header)
    writer.writerow([first[name] for name in header])
    path = tmp_path / "minimal.csv"
    path.write_bytes(buf.getvalue().encode("cp1252"))

    result = read_geomos(path)
    obs = result.observations
    assert len(obs) == 1
    for column in ("temp_c", "ppm", "horz_dist_m", "null_meas_m"):
        assert obs[column].isna().all()
    assert not obs["hz_deg"].isna().any()


# ---------------------------------------------------------------------------
#  Diagnostics and physical lines
# ---------------------------------------------------------------------------
def test_malformed_record_and_bad_time_diagnostics(tmp_path):
    _, rows, data = _write(tmp_path, days=1)
    lines = data.decode("cp1252").split("\r\n")
    header, body = lines[0], [line for line in lines[1:] if line]

    wrong_count = ";".join(body[0].split(";")[:-1])
    bad_time_parts = body[1].split(";")
    bad_time_parts[1] = "not-a-time"
    bad_time = ";".join(bad_time_parts)

    injected = "\r\n".join([header, wrong_count, bad_time, *body]) + "\r\n"
    path = tmp_path / "injected.csv"
    path.write_bytes(injected.encode("cp1252"))
    result = read_geomos(path)

    obs = result.observations
    diags = result.diagnostics
    assert len(obs) == len(rows) + 1  # wrong field count skipped, bad time kept
    assert set(diags["severity"]).issubset({"warning", "error"})
    assert "error" in set(diags["severity"])
    assert {2, 3}.issubset(set(diags["line"]))
    assert obs["src_line"].iloc[0] == 3
    assert obs["src_line"].iloc[1] == 4
    assert obs["time"].isna().sum() == 1
    assert obs.loc[obs["time"].isna(), "src_line"].tolist() == [3]
    messages = " | ".join(diags["message"])
    assert "expected" in messages
    assert "time" in messages


def test_multiline_quoted_field_keeps_true_start_lines(tmp_path):
    rows = synthetic.build_observations(days=1)
    first = dict(rows[0])
    first["Point ID"] = "MULTI\nLINE"
    second = rows[1]
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=synthetic.HEADER, delimiter=";", lineterminator="\r\n")
    writer.writeheader()
    writer.writerow(first)
    writer.writerow(second)
    path = tmp_path / "multiline.csv"
    path.write_bytes(buf.getvalue().encode("cp1252"))

    obs = read_geomos(path).observations
    assert len(obs) == 2
    assert obs["point_id"].tolist() == ["MULTI\nLINE", second["Point ID"]]
    assert obs["src_line"].tolist() == [2, 4]


# ---------------------------------------------------------------------------
#  Format detection
# ---------------------------------------------------------------------------
def test_detect_format_synthetic_bytes(tmp_path):
    path, _, data = _write(tmp_path, days=1)
    spec = detect_format(data)
    assert spec.delimiter == ";"
    assert spec.encoding == "cp1252"
    assert spec.time_format == "%d/%m/%Y %H:%M"
    assert spec.column_map["point_id"] == "Point ID"
    assert spec.column_map["time"] == "Time"
    assert spec.column_map["hz_deg"] == "Hz [dms]"
    assert spec.column_map["v_deg"] == "V [dms]"
    assert spec.column_map["d_m"] == "D [m]"
    assert spec.column_map["te_m"] == "Target Easting [m]"
    assert spec.column_map["tn_m"] == "Target Northing [m]"
    assert spec.column_map["tz_m"] == "Target Elevation [m]"
    assert spec.column_map["se_m"] == "Station Easting [m]"
    assert spec.column_map["sn_m"] == "Station Northing [m]"
    assert spec.column_map["sh_m"] == "Station Height [m]"
    assert spec.column_map["null_meas_m"] == "Null Measurement [m]"
    assert spec.column_map["horz_dist_m"] == "Horz Distance [m]"
    assert spec.column_map["ppm"] == "PPM"
    assert spec.column_map["temp_c"] == "Av Temp [C]"
    assert detect_format(path).delimiter == ";"
    assert detect_format(("export.csv", data)).delimiter == ";"


def test_detect_format_tab_separated(tmp_path):
    _, rows, data = _write(tmp_path, days=1)
    tab_data = data.decode("cp1252").replace(";", "\t").encode("cp1252")
    spec = detect_format(tab_data)
    assert spec.delimiter == "\t"
    assert len(read_geomos(tab_data).observations) == len(rows)


def test_detect_format_missing_required_columns():
    data = b"Point ID;Time\r\nP1;01/01/2026 00:00\r\n"
    with pytest.raises(ValueError, match="hz_deg"):
        detect_format(data)


# ---------------------------------------------------------------------------
#  Provenance, sources and immutability
# ---------------------------------------------------------------------------
def test_provenance_ids_and_source_links(tmp_path):
    path, rows, _ = _write(tmp_path, days=1)
    result = read_geomos(path)
    obs = result.observations
    sha = result.sources[0].sha256

    for record in obs.itertuples(index=False):
        prefix, _, line = record.obs_id.partition(":")
        assert prefix == sha[:16]
        assert int(line) == record.src_line
        assert record.obs_id == make_observation_id(record.source_sha256, record.src_line)

    links = result.source_links
    assert list(links.columns) == list(SOURCE_LINK_COLUMNS)
    assert len(links) == len(obs) == len(rows)
    assert links["table_name"].eq("observations").all()
    assert links["role"].eq("observation").all()
    assert links["row_key"].tolist() == obs["obs_id"].tolist()
    assert links["src_line"].tolist() == obs["src_line"].tolist()
    assert links["source_sha256"].eq(sha[:16]).all()


def test_read_geomos_accepts_bytes_tuple_and_list(tmp_path):
    path_a, rows_a, data_a = _write(tmp_path, "a.csv", days=1)
    path_b, rows_b, _ = _write(tmp_path, "b.csv", days=1, start="2026-02-01")

    from_bytes = read_geomos(data_a)
    from_tuple = read_geomos(("named.csv", data_a))
    assert len(from_bytes.observations) == len(rows_a)
    assert len(from_tuple.observations) == len(rows_a)
    assert from_tuple.sources[0].name == "named.csv"

    combined = read_geomos([path_a, path_b])
    assert len(combined.observations) == len(rows_a) + len(rows_b)
    assert [record.name for record in combined.sources] == ["a.csv", "b.csv"]
    assert set(combined.raw_bytes) == {"a.csv", "b.csv"}
    names = combined.observations["source_name"].tolist()
    assert names == ["a.csv"] * len(rows_a) + ["b.csv"] * len(rows_b)
    second_file = combined.observations[combined.observations["source_name"] == "b.csv"]
    assert second_file["src_line"].iloc[0] == 2


def test_read_geomos_returns_new_frames(tmp_path):
    path, _, _ = _write(tmp_path, days=1)
    first = read_geomos(path)
    second = read_geomos(path)
    assert first.observations is not second.observations
    assert first.source_links is not second.source_links
    assert first.diagnostics is not second.diagnostics
    assert first.raw_bytes is not second.raw_bytes
    assert first.observations.equals(second.observations)
    first.observations.loc[0, "point_id"] = "MUTATED"
    assert second.observations.loc[0, "point_id"] != "MUTATED"
