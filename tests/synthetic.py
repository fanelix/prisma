"""
synthetic.py — independent forward model for synthetic RTS exports.

The generator deliberately shares no code with the production pipeline so that
sign/unit mistakes in the pipeline cannot be cancelled by the same mistake in
the fixture. It produces the documented GeoMoS-style export: cp1252, CRLF,
``;`` separator, 18 columns, quoted DMS strings.

Injected effects are explicit arguments, never hidden defaults:
rotation step/drift, station translation, scale, local prism motion, daytime
LOS bias and dropouts. Tests configure real numbers; the package never does.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

HEADER = [
    "Point ID", "Time", "Hz [dms]", "V [dms]", "D [m]",
    "PPM Type", "PPM", "Pressure [mBar]", "Av Temp [C]", "Add Const [m]",
    "Target Easting [m]", "Target Northing [m]", "Target Elevation [m]",
    "Station Easting [m]", "Station Northing [m]", "Station Height [m]",
    "Null Measurement [m]", "Horz Distance [m]",
]

NIGHT_HOURS = {20, 0, 4}


@dataclass
class Station:
    label: str
    east: float
    north: float
    height: float
    orientation_deg: float
    gap_minutes: int
    cycle_hours: list[int]


@dataclass
class Prism:
    pid: str
    station: str
    east: float
    north: float
    z: float


@dataclass
class Effects:
    """All injected effects are opt-in; zero means a physically clean export."""

    seed: int = 0
    noise_hz_arcsec: float = 0.0
    noise_v_arcsec: float = 0.0
    noise_d_mm: float = 0.0
    rotation_step_arcsec: float = 0.0
    rotation_drift_arcsec_per_day: float = 0.0
    rotation_applied_in_coordinates: bool = True
    translation_mm: tuple[float, float] = (0.0, 0.0)
    scale_ppm: float = 0.0
    local_motion_mm_per_day: dict[str, float] = field(default_factory=dict)
    diurnal_los_mm: dict[str, float] = field(default_factory=dict)  # prism -> day-cycle bias
    dropout_after: dict[str, str] = field(default_factory=dict)     # prism -> ISO time
    missing_night: list[str] = field(default_factory=list)
    repeats: int = 1


def default_stations() -> dict[str, Station]:
    return {
        "E": Station("E", 1000.0, 2000.0, 50.0, 30.0, 30, [0, 4, 8, 12, 16, 20]),
        "W": Station("W", 0.0, 1500.0, 45.0, -10.0, 15,
                     [17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6]),
    }


def default_prisms() -> dict[str, Prism]:
    return {
        "P_A": Prism("P_A", "E", 1400.0, 2600.0, 58.0),
        "P_B": Prism("P_B", "E", 1600.0, 2200.0, 55.0),
        "P_C": Prism("P_C", "E", 900.0, 2800.0, 70.0),
        "P_D": Prism("P_D", "E", 1000.0, 1500.0, 40.0),
        "P_E": Prism("P_E", "E", 1300.0, 2400.0, 52.0),
        "P_F": Prism("P_F", "E", 1500.0, 2500.0, 57.0),
        "P_W1": Prism("P_W1", "W", 200.0, 1200.0, 42.0),
        "P_W2": Prism("P_W2", "W", -100.0, 1300.0, 44.0),
    }


def _format_dms(degrees: float) -> str:
    sign = "-" if degrees < 0 else ""
    a = abs(degrees)
    d = int(a)
    m = int((a - d) * 60)
    s = (a - d - m / 60) * 3600
    return f"{sign}{d}° {m:02d}' {s:08.5f}\""


def build_observations(*, days: int = 3, start: str = "2026-01-01",
                       stations: dict[str, Station] | None = None,
                       prisms: dict[str, Prism] | None = None,
                       effects: Effects | None = None) -> list[dict]:
    """Return source-shaped row dictionaries (one per observation)."""
    stations = stations or default_stations()
    prisms = prisms or default_prisms()
    eff = effects or Effects()
    rng = np.random.default_rng(eff.seed)
    t0 = datetime.fromisoformat(start)
    rows: list[dict] = []

    by_station: dict[str, list[Prism]] = {}
    for p in prisms.values():
        by_station.setdefault(p.station, []).append(p)

    for day in range(days):
        for label, st in stations.items():
            for hour in sorted(st.cycle_hours):
                t = t0 + timedelta(days=day, hours=hour)
                elapsed_days = (t - t0).total_seconds() / 86400.0
                rotation_as = (eff.rotation_step_arcsec
                               + eff.rotation_drift_arcsec_per_day * elapsed_days)
                st_e = st.east + eff.translation_mm[0] / 1000.0
                st_n = st.north + eff.translation_mm[1] / 1000.0
                st_h = st.height
                orientation = st.orientation_deg
                applied = orientation + (rotation_as / 3600.0
                                         if eff.rotation_applied_in_coordinates else 0.0)

                for p in by_station.get(label, []):
                    if p.pid in eff.dropout_after and \
                            t >= datetime.fromisoformat(eff.dropout_after[p.pid]):
                        continue
                    if p.pid in eff.missing_night and hour in NIGHT_HOURS:
                        continue

                    motion_m = eff.local_motion_mm_per_day.get(p.pid, 0.0) * day / 1000.0
                    de, dn = p.east - st_e, p.north - st_n
                    dz = p.z - st_h
                    hd_true = math.hypot(de, dn) + motion_m
                    slope_true = math.hypot(hd_true, dz)
                    az_true = math.degrees(math.atan2(de, dn)) % 360
                    zen_true = math.degrees(math.atan2(hd_true, dz))

                    for _ in range(max(1, eff.repeats)):
                        hz = (az_true - orientation
                              + rotation_as / 3600.0
                              + rng.normal(0, eff.noise_hz_arcsec) / 3600.0) % 360
                        v = zen_true + rng.normal(0, eff.noise_v_arcsec) / 3600.0
                        d = (slope_true * (1 + eff.scale_ppm * 1e-6)
                             + rng.normal(0, eff.noise_d_mm) / 1000.0)
                        if p.pid in eff.diurnal_los_mm and hour in (8, 12, 16):
                            d -= eff.diurnal_los_mm[p.pid] / 1000.0

                        hd = d * math.sin(math.radians(v))
                        te = st_e + hd * math.sin(math.radians(applied))
                        tn = st_n + hd * math.cos(math.radians(applied))
                        tz = st_h + d * math.cos(math.radians(v))
                        rows.append({
                            "Point ID": p.pid,
                            "Time": t.strftime("%d/%m/%Y %H:%M"),
                            "Hz [dms]": _format_dms(hz),
                            "V [dms]": _format_dms(v),
                            "D [m]": f"{d:.3f}",
                            "PPM Type": "Atmos PPM",
                            "PPM": "0",
                            "Pressure [mBar]": "1013.25",
                            "Av Temp [C]": "11.1012",
                            "Add Const [m]": "0",
                            "Target Easting [m]": f"{te:.3f}",
                            "Target Northing [m]": f"{tn:.3f}",
                            "Target Elevation [m]": f"{tz:.3f}",
                            "Station Easting [m]": f"{st_e:.3f}",
                            "Station Northing [m]": f"{st_n:.3f}",
                            "Station Height [m]": f"{st_h:.3f}",
                            "Null Measurement [m]": f"{hd_true:.3f}",
                            "Horz Distance [m]": f"{hd:.3f}",
                        })
    rows.sort(key=lambda r: (datetime.strptime(r["Time"], "%d/%m/%Y %H:%M"),
                             r["Point ID"]))
    return rows


def rows_to_bytes(rows: list[dict], *, encoding: str = "cp1252") -> bytes:
    """Serialize rows to the documented CSV bytes (CRLF, quoted DMS)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=HEADER, delimiter=";",
                       lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode(encoding)


def write_fixture(path, **kwargs) -> tuple[str, list[dict]]:
    rows = build_observations(**kwargs)
    data = rows_to_bytes(rows)
    with open(path, "wb") as f:
        f.write(data)
    return str(path), rows
