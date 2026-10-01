"""
geometry.py — angles and local grid geometry.

Conventions (documented once, used everywhere):

* grid axes are East and North in the supplied Cartesian site grid;
* azimuth is measured clockwise from North: ``atan2(east, north)``;
* radial direction points from the RTS station towards the prism;
* tangential is the in-plane direction 90 degrees clockwise from radial
  (right-handed with respect to the vertical axis);
* internal angles are radians in exported quantities only where stated;
  observation angles remain degrees and Hz is aggregated circularly.
"""

from __future__ import annotations

import numpy as np


def angular_difference_deg(a: np.ndarray | float,
                           b: np.ndarray | float) -> np.ndarray | float:
    """Signed shortest difference ``a - b`` in degrees, within (-180, 180]."""
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0


def circular_mean_deg(values, weights=None) -> float:
    """Circular mean of angles in degrees, correct across the 0/360 wrap."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan")
    rad = np.radians(v)
    if weights is None:
        c, s = np.cos(rad).mean(), np.sin(rad).mean()
    else:
        w = np.asarray(weights, dtype=float)
        c = np.average(np.cos(rad), weights=w)
        s = np.average(np.sin(rad), weights=w)
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


def circular_std_deg(values) -> float:
    """Circular standard deviation in degrees."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 2:
        return float("nan")
    rad = np.radians(v)
    r = np.hypot(np.cos(rad).mean(), np.sin(rad).mean())
    r = min(max(r, 1e-12), 1.0)
    return float(np.degrees(np.sqrt(-2.0 * np.log(r))))


def azimuth_deg(east, north):
    """Grid azimuth clockwise from North, degrees [0, 360)."""
    return np.degrees(np.arctan2(np.asarray(east, dtype=float),
                                 np.asarray(north, dtype=float))) % 360.0


def horizontal_distance(east, north):
    return np.hypot(np.asarray(east, dtype=float), np.asarray(north, dtype=float))


def polar_components(d_east, d_north, azimuth_deg_):
    """Project a local displacement onto radial and tangential unit vectors.

    radial  = +d·u_r  (away from the station)
    tangential = +d·u_t with u_t = (cos az, -sin az) in (E, N)
    """
    az = np.radians(np.asarray(azimuth_deg_, dtype=float))
    de = np.asarray(d_east, dtype=float)
    dn = np.asarray(d_north, dtype=float)
    radial = de * np.sin(az) + dn * np.cos(az)
    tangential = de * np.cos(az) - dn * np.sin(az)
    return radial, tangential


def enu_from_polar(radial, tangential, azimuth_deg_):
    """Inverse of :func:`polar_components`."""
    az = np.radians(np.asarray(azimuth_deg_, dtype=float))
    r = np.asarray(radial, dtype=float)
    t = np.asarray(tangential, dtype=float)
    return r * np.sin(az) + t * np.cos(az), r * np.cos(az) - t * np.sin(az)
