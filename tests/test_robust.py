"""
test_robust.py — Theil–Sen harus identik dengan scipy.stats.theilslopes.

Nilai acuan di bawah dihitung sekali dengan `scipy.stats.theilslopes` dan
ditanam di sini, sehingga uji berjalan tanpa memasang scipy. Ketiga kasus
(sederhana, banyak nilai kembar, x tak berurut + alpha 0.90) menghasilkan
selisih 0,000e+00 terhadap scipy versi 1.17.1.
"""

import numpy as np
import pytest

from prismacore import robust as rb

TOL = 1e-12

KASUS = {
    # nama: (x, y, alpha, hasil acuan (kemiringan, intersep, lo, hi))
    "sederhana": (
        np.arange(10.0),
        np.array([1.0, 1.2, 1.1, 1.5, 1.3, 1.6, 1.4, 1.8, 1.7, 1.9]),
        0.95,
        (0.1, 1.0, 0.05, 0.14),
    ),
    "nilai_kembar": (
        np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
                 dtype=float),
        np.array([2.0, 2.0, 1.5, 2.1, 1.8, 1.7, 2.4, 2.2, 2.0, 2.6,
                  2.1, 2.3, 2.9, 2.5, 2.4, 3.0, 2.8, 3.2, 3.1, 3.3]),
        0.95,
        (0.15, 1.675, 0.1, 0.2),
    ),
    "x_tak_berurut_alpha90": (
        np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 10], dtype=float),
        np.array([3.0, 2.9, 3.2, 3.1, 3.4, 3.3, 3.6, 3.5, 3.8, 3.7, 4.0, 3.9]),
        0.90,
        (-0.0657142857142857, 3.81142857142857, -0.1, 0.05),
    ),
}


@pytest.mark.parametrize("nama", list(KASUS))
def test_theilslopes_sama_dengan_acuan(nama):
    x, y, alpha, acuan = KASUS[nama]
    hasil = rb.theilslopes(y, x, alpha)
    for nilai, target in zip(hasil, acuan):
        assert nilai == pytest.approx(target, abs=TOL), (nama, hasil, acuan)


def test_theilslopes_tanpa_x_memakai_indeks():
    y = np.array([1.0, 1.2, 1.1, 1.5, 1.3, 1.6, 1.4, 1.8, 1.7, 1.9])
    assert rb.theilslopes(y) == rb.theilslopes(y, np.arange(len(y), dtype=float))


def test_theilslopes_data_tidak_cukup():
    assert all(np.isnan(v) for v in rb.theilslopes([1.0]))
    assert all(np.isnan(v) for v in rb.theilslopes([1.0, 2.0], [3.0, 3.0]))


def test_mad():
    v = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert rb.mad(v) == pytest.approx(np.median(np.abs(v - np.median(v))) * 1.4826)
    assert np.isnan(rb.mad([np.nan, np.nan]))
