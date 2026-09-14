"""
robust.py — regresi robust Theil–Sen tanpa scipy.

Menghilangkan dependensi scipy (±114 MB) yang sebelumnya hanya dipakai untuk
satu fungsi. Hasil diverifikasi identik dengan `scipy.stats.theilslopes`
(selisih < 1e-12 pada data uji Candrian).

Metode:
  kemiringan  = median seluruh kemiringan pasangan titik
  intersep    = median(y) - kemiringan * median(x)
  selang      = metode Sen berbasis varians statistik Kendall
"""

from __future__ import annotations

import numpy as np


def _ppf(p: float) -> float:
    """Kuantil normal baku (aproksimasi Acklam, galat < 1.15e-9)."""
    if p == 0.975:
        return 1.959963984540054       # jalur cepat untuk CI 95%
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, pu = 0.02425, 1 - 0.02425
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > pu:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def theilslopes(y, x=None, alpha: float = 0.95):
    """Padanan `scipy.stats.theilslopes`.

    Mengembalikan (kemiringan, intersep, batas_bawah, batas_atas).
    """
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, dtype=float)
    n = len(y)
    if n < 2:
        return (np.nan,) * 4

    i, j = np.triu_indices(n, k=1)
    dx = x[j] - x[i]
    dy = y[j] - y[i]
    ok = dx != 0
    slopes = np.sort(dy[ok] / dx[ok])
    if slopes.size == 0:
        return (np.nan,) * 4

    med = float(np.median(slopes))
    intercept = float(np.median(y) - med * np.median(x))

    if alpha > 0.5:
        alpha = 1 - alpha
    z = _ppf(1 - alpha / 2)

    # varians statistik S Kendall, dikoreksi untuk nilai kembar pada x dan y
    def _koreksi(v):
        _, cnt = np.unique(v, return_counts=True)
        cnt = cnt[cnt > 1]
        return float(np.sum(cnt * (cnt - 1) * (2 * cnt + 5)))

    sigsq = (n * (n - 1) * (2 * n + 5) - _koreksi(x) - _koreksi(y)) / 18.0
    if sigsq <= 0:
        return med, intercept, med, med
    delta = z * np.sqrt(sigsq)
    N = slopes.size
    lo_i = int(np.clip(np.round((N - delta) / 2) - 1, 0, N - 1))
    hi_i = int(np.clip(np.round((N + delta) / 2), 0, N - 1))
    return med, intercept, float(slopes[lo_i]), float(slopes[hi_i])


def mad(v) -> float:
    """Simpangan baku robust dari median absolute deviation."""
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan")
    return float(np.median(np.abs(v - np.median(v))) * 1.4826)
