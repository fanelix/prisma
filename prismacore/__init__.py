"""
prismacore — mesin analisis pemantauan prisma RTS.
===================================================
Paket inti tanpa scipy/GDAL. Hanya butuh pandas + numpy.

Pemakaian:
    import prismacore

    hasil = prismacore.jalankan(["data/Candrian_Sep_w1_w2_2026.csv"])
    print(hasil["info"])

Konfigurasi situs cukup ditulis sebagai JSON (lihat `muat_konfigurasi`):

    cfg = prismacore.muat_konfigurasi("konfigurasi/candrian.json")
    hasil = prismacore.jalankan(berkas, cfg)
"""

from .core import CONFIG, VERSI, jalankan, muat_konfigurasi
from . import core, export, gpkg_lite, robust

__version__ = VERSI

__all__ = [
    "jalankan",
    "muat_konfigurasi",
    "CONFIG",
    "VERSI",
    "core",
    "export",
    "gpkg_lite",
    "robust",
]
