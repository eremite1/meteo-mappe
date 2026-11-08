"""
Climatologia 1991-2020 (rianalisi NCEP/NCAR) per le mappe di anomalia.
La media del giorno si ottiene interpolando tra le medie mensili (riferite a metà mese).
"""
import os
from datetime import datetime
import numpy as np
from scipy.interpolate import RegularGridInterpolator

FILE = os.path.join(os.path.dirname(__file__), "..", "dati", "clima_ncep_1991_2020.npz")
_z = None
_cache = {}


def disponibile():
    return os.path.exists(FILE)


def _dati():
    global _z
    if _z is None:
        _z = np.load(FILE)
    return _z


def _pesi_mese(t):
    """Due mesi e i loro pesi per il giorno t (medie mensili riferite al giorno 15)."""
    g = t.timetuple().tm_yday
    meta = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    for k in range(12):
        a, b = meta[k], meta[(k + 1) % 12] + (365 if k == 11 else 0)
        gg = g + (365 if (k == 11 and g < meta[0]) else 0)
        if a <= gg < b:
            w = (gg - a) / (b - a)
            return k, (k + 1) % 12, w
    return 11, 0, (g + 365 - 349) / (380 - 349)


def giorno(nome, t):
    """Campo climatologico del giorno sulla griglia NCEP: (lat, lon, valori)."""
    z = _dati()
    m1, m2, w = _pesi_mese(t)
    v = (1 - w) * z[nome][m1] + w * z[nome][m2]
    return z[nome + "_lat"], z[nome + "_lon"], v


def su_griglia(nome, t, lat, lon):
    """Climatologia interpolata sulla griglia regolare (lat, lon 1D) di un'area."""
    chiave = (nome, t.strftime("%Y%m%d%H"), lat.size, lon.size, float(lat[0]), float(lon[0]))
    if chiave not in _cache:
        if len(_cache) > 40:
            _cache.clear()
        la, lo, v = giorno(nome, t)
        f = RegularGridInterpolator((la, lo), v, bounds_error=False, fill_value=np.nan)
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        _cache[chiave] = f(np.stack([LA, LO], -1)).astype("float64")
    return _cache[chiave]


def area(nome, c):
    """Climatologia per il contesto c (modello, area, istante)."""
    if not disponibile():
        raise KeyError("climatologia non disponibile")
    lat, lon = c.L.griglie[c.area]
    return su_griglia(nome, c.t, lat, lon)
