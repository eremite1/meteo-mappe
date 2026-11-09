"""
EFI (Extreme Forecast Index) del vapore acqueo totale in colonna, calcolato da MeteoNerola con la stessa
formula di ECMWF (Lalaurette 2003; Zsoter 2006), ma confrontando l'ensemble ECMWF (51 membri, dati aperti
CC BY 4.0) con il clima 1991-2020 della rianalisi NCEP/NCAR invece che con il "clima del modello" di ECMWF
(che non e' pubblico).

  EFI = 2/pi * integrale_0^1 (p - F(p)) / sqrt(p (1 - p)) dp
        p = percentile del clima, F(p) = frazione dei membri sotto quel percentile del clima.
  Va da -1 a +1: oltre 0,5 il vapore previsto e' insolito, oltre 0,8 e' molto raro per la stagione.
  SOT (Shift Of Tails, 90): quanto la coda alta dell'ensemble supera il 99% del clima; se positivo
  almeno il 10% dei membri va oltre gli estremi del clima.

Uso:
  python -m motore.efi --clima dati/clima_tcwv.npz      (una volta: percentili giornalieri 1991-2020)
  python -m motore.efi --out uscita_efi                  (ogni corsa)
"""
import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

OPENDAP = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/surface/pr_wtr.eatm.{anno}.nc"
ECMWF = "https://data.ecmwf.int/forecasts/"
LAT = (25.0, 75.0)
LON = (-35.0, 50.0)
PERC = np.arange(1, 100)                  # percentili del clima
FILE_CLIMA = os.path.join(os.path.dirname(__file__), "..", "dati", "clima_tcwv.npz")
GIORNI = 5


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ clima
def prepara_clima(out):
    """Percentili 1-99 del vapore totale in colonna per ogni pentade (5 giorni) dell'anno, usando i giorni
    entro +-15 giorni dal centro della pentade in tutti gli anni 1991-2020 (circa 930 valori per punto)."""
    import xarray as xr
    serie, giorni = [], []
    lat = lon = None
    for anno in range(1991, 2021):
        ds = xr.open_dataset(OPENDAP.format(anno=anno))
        da = ds["pr_wtr"]
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon").sortby("lat")
        da = da.sel(lat=slice(*LAT), lon=slice(*LON)).load()
        if lat is None:
            lat, lon = da.lat.values, da.lon.values
        for t, v in zip(da["time"].values, da.values):
            d = np.datetime64(t, "D").astype(object)
            g = min(d.timetuple().tm_yday, 365)
            giorni.append(g)
            serie.append(v.astype("float32"))
        log(anno, da.shape)
    serie = np.array(serie)
    giorni = np.array(giorni)
    q = np.zeros((73, len(PERC)) + serie.shape[1:], dtype="float32")
    for k in range(73):
        centro = k * 5 + 3
        dist = np.minimum(np.abs(giorni - centro), 365 - np.abs(giorni - centro))
        sel = serie[dist <= 15]
        q[k] = np.percentile(sel, PERC, axis=0)
    np.savez_compressed(out, q=q.astype("float16"), lat=lat.astype("float32"), lon=lon.astype("float32"))
    log("clima salvato:", q.shape)


def clima_giorno(t, lat_a, lon_a):
    from scipy.interpolate import RegularGridInterpolator
    z = np.load(FILE_CLIMA)
    k = min((t.timetuple().tm_yday - 1) // 5, 72)
    q = z["q"][k].astype("float32")
    LA, LO = np.meshgrid(lat_a, lon_a, indexing="ij")
    pts = np.stack([LA, LO], -1)
    return np.array([RegularGridInterpolator((z["lat"], z["lon"]), q[i], bounds_error=False, fill_value=None)(pts)
                     for i in range(q.shape[0])])


# ------------------------------------------------------------------ ensemble ECMWF
def ultima_corsa():
    oggi = datetime.now(timezone.utc)
    for d in range(3):
        g = (oggi - timedelta(days=d)).strftime("%Y%m%d")
        for h in ("12", "00"):
            u = f"{ECMWF}{g}/{h}z/ifs/0p25/enfo/{g}{h}0000-{GIORNI * 24}h-enfo-ef.index"
            if requests.head(u, timeout=30).status_code == 200:
                return g, h
    raise SystemExit("nessuna corsa ensemble ECMWF completa")


def scarica_tcwv(g, h, passo):
    """I 51 membri (50 perturbati + controllo) del vapore totale in colonna a una scadenza, ritagliati."""
    import eccodes
    base = f"{ECMWF}{g}/{h}z/ifs/0p25/enfo/{g}{h}0000-{passo}h-enfo-ef"
    righe = [json.loads(r) for r in requests.get(base + ".index", timeout=60).text.splitlines() if r.strip()]
    pezzi = [r for r in righe if r.get("param") == "tcwv" and r.get("type") in ("pf", "cf")]
    campi = []
    for r in pezzi:
        a, n = int(r["_offset"]), int(r["_length"])
        b = requests.get(base + ".grib2", headers={"Range": f"bytes={a}-{a + n - 1}"}, timeout=120).content
        msg = eccodes.codes_new_from_message(b)
        ni, nj = eccodes.codes_get(msg, "Ni"), eccodes.codes_get(msg, "Nj")
        v = eccodes.codes_get_values(msg).reshape(nj, ni)
        lat0 = eccodes.codes_get(msg, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(msg, "longitudeOfFirstGridPointInDegrees")
        dl = eccodes.codes_get(msg, "iDirectionIncrementInDegrees")
        eccodes.codes_release(msg)
        lats = lat0 - dl * np.arange(nj)
        lons = ((lon0 + dl * np.arange(ni)) + 180) % 360 - 180
        oi = np.argsort(lons)
        v, lons = v[:, oi], lons[oi]
        j = (lats >= LAT[0]) & (lats <= LAT[1])
        i = (lons >= LON[0]) & (lons <= LON[1])
        campi.append(v[np.ix_(j, i)][::-1])
        lat_s, lon_s = lats[j][::-1], lons[i]
    log(f"  +{passo} h: {len(campi)} membri")
    return np.array(campi), lat_s, lon_s


def efi(membri, q):
    """membri [m, y, x], q [99, y, x] percentili del clima -> EFI e SOT(90)."""
    p = PERC / 100.0
    F = (membri[None] < q[:, None]).mean(axis=1)                 # [99, y, x]
    w = 1 / np.sqrt(p * (1 - p))
    integra = getattr(np, "trapezoid", None) or np.trapz
    e = (2 / np.pi) * integra((p[:, None, None] - F) * w[:, None, None], p, axis=0)
    q90m = np.percentile(membri, 90, axis=0)
    q90, q99 = q[89], q[98]
    sot = (q90m - q99) / np.maximum(q99 - q90, 0.1)
    return np.clip(e, -1, 1), sot


def disegna(lat, lon, e, sot, titolo, sotto, file, riquadro):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import ListedColormap, BoundaryNorm
    livelli = [-1, -0.8, -0.6, -0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    colori = ["#6a3d9a", "#9e7cc1", "#cbb8e0", "#ffffff00", "#fff2a8", "#ffd23f", "#ff9f1c", "#f25c05", "#c1121f"]
    fig = plt.figure(figsize=(9, 7.4))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=12, central_latitude=48))
    ax.set_extent(riquadro, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2f0eb")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dfeaf2")
    cf = ax.contourf(lon, lat, e, levels=livelli, colors=colori, transform=ccrs.PlateCarree())
    cs = ax.contour(lon, lat, sot, levels=[0, 1, 2], colors=["#000"], linewidths=[0.8, 1.4, 2.0], transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt="SOT %g", fontsize=7)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)
    ax.plot(12.7856, 42.1606, "k*", ms=8, transform=ccrs.PlateCarree())
    cb = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.03, fraction=0.05, aspect=40, ticks=[-1, -0.8, -0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    cb.set_label("EFI vapore acqueo totale in colonna (0,5 insolito · 0,8 molto raro · linee nere SOT ≥ 0)")
    ax.set_title(titolo, fontsize=13, fontweight="bold", loc="left")
    fig.text(0.02, 0.012, sotto, fontsize=7.5, color="#555")
    plt.savefig(file, dpi=100, bbox_inches="tight")
    plt.close(fig)


def calcola(out):
    os.makedirs(out, exist_ok=True)
    g, h = ultima_corsa()
    corsa = datetime.strptime(g + h, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    log("corsa ensemble", corsa)
    indice = {"corsa": corsa.strftime("%Y%m%d%H"), "creato": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "giorni": []}
    for d in range(1, GIORNI + 1):
        passi = [24 * (d - 1) + k for k in (6, 12, 18, 24)]
        membri = None
        for p in passi:
            m, lat, lon = scarica_tcwv(g, h, p)
            membri = m if membri is None else membri + m
        membri /= len(passi)                                   # media del giorno, membro per membro
        centro = corsa + timedelta(hours=24 * (d - 1) + 15)
        q = clima_giorno(centro, lat, lon)
        e, sot = efi(membri, q)
        inizio, fine = corsa + timedelta(hours=24 * (d - 1)), corsa + timedelta(hours=24 * d)
        titolo = f"EFI vapore acqueo · {inizio:%d/%m %H}-{fine:%d/%m %H} UTC"
        sotto = (f"Calcolo MeteoNerola con la formula EFI di ECMWF: ensemble ECMWF {corsa:%d/%m %H} UTC (51 membri, CC BY 4.0) "
                 f"contro il clima 1991-2020 della rianalisi NCEP/NCAR (non il clima del modello ECMWF)")
        for area, riq in (("europa", [-15, 40, 33, 68]), ("italia", [5, 20, 36, 48])):
            f = f"efi_tcwv_{area}_g{d}.png"
            disegna(lat, lon, e, sot, titolo, sotto, os.path.join(out, f), riq)
        j = np.abs(lat - 42.16).argmin(); i = np.abs(lon - 12.79).argmin()
        indice["giorni"].append({"giorno": d, "da": inizio.strftime("%Y-%m-%dT%H:%MZ"), "a": fine.strftime("%Y-%m-%dT%H:%MZ"),
                                 "nerola_efi": round(float(e[j, i]), 2), "nerola_sot": round(float(sot[j, i]), 2),
                                 "immagini": {"europa": f"efi_tcwv_europa_g{d}.png", "italia": f"efi_tcwv_italia_g{d}.png"}})
        log(f"giorno {d}: EFI a Nerola {e[j, i]:+.2f}, SOT {sot[j, i]:+.2f}")
    json.dump(indice, open(os.path.join(out, "indice.json"), "w"), indent=1)
    import shutil
    shutil.copy(os.path.join(os.path.dirname(__file__), "efi.html"), os.path.join(out, "index.html"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clima")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.clima:
        prepara_clima(a.clima)
    if a.out:
        calcola(a.out)


if __name__ == "__main__":
    main()
