"""
Prepara la climatologia 1991-2020 (medie mensili) dalla rianalisi NCEP/NCAR R1 (NOAA PSL, dominio pubblico),
usata per le mappe di anomalia e per il modello stagionale.

Uso: python -m motore.prepara_clima dati/clima_ncep_1991_2020.npz
Scarica solo il riquadro euro-atlantico, tramite OPeNDAP (pochi MB).
"""
import sys
import numpy as np
import xarray as xr

URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/"
CAMPI = {
    # nome: (file, variabile, livello, conversione)
    "t850": ("pressure/air.mon.mean.nc", "air", 850, lambda x, u: x - 273.15 if "K" in u else x),
    "z500": ("pressure/hgt.mon.mean.nc", "hgt", 500, lambda x, u: x),
    "mslp": ("surface/slp.mon.mean.nc", "slp", None, lambda x, u: x / 100.0 if "Pa" in u and "milli" not in u.lower() and "hPa" not in u else x),
    "t2m": ("surface_gauss/air.2m.mon.mean.nc", "air", None, lambda x, u: x - 273.15 if "K" in u else x),
    "pioggia": ("surface_gauss/prate.sfc.mon.mean.nc", "prate", None, lambda x, u: x * 86400.0),   # mm/giorno
}
LAT = (15.0, 85.0)
LON = (-70.0, 70.0)


def leggi(nome):
    f, var, liv, conv = CAMPI[nome]
    ds = xr.open_dataset(URL + f)
    da = ds[var]
    if liv is not None:
        da = da.sel(level=liv)
    da = da.sel(time=slice("1991-01-01", "2020-12-31"))
    da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon").sortby("lat")
    da = da.sel(lat=slice(*LAT), lon=slice(*LON))
    u = str(da.attrs.get("units", ""))
    clim = da.groupby("time.month").mean("time").values
    dev = da.groupby("time.month").std("time").values
    print(nome, da.shape, "unita'", u, flush=True)
    return conv(clim, u), (dev * 86400.0 if nome == "pioggia" else dev), da.lat.values, da.lon.values


def main(out):
    salva = {}
    for n in CAMPI:
        c, s, la, lo = leggi(n)
        salva[n] = c.astype("float32")
        salva[n + "_dev"] = s.astype("float32")
        salva[n + "_lat"] = la.astype("float32")
        salva[n + "_lon"] = lo.astype("float32")
        print(f"  {n}: gennaio media {np.nanmean(c[0]):.2f}, luglio {np.nanmean(c[6]):.2f}")
    np.savez_compressed(out, **salva)


if __name__ == "__main__":
    main(sys.argv[1])
