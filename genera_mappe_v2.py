"""
MAPPE METEO LAZIO - PIÙ MODELLI E PIÙ PARAMETRI
===============================================

Modelli:   ICON-2I, AROME, ECMWF IFS, UKMO
Parametri: temperatura, pioggia, neve, vento, raffiche, nuvolosità, pressione,
           umidità, zero termico, CAPE (energia per i temporali)
Durata:    fino a 72 ore (o meno, se il modello si ferma prima)

Uso:   python genera_mappe_v2.py icon2i            (un modello)
       python genera_mappe_v2.py icon2i,ecmwf      (più modelli)
       python genera_mappe_v2.py tutti

Le mappe finiscono in  mappe/<modello>/<parametro>_hNN.png
(NN = ore dall'inizio della previsione), più un file indice.json per modello.

Principi (vedi anche le versioni precedenti):
- si chiede a Open-Meteo il valore vero della cella del modello (elevation=nan,
  cell_selection=nearest): niente "chiazze";
- il dettaglio del terreno della temperatura lo aggiungiamo noi con un DEM a ~500 m;
- le richieste rispettano i limiti gratuiti (600/minuto, 5000/ora);
- dove un modello non ha dati (es. AROME ai bordi del suo territorio) la mappa resta vuota.
"""

import os
import sys
import json
import time
import glob
from collections import deque
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import requests
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, distance_transform_edt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

# ----------------------------------------------------------------------------
# IMPOSTAZIONI
# ----------------------------------------------------------------------------
OUTPUT_DIR = "mappe"
CACHE_DIR = "dati"
DEM_CACHE = os.path.join(CACHE_DIR, "dem_lazio_500m.npz")

ORE_PREVISIONE = 72
EXTENT = [11.3, 14.1, 41.0, 42.8]          # lon_min, lon_max, lat_min, lat_max (area disegnata)
AREA_DATI = [11.2, 14.2, 40.92, 42.92]     # area in cui chiediamo i dati (un po' più larga)

PASSO_FINE = 0.005                          # griglia finale ~500 m
LAT_FINE = np.round(np.arange(EXTENT[2], EXTENT[3] + 1e-9, PASSO_FINE), 4)
LON_FINE = np.round(np.arange(EXTENT[0], EXTENT[1] + 1e-9, PASSO_FINE), 4)

# passo   = distanza tra i punti chiesti al modello (gradi)
# sigma   = quanto "lisciare" il terreno per imitare quello che vede il modello (celle da 500 m)
# ogni_ore= ogni quante ore disegnare una mappa
MODELLI = {
    "icon2i": dict(nome="ICON-2I", api="italia_meteo_arpae_icon_2i", ris="2,2 km",
                   passo=0.05, sigma=5, ogni_ore=1),
    "arome":  dict(nome="AROME", api="meteofrance_arome_france", ris="2,5 km",
                   passo=0.06, sigma=5, ogni_ore=1),
    "ecmwf":  dict(nome="ECMWF IFS", api="ecmwf_ifs", ris="9 km",
                   passo=0.10, sigma=15, ogni_ore=3),
    "ukmo":   dict(nome="UKMO", api="ukmo_global_deterministic_10km", ris="10 km",
                   passo=0.10, sigma=17, ogni_ore=3),
}

VARIABILI_API = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m",
                 "wind_direction_10m", "wind_gusts_10m", "cloud_cover", "pressure_msl",
                 "relative_humidity_2m", "freezing_level_height", "cape"]

# Limiti gratuiti Open-Meteo (con un po' di margine)
LIMITE_MINUTO = 540
LIMITE_ORA = 4700
PUNTI_PER_RICHIESTA = 100

GRADIENTE_MIN, GRADIENTE_MAX, GRADIENTE_STANDARD = -0.0098, 0.0040, -0.0065

CAPOLUOGHI = {
    "RM": (12.4964, 41.9028), "LT": (12.9043, 41.4676), "FR": (13.3441, 41.6390),
    "RI": (12.8634, 42.4036), "VT": (12.1081, 42.4204), "Nerola": (12.7856, 42.1606),
}
MESI = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
        "agosto", "settembre", "ottobre", "novembre", "dicembre"]
GIORNI = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
ROMA = ZoneInfo("Europe/Rome")


def log(msg):
    print(msg, flush=True)


def interpola(lat_src, lon_src, campo, lat_dst, lon_dst, metodo="linear"):
    try:
        f = RegularGridInterpolator((lat_src, lon_src), campo, method=metodo,
                                    bounds_error=False, fill_value=None)
    except ValueError:
        f = RegularGridInterpolator((lat_src, lon_src), campo, method="linear",
                                    bounds_error=False, fill_value=None)
    LA, LO = np.meshgrid(lat_dst, lon_dst, indexing="ij")
    return f((LA, LO)).astype(np.float32)


def riempi_buchi(campo):
    """I punti mancanti prendono il valore del punto valido più vicino."""
    buchi = ~np.isfinite(campo)
    if not buchi.any() or buchi.all():
        return campo
    idx = distance_transform_edt(buchi, return_distances=False, return_indices=True)
    return campo[tuple(idx)]


# ----------------------------------------------------------------------------
# 1. TERRENO (Copernicus DEM, mediato a ~500 m, scaricato una sola volta)
# ----------------------------------------------------------------------------
DEM_LAT0, DEM_LAT1, DEM_LON0, DEM_LON1 = 40, 43, 11, 15
DEM_URL = ("https://copernicus-dem-90m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM.tif")
DEM_BLOCCO, DEM_PIX = 6, 200


def scarica_dem():
    import rasterio
    from scipy.ndimage import zoom
    mosaico = np.zeros(((DEM_LAT1 - DEM_LAT0) * DEM_PIX, (DEM_LON1 - DEM_LON0) * DEM_PIX),
                       dtype=np.float32)
    trovate = 0
    for la in range(DEM_LAT0, DEM_LAT1):
        for lo in range(DEM_LON0, DEM_LON1):
            try:
                r = requests.get(DEM_URL.format(lat=la, lon=lo), timeout=120)
            except Exception as e:
                log(f"  DEM N{la} E{lo}: errore di rete ({e})")
                continue
            if r.status_code != 200:
                log(f"  DEM N{la} E{lo}: non disponibile (probabilmente solo mare)")
                continue
            with rasterio.MemoryFile(r.content) as mf, mf.open() as ds:
                z = ds.read(1).astype(np.float32)
                nod = ds.nodata
            if nod is not None:
                z[z == nod] = 0.0
            z[z < -100] = 0.0
            if z.shape != (1200, 1200):
                z = zoom(z, (1200 / z.shape[0], 1200 / z.shape[1]), order=1)[:1200, :1200]
            z = z.reshape(DEM_PIX, DEM_BLOCCO, DEM_PIX, DEM_BLOCCO).mean(axis=(1, 3))
            r0 = (DEM_LAT1 - 1 - la) * DEM_PIX
            c0 = (lo - DEM_LON0) * DEM_PIX
            mosaico[r0:r0 + DEM_PIX, c0:c0 + DEM_PIX] = z
            trovate += 1
    if trovate == 0:
        raise RuntimeError("nessuna piastrella DEM scaricata")
    lat = DEM_LAT1 - (np.arange(mosaico.shape[0]) + 0.5) / DEM_PIX
    lon = DEM_LON0 + (np.arange(mosaico.shape[1]) + 0.5) / DEM_PIX
    return lat[::-1].copy(), lon, mosaico[::-1].copy()


def carica_dem():
    if os.path.exists(DEM_CACHE):
        d = np.load(DEM_CACHE)
        return d["lat"], d["lon"], np.clip(d["z"], 0, None)
    log("DEM non presente: lo scarico (solo la prima volta)...")
    try:
        lat, lon, z = scarica_dem()
    except Exception as e:
        log(f"ATTENZIONE: DEM non disponibile ({e}). Mappe senza dettaglio del terreno.")
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(DEM_CACHE, lat=lat, lon=lon, z=z)
    return lat, lon, np.clip(z, 0, None)


# ----------------------------------------------------------------------------
# 2. DOWNLOAD DATI (con rispetto dei limiti gratuiti)
# ----------------------------------------------------------------------------
API = "https://api.open-meteo.com/v1/forecast"


class Limitatore:
    """Tiene il conto delle richieste fatte e aspetta quando ci si avvicina ai limiti."""

    def __init__(self):
        self.storico = deque()          # (istante, peso)

    def _somma(self, secondi):
        ora = time.time()
        return sum(p for t, p in self.storico if ora - t < secondi)

    def attendi(self, peso):
        while self.storico and time.time() - self.storico[0][0] > 3600:
            self.storico.popleft()
        avvisato = False
        while self._somma(60) + peso > LIMITE_MINUTO or self._somma(3600) + peso > LIMITE_ORA:
            if not avvisato and self._somma(3600) + peso > LIMITE_ORA:
                log("  limite orario vicino: attendo...")
                avvisato = True
            time.sleep(5)
        self.storico.append((time.time(), peso))


LIMITATORE = Limitatore()


def scarica_blocco(cfg, punti):
    lat_s = ",".join(f"{p[0]:.4f}" for p in punti)
    lon_s = ",".join(f"{p[1]:.4f}" for p in punti)
    ele_s = ",".join("nan" for _ in punti)
    url = (f"{API}?latitude={lat_s}&longitude={lon_s}&elevation={ele_s}"
           f"&hourly={','.join(VARIABILI_API)}&models={cfg['api']}"
           f"&forecast_hours={ORE_PREVISIONE}&past_hours=2"
           f"&cell_selection=nearest&timezone=GMT")
    peso = len(punti) * max(1.0, len(VARIABILI_API) / 10)
    for tentativo in range(6):
        LIMITATORE.attendi(peso)
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                d = r.json()
                return d if isinstance(d, list) else [d]
            if r.status_code == 429:
                log("  Open-Meteo dice: troppe richieste. Attendo 65 secondi...")
                time.sleep(65)
                continue
            log(f"  risposta {r.status_code}: {r.text[:300]}")
            if r.status_code == 400:
                return None
        except Exception as e:
            log(f"  errore di rete: {e}")
        time.sleep(10 * (tentativo + 1))
    return None


def griglia_modello(cfg):
    p = cfg["passo"]
    lat = np.round(np.arange(AREA_DATI[2], AREA_DATI[3] + 1e-9, p), 4)
    lon = np.round(np.arange(AREA_DATI[0], AREA_DATI[1] + 1e-9, p), 4)
    return lat, lon


def scarica_modello(cfg):
    lat, lon = griglia_modello(cfg)
    punti = [(la, lo, i, j) for i, la in enumerate(lat) for j, lo in enumerate(lon)]
    blocchi = [punti[k:k + PUNTI_PER_RICHIESTA] for k in range(0, len(punti), PUNTI_PER_RICHIESTA)]
    log(f"[{cfg['nome']}] scarico {len(punti)} punti in {len(blocchi)} richieste...")
    tempi, dati = None, {}
    for n, blocco in enumerate(blocchi):
        risp = scarica_blocco(cfg, blocco)
        if risp is None:
            log(f"  blocco {n + 1}/{len(blocchi)} perso")
            if n == 0:
                return None
            continue
        for (la, lo, i, j), item in zip(blocco, risp):
            h = item.get("hourly", {})
            if tempi is None and "time" in h:
                tempi = h["time"]
                dati = {v: np.full((len(tempi), len(lat), len(lon)), np.nan, np.float32)
                        for v in VARIABILI_API}
            if tempi is None:
                continue
            for v in VARIABILI_API:
                try:
                    arr = np.atleast_1d(np.array(h.get(v) or [], dtype=float))
                except (TypeError, ValueError):
                    continue
                k = min(len(tempi), len(arr))
                if k:
                    dati[v][:k, i, j] = arr[:k]
        if (n + 1) % 10 == 0 or n == len(blocchi) - 1:
            log(f"  {n + 1}/{len(blocchi)} richieste completate")
    if tempi is None:
        return None
    return dict(lat=lat, lon=lon, tempi=tempi, dati=dati)


# ----------------------------------------------------------------------------
# 3. ELABORAZIONE: dalla griglia del modello alla griglia fine da 500 m
# ----------------------------------------------------------------------------
def gradiente_termico(T_mod, z_mod):
    ok = np.isfinite(T_mod) & (z_mod > 30)
    if ok.sum() < 30 or np.ptp(z_mod[ok]) < 300:
        return GRADIENTE_STANDARD
    g = np.polyfit(z_mod[ok], T_mod[ok], 1)[0]
    return float(np.clip(g, GRADIENTE_MIN, GRADIENTE_MAX))


class Elaboratore:
    def __init__(self, cfg, mod, dem):
        self.lat, self.lon = mod["lat"], mod["lon"]
        T = mod["dati"]["temperature_2m"]
        # punti in cui il modello ha dati (per AROME: solo il suo territorio)
        copre = np.isfinite(T).any(axis=0).astype(np.float32)
        self.maschera = interpola(self.lat, self.lon, copre, LAT_FINE, LON_FINE) >= 0.5
        self.terreno = None
        if dem is not None:
            lat_d, lon_d, z_d = dem
            z_fine = interpola(lat_d, lon_d, z_d, LAT_FINE, LON_FINE)
            z_mod = interpola(lat_d, lon_d, gaussian_filter(z_d, cfg["sigma"]), self.lat, self.lon)
            z_mod_fine = interpola(self.lat, self.lon, z_mod, LAT_FINE, LON_FINE, "cubic")
            self.terreno = dict(z_fine=z_fine, z_mod=z_mod, diff=z_fine - z_mod_fine)

    def _fine(self, campo, metodo="cubic", minimo=None, massimo=None, liscia=0):
        f = interpola(self.lat, self.lon, riempi_buchi(campo), LAT_FINE, LON_FINE, metodo)
        if liscia:
            f = gaussian_filter(f, liscia)
        if minimo is not None or massimo is not None:
            f = np.clip(f, minimo, massimo)
        f[~self.maschera] = np.nan
        return f

    def temperatura(self, T):
        base = self._fine(T)
        if self.terreno is not None:
            g = gradiente_termico(T, self.terreno["z_mod"])
            base = base + g * self.terreno["diff"]
        return base

    def accumulo(self, A3):
        return self._fine(A3, "linear", minimo=0, liscia=1.5)

    def scalare(self, X, minimo=None, massimo=None):
        return self._fine(X, "cubic", minimo, massimo)

    def vento(self, vel, dirz):
        rad = np.radians(dirz)
        u = -vel * np.sin(rad)
        v = -vel * np.cos(rad)
        return (self._fine(vel, minimo=0), self._fine(u, "linear"), self._fine(v, "linear"))


# ----------------------------------------------------------------------------
# 4. STILI DELLE MAPPE
# ----------------------------------------------------------------------------
def cmap_da(colori, nome):
    return LinearSegmentedColormap.from_list(nome, colori)


STILI = {
    "temperatura": dict(titolo="Temperatura", unita="°C", cmap=plt.get_cmap("Spectral_r"),
                        extend="both"),
    "pioggia": dict(titolo="Pioggia", unita="mm caduti nelle 3 ore precedenti",
                    livelli=[0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150],
                    colori=["#c8e6ff", "#96cdff", "#5aaaf5", "#1e78dc", "#00b45a", "#5ad200",
                            "#c8e600", "#ffdc00", "#ffa000", "#ff3c00", "#c80078", "#800080"],
                    extend="max", ombra=True),
    "neve": dict(titolo="Neve fresca", unita="cm caduti nelle 3 ore precedenti",
                 livelli=[0.2, 0.5, 1, 2, 3, 5, 10, 15, 20, 30, 50],
                 colori=["#e8f4ff", "#c6e2ff", "#a0c8f0", "#78aee6", "#5590d2", "#6a6ad2",
                         "#8c55c8", "#aa3cb4", "#c8288c", "#dc1e5a"],
                 extend="max", ombra=True),
    "vento": dict(titolo="Vento a 10 m", unita="km/h (le frecce indicano la direzione)",
                  livelli=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110],
                  cmap=cmap_da(["#f7fbff", "#c6e9c2", "#7fd0a8", "#3fb0c8", "#2b7fd8",
                                "#7b5cd6", "#c53fb3", "#e8355f", "#f08a1f"], "vento"),
                  extend="max"),
    "raffiche": dict(titolo="Raffiche di vento", unita="km/h",
                     livelli=[20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140],
                     cmap=plt.get_cmap("YlOrRd"), extend="max", ombra=True),
    "nuvole": dict(titolo="Nuvolosità", unita="% di cielo coperto",
                   livelli=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                   cmap=cmap_da(["#eef2f6", "#c3cbd3", "#8e99a6", "#56606c"], "nuvole"),
                   extend="neither", ombra=True),
    "pressione": dict(titolo="Pressione al livello del mare", unita="hPa",
                      cmap=plt.get_cmap("RdBu_r"), extend="both"),
    "umidita": dict(titolo="Umidità relativa", unita="%",
                    livelli=list(range(0, 101, 10)), cmap=plt.get_cmap("BrBG"), extend="neither"),
    "zero_termico": dict(titolo="Zero termico", unita="metri di quota",
                         cmap=plt.get_cmap("Spectral_r"), extend="both"),
    "cape": dict(titolo="Energia per i temporali (CAPE)", unita="J/kg",
                 livelli=[50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000],
                 cmap=cmap_da(["#fff7bc", "#fec44f", "#f03b20", "#bd0026", "#7a0177"], "cape"),
                 extend="max", ombra=True),
}
ORDINE = ["temperatura", "pioggia", "neve", "vento", "raffiche", "nuvole", "pressione",
          "umidita", "zero_termico", "cape"]


def livelli_dinamici(campi, passo_base, maschera_terra=None, max_livelli=30):
    valori = np.concatenate([(c[maschera_terra] if maschera_terra is not None else c).ravel()
                             for c in campi])
    valori = valori[np.isfinite(valori)]
    if valori.size == 0:
        return None
    lo, hi = np.percentile(valori, 0.05), np.percentile(valori, 99.95)
    passo = passo_base
    while (hi - lo) / passo > max_livelli:
        passo *= 2
    lo = np.floor(lo / passo) * passo
    hi = np.ceil(hi / passo) * passo
    if hi - lo < 4 * passo:
        hi = lo + 4 * passo
    return np.arange(lo, hi + passo / 2, passo)


def testo_ora(t_iso):
    utc = datetime.fromisoformat(t_iso).replace(tzinfo=timezone.utc)
    loc = utc.astimezone(ROMA)
    return (f"{GIORNI[loc.weekday()]} {loc.day} {MESI[loc.month - 1]}, ore {loc:%H:%M}",
            f"{utc:%H} UTC")


# ----------------------------------------------------------------------------
# 5. DISEGNO
# ----------------------------------------------------------------------------
class Mappa:
    """Prepara una volta sola coste, confini e ombreggiatura del rilievo."""

    def __init__(self, dem):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cartopy.io.shapereader as shpreader
        self.ccrs, self.cfeature = ccrs, cfeature
        self.province = None
        try:
            shp = shpreader.natural_earth(resolution="10m", category="cultural",
                                          name="admin_1_states_provinces")
            self.province = [r.geometry for r in shpreader.Reader(shp).records()
                             if r.attributes.get("admin") == "Italy"]
        except Exception as e:
            log(f"Confini provinciali non disponibili: {e}")
        self.geo = None
        try:
            from shapely.geometry import box
            area = box(EXTENT[0] - 0.2, EXTENT[2] - 0.2, EXTENT[1] + 0.2, EXTENT[3] + 0.2)
            ext = [EXTENT[0] - 0.2, EXTENT[1] + 0.2, EXTENT[2] - 0.2, EXTENT[3] + 0.2]

            def ritaglia(geoms):
                out = []
                for g in geoms:
                    if g.intersects(area):
                        g = g.intersection(area)
                        if not g.is_empty:
                            out.append(g)
                return out

            self.geo = dict(
                mare=ritaglia(cfeature.OCEAN.with_scale("10m").intersecting_geometries(ext)),
                laghi=ritaglia(cfeature.LAKES.with_scale("10m").intersecting_geometries(ext)),
                coste=ritaglia(cfeature.COASTLINE.with_scale("10m").intersecting_geometries(ext)),
                province=ritaglia(self.province or []))
        except Exception as e:
            log(f"Ritaglio coste non riuscito ({e}): uso il metodo normale, più lento.")
        self.ombra = None
        if dem is not None:
            lat_d, lon_d, z_d = dem
            z = interpola(lat_d, lon_d, z_d, LAT_FINE, LON_FINE)
            dy = PASSO_FINE * 111000.0
            dx = dy * np.cos(np.radians(LAT_FINE.mean()))
            gy, gx = np.gradient(z, dy, dx)
            pend = np.arctan(np.hypot(gx, gy) * 1.5)
            espo = np.arctan2(-gx, gy)
            alt, az = np.radians(45), np.radians(315)
            luce = np.sin(alt) * np.cos(pend) + np.cos(alt) * np.sin(pend) * np.cos(az - espo)
            self.ombra = np.clip(luce / np.sin(alt), 0, 1)

    def disegna(self, file, campo, stile, livelli, titolo_modello, data, scadenza, info,
                frecce=None, isobare=False):
        pc = self.ccrs.PlateCarree()
        fig = plt.figure(figsize=(10, 7.6), facecolor="white")
        ax = plt.axes(projection=pc)
        ax.set_extent(EXTENT, crs=pc)
        ax.set_facecolor("#f2f2f2")         # grigio chiaro = zona senza dati del modello
        ombra = stile.get("ombra", False)

        # fondo bianco solo dove il modello ha dati
        dentro = np.where(np.isfinite(campo), 1.0, np.nan)
        ax.contourf(LON_FINE, LAT_FINE, dentro, levels=[0.5, 1.5], colors=["white"],
                    transform=pc, zorder=0)
        if ombra and self.ombra is not None:
            o = np.where(np.isfinite(campo), self.ombra, np.nan)
            ax.imshow(o, extent=[LON_FINE[0], LON_FINE[-1], LAT_FINE[0], LAT_FINE[-1]],
                      origin="lower", cmap="gray", vmin=0.2, vmax=1.0, alpha=0.5,
                      transform=pc, zorder=0.5)

        if "colori" in stile:
            cmap = ListedColormap(stile["colori"])
            norm = BoundaryNorm(livelli, cmap.N)
        else:
            cmap, norm = stile["cmap"], None
        mesh = ax.contourf(LON_FINE, LAT_FINE, campo, levels=livelli, cmap=cmap, norm=norm,
                           extend=stile["extend"], alpha=0.85 if ombra else 0.95,
                           transform=pc, zorder=1)

        if isobare:
            cs = ax.contour(LON_FINE, LAT_FINE, campo, levels=livelli, colors="#222222",
                            linewidths=0.6, transform=pc, zorder=4)
            ax.clabel(cs, cs.levels[::2], fmt="%d", fontsize=7, inline=True)

        if frecce is not None:
            u, v = frecce
            k = 24                               # una freccia ogni ~12 km
            uu, vv = u[k // 2::k, k // 2::k], v[k // 2::k, k // 2::k]
            mod = np.hypot(uu, vv)
            mod[mod < 0.1] = np.nan
            ax.quiver(LON_FINE[k // 2::k], LAT_FINE[k // 2::k], uu / mod, vv / mod,
                      transform=pc, zorder=4, color="#1a1a1a", alpha=0.75, scale=38,
                      width=0.0022, headwidth=4, headlength=4.5, headaxislength=4)

        if self.geo is not None:
            ax.add_geometries(self.geo["mare"], pc, facecolor="#dceefc", edgecolor="none", zorder=2)
            ax.add_geometries(self.geo["laghi"], pc, facecolor="#dceefc", edgecolor="#5a7a99",
                              linewidth=0.4, zorder=2)
            ax.add_geometries(self.geo["coste"], pc, facecolor="none", edgecolor="#333333",
                              linewidth=0.8, zorder=3)
            ax.add_geometries(self.geo["province"], pc, facecolor="none", edgecolor="#555555",
                              linewidth=0.45, zorder=3)
        else:
            cf = self.cfeature
            ax.add_feature(cf.OCEAN.with_scale("10m"), facecolor="#dceefc", zorder=2)
            ax.add_feature(cf.LAKES.with_scale("10m"), facecolor="#dceefc", edgecolor="#5a7a99",
                           linewidth=0.4, zorder=2)
            ax.add_feature(cf.COASTLINE.with_scale("10m"), linewidth=0.8, edgecolor="#333333",
                           zorder=3)
            if self.province:
                ax.add_geometries(self.province, pc, edgecolor="#555555", facecolor="none",
                                  linewidth=0.45, zorder=3)

        for sigla, (lo, la) in CAPOLUOGHI.items():
            ax.plot(lo, la, marker="o", color="black", markersize=3.5, transform=pc, zorder=5)
            ax.text(lo + 0.03, la, sigla, transform=pc, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white", alpha=0.85,
                              edgecolor="#cccccc"), zorder=6)

        cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.03, shrink=0.85,
                            aspect=28, extend=stile["extend"])
        cbar.set_label(stile["unita"], fontsize=9, fontweight="bold")
        cbar.ax.tick_params(labelsize=8)

        # titolo: parametro a sinistra, modello a destra; sotto data e scadenza
        ax.set_title(stile["titolo"], loc="left", fontsize=15, fontweight="bold", pad=24)
        ax.set_title(titolo_modello, loc="right", fontsize=12, fontweight="bold",
                     color="#1f4e79", pad=24)
        ax.text(0, 1.012, data, transform=ax.transAxes, fontsize=11, color="#333333",
                ha="left", va="bottom")
        ax.text(1, 1.012, scadenza, transform=ax.transAxes, fontsize=11, fontweight="bold",
                color="#555555", ha="right", va="bottom")
        # dettagli tecnici: in piccolo, in basso a sinistra dentro la mappa
        ax.text(0.012, 0.022, info, transform=ax.transAxes, fontsize=6.5, color="#555555",
                ha="left", va="bottom", zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"))
        ax.text(0.985, 0.022, "www.meteonerola.it", transform=ax.transAxes, fontsize=8,
                fontweight="bold", ha="right", va="bottom", zorder=10,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9,
                          edgecolor="#aaaaaa"))

        fig.savefig(file, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)


# I processi paralleli leggono questi dati (copiati alla partenza, non ritrasmessi)
_LAVORO = {}


def _disegna_uno(compito):
    try:
        _LAVORO["mappa"].disegna(**_LAVORO["compiti"][compito])
        return None
    except Exception as e:
        return f"{compito}: {e}"


# ----------------------------------------------------------------------------
# 6. UN MODELLO DALL'INIZIO ALLA FINE
# ----------------------------------------------------------------------------
def elabora_modello(chiave, dem, mappa):
    cfg = MODELLI[chiave]
    inizio = time.time()
    mod = scarica_modello(cfg)
    if mod is None:
        log(f"[{cfg['nome']}] nessun dato ricevuto: salto questo modello.")
        return False
    D, tempi = mod["dati"], mod["tempi"]
    T = D["temperature_2m"]
    copertura = np.isfinite(T).mean(axis=(1, 2))
    if copertura.max() == 0:
        log(f"[{cfg['nome']}] nessun dato di temperatura: salto.")
        return False
    # ore valide: dalla terza in poi (le prime 2 servono solo per gli accumuli su 3 ore)
    ore = [k for k in range(2, len(tempi)) if copertura[k] >= 0.9 * copertura.max()]
    ore = [k for n, k in enumerate(ore) if (k - 2) % cfg["ogni_ore"] == 0]
    log(f"[{cfg['nome']}] {len(tempi) - 2} ore ricevute, copertura area {copertura.max() * 100:.0f}%, "
        f"{len(ore)} istanti da disegnare")

    el = Elaboratore(cfg, mod, dem)
    disponibili = {v: np.isfinite(D[v][ore]).any() for v in VARIABILI_API}
    mancanti = [v for v, ok in disponibili.items() if not ok]
    if mancanti:
        log(f"[{cfg['nome']}] parametri non forniti da questo modello: {', '.join(mancanti)}")

    def acc3(arr, k):
        blocco = arr[k - 2:k + 1]
        s = np.nansum(blocco, axis=0)
        s[~np.isfinite(blocco).any(axis=0)] = np.nan
        return s

    campi = {p: {} for p in ORDINE}
    frecce = {}
    for k in ore:
        if disponibili["temperature_2m"]:
            campi["temperatura"][k] = el.temperatura(T[k])
        if disponibili["precipitation"]:
            campi["pioggia"][k] = el.accumulo(acc3(D["precipitation"], k))
        if disponibili["snowfall"]:
            campi["neve"][k] = el.accumulo(acc3(D["snowfall"], k))
        if disponibili["wind_speed_10m"] and disponibili["wind_direction_10m"]:
            vel, u, v = el.vento(D["wind_speed_10m"][k], D["wind_direction_10m"][k])
            campi["vento"][k] = vel
            frecce[k] = (u, v)
        if disponibili["wind_gusts_10m"]:
            campi["raffiche"][k] = el.scalare(D["wind_gusts_10m"][k], minimo=0)
        if disponibili["cloud_cover"]:
            campi["nuvole"][k] = el.scalare(D["cloud_cover"][k], 0, 100)
        if disponibili["pressure_msl"]:
            # la pressione ridotta al livello del mare è "rumorosa" sui rilievi:
            # la lisciamo (~8 km) per avere isobare pulite, come nelle carte sinottiche
            pr = el.scalare(D["pressure_msl"][k])
            dentro = np.isfinite(pr)
            pr = gaussian_filter(np.where(dentro, pr, np.nanmean(pr)), 16)
            pr[~dentro] = np.nan
            campi["pressione"][k] = pr
        if disponibili["relative_humidity_2m"]:
            campi["umidita"][k] = el.scalare(D["relative_humidity_2m"][k], 0, 100)
        if disponibili["freezing_level_height"]:
            campi["zero_termico"][k] = el.scalare(D["freezing_level_height"][k], minimo=0)
        if disponibili["cape"]:
            campi["cape"][k] = el.scalare(D["cape"][k], minimo=0)

    terra = (el.terreno["z_fine"] > 1) if el.terreno is not None else None
    livelli = {}
    for p in ORDINE:
        if not campi[p]:
            continue
        if p == "temperatura":
            livelli[p] = livelli_dinamici(campi[p].values(), 1, terra)
        elif p == "pressione":
            livelli[p] = livelli_dinamici(campi[p].values(), 1)
        elif p == "zero_termico":
            livelli[p] = livelli_dinamici(campi[p].values(), 100)
        else:
            livelli[p] = STILI[p]["livelli"]

    cartella = os.path.join(OUTPUT_DIR, chiave)
    os.makedirs(cartella, exist_ok=True)
    for f in glob.glob(os.path.join(cartella, "*")):
        os.remove(f)

    compiti, indice = {}, {p: [] for p in ORDINE}
    for n, k in enumerate(ore):
        scad = k - 2
        data, ora_utc = testo_ora(tempi[k])
        base_info = f"Modello {cfg['nome']} ({cfg['ris']}) · ora italiana ({ora_utc})"
        for p in ORDINE:
            if k not in campi[p] or livelli.get(p) is None:
                continue
            nome_file = f"{p}_h{scad:02d}.png"
            info = base_info
            if p == "temperatura" and el.terreno is not None:
                info = f"Modello {cfg['nome']} ({cfg['ris']}) · dettaglio del terreno a 500 m · ora italiana ({ora_utc})"
            compiti[f"{chiave}/{nome_file}"] = dict(
                file=os.path.join(cartella, nome_file), campo=campi[p][k], stile=STILI[p],
                livelli=livelli[p], titolo_modello=cfg["nome"], data=data,
                scadenza=f"+{scad}h", info=info,
                frecce=frecce.get(k) if p == "vento" else None, isobare=(p == "pressione"))
            indice[p].append(dict(file=nome_file, scadenza=scad, valido_utc=tempi[k] + "Z"))

    log(f"[{cfg['nome']}] disegno {len(compiti)} mappe...")
    t0 = time.time()
    _LAVORO["mappa"], _LAVORO["compiti"] = mappa, compiti
    errori = []
    nproc = max(1, min(4, os.cpu_count() or 1))
    try:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx) as ex:
            for e in ex.map(_disegna_uno, list(compiti), chunksize=4):
                if e:
                    errori.append(e)
    except Exception as e:
        log(f"  disegno in parallelo non riuscito ({e}), procedo uno alla volta")
        errori = [e for e in map(_disegna_uno, compiti) if e]
    for e in errori[:5]:
        log(f"  errore: {e}")
    log(f"[{cfg['nome']}] mappe disegnate in {time.time() - t0:.0f} s "
        f"(totale modello {time.time() - inizio:.0f} s), errori: {len(errori)}")

    with open(os.path.join(cartella, "indice.json"), "w") as f:
        json.dump(dict(modello=cfg["nome"], risoluzione=cfg["ris"],
                       aggiornato_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                       mappe={p: v for p, v in indice.items() if v}), f, indent=1)
    return True


def main():
    scelta = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODELLI", "tutti")).strip()
    chiavi = list(MODELLI) if scelta in ("", "tutti") else [c.strip() for c in scelta.split(",")]
    sconosciuti = [c for c in chiavi if c not in MODELLI]
    if sconosciuti:
        raise SystemExit(f"Modelli sconosciuti: {sconosciuti}. Disponibili: {list(MODELLI)}")
    log(f"=== MAPPE METEO LAZIO - modelli: {', '.join(MODELLI[c]['nome'] for c in chiavi)} ===")
    dem = carica_dem()
    mappa = Mappa(dem)
    riusciti = [c for c in chiavi if elabora_modello(c, dem, mappa)]
    log(f"=== Fine. Modelli completati: {riusciti or 'nessuno'} ===")
    if not riusciti:
        sys.exit(1)


if __name__ == "__main__":
    main()
