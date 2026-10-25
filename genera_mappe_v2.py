"""
MAPPE ICON-2I LAZIO - VERSIONE 2
================================

Cosa cambia rispetto alle versioni precedenti (genera_mappe.py e genera_mappe_native.py):

1. DATI GREZZI DEL MODELLO
   Prima Open-Meteo "correggeva" la temperatura di ogni punto sulla quota esatta
   (DEM a 90 m) di quel singolo punto. Campionando un punto ogni 3-4 km, alcuni punti
   cadevano in fondo a una valle e altri su un crinale: ne uscivano le "chiazze"
   rotonde. Ora chiediamo il valore vero della cella del modello
   (elevation=nan + cell_selection=nearest): campo liscio e coerente.

2. DETTAGLIO DEL TERRITORIO FATTO BENE
   Il dettaglio di valli e rilievi lo aggiungiamo noi, su TUTTA la mappa e non punto
   per punto, usando un modello digitale del terreno (Copernicus DEM) mediato a ~500 m
   e un gradiente termico verticale calcolato ora per ora dal modello stesso
   (così di notte le inversioni termiche vengono rispettate).

3. MENO RICHIESTE ALL'API
   Le coordinate vengono inviate a gruppi di 100 e con le pause giuste, per non
   superare i limiti gratuiti di Open-Meteo (che prima venivano superati: i punti
   persi venivano riempiti con la media, altra fonte di chiazze).

4. SCALA COLORI STABILE
   La scala della temperatura è la stessa per tutte le ore di una corsa, così
   scorrendo le mappe i colori sono confrontabili.
"""

import os
import io
import sys
import time
import glob
import tempfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import requests
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, distance_transform_edt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ----------------------------------------------------------------------------
# IMPOSTAZIONI (si possono cambiare qui senza toccare il resto)
# ----------------------------------------------------------------------------
OUTPUT_DIR = "mappe_v2"
CACHE_DIR = "dati"
DEM_CACHE = os.path.join(CACHE_DIR, "dem_lazio_500m.npz")

MODELLO = "italia_meteo_arpae_icon_2i"
ORE_PREVISIONE = 72

# Area disegnata sulla mappa
EXTENT = [11.3, 14.1, 41.0, 42.8]          # lon_min, lon_max, lat_min, lat_max

# Punti in cui chiediamo i dati al modello (un po' più larghi dell'area disegnata)
PASSO_MODELLO = 0.04                        # gradi, circa 4 km
LAT_MOD = np.round(np.arange(40.92, 42.92 + 1e-9, PASSO_MODELLO), 4)
LON_MOD = np.round(np.arange(11.20, 14.20 + 1e-9, PASSO_MODELLO), 4)

# Griglia fine della mappa finale
PASSO_FINE = 0.005                          # gradi, circa 500 m
LAT_FINE = np.round(np.arange(EXTENT[2], EXTENT[3] + 1e-9, PASSO_FINE), 4)
LON_FINE = np.round(np.arange(EXTENT[0], EXTENT[1] + 1e-9, PASSO_FINE), 4)

# Limiti del gradiente termico verticale (°C per metro)
GRADIENTE_MIN = -0.0098                     # massimo raffreddamento con la quota
GRADIENTE_MAX = +0.0040                     # inversione termica moderata
GRADIENTE_STANDARD = -0.0065

PUNTI_PER_RICHIESTA = 100
PAUSA_TRA_RICHIESTE = 11                    # secondi (limite: 600 punti al minuto)

CAPOLUOGHI = {
    "RM": (12.4964, 41.9028),
    "LT": (12.9043, 41.4676),
    "FR": (13.3441, 41.6390),
    "RI": (12.8634, 42.4036),
    "VT": (12.1081, 42.4204),
    "Nerola": (12.7856, 42.1606),
}

MESI = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
        "agosto", "settembre", "ottobre", "novembre", "dicembre"]
GIORNI = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
ROMA = ZoneInfo("Europe/Rome")


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# 1. MODELLO DIGITALE DEL TERRENO (DEM)
# ----------------------------------------------------------------------------
# Copernicus DEM GLO-90: piastrelle da 1 grado, 1200 x 1200 pixel (3 secondi d'arco).
# Le mediamo a blocchi di 6x6 pixel -> 0.005 gradi (~500 m) e salviamo il risultato
# nella cartella "dati", così il download avviene una sola volta.
DEM_LAT0, DEM_LAT1 = 40, 43                 # piastrelle N40, N41, N42
DEM_LON0, DEM_LON1 = 11, 15                 # piastrelle E011 ... E014
DEM_URL = ("https://copernicus-dem-90m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM.tif")
DEM_BLOCCO = 6
DEM_PIX = 1200 // DEM_BLOCCO                 # 200 celle per grado


def scarica_dem():
    import rasterio
    from scipy.ndimage import zoom

    n_lat = DEM_LAT1 - DEM_LAT0
    n_lon = DEM_LON1 - DEM_LON0
    mosaico = np.zeros((n_lat * DEM_PIX, n_lon * DEM_PIX), dtype=np.float32)
    trovate = 0
    for la in range(DEM_LAT0, DEM_LAT1):
        for lo in range(DEM_LON0, DEM_LON1):
            url = DEM_URL.format(lat=la, lon=lo)
            try:
                r = requests.get(url, timeout=120)
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
            # le righe del mosaico vanno da nord a sud
            r0 = (DEM_LAT1 - 1 - la) * DEM_PIX
            c0 = (lo - DEM_LON0) * DEM_PIX
            mosaico[r0:r0 + DEM_PIX, c0:c0 + DEM_PIX] = z
            trovate += 1
            log(f"  DEM N{la} E{lo}: ok")
    if trovate == 0:
        raise RuntimeError("nessuna piastrella DEM scaricata")
    lat = DEM_LAT1 - (np.arange(mosaico.shape[0]) + 0.5) / DEM_PIX
    lon = DEM_LON0 + (np.arange(mosaico.shape[1]) + 0.5) / DEM_PIX
    return lat[::-1].copy(), lon, mosaico[::-1].copy()   # lat crescente


def carica_dem():
    """Restituisce (lat, lon, quota) del DEM a ~500 m, oppure None se non disponibile."""
    if os.path.exists(DEM_CACHE):
        d = np.load(DEM_CACHE)
        log("DEM caricato dalla cartella dati.")
        return d["lat"], d["lon"], d["z"]
    log("DEM non presente: lo scarico (solo la prima volta)...")
    try:
        lat, lon, z = scarica_dem()
    except Exception as e:
        log(f"ATTENZIONE: DEM non disponibile ({e}). Mappe senza dettaglio del terreno.")
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(DEM_CACHE, lat=lat, lon=lon, z=z)
    log("DEM salvato.")
    return lat, lon, z


def interpola(lat_src, lon_src, campo, lat_dst, lon_dst, metodo="linear"):
    """Interpola un campo da una griglia regolare a un'altra."""
    try:
        f = RegularGridInterpolator((lat_src, lon_src), campo, method=metodo,
                                    bounds_error=False, fill_value=None)
    except ValueError:
        f = RegularGridInterpolator((lat_src, lon_src), campo, method="linear",
                                    bounds_error=False, fill_value=None)
    LA, LO = np.meshgrid(lat_dst, lon_dst, indexing="ij")
    return f((LA, LO))


# ----------------------------------------------------------------------------
# 2. DOWNLOAD DEI DATI DEL MODELLO
# ----------------------------------------------------------------------------
API = "https://api.open-meteo.com/v1/forecast"


def scarica_blocco(punti):
    lat_s = ",".join(f"{p[0]:.4f}" for p in punti)
    lon_s = ",".join(f"{p[1]:.4f}" for p in punti)
    # elevation=nan disattiva la correzione punto per punto (causa delle chiazze);
    # va ripetuto una volta per ogni coordinata
    ele_s = ",".join("nan" for _ in punti)
    url = (f"{API}?latitude={lat_s}&longitude={lon_s}"
           f"&hourly=temperature_2m,precipitation&models={MODELLO}"
           f"&forecast_hours={ORE_PREVISIONE}&past_hours=2"
           f"&elevation={ele_s}&cell_selection=nearest&timezone=GMT")
    for tentativo in range(6):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 200:
                dati = r.json()
                return dati if isinstance(dati, list) else [dati]
            if r.status_code == 429:
                log("  limite richieste raggiunto, attendo 65 secondi...")
                time.sleep(65)
                continue
            log(f"  risposta {r.status_code}: {r.text[:200]}")
        except Exception as e:
            log(f"  errore di rete: {e}")
        time.sleep(10 * (tentativo + 1))
    return None


def scarica_modello():
    punti = [(la, lo, i, j) for i, la in enumerate(LAT_MOD) for j, lo in enumerate(LON_MOD)]
    blocchi = [punti[k:k + PUNTI_PER_RICHIESTA] for k in range(0, len(punti), PUNTI_PER_RICHIESTA)]
    log(f"Scarico {len(punti)} punti del modello in {len(blocchi)} richieste "
        f"(circa {len(blocchi) * PAUSA_TRA_RICHIESTE // 60 + 1} minuti)...")

    T = P = None
    tempi = None
    for n, blocco in enumerate(blocchi):
        risposta = scarica_blocco(blocco)
        if risposta is None:
            log(f"  blocco {n + 1}/{len(blocchi)} perso")
        else:
            for (la, lo, i, j), item in zip(blocco, risposta):
                h = item.get("hourly", {})
                if tempi is None and "time" in h:
                    tempi = h["time"]
                    T = np.full((len(tempi), len(LAT_MOD), len(LON_MOD)), np.nan, np.float32)
                    P = np.full_like(T, np.nan)
                if tempi is None:
                    continue
                t = np.array(h.get("temperature_2m", []), dtype=float)
                p = np.array(h.get("precipitation", []), dtype=float)
                k = min(len(tempi), len(t))
                T[:k, i, j] = t[:k]
                k = min(len(tempi), len(p))
                P[:k, i, j] = p[:k]
            log(f"  blocco {n + 1}/{len(blocchi)} ok")
        if n < len(blocchi) - 1:
            time.sleep(PAUSA_TRA_RICHIESTE)
    if tempi is None:
        raise RuntimeError("Nessun dato ricevuto dal modello.")
    return tempi, T, P


def riempi_buchi(campo):
    """I punti mancanti prendono il valore del punto valido più vicino (non la media)."""
    buchi = np.isnan(campo)
    if not buchi.any() or buchi.all():
        return campo
    idx = distance_transform_edt(buchi, return_distances=False, return_indices=True)
    return campo[tuple(idx)]


# ----------------------------------------------------------------------------
# 3. ELABORAZIONE
# ----------------------------------------------------------------------------
def prepara_terreno(dem):
    """Quote necessarie al downscaling della temperatura."""
    if dem is None:
        return None
    lat_d, lon_d, z_d = dem
    z_d = np.clip(z_d, 0, None)
    # quota "fine" della mappa (~500 m)
    z_fine = interpola(lat_d, lon_d, z_d, LAT_FINE, LON_FINE)
    # quota "vista dal modello": terreno lisciato alla scala del modello (~2-3 km)
    z_liscio = gaussian_filter(z_d, sigma=5)
    z_mod = interpola(lat_d, lon_d, z_liscio, LAT_MOD, LON_MOD)
    # la stessa quota liscia riportata sulla griglia fine con la stessa interpolazione della temperatura
    z_mod_fine = interpola(LAT_MOD, LON_MOD, z_mod, LAT_FINE, LON_FINE, "cubic")
    # rilievo per l'ombreggiatura
    return {"z_fine": z_fine, "z_mod": z_mod, "diff": z_fine - z_mod_fine}


def gradiente_termico(T_mod, z_mod):
    """Gradiente verticale della temperatura (°C/m) stimato dal modello per questa ora."""
    terra = z_mod > 30
    if terra.sum() < 50:
        return GRADIENTE_STANDARD
    z = z_mod[terra]
    t = T_mod[terra]
    ok = np.isfinite(t)
    if ok.sum() < 50 or np.ptp(z[ok]) < 300:
        return GRADIENTE_STANDARD
    g = np.polyfit(z[ok], t[ok], 1)[0]
    return float(np.clip(g, GRADIENTE_MIN, GRADIENTE_MAX))


def temperatura_fine(T_mod, terreno):
    T_mod = riempi_buchi(T_mod)
    base = interpola(LAT_MOD, LON_MOD, T_mod, LAT_FINE, LON_FINE, "cubic")
    if terreno is None:
        return base, None
    g = gradiente_termico(T_mod, terreno["z_mod"])
    return base + g * terreno["diff"], g


def pioggia_fine(P_mod):
    P_mod = np.nan_to_num(riempi_buchi(P_mod), nan=0.0)
    campo = interpola(LAT_MOD, LON_MOD, P_mod, LAT_FINE, LON_FINE, "linear")
    campo = gaussian_filter(campo, sigma=1.5)
    return np.clip(campo, 0, None)


# ----------------------------------------------------------------------------
# 4. GRAFICA
# ----------------------------------------------------------------------------
PIOGGIA_LIVELLI = [0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150]
PIOGGIA_COLORI = ["#c8e6ff", "#96cdff", "#5aaaf5", "#1e78dc", "#00b45a", "#5ad200",
                  "#c8e600", "#ffdc00", "#ffa000", "#ff3c00", "#c80078", "#800080"]


def testo_ora(t_iso):
    utc = datetime.fromisoformat(t_iso).replace(tzinfo=timezone.utc)
    loc = utc.astimezone(ROMA)
    return (f"{GIORNI[loc.weekday()]} {loc.day} {MESI[loc.month - 1]} {loc.year} - "
            f"ore {loc:%H:%M} ora italiana ({utc:%H} UTC)")


class Mappa:
    """Prepara una volta sola gli elementi fissi (confini, ombreggiatura)."""

    def __init__(self, terreno):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cartopy.io.shapereader as shpreader
        self.ccrs = ccrs
        self.cfeature = cfeature
        self.terreno = terreno
        self.province = None
        try:
            shp = shpreader.natural_earth(resolution="10m", category="cultural",
                                          name="admin_1_states_provinces")
            self.province = [r.geometry for r in shpreader.Reader(shp).records()
                             if r.attributes.get("admin") == "Italy"]
        except Exception as e:
            log(f"Confini provinciali non disponibili: {e}")
        self.ombra = None
        if terreno is not None:
            # ombreggiatura del rilievo: 1 = pianura (bianco), valori minori = versanti in ombra
            z = terreno["z_fine"]
            dy = PASSO_FINE * 111000.0
            dx = dy * np.cos(np.radians(LAT_FINE.mean()))
            gy, gx = np.gradient(z, dy, dx)
            pend = np.arctan(np.hypot(gx, gy) * 1.5)
            espo = np.arctan2(-gx, gy)
            alt, az = np.radians(45), np.radians(315)
            luce = (np.sin(alt) * np.cos(pend)
                    + np.cos(alt) * np.sin(pend) * np.cos(az - espo))
            self.ombra = np.clip(luce / np.sin(alt), 0, 1)

    def disegna(self, campo, levels, cmap, norm, extend, titolo, sottotitolo,
                etichetta, file, ombra=False, nota=None):
        ccrs = self.ccrs
        pc = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10, 7.6), facecolor="white")
        ax = plt.axes(projection=pc)
        ax.set_extent(EXTENT, crs=pc)
        ax.set_facecolor("white")
        img_ext = [LON_FINE[0], LON_FINE[-1], LAT_FINE[0], LAT_FINE[-1]]

        if ombra and self.ombra is not None:
            ax.imshow(self.ombra, extent=img_ext, origin="lower", cmap="gray",
                      vmin=0.2, vmax=1.0, alpha=0.5, transform=pc, zorder=0)

        mesh = ax.contourf(LON_FINE, LAT_FINE, campo, levels=levels, cmap=cmap, norm=norm,
                           extend=extend, alpha=0.85 if ombra else 0.95,
                           transform=pc, zorder=1, antialiased=True)

        ax.add_feature(self.cfeature.OCEAN.with_scale("10m"), facecolor="#dceefc", zorder=2)
        ax.add_feature(self.cfeature.LAKES.with_scale("10m"), facecolor="#dceefc",
                       edgecolor="#5a7a99", linewidth=0.4, zorder=2)
        ax.add_feature(self.cfeature.COASTLINE.with_scale("10m"), linewidth=0.8,
                       edgecolor="#333333", zorder=3)
        if self.province:
            ax.add_geometries(self.province, pc, edgecolor="#555555", facecolor="none",
                              linewidth=0.45, zorder=3)

        for sigla, (lo, la) in CAPOLUOGHI.items():
            ax.plot(lo, la, marker="o", color="black", markersize=3.5, transform=pc, zorder=5)
            ax.text(lo + 0.03, la, sigla, transform=pc, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="white", alpha=0.85,
                              edgecolor="#cccccc"), zorder=6)

        cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.03, shrink=0.85,
                            aspect=28, extend=extend)
        cbar.set_label(etichetta, fontsize=9, fontweight="bold")
        cbar.ax.tick_params(labelsize=8)

        ax.set_title(f"{titolo}\n{sottotitolo}", fontsize=10.5, fontweight="bold", loc="left")
        ax.text(0.985, 0.025, "www.meteonerola.it", transform=ax.transAxes, fontsize=8,
                fontweight="bold", ha="right", va="bottom", zorder=10,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9,
                          edgecolor="#aaaaaa"))
        if nota:
            ax.text(0.015, 0.025, nota, transform=ax.transAxes, fontsize=6.5, color="#444444",
                    ha="left", va="bottom", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8,
                              edgecolor="none"))

        fig.savefig(file, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)


# ----------------------------------------------------------------------------
# 5. PROGRAMMA PRINCIPALE
# ----------------------------------------------------------------------------
def main():
    log("=== MAPPE ICON-2I LAZIO - VERSIONE 2 ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dem = carica_dem()
    terreno = prepara_terreno(dem)

    tempi, T, P = scarica_modello()
    validi = np.isfinite(T).mean(axis=(1, 2))
    log(f"Dati ricevuti: {len(tempi)} ore, copertura media {validi.mean() * 100:.0f}%")

    # I primi 2 valori sono le 2 ore passate (servono solo per la pioggia su 3 ore)
    ore = [k for k in range(2, len(tempi)) if validi[k] > 0.6]
    if not ore:
        raise RuntimeError("Dati insufficienti per disegnare le mappe.")

    # --- temperatura: calcolo tutte le ore, poi una scala unica per la corsa
    temp_fine = {}
    gradienti = {}
    for k in ore:
        temp_fine[k], gradienti[k] = temperatura_fine(T[k], terreno)
    tutte = np.concatenate([temp_fine[k].ravel() for k in ore])
    t_lo = int(np.floor(np.nanpercentile(tutte, 0.5)))
    t_hi = int(np.ceil(np.nanpercentile(tutte, 99.5)))
    passo = 1 if t_hi - t_lo <= 30 else 2
    t_lo -= t_lo % passo
    livelli_t = np.arange(t_lo, t_hi + passo, passo)
    log(f"Scala temperatura per questa corsa: da {t_lo} a {t_hi} °C")

    cmap_p = ListedColormap(PIOGGIA_COLORI)
    norm_p = BoundaryNorm(PIOGGIA_LIVELLI, cmap_p.N)

    # cancello le mappe vecchie per non lasciare ore non più valide
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*.png")):
        os.remove(f)

    mappa = Mappa(terreno)
    for n, k in enumerate(ore):
        sotto = f"Valido: {testo_ora(tempi[k])}  |  +{n}h"
        g = gradienti[k]
        nota = None if g is None else f"Gradiente verticale usato: {g * 1000:+.1f} °C/km"
        mappa.disegna(temp_fine[k], livelli_t, plt.get_cmap("Spectral_r"), None, "both",
                      "Modello ICON-2I (2,2 km) con dettaglio del terreno a 500 m - Temperatura a 2 m",
                      sotto, "Temperatura (°C)",
                      os.path.join(OUTPUT_DIR, f"temperature_2m_h{n:02d}.png"), nota=nota)

        pioggia3 = np.nansum(P[k - 2:k + 1], axis=0)
        pioggia3[np.isnan(P[k - 2:k + 1]).all(axis=0)] = np.nan
        mappa.disegna(pioggia_fine(pioggia3), PIOGGIA_LIVELLI, cmap_p, norm_p, "max",
                      "Modello ICON-2I (2,2 km) - Precipitazione cumulata nelle 3 ore precedenti",
                      sotto, "Precipitazione (mm / 3h)",
                      os.path.join(OUTPUT_DIR, f"precipitation_h{n:02d}.png"), ombra=True)
        if n % 12 == 0:
            log(f"  mappe disegnate: {n + 1}/{len(ore)} ore")

    log(f"Fatto: {len(ore)} ore x 2 variabili in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
