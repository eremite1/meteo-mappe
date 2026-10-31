"""
Genera tutte le mappe di un modello per le sue aree (Lazio, Italia, Europa).

Uso:  python -m motore.genera icon2i        (oppure ecmwf, ukmo, arome, iconeu)

Fasi:
 1. legge latest.json dei dataset del modello (ultima corsa completa);
 2. legge pioggia/neve di tutte le ore e calcola le cumulate (servono per gli accumuli);
 3. in parallelo, per ogni istante: legge i parametri, calcola gli indici, disegna le mappe;
 4. scrive indice.json con l'elenco delle mappe.
"""
import os
import sys
import json
import time
import glob
import shutil
import traceback
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import Delaunay

from . import calcoli as C
from .catalogo import MODELLI, AREE, PARAMETRI, CATEGORIE, fonti_di
from . import fonti

USCITA = "uscita"
DEM_FILE = os.path.join("dati", "dem_lazio_500m.npz")
ROMA = ZoneInfo("Europe/Rome")
MESI = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
        "agosto", "settembre", "ottobre", "novembre", "dicembre"]
GIORNI = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
VAR_ACCUMULO = ["precipitation", "showers", "snowfall_water_equivalent"]
GRAD_MIN, GRAD_MAX, GRAD_STD = -0.0098, 0.0040, -0.0065


def log(m):
    print(m, flush=True)


def avviso(m):
    log(m)
    print(f"::warning::{m}", flush=True)


def t_iso(s):
    s = s.replace("Z", "")
    if len(s) == 16:
        s += ":00"
    return datetime.fromisoformat(s[:19]).replace(tzinfo=timezone.utc)


class Mancante(Exception):
    pass


# ---------------------------------------------------------------------------
# Griglie di destinazione e interpolazione
# ---------------------------------------------------------------------------
def griglia_area(area):
    a = AREE[area]
    e, p = a.get("dati", a["extent"]), a["passo"]
    m = 0 if area == "lazio" else 2 * p
    lat = np.round(np.arange(e[2] - m, e[3] + m + 1e-9, p), 4)
    lon = np.round(np.arange(e[0] - m, e[1] + m + 1e-9, p), 4)
    return lat, lon


def riquadro_area(area, margine=None):
    if "dati" in AREE[area] and margine is None:
        return list(AREE[area]["dati"])
    e = AREE[area]["extent"]
    m = margine if margine is not None else {"lazio": 0.4, "italia": 1.0, "europa": 4.0}[area]
    return [e[0] - m, e[1] + m, e[2] - m, e[3] + m]


def riempi(campo):
    buchi = ~np.isfinite(campo)
    if not buchi.any() or buchi.all():
        return campo
    idx = distance_transform_edt(buchi, return_distances=False, return_indices=True)
    return campo[tuple(idx)]


class Interpolatore:
    """Porta i dati di un dataset sulla griglia di un'area. Calcolato una volta e riusato."""

    def __init__(self, lat_s, lon_s, lat_d, lon_d, sparsa=False):
        self.sparsa = sparsa
        self.forma = (lat_d.size, lon_d.size)
        LA, LO = np.meshgrid(lat_d, lon_d, indexing="ij")
        if sparsa:
            pts = np.column_stack([lon_s, lat_s])
            tri = Delaunay(pts)
            dst = np.column_stack([LO.ravel(), LA.ravel()])
            s = tri.find_simplex(dst)
            T = tri.transform[s]
            b = np.einsum("ijk,ik->ij", T[:, :2], dst - T[:, 2])
            w = np.column_stack([b, 1 - b.sum(axis=1)])
            self.idx = tri.simplices[s]
            self.w = np.where((s >= 0)[:, None], w, np.nan)
        else:
            # indici "frazionari" dei punti di destinazione nella griglia di partenza (calcolati una volta)
            dy = lat_s[1] - lat_s[0] if lat_s.size > 1 else 1.0
            dx = lon_s[1] - lon_s[0] if lon_s.size > 1 else 1.0
            fy = (LA - lat_s[0]) / dy
            fx = (LO - lon_s[0]) / dx
            # escludo anche 2 celle sul bordo del dominio del modello (zona di "raccordo" poco affidabile)
            self.fuori = (fy < 2) | (fy > lat_s.size - 3) | (fx < 2) | (fx > lon_s.size - 3)
            if lat_s[0] <= -89 or lat_s[-1] >= 89:            # modelli globali: nessun bordo vero
                self.fuori = (fy < -0.5) | (fy > lat_s.size - 0.5) | (fx < -0.5) | (fx > lon_s.size - 0.5)
            self.coord = np.array([fy.ravel(), fx.ravel()])

    def __call__(self, valori, metodo="linear"):
        if self.sparsa:
            v = np.asarray(valori, dtype=np.float32).reshape(-1)
            return np.sum(v[self.idx] * self.w, axis=1).reshape(self.forma).astype(np.float32)
        from scipy.ndimage import map_coordinates
        dati = np.asarray(valori, dtype=np.float32)
        buchi = ~np.isfinite(dati)
        if buchi.any():
            if buchi.all():
                return np.full(self.forma, np.nan, np.float32)
            dati = riempi(dati)
        ordine = 3 if metodo == "cubic" else 1
        out = map_coordinates(dati, self.coord, order=ordine, mode="nearest").reshape(self.forma)
        if buchi.any():
            m = map_coordinates(buchi.astype(np.float32), self.coord, order=1, mode="nearest").reshape(self.forma)
            out[m > 0.5] = np.nan
        out[self.fuori] = np.nan
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Terreno del Lazio (DEM a 500 m) per il dettaglio della temperatura
# ---------------------------------------------------------------------------
def carica_dem():
    if not os.path.exists(DEM_FILE):
        return None
    d = np.load(DEM_FILE)
    return d["lat"], d["lon"], np.clip(d["z"], 0, None)


def prepara_terreno(dem, lat_f, lon_f):
    lat_d, lon_d, z_d = dem
    f = RegularGridInterpolator((lat_d, lon_d), z_d, bounds_error=False, fill_value=0)
    LA, LO = np.meshgrid(lat_f, lon_f, indexing="ij")
    z = f((LA, LO)).astype(np.float32)
    dy = 0.005 * 111000.0
    dx = dy * np.cos(np.radians(lat_f.mean()))
    gy, gx = np.gradient(z, dy, dx)
    pend = np.arctan(np.hypot(gx, gy) * 1.5)
    espo = np.arctan2(-gx, gy)
    alt, az = np.radians(45), np.radians(315)
    luce = np.sin(alt) * np.cos(pend) + np.cos(alt) * np.sin(pend) * np.cos(az - espo)
    return dict(z=z, ombra=np.clip(luce / np.sin(alt), 0, 1), dem=dem)


# ---------------------------------------------------------------------------
# Contesto: dà al catalogo i parametri già sulla griglia dell'area
# ---------------------------------------------------------------------------
class Contesto:
    def __init__(self, lavoro, area, t):
        self.L = lavoro
        self.area = area
        self.t = t                                   # istante (datetime UTC)
        self.cache = {}

    # --- lettura di base ---------------------------------------------------
    def _sorgente(self, nome):
        for ds in self.L.fonti[self.area]:
            f = self.L.dati(ds, self.area, self.t)
            if f is not None and f.ha(nome):
                return ds, f
        raise Mancante(nome)

    def grezzo(self, nome):
        """Valori sulla griglia del modello (ritagliata) per l'area: (lat, lon, valori, ds)."""
        chiave = ("grezzo", nome)
        if chiave not in self.cache:
            ds, f = self._sorgente(nome)
            self.cache[chiave] = f.leggi(nome) + (ds,)
        return self.cache[chiave]

    def v(self, nome, minimo=None, massimo=None, unita=None, metodo=None):
        chiave = ("v", nome, metodo)
        if chiave not in self.cache:
            lat, lon, val, ds = self.grezzo(nome)
            interp = self.L.interpolatore(ds, self.area, lat, lon)
            if metodo is None:
                metodo = "cubic" if nome.startswith(("temperature", "geopotential", "pressure", "dew")) else "linear"
            if self.L.info[ds]["ridotta"]:
                campo = interp(val)
            else:
                campo = interp(val, metodo)
            campo = campo * self.L.scala(nome)
            campo = self.L.maschera(self.area, campo)
            self.cache[chiave] = campo
        c = self.cache[chiave]
        if minimo is not None or massimo is not None:
            c = np.clip(c, minimo, massimo)
        return c

    def ha(self, nome):
        try:
            self._sorgente(nome)
            return True
        except Mancante:
            return False

    def liscia(self, campo, sigma):
        if not sigma:
            return campo
        ok = np.isfinite(campo)
        if not ok.any():
            return campo
        out = gaussian_filter(np.where(ok, campo, np.nanmean(campo)), sigma)
        return np.where(ok, out, np.nan)

    # --- temperatura e umidità vicino al suolo ------------------------------
    def t2m(self):
        if "t2m" in self.cache:
            return self.cache["t2m"]
        base = self.v("temperature_2m")
        if self.area == "lazio" and self.L.terreno is not None:
            lat, lon, T, ds = self.grezzo("temperature_2m")
            zmod = self.L.quota_modello(ds, lat, lon)
            g = GRAD_STD
            ok = np.isfinite(T) & (zmod > 30)
            if ok.sum() > 30 and np.ptp(zmod[ok]) > 300:
                g = float(np.clip(np.polyfit(zmod[ok], T[ok], 1)[0], GRAD_MIN, GRAD_MAX))
            interp = self.L.interpolatore(ds, self.area, lat, lon)
            zmod_f = interp(zmod.astype(np.float32), "linear") if not self.L.info[ds]["ridotta"] else interp(zmod)
            base = base + g * (self.L.terreno["z"] - zmod_f)
        self.cache["t2m"] = base
        return base

    def ur2m(self):
        if self.ha("relative_humidity_2m"):
            return self.v("relative_humidity_2m", 0, 100)
        return C.umidita_relativa(self.v("temperature_2m"), self.v("dew_point_2m"))

    def td2m(self):
        if self.area == "lazio":
            return C.punto_rugiada(self.t2m(), self.ur2m())
        if self.ha("dew_point_2m"):
            return self.v("dew_point_2m")
        return C.punto_rugiada(self.v("temperature_2m"), self.ur2m())

    def td_livello(self, liv):
        return C.punto_rugiada(self.v(f"temperature_{liv}"), self.v(f"relative_humidity_{liv}", 0, 100))

    def nuvole(self):
        """Nuvolosità totale; se il modello dà solo i tre strati, la stimo combinandoli."""
        if self.ha("cloud_cover"):
            return self.v("cloud_cover", 0, 100)
        strati = [self.v(f"cloud_cover_{s}", 0, 100) / 100 for s in ("low", "mid", "high")]
        return 100 * (1 - (1 - strati[0]) * (1 - strati[1]) * (1 - strati[2]))

    # --- vento -----------------------------------------------------------
    def uv(self, liv):
        chiave = ("uv", liv)
        if chiave not in self.cache:
            if self.ha(f"wind_u_component_{liv}"):
                u = self.v(f"wind_u_component_{liv}", unita="vento")
                v = self.v(f"wind_v_component_{liv}", unita="vento")
            else:
                vel = self.v(f"wind_speed_{liv}")
                lat, lon, d, ds = self.grezzo(f"wind_direction_{liv}")
                us, vs = C.uv_da_vel_dir(self.grezzo(f"wind_speed_{liv}")[2], d)
                interp = self.L.interpolatore(ds, self.area, lat, lon)
                if self.L.info[ds]["ridotta"]:
                    u, v = interp(us), interp(vs)
                else:
                    u, v = interp(us, "linear"), interp(vs, "linear")
                sc = self.L.scala(f"wind_speed_{liv}")
                u, v = self.L.maschera(self.area, u * sc), self.L.maschera(self.area, v * sc)
            self.cache[chiave] = (u, v)
        return self.cache[chiave]

    def vel(self, liv):
        if self.ha(f"wind_speed_{liv}"):
            return self.v(f"wind_speed_{liv}", minimo=0)
        return np.hypot(*self.uv(liv))

    def livello_getto(self):
        for liv in ("300hPa", "250hPa"):
            if self.ha(f"wind_u_component_{liv}") or self.ha(f"wind_speed_{liv}"):
                return liv
        raise Mancante("vento in quota")

    # --- accumuli --------------------------------------------------------
    def acc(self, nome, totale=False):
        ds, cum, tempi = self.L.cumulata(self.area, nome)
        t = self.t
        if t <= self.L.corsa:
            raise Mancante(nome)                     # all'istante iniziale non c'è accumulo
        if t not in tempi:
            raise Mancante(nome)
        i = tempi.index(t)
        if totale:
            val = cum[i]
        else:
            passo = self.L.passo[self.area]
            t0 = t - timedelta(hours=passo)
            j = max([n for n, tt in enumerate(tempi) if tt <= t0], default=None)
            val = cum[i] - (cum[j] if j is not None else 0)
        lat, lon = self.L.coord_cum[(ds, self.area)]
        interp = self.L.interpolatore(ds, self.area, lat, lon)
        campo = interp(val) if self.L.info[ds]["ridotta"] else interp(val, "linear")
        dentro = np.isfinite(campo)
        campo = np.clip(gaussian_filter(np.nan_to_num(campo), 1.0 if self.area == "lazio" else 0.6), 0, None)
        if self.ha("temperature_2m"):
            dentro &= np.isfinite(self.v("temperature_2m"))
        return np.where(dentro, campo, np.nan)

    def acc_totale(self, nome):
        return self.acc(nome, totale=True)

    def neve_acc(self, totale=False):
        return self.acc("snowfall_water_equivalent", totale)       # 1 mm d'acqua ~ 1 cm di neve

    def neve_suolo(self):
        return self.v("snow_depth", minimo=0) * 100.0              # metri -> cm

    # --- altre grandezze ---------------------------------------------------
    def zero_termico(self):
        if self.ha("freezing_level_height"):
            return self.v("freezing_level_height", minimo=0)
        livelli = [l for l in ("1000hPa", "925hPa", "850hPa", "700hPa", "600hPa", "500hPa")
                   if self.ha(f"temperature_{l}") and self.ha(f"geopotential_height_{l}")]
        if len(livelli) < 3:
            raise Mancante("zero termico")
        return C.zero_termico_da_livelli([self.v(f"temperature_{l}") for l in livelli],
                                         [self.v(f"geopotential_height_{l}") for l in livelli])

    def visibilita(self):
        return self.v("visibility", minimo=0) / 1000.0              # metri -> km

    def ascesa(self, liv):
        """Velocità verticale in cm/s, positiva quando l'aria sale."""
        w = self.v(f"vertical_velocity_{liv}")
        return w * self.L.segno_ascesa * 100.0

    def pressione_suolo(self):
        p = self.v("pressure_msl")
        z = self.L.terreno["z"] if (self.area == "lazio" and self.L.terreno is not None) else 0
        return p * (1 - 2.25577e-5 * z) ** 5.25588

    def sweat(self):
        t850, t500 = self.v("temperature_850hPa"), self.v("temperature_500hPa")
        td850 = self.td_livello("850hPa")
        tt = C.total_totals(t850, td850, t500)
        v8, d8 = C.vel_dir_da_uv(*self.uv("850hPa"))
        v5, d5 = C.vel_dir_da_uv(*self.uv("500hPa"))
        return C.sweat(td850, tt, v8 / 1.852, v5 / 1.852, d8, d5)


# ---------------------------------------------------------------------------
# Dati già scaricati e ritagliati (uno per dataset, area e istante)
# ---------------------------------------------------------------------------
LIVELLI = ["1000hPa", "925hPa", "850hPa", "700hPa", "600hPa", "500hPa", "300hPa", "250hPa"]
VARIABILI_UTILI = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "showers",
    "snowfall_water_equivalent", "snow_depth", "snowfall_height", "freezing_level_height",
    "wind_speed_10m", "wind_direction_10m", "wind_u_component_10m", "wind_v_component_10m",
    "wind_gusts_10m", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "pressure_msl", "visibility", "shortwave_radiation", "cape", "convective_inhibition",
    "lightning_potential", "total_column_integrated_water_vapour", "sea_surface_temperature",
    "boundary_layer_height"] + [f"{x}_{l}" for l in LIVELLI for x in (
        "temperature", "geopotential_height", "relative_humidity", "wind_u_component",
        "wind_v_component", "wind_speed", "wind_direction", "vertical_velocity")]
CARTELLA_DATI = "dati_corsa"


class DatiSalvati:
    """Stessa interfaccia di FileOM, ma legge i ritagli già salvati su disco."""

    def __init__(self, percorso):
        self.z = np.load(percorso)
        self.nomi = {k for k in self.z.files if not k.startswith("_")}

    def ha(self, nome):
        return nome in self.nomi

    def leggi(self, nome, riquadro=None):
        return self.z["_lat"], self.z["_lon"], self.z[nome]


# ---------------------------------------------------------------------------
# Lavoro: tutto ciò che serve per una corsa di un modello
# ---------------------------------------------------------------------------
class Lavoro:
    def __init__(self, chiave, aree=None):
        self.chiave = chiave
        self.cfg = MODELLI[chiave]
        self.aree = [a for a in self.cfg["aree"] if not aree or a in aree]
        self.fonti = {a: fonti_di(chiave, a) for a in self.aree}
        self.dataset = sorted({d for a in self.aree for d in self.fonti[a]})
        from .controlla import scegli_corsa
        scelta = scegli_corsa(chiave)
        if not scelta:
            raise RuntimeError("nessuna corsa completa e abbastanza lunga disponibile")
        self.info = {d: scelta[d] for d in self.dataset}
        corse = {self.info[d]["reference_time"] for d in self.dataset}
        if len(corse) > 1:
            raise RuntimeError(f"corse non allineate tra i dataset: {corse}")
        self.corsa = t_iso(next(iter(corse)))
        self.tempi = {d: sorted({t_iso(t) for t in self.info[d]["valid_times"]}) for d in self.dataset}
        self.griglie = {a: griglia_area(a) for a in self.aree}
        self.passo = {a: self.cfg["aree"][a][0] for a in self.aree}
        self._interp = {}
        self._dati = {}
        self._zmod = {}
        self._cum = {}
        self.coord_cum = {}
        self.terreno = None
        self.segno_ascesa = 1.0
        self.scale_corsa = {}
        self.disponibili = {}          # (ds, area, t) -> percorso del file salvato

    # --- istanti da disegnare --------------------------------------------
    def istanti(self, area):
        d = self.fonti[area][0]
        passo, massimo = self.cfg["aree"][area]
        out = []
        for k, t in enumerate(self.tempi[d]):
            h = int(round((t - self.corsa).total_seconds() / 3600))
            if 0 <= h <= massimo and h % passo == 0:
                if all(t in self.tempi[x] for x in self.fonti[area][1:]):
                    out.append((k, h, t))
        return out

    # --- fase 1: scarico i file e salvo solo i ritagli utili ----------------------
    def _diradamento(self, ds, area):
        """Per l'Europa non serve tutta la risoluzione dei modelli fini: prendo un punto ogni N."""
        if area != "europa" or self.info[ds]["ridotta"]:
            return 1
        b = self.info[ds]["bbox"]
        return 1   # calcolato dopo la lettura (serve la forma della griglia)

    def scarica(self, paralleli=6):
        os.makedirs(CARTELLA_DATI, exist_ok=True)
        lavori = {}   # (ds, t) -> {area: "tutto" | "accumulo"}
        for area in self.aree:
            for k, h, t in self.istanti(area):
                for ds in self.fonti[area]:
                    lavori.setdefault((ds, t), {})[area] = "tutto"
            t_lim = self.corsa + timedelta(hours=self.cfg["aree"][area][1])
            for ds in self.fonti[area][:1]:
                for t in self.tempi[ds]:
                    if t <= t_lim:
                        lavori.setdefault((ds, t), {}).setdefault(area, "accumulo")
        log(f"File da scaricare: {len(lavori)}")

        def uno(chiave):
            ds, t = chiave
            v = t.strftime("%Y-%m-%dT%H:%MZ")
            try:
                f = fonti.FileOM(ds, self.info[ds], v)
            except Exception as e:
                return f"{ds} {v}: {e}"
            try:
                for area, cosa in lavori[chiave].items():
                    nomi = VARIABILI_UTILI if cosa == "tutto" else VAR_ACCUMULO
                    salva = {}
                    lat = lon = None
                    for n in nomi:
                        if f.ha(n):
                            la, lo, val = f.leggi(n, riquadro_area(area))
                            if area == "europa" and not self.info[ds]["ridotta"]:
                                passo_src = abs(la[1] - la[0]) if la.size > 1 else 1
                                d = max(1, int(round(AREE["europa"]["passo"] / passo_src / 1.2)))
                                la, lo, val = la[::d], lo[::d], val[::d, ::d]
                            salva[n] = val.astype(np.float32)
                            lat, lon = la, lo
                    if salva:
                        salva["_lat"], salva["_lon"] = lat, lon
                        cartella = os.path.join(CARTELLA_DATI, ds, area)
                        os.makedirs(cartella, exist_ok=True)
                        pf = os.path.join(cartella, t.strftime("%Y%m%d%H%M") + ".npz")
                        np.savez(pf, **salva)
                        self.disponibili[(ds, area, t)] = pf
            except Exception as e:
                return f"{ds} {v}: {type(e).__name__} {e}"
            finally:
                if hasattr(f, "chiudi"):
                    f.chiudi()
            return None

        t0 = time.time()
        errori = []
        with ThreadPoolExecutor(paralleli) as ex:
            for n, e in enumerate(ex.map(uno, sorted(lavori, key=lambda x: (x[1], x[0])))):
                if e:
                    errori.append(e)
                if (n + 1) % 25 == 0:
                    log(f"  scaricati {n + 1}/{len(lavori)} ({time.time() - t0:.0f} s)")
        if errori:
            avviso(f"[{self.cfg['nome']}] {len(errori)} file non scaricati, es.: {errori[0]}")
        log(f"  download completato in {time.time() - t0:.0f} s")

    def dati(self, ds, area, t):
        chiave = (ds, area, t)
        if chiave not in self._dati:
            pf = self.disponibili.get(chiave)
            self._dati[chiave] = DatiSalvati(pf) if pf else None
        return self._dati[chiave]

    def libera(self, t):
        for k in [k for k in self._dati if k[2] == t]:
            self._dati.pop(k, None)

    # --- interpolazioni, unità, terreno ---------------------------------------
    def interpolatore(self, ds, area, lat, lon):
        chiave = (ds, area, lat.size, lon.size, float(lat[0]), float(lon[0]))
        if chiave not in self._interp:
            la, lo = self.griglie[area]
            self._interp[chiave] = Interpolatore(lat, lon, la, lo, sparsa=self.info[ds]["ridotta"])
        return self._interp[chiave]

    def maschera(self, area, campo):
        return campo

    def scala(self, nome):
        """Il vento nell'archivio è in m/s; sulle mappe in km/h."""
        if nome.startswith(("wind_speed", "wind_gusts", "wind_u_component", "wind_v_component")):
            return 3.6
        return 1.0

    def quota_modello(self, ds, lat, lon):
        chiave = (ds, lat.size, lon.size)
        if chiave not in self._zmod:
            lat_d, lon_d, z_d = self.terreno["dem"]
            zl = gaussian_filter(z_d, self.cfg["sigma"])
            f = RegularGridInterpolator((lat_d, lon_d), zl, bounds_error=False, fill_value=0)
            if self.info[ds]["ridotta"]:
                z = f(np.column_stack([lat, lon]))
            else:
                LA, LO = np.meshgrid(lat, lon, indexing="ij")
                z = f((LA, LO))
            self._zmod[chiave] = z
        return self._zmod[chiave]

    # --- scale di colore adattate alla corsa (temperature) ---------------------
    FAMIGLIE = {"temperature_2m": ["temperatura", "percepita", "bulbo_umido"],
                "temperature_850hPa": ["t850"], "temperature_500hPa": ["z500_t500"]}

    def calcola_scale(self):
        for area in self.aree:
            ist = self.istanti(area)
            campione = ist[::max(1, len(ist) // 12)]
            e = AREE[area]["extent"]
            for var, params in self.FAMIGLIE.items():
                valori = []
                for k, h, t in campione:
                    for ds in self.fonti[area]:
                        d = self.dati(ds, area, t)
                        if d is not None and d.ha(var):
                            la, lo, v = d.leggi(var)
                            if v.ndim == 2:
                                iy = (la >= e[2]) & (la <= e[3])
                                ix = (lo >= e[0]) & (lo <= e[1])
                                v = v[np.ix_(iy, ix)]
                            else:
                                v = v[(la >= e[2]) & (la <= e[3]) & (lo >= e[0]) & (lo <= e[1])]
                            valori.append(v[np.isfinite(v)].ravel())
                            break
                    self.libera(t)
                valori = [v for v in valori if v.size]
                if not valori:
                    continue
                tutto = np.concatenate(valori)
                lo_, hi_ = np.percentile(tutto, 0.3), np.percentile(tutto, 99.7)
                if var == "temperature_2m":
                    lo_ -= 5 if area == "lazio" else 2
                    hi_ += 2
                passo = 1
                while (hi_ - lo_) / passo > 32:
                    passo *= 2
                lo_ = np.floor(lo_ / passo) * passo
                hi_ = np.ceil(hi_ / passo) * passo
                livelli = list(np.arange(lo_, hi_ + passo / 2, passo))
                for pid in params:
                    self.scale_corsa[(area, pid)] = livelli

    # --- cumulate di pioggia e neve ----------------------------------------------
    def prepara_cumulate(self):
        for area in self.aree:
            ds = self.fonti[area][0]
            t_lim = self.corsa + timedelta(hours=self.cfg["aree"][area][1])
            tempi = [t for t in self.tempi[ds] if t <= t_lim]
            for nome in VAR_ACCUMULO:
                cum, s, lat, lon = [], None, None, None
                for t in tempi:
                    d = self.dati(ds, area, t)
                    if d is not None and d.ha(nome):
                        lat, lon, val = d.leggi(nome)
                        val = np.nan_to_num(np.clip(val, 0, None))
                        s = val.copy() if s is None else s + val
                    cum.append(None if s is None else s.copy())
                    self.libera(t)
                if s is None:
                    continue
                cum = [c if c is not None else np.zeros_like(s) for c in cum]
                self._cum[(ds, area, nome)] = (cum, tempi)
                self.coord_cum[(ds, area)] = (lat, lon)

    def cumulata(self, area, nome):
        ds = self.fonti[area][0]
        if (ds, area, nome) in self._cum:
            cum, tempi = self._cum[(ds, area, nome)]
            return ds, cum, tempi
        raise Mancante(nome)


# ---------------------------------------------------------------------------
# Lavoro di un processo: un istante, tutte le aree e tutti i parametri
# ---------------------------------------------------------------------------
_G = {}


def testo_ora(t):
    loc = t.astimezone(ROMA)
    return (f"{GIORNI[loc.weekday()]} {loc.day} {MESI[loc.month - 1]}, ore {loc:%H:%M}",
            f"{t:%H} UTC")


def _avvia_processo():
    from .disegno import Disegnatore
    L = _G["lavoro"]
    _G["disegnatori"] = {}
    for a in L.aree:
        la, lo = L.griglie[a]
        ombra = L.terreno["ombra"] if (a == "lazio" and L.terreno is not None) else None
        _G["disegnatori"][a] = Disegnatore(a, la, lo, ombra)


def _fai_istante(compito):
    area_istanti = compito
    L = _G["lavoro"]
    risultati, errori = [], []
    for area, k, h, t in area_istanti:
        ctx = Contesto(L, area, t)
        dis = _G["disegnatori"][area]
        data, ora_utc = testo_ora(t)
        cartella = os.path.join(USCITA, area)
        for p in PARAMETRI:
            if area not in p["aree"]:
                continue
            if (area, p["id"]) in L.scale_corsa:
                st = dict(p["stile"], livelli=L.scale_corsa[(area, p["id"])])
                p = dict(p, stile=st, stile_europa=None)
            try:
                r = p["calcolo"](ctx)
                if isinstance(r, dict):
                    campo, contorni, frecce = r["campo"], r.get("contorni"), r.get("frecce")
                else:
                    campo, contorni, frecce = r, None, None
                if campo is None or not np.isfinite(campo).any():
                    continue
                unita = p["unita"]
                if p["id"] in ("pioggia", "pioggia_isobare", "rovesci", "neve") and h > 0:
                    passo_eff = L.passo[area]
                    unita = unita.replace("nelle ore precedenti", f"nelle {passo_eff} ore precedenti" if passo_eff > 1 else "nell'ora precedente")
                info = f"Modello {L.cfg['nome']} ({L.cfg['ris']}) · corsa {L.corsa:%d/%m %H} UTC · ora italiana ({ora_utc})"
                if p["id"] in ("temperatura", "percepita", "bulbo_umido", "gelate", "incendi") and area == "lazio" and L.terreno is not None:
                    info = f"Modello {L.cfg['nome']} ({L.cfg['ris']}) · dettaglio del terreno a 500 m · corsa {L.corsa:%d/%m %H} UTC"
                nome = f"{p['id']}_h{h:03d}.png"
                dis.disegna(os.path.join(cartella, nome), campo, p, L.cfg["nome"], data,
                            f"+{h}h", info, contorni=contorni, frecce=frecce, unita=unita)
                risultati.append((area, p["id"], nome, h, t.strftime("%Y-%m-%dT%H:%MZ")))
            except Exception as e:
                if not isinstance(e, (Mancante, KeyError)):
                    errori.append(f"{area}/{p['id']}/+{h}h: {type(e).__name__}: {e}")
        L.libera(t)
    return risultati, errori


# ---------------------------------------------------------------------------
def esegui(chiave, max_istanti=None, processi=None, aree=None):
    t0 = time.time()
    L = Lavoro(chiave, aree)
    log(f"=== {L.cfg['nome']} · corsa {L.corsa:%Y-%m-%d %H} UTC · aree: {', '.join(L.aree)} ===")
    dem = carica_dem()
    if dem is not None and "lazio" in L.aree:
        L.terreno = prepara_terreno(dem, *L.griglie["lazio"])

    log("Scarico i dati della corsa...")
    if os.path.exists(CARTELLA_DATI):
        shutil.rmtree(CARTELLA_DATI)
    L.scarica()
    log("Scale di colore e cumulate di pioggia e neve...")
    L.calcola_scale()
    L.prepara_cumulate()
    log(f"  preparazione completata in {time.time() - t0:.0f} s")

    # compiti: raggruppo per istante (stesso file letto una volta per tutte le aree)
    per_t = {}
    for area in L.aree:
        ist = L.istanti(area)
        if max_istanti:
            ist = ist[:max_istanti]
        for k, h, t in ist:
            per_t.setdefault(t, []).append((area, k, h, t))
    compiti = [per_t[t] for t in sorted(per_t)]
    log(f"Istanti da elaborare: {len(compiti)}")

    if os.path.exists(USCITA):
        shutil.rmtree(USCITA)
    for a in L.aree:
        os.makedirs(os.path.join(USCITA, a), exist_ok=True)

    _G["lavoro"] = L
    nproc = processi or max(1, min(4, os.cpu_count() or 1))
    tutti, errori = [], []
    t1 = time.time()
    if nproc == 1:
        _avvia_processo()
        for n, c in enumerate(compiti):
            r, e = _fai_istante(c)
            tutti += r
            errori += e
    else:
        # coste e confini preparati una volta sola qui; i processi figli li ricevono già pronti
        _avvia_processo()
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(nproc, mp_context=ctx) as ex:
            for n, (r, e) in enumerate(ex.map(_fai_istante, compiti)):
                tutti += r
                errori += e
                if (n + 1) % 10 == 0:
                    log(f"  istanti completati: {n + 1}/{len(compiti)} ({time.time() - t1:.0f} s)")
    log(f"Mappe disegnate: {len(tutti)} in {time.time() - t1:.0f} s")
    if errori:
        avviso(f"[{L.cfg['nome']}] {len(errori)} mappe con errori, es.: " + " | ".join(errori[:4]))

    # diagnostica: valori minimi/medi/massimi di alcuni parametri al primo istante (per controllo)
    diag = {}
    try:
        area = L.aree[0]
        k, h, t = L.istanti(area)[min(3, len(L.istanti(area)) - 1)]
        for ds in L.fonti[area]:
            d = L.dati(ds, area, t)
            if d is None:
                continue
            for n in sorted(d.nomi):
                v = d.leggi(n)[2]
                v = v[np.isfinite(v)]
                if v.size:
                    diag[f"{ds}:{n}"] = [round(float(v.min()), 3), round(float(v.mean()), 3), round(float(v.max()), 3)]
            la, lo, _ = d.leggi(sorted(d.nomi)[0])
            diag[f"{ds}:_coord"] = [float(la[0]), float(la[-1]), float(lo[0]), float(lo[-1])]
    except Exception as e:
        diag["errore"] = str(e)

    # indice
    corsa_txt = L.corsa.strftime("%Y-%m-%dT%H:%MZ")
    indice = dict(modello=chiave, nome=L.cfg["nome"], risoluzione=L.cfg["ris"],
                  corsa=corsa_txt, corse={a: corsa_txt for a in L.aree},
                  aggiornato_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                  aree={}, diagnostica=diag)
    for area, pid, nome, h, v in sorted(tutti, key=lambda x: (x[0], x[1], x[3])):
        indice["aree"].setdefault(area, {}).setdefault(pid, []).append(
            dict(file=f"{area}/{nome}", scadenza=h, valido_utc=v))
    with open(os.path.join(USCITA, "indice_parziale.json"), "w") as f:
        json.dump(indice, f, separators=(",", ":"))
    log(f"=== Fine {L.cfg['nome']}: {len(tutti)} mappe in {time.time() - t0:.0f} s ===")
    return len(tutti)


if __name__ == "__main__":
    modello = sys.argv[1]
    aree = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] not in ("", "tutte") else None
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else None
    # se nel frattempo la stessa corsa è già stata pubblicata, non rifaccio il lavoro
    if os.environ.get("FORZA", "false").lower() != "true":
        from .controlla import corsa_pubblicata, corsa_disponibile
        nuova = corsa_disponibile(modello)
        if nuova and all(corsa_pubblicata(modello, a) == nuova for a in (aree or MODELLI[modello]["aree"])):
            log(f"{modello}: corsa già pubblicata, niente da fare.")
            sys.exit(0)
    n = esegui(modello, max_istanti=mx, aree=aree)
    if n == 0:
        sys.exit(1)
