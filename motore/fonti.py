"""
Lettura dei dati dall'archivio pubblico di Open-Meteo su Amazon S3 (data_spatial).

Ogni file .om contiene TUTTI i parametri di un istante di previsione di un modello.
Si leggono solo i parametri e il riquadro geografico che servono.
"""
import os
import json
import time
import numpy as np
import requests

URL_BASE = "https://openmeteo.s3.amazonaws.com/data_spatial/"
S3_BASE = "openmeteo/data_spatial/"

_fs = None


def fs():
    global _fs
    if _fs is None:
        import fsspec
        _fs = fsspec.filesystem("s3", anon=True)
    return _fs


def ultimo(dataset):
    """Contenuto di latest.json (ultima corsa completa) del dataset, o None."""
    try:
        r = requests.get(f"{URL_BASE}{dataset}/latest.json", timeout=30)
        if r.status_code != 200:
            return None
        j = r.json()
        j["bbox"] = _bbox(j.get("crs_wkt", ""))
        j["ridotta"] = "Reduced Gaussian" in j.get("crs_wkt", "")
        return j
    except Exception:
        return None


def tempi_corsa(dataset, corsa):
    """Elenca gli istanti disponibili di una corsa leggendo l'elenco dei file nell'archivio.
    corsa: 'YYYY-MM-DDTHH:MM...'. Restituisce una lista di stringhe 'YYYY-MM-DDTHH:MMZ' (o None)."""
    import re
    prefisso = f"data_spatial/{dataset}/{corsa[:4]}/{corsa[5:7]}/{corsa[8:10]}/{corsa[11:13]}{corsa[14:16]}Z/"
    chiavi, token = [], None
    try:
        for _ in range(10):
            params = {"list-type": "2", "prefix": prefisso}
            if token:
                params["continuation-token"] = token
            r = requests.get("https://openmeteo.s3.amazonaws.com/", params=params, timeout=30)
            if r.status_code != 200:
                return None
            chiavi += re.findall(r"<Key>([^<]+)</Key>", r.text)
            m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
            if not m:
                break
            token = m.group(1)
    except Exception:
        return None
    tempi = []
    for k in chiavi:
        nome = k.rsplit("/", 1)[-1]
        if nome.endswith(".om") and len(nome) >= 18:
            t = nome[:-3]                                   # YYYY-MM-DDTHHMM
            tempi.append(f"{t[:13]}:{t[13:15]}Z")
    return sorted(set(tempi)) or None


def info_corsa(dataset, corsa, base=None):
    """Come ultimo(), ma per una corsa specifica (anche non la più recente)."""
    base = base or ultimo(dataset)
    if base is None:
        return None
    if base["reference_time"][:16] == corsa[:16]:
        return base
    tempi = tempi_corsa(dataset, corsa)
    if not tempi:
        return None
    j = dict(base)
    j["reference_time"] = corsa[:16] + ":00Z"
    j["valid_times"] = tempi
    j["completed"] = True
    return j


def _bbox(wkt):
    try:
        b = wkt.split("BBOX[")[1].split("]")[0]
        return [float(x) for x in b.split(",")]      # lat0, lon0, lat1, lon1
    except Exception:
        return None


def percorso(dataset, corsa, valido):
    """corsa e valido in formato ISO (es. 2026-10-30T12:00:00Z / 2026-10-30T15:00Z)."""
    d = corsa[:10].replace("-", "/")
    hh = corsa[11:13] + corsa[14:16]
    t = valido.replace(":", "").replace("Z", "")[:15]
    return f"{S3_BASE}{dataset}/{d}/{hh}Z/{t}.om"


# ---------------------------------------------------------------------------
# Griglia ridotta gaussiana O1280 (ECMWF IFS 9 km)
# ---------------------------------------------------------------------------
_O1280 = None


def griglia_o1280():
    """Coordinate dei 6.599.680 punti della griglia O1280, righe da nord a sud."""
    global _O1280
    if _O1280 is None:
        n = 1280
        x, _ = np.polynomial.legendre.leggauss(2 * n)
        lat_righe = np.degrees(np.arcsin(x))[::-1]
        lat, lon, inizio = [], [], [0]
        for i in range(2 * n):
            k = i + 1 if i < n else 2 * n - i
            npt = 4 * k + 16
            lat.append(np.full(npt, lat_righe[i], np.float32))
            lo = np.arange(npt, dtype=np.float64) * 360.0 / npt
            lon.append(np.where(lo >= 180, lo - 360, lo).astype(np.float32))
            inizio.append(inizio[-1] + npt)
        _O1280 = (np.concatenate(lat), np.concatenate(lon), np.array(inizio), lat_righe)
    return _O1280


# ---------------------------------------------------------------------------
# Lettura di un file
# ---------------------------------------------------------------------------
class FileOM:
    """Scarica il file .om intero (molto più veloce che leggerlo a pezzi) e lo legge in locale."""

    def __init__(self, dataset, info, valido, tentativi=4, cartella="/tmp/om"):
        import omfiles
        os.makedirs(cartella, exist_ok=True)
        self.dataset = dataset
        self.info = info
        self.valido = valido
        p = percorso(dataset, info["reference_time"], valido)
        url = URL_BASE + p[len(S3_BASE):]
        self.locale = os.path.join(cartella, f"{dataset}_{os.getpid()}_{abs(hash(valido))}.om")
        ultimo_err = None
        for k in range(tentativi):
            try:
                with requests.get(url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(self.locale, "wb") as f:
                        for pezzo in r.iter_content(8 << 20):
                            f.write(pezzo)
                self.rd = omfiles.OmFileReader(self.locale)
                break
            except Exception as e:
                ultimo_err = e
                time.sleep(5 * (k + 1))
        else:
            self.chiudi()
            raise IOError(f"impossibile scaricare {url}: {ultimo_err}")
        self.nomi = set()
        for i in range(self.rd.num_children):
            c = self.rd.get_child_by_index(i)
            if c.is_array:
                self.nomi.add(c.name)

    def chiudi(self):
        try:
            self.rd.close()
        except Exception:
            pass
        try:
            os.remove(self.locale)
        except Exception:
            pass

    def ha(self, nome):
        return nome in self.nomi

    def leggi(self, nome, riquadro):
        """Legge il parametro nel riquadro [lon0, lon1, lat0, lat1] (con margine).
        Restituisce (lat, lon, valori): per le griglie regolari lat/lon sono 1D e valori 2D;
        per la griglia ridotta ECMWF sono tre array 1D di punti sparsi."""
        c = self.rd.get_child_by_name(nome)
        lo0, lo1, la0, la1 = riquadro
        if self.info["ridotta"]:
            lat_p, lon_p, inizio, lat_righe = griglia_o1280()
            righe = np.where((lat_righe >= la0) & (lat_righe <= la1))[0]
            a, b = inizio[righe[0]], inizio[righe[-1] + 1]
            dati = self._leggi(c, (slice(0, 1), slice(int(a), int(b)))).reshape(-1)
            la, lo = lat_p[a:b], lon_p[a:b]
            sel = (lo >= lo0) & (lo <= lo1)
            return la[sel], lo[sel], dati[sel]
        lat0, lon0, lat1, lon1 = self.info["bbox"]
        ny, nx = c.shape[-2], c.shape[-1]
        lat = np.linspace(lat0, lat1, ny)
        lon = np.linspace(lon0, lon1, nx)
        iy = np.where((lat >= la0) & (lat <= la1))[0]
        ix = np.where((lon >= lo0) & (lon <= lo1))[0]
        if iy.size == 0 or ix.size == 0:
            raise KeyError("riquadro fuori dal dominio del modello")
        sl = (slice(int(iy[0]), int(iy[-1]) + 1), slice(int(ix[0]), int(ix[-1]) + 1))
        if len(c.shape) == 3:
            sl = (slice(0, 1),) + sl
        dati = self._leggi(c, sl).reshape(iy.size, ix.size)
        return lat[iy], lon[ix], dati

    @staticmethod
    def _leggi(c, sl):
        try:
            a = c[sl]
        except Exception:
            a = c.read_array(sl)
        return np.asarray(a, dtype=np.float32)
