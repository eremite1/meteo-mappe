#!/usr/bin/env python3
"""PAGELLA UNICA DI METEONEROLA: tutti i modelli (Mappe Lazio, multimodello, tutti i WRF, varianti di fisica, WRF corretto
e qualsiasi modello aggiunto in futuro) confrontati con la realta' con UN SOLO METODO, uguale per tutti.

Cosa si confronta
  1. TEMPERATURA e VENTO con le stazioni (aeroporti, Protezione Civile, MeteoNetwork, stazione di Nerola).
  2. «CHE TEMPO FA» con le stazioni: il tempo di ogni ora (sereno, nuvoloso, nebbia o foschia, pioggia, neve) calcolato
     con la STESSA formula per tutti i modelli (nuvolosita', umidita', pioggia, temperatura) e confrontato con i bollettini
     degli aeroporti; in piu' la classe di pioggia ora per ora con tutte le stazioni.
  3. PIOGGIA ALLE STAZIONI: pioggia caduta in 3 ore, prevista contro misurata.
  4. PIOGGIA CONTRO IL RADAR, metodo 1 «istante per istante»: la mappa di ogni ora del radar (prodotto SRT1 della
     Protezione Civile) sovrapposta alla mappa della stessa ora di ogni modello. Quattro voti:
       - ESTENSIONE: quanto territorio bagnato prevede il modello rispetto al radar;
       - DISTRIBUZIONE: se la pioggia cade nei posti giusti (Fractions Skill Score su quadrati di ~15 km, a tre soglie,
         piu' l'errore di posizione misurato spostando la mappa finche' combacia col radar);
       - INTENSITA': forza dei rovesci (95° percentile della pioggia dove piove) e quantita' totale;
       - TEMPISTICHE: la stessa previsione spostata di 1, 2, 3 ore prima e dopo: se combacia meglio, il modello e' in
         anticipo o in ritardo.
  5. PIOGGIA CONTRO IL RADAR, metodo 2 «accumuli»: accumulo di 6 ore (finestre 00-06-12-18 UTC) e di 24 ore (da
     mezzanotte a mezzanotte UTC) del radar contro quello del modello, con gli stessi quattro voti.

Situazioni (per capire quale modello va meglio in quali condizioni)
  Ogni ora e ogni finestra vengono classificate guardando il RADAR (cioe' quello che e' successo davvero):
    asciutto · piogge deboli e sparse · rovesci e temporali isolati · temporali organizzati (linee, sistemi) ·
    perturbazione estesa · perturbazione estesa con nuclei intensi
  piu' alcune etichette che si sommano: pioggia sui rilievi (orografica), vento forte, calma notturna (inversioni),
  nebbia. Stagione e scadenza (primo, secondo, terzo giorno) si scelgono nella pagina.

Voti: ogni misura diventa un voto 0-100 con la stessa formula per tutti; il voto generale e' la media pesata dei voti
disponibili (temperatura 20, vento 10, che tempo fa 15, pioggia alle stazioni 15, radar istante 20, radar accumuli 20).

Storico: ogni giorno viene sommato e conservato (dopo 60 giorni i giorni vengono riuniti per mese, per sempre).
Le pagelle calcolate prima di questo sistema restano nel file «precedente.json», con le loro classifiche.

  python -m motore.unica --storico DIR [--radar radar_orario.npz] [--osservazioni osservazioni.csv]
"""
import argparse
import csv
import gzip
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

from motore import verifica as V

# ----------------------------------------------------------------------------------------------- modelli
_FORZ = {"ecmwf": "ECMWF", "gfs": "GFS", "iconeu": "ICON-EU", "icon2i": "ICON-2I"}
# f2 sospesa a fine novembre 2026 (resta per i voti passati); f5, f6, g1 nuove prove
_FIS = {"f1": "WSM6 + MYNN", "f2": "Thompson + YSU", "f3": "Morrison + ACM2", "f4": "Thompson + MYJ",
        "f5": "WSM6 + MYNN + cumuli a 3 km", "f6": "WSM6 + ACM2", "g1": "WSM6 + MYNN, dominio esterno grande"}
REGISTRO = {
    "icon2i": ("ICON-2I", "2 km", "Modelli", 2.2), "arome": ("AROME", "1,3 km", "Modelli", 1.3),
    "ecmwf": ("ECMWF IFS", "9 km", "Modelli", 9), "ukmo": ("UKMO", "10 km", "Modelli", 10),
    "iconeu": ("ICON-EU", "7 km", "Modelli", 7), "gfs": ("GFS", "13 km", "Modelli", 13),
    "multi": ("Multimodello", "media pesata", "Multimodello", 2), "media": ("Media semplice", "pesi uguali", "Multimodello", 2),
    "wrf3km": ("WRF 3 km · ECMWF", "3 km", "WRF Nerola", 3), "wrf1km": ("WRF 1 km · ECMWF", "1 km", "WRF Nerola", 1),
    "wrf333m": ("WRF 333 m · ECMWF", "333 m", "WRF Nerola", 0.33),
    "wrfcorretto": ("WRF corretto", "corretto con stazioni e radar", "WRF Nerola", 1),
}
for _k, _n in _FORZ.items():
    if _k != "ecmwf":
        REGISTRO[f"wrf{_k}3km"] = (f"WRF 3 km · {_n}", "3 km", "WRF Nerola", 3)
        REGISTRO[f"wrf{_k}1km"] = (f"WRF 1 km · {_n}", "1 km", "WRF Nerola", 1)
    for _f, _d in _FIS.items():
        REGISTRO[f"wrffis_{_k}_{_f}"] = (f"WRF 3 km · {_n} · {_f.upper()}", _d, "WRF varianti di fisica", 3)

ALTA_RIS = 3.5         # km: sotto questa maglia un modello e' «ad alta risoluzione»


def canonico(k):
    """Stesso modello, nomi diversi nei vari file: un nome solo."""
    k = k.split("@")[0]
    if k.startswith("fis_"):
        return "wrf" + k
    return None if k in ("wrf",) else k


def info(k):
    r = REGISTRO.get(k)
    if r:
        return {"id": k, "nome": r[0], "ris": r[1], "gruppo": r[2], "alta_ris": r[3] <= ALTA_RIS}
    return {"id": k, "nome": k, "ris": "", "gruppo": "Altri", "alta_ris": False}


# ---------------------------------------------------------------------------------------------- costanti
BANDE = (("0-24", 0, 24), ("24-48", 24, 48), ("48-72", 48, 72))
SIT_PIOGGIA = ("asciutto", "deboli", "isolati", "organizzati", "estesa", "estesa_intensa")
ETICHETTE = ("orografica", "vento_forte", "calma_notturna", "nebbia")
NOMI_SIT = {"tutte": "Tutte le situazioni", "asciutto": "Tempo asciutto (falsi allarmi)",
            "deboli": "Piogge deboli e sparse", "isolati": "Rovesci e temporali isolati",
            "organizzati": "Temporali organizzati (linee, sistemi)", "estesa": "Perturbazione estesa",
            "estesa_intensa": "Perturbazione estesa con nuclei intensi", "orografica": "Piogge sui rilievi",
            "vento_forte": "Vento forte", "calma_notturna": "Notti calme (inversioni)", "nebbia": "Nebbia e foschia"}
# soglie per durata: bagnato, intenso, soglie del FSS
DURATE = {"R1": dict(ore=1, bagnato=0.5, intenso=10.0, soglie=(0.5, 2.0, 5.0), lag=(-3, -2, -1, 1, 2, 3), passo_lag=25.0),
          "R6": dict(ore=6, bagnato=1.0, intenso=20.0, soglie=(1.0, 5.0, 15.0), lag=(-6, -3, 3, 6), passo_lag=12.5),
          "R24": dict(ore=24, bagnato=1.0, intenso=40.0, soglie=(1.0, 10.0, 30.0), lag=(-12, -6, 6, 12), passo_lag=100 / 24)}
SCALA = 7                      # celle della griglia comune: ~15 km
PESI = {"T": 20, "V": 10, "CT": 15, "PST": 15, "R1": 20, "RACC": 20}
NOMI_PARTI = {"T": "Temperatura", "V": "Vento", "CT": "Che tempo fa", "PST": "Pioggia alle stazioni",
              "R1": "Radar ora per ora", "RACC": "Radar accumuli 6 e 24 ore"}
RIF_ERR = {"T": 2.0, "V": 7.0}         # errore che vale 50: 2 gradi, 7 km/h
GIORNI_DETTAGLIO = 60          # poi i giorni si riuniscono per mese
GIORNI_RICALCOLO_STAZIONI = 8  # i confronti con le stazioni si ricalcolano per questi giorni (dati completi)
# campi dei contatori (liste, per tenere piccolo il file)
CR = ["n", "voto", "est", "dis", "int", "tem", "nt", "fss", "lnA", "pf", "po", "lag", "np", "dx", "dy", "dd", "nw"]
CS = {"T": ["n", "sae", "se"], "V": ["n", "sae", "se"], "PST": ["n", "sae", "se", "a", "b", "c", "pf", "po"],
      "CT": ["nc", "cok", "np", "pok"]}


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%MZ")


def ora(s):
    return V.ora(s)


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def banda_di(h):
    return next((b for b, a, c in BANDE if a < h <= c or (a == 0 and h == 0)), None)


def vuoto(campi):
    return [0.0] * len(campi)


# ----------------------------------------------------------------------------------- «che tempo fa» comune
GRUPPI = ["sereno o poco nuvoloso", "nuvoloso o coperto", "nebbia o foschia", "pioggia", "neve"]
GRUPPO_METAR = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3, 10: 4}


def classe_comune(nub, T, Td, p_ora):
    """Il tempo di un'ora con la stessa formula per tutti i modelli (gruppo 0-4 come GRUPPI). None se mancano dati."""
    if T is None or p_ora is None:
        return None
    if p_ora >= 0.1 and T <= 0.5:
        return 4
    if p_ora >= 0.3:
        return 3
    if Td is not None:
        rh = 100 * math.exp(17.625 * Td / (243.04 + Td)) / math.exp(17.625 * T / (243.04 + T))
        if p_ora >= 0.05 and rh >= 90 and (nub is None or nub >= 60):
            return 3                                     # pioviggine
        vis = 1609.0 * 6000.0 * max((T - Td) * 1.8, 0) / max(rh, 1) ** 1.75
        if rh >= 99.5:
            vis = 400.0
        if vis < 5000 and (nub is None or nub >= 30):
            return 2
    if nub is None:
        return None
    return 0 if nub < 45 else 1


def classe_pioggia_ora(p):
    return 0 if p < 0.1 else 1 if p < 2 else 2 if p < 6 else 3


# ------------------------------------------------------------------------------------ misure sulle mappe
def _fss(f, o, m, soglia, pm):
    bf = ((f >= soglia) & m).astype("float32")
    bo = ((o >= soglia) & m).astype("float32")
    if bf.sum() == 0 and bo.sum() == 0:
        return None
    ff = ndimage.uniform_filter(bf, SCALA, mode="constant") / np.maximum(pm, 1e-6)
    fo = ndimage.uniform_filter(bo, SCALA, mode="constant") / np.maximum(pm, 1e-6)
    ok = pm > 0.5
    den = np.mean(ff[ok] ** 2) + np.mean(fo[ok] ** 2)
    return float(1 - np.mean((ff[ok] - fo[ok]) ** 2) / den) if den > 0 else None


def fss_medio(f, o, m, soglie):
    pm = ndimage.uniform_filter(m.astype("float32"), SCALA, mode="constant")
    vv = [x for x in (_fss(f, o, m, s, pm) for s in soglie) if x is not None]
    return float(np.mean(vv)) if vv else None


def voto_rapporto(r, limite=4.0):
    """1 = perfetto; 4 volte di piu' o di meno = 0."""
    return 100.0 * max(0.0, 1.0 - abs(math.log(r)) / math.log(limite))


def misura(f, o, campi_lag, d):
    """Confronto di una mappa prevista f con quella del radar o. campi_lag: {spostamento in ore: mappa prevista}.
    Ritorna il contatore di un caso (lista CR) o None se il confronto non dice nulla."""
    m = np.isfinite(f) & np.isfinite(o)
    if m.sum() < 500:
        return None
    f0, o0 = np.where(m, f, 0.0), np.where(m, o, 0.0)
    bag = d["bagnato"]
    af, ao = float((f0 >= bag)[m].mean()), float((o0 >= bag)[m].mean())
    if af < 0.002 and ao < 0.002:
        return None                                       # asciutto per tutti e due: non dice nulla
    c = vuoto(CR)
    c[0] = 1
    # estensione
    eps = 0.005
    ra = (af + eps) / (ao + eps)
    est = voto_rapporto(ra)
    c[CR.index("lnA")] = math.log(ra)
    # distribuzione
    fs = fss_medio(f0, o0, m, d["soglie"]) or 0.0
    dis = 100 * fs
    if af >= 0.01 and ao >= 0.01:
        sp = V.spostamento(np.where(m, f, np.nan), np.where(m, o, np.nan))
        if sp is not None:
            dd = math.hypot(*sp)
            dis = 0.75 * dis + 0.25 * 100 * max(0.0, 1 - dd / 60.0)
            c[CR.index("np")] = 1
            c[CR.index("dx")], c[CR.index("dy")], c[CR.index("dd")] = sp[0], sp[1], dd
    # intensita': picchi (95° percentile dove piove) e quantita' totale
    pf, po = float(f0[m].sum()), float(o0[m].sum())
    if af >= 0.002 and ao >= 0.002:
        qf = float(np.percentile(f0[m & (f0 >= bag)], 95))
        qo = float(np.percentile(o0[m & (o0 >= bag)], 95))
        inten = 0.5 * voto_rapporto((qf + 0.1) / (qo + 0.1)) + 0.5 * voto_rapporto((pf + 1) / (po + 1))
    else:
        inten = 0.0
    # tempistiche: la previsione spostata nel tempo combacia meglio?
    tem, lag_m = None, 0
    migliore = fs
    for lag, fl in sorted(campi_lag.items(), key=lambda x: abs(x[0])):
        ml = np.isfinite(fl) & m
        if ml.sum() < 500:
            continue
        v = fss_medio(np.where(ml, fl, 0.0), np.where(ml, o, 0.0), ml, d["soglie"])
        if v is not None and v > migliore + 0.02:
            migliore, lag_m = v, lag
    if migliore >= 0.15 and campi_lag:
        tem = max(0.0, 100 - d["passo_lag"] * abs(lag_m))
        c[CR.index("nt")] = 1
        c[CR.index("tem")] = tem
        c[CR.index("lag")] = lag_m
    parti = [est, dis, inten] + ([tem] if tem is not None else [])
    c[CR.index("voto")] = float(np.mean(parti))
    c[CR.index("est")], c[CR.index("dis")], c[CR.index("int")] = est, dis, inten
    c[CR.index("fss")] = fs
    c[CR.index("pf")], c[CR.index("po")] = pf, po
    c[CR.index("nw")] = 1 if ao >= 0.002 else 0
    return c


_DEM = {}


def dem_comune():
    """Quota del terreno sulla griglia comune (dal DEM a 500 m del Lazio)."""
    if "z" not in _DEM:
        p = Path(__file__).resolve().parent.parent / "dati" / "dem_lazio_500m.npz"
        try:
            d = np.load(p)
            _DEM["z"] = V.su_griglia(d["lat"], d["lon"], [d["z"]], massima_km=3)[0]
        except Exception:  # noqa: BLE001
            _DEM["z"] = None
    return _DEM["z"]


def situazione(o, d):
    """Tipo di tempo dalla mappa del radar (pioggia di 1, 6 o 24 ore). Ritorna (classe, [etichette]) o (None, [])."""
    m = np.isfinite(o)
    if m.sum() < 500:
        return None, []
    o0 = np.where(m, o, 0.0)
    bag = o0 >= d["bagnato"]
    area = float(bag[m].mean())
    if area < 0.01:
        return "asciutto", []
    forte = o0 >= d["intenso"]
    vol = float(o0[bag].sum())
    frac_conv = float(o0[forte].sum()) / vol if vol > 0 else 0.0
    lab, n = ndimage.label(bag)
    dim = np.bincount(lab.ravel())[1:] if n else np.array([0])
    grande = float(dim.max()) / float(m.sum()) if n else 0.0
    intensa = frac_conv >= 0.25 or float(forte[m].mean()) >= 0.004
    if area >= 0.25:
        cl = "estesa_intensa" if intensa else "estesa"
    elif intensa:
        cl = "organizzati" if (grande >= 0.04 or area >= 0.10) else "isolati"
    else:
        cl = "estesa" if grande >= 0.15 else "deboli"
    et = []
    z = dem_comune()
    if z is not None and area >= 0.03:
        alto, basso = (z >= 600) & m, (z < 300) & m & np.isfinite(z)
        if alto.sum() > 50 and basso.sum() > 50:
            ra = (float(o0[alto].mean()) + 0.05) / (float(o0[basso].mean()) + 0.05)
            if ra >= 1.8:
                et.append("orografica")
    return cl, et


# ------------------------------------------------------------------------------------------ stato
def carica_stato(cart):
    p = cart / "stato.json.gz"
    if p.exists():
        with gzip.open(p, "rt") as f:
            return json.load(f)
    return {"giorni": {}, "mesi": {}, "fatti": {}, "situazioni": {}}


def salva_stato(cart, st):
    cart.mkdir(parents=True, exist_ok=True)
    with gzip.open(cart / "stato.json.gz", "wt") as f:
        json.dump(st, f, separators=(",", ":"))


def aggiungi(dest, c):
    if not dest:
        dest.extend(c)
        return
    for i, x in enumerate(c):
        dest[i] += x


def nodo(st, giorno, modello, banda, sit, metodo, campi):
    g = st["giorni"].setdefault(giorno, {}).setdefault(modello, {}).setdefault(banda, {}).setdefault(sit, {})
    return g.setdefault(metodo, vuoto(campi))


def compatta(st, adesso):
    """I giorni piu' vecchi di GIORNI_DETTAGLIO si sommano nel loro mese (cosi' lo storico non si perde mai)."""
    limite = (adesso - timedelta(days=GIORNI_DETTAGLIO)).strftime("%Y-%m-%d")
    for g in [g for g in st["giorni"] if g < limite]:
        mese = st["mesi"].setdefault(g[:7], {})
        for mod, bb in st["giorni"].pop(g).items():
            for b, ss in bb.items():
                for s, mm in ss.items():
                    for met, c in mm.items():
                        dest = mese.setdefault(mod, {}).setdefault(b, {}).setdefault(s, {}).setdefault(met, [])
                        aggiungi(dest, c)
    lim_f = (adesso - timedelta(days=6)).strftime("%Y%m%d")
    for k in [k for k in st["fatti"] if k.split("__")[-1][:8] < lim_f]:
        st["fatti"].pop(k)
    lim_s = (adesso - timedelta(days=400)).strftime("%Y-%m-%dT%H")
    st["situazioni"] = {t: v for t, v in st["situazioni"].items() if t >= lim_s}


# ------------------------------------------------------------------------------ confronto con il radar
def radar_somma(per_ora, fine, ore):
    x = [per_ora.get(fine - timedelta(hours=k)) for k in range(ore)]
    if any(v is None for v in x):
        return None
    s = np.stack(x)
    out = s.sum(axis=0)
    out[~np.isfinite(s).all(axis=0)] = np.nan
    return out


class Corsa:
    """Una corsa archiviata (pioggia oraria fino a 48 ore, di 3 ore fino a 72 ore) sulla griglia comune."""

    def __init__(self, percorso):
        z = np.load(percorso)
        self.chiave = str(z["chiave"])
        self.corsa = ora(str(z["corsa"]))
        self.f3 = {ora(str(t)): k for k, t in enumerate(z["fini"])}
        self.f1 = {ora(str(t)): k for k, t in enumerate(z["fini1"])} if "fini1" in z.files else {}
        self.z = z
        self._c = {}

    def ora1(self, t):
        if t not in self.f1:
            return None
        if ("1", t) not in self._c:
            self._c[("1", t)] = V.espandi(self.z["mm1"][self.f1[t]])
        return self._c[("1", t)]

    def tre(self, t):
        if t not in self.f3:
            return None
        if ("3", t) not in self._c:
            self._c[("3", t)] = V.espandi(self.z["mm"][self.f3[t]])
        return self._c[("3", t)]

    def somma(self, fine, ore):
        """Accumulo previsto di 'ore' ore che finisce a 'fine' (da finestre orarie o di 3 ore)."""
        if ore == 1:
            return self.ora1(fine)
        if ore % 3 == 0:
            pezzi = [self.tre(fine - timedelta(hours=3 * k)) for k in range(ore // 3)]
            if all(p is not None for p in pezzi):
                return np.sum(pezzi, axis=0)
        pezzi = [self.ora1(fine - timedelta(hours=k)) for k in range(ore)]
        if all(p is not None for p in pezzi):
            return np.sum(pezzi, axis=0)
        return None


def finestre(corsa, metodo):
    """Le finestre da verificare di una corsa per un metodo: (fine, ore di anticipo)."""
    d = DURATE[metodo]
    out = []
    if metodo == "R1":
        for t in sorted(corsa.f1):
            out.append(t)
    elif metodo == "R6":
        for t in sorted(corsa.f3):
            if t.hour % 6 == 0:
                out.append(t)
    else:
        for t in sorted(corsa.f3):
            if t.hour == 0:
                out.append(t)
    return [(t, int(round((t - corsa.corsa).total_seconds() / 3600))) for t in out if (t - corsa.corsa) >= timedelta(hours=d["ore"])]


# --------------------------------------------------------------- memoria zona per zona (per il WRF corretto)
CANDIDATI_CORRETTO = ("wrf3km", "wrf1km", "wrfgfs3km", "wrfgfs1km", "wrficoneu3km", "wrficoneu1km",
                      "wrficon2i3km", "wrficon2i1km")
MEZZA_VITA_GIORNI = 30.0                 # gli errori vecchi contano sempre meno: la correzione si adatta man mano
BIN = np.concatenate([[0.0], np.logspace(-1, 2.2, 40)])     # classi di intensita' oraria (mm), per l'intensita'
ZONE_QUOTA = ((-1e9, 300), (300, 800), (800, 1e9))          # pianura e costa, colline, montagna


class Celle:
    """Per ogni WRF candidato: pioggia prevista e vista dal radar sommate cella per cella (finestre di 6 ore, divise
    per il tipo di tempo che il modello PREVEDEVA) e istogrammi dell'intensita' oraria per fascia di quota."""

    def __init__(self, cart, mod, adesso):
        self.p = Path(cart) / "celle" / f"{mod}.npz"
        self.adesso = adesso
        shape = (V.NY, V.NX)
        k = len(SIT_PIOGGIA) + 1
        if self.p.exists():
            z = np.load(self.p)
            self.sf, self.so, self.n = z["sf"], z["so"], z["n"]
            self.hf, self.ho = z["hf"], z["ho"]
            self.t = ora(str(z["t"]))
            self.decadi()
        else:
            self.sf, self.so = np.zeros((k,) + shape, np.float32), np.zeros((k,) + shape, np.float32)
            self.n = np.zeros((k,) + shape, np.float32)
            self.hf = np.zeros((len(ZONE_QUOTA), len(BIN)), np.float64)
            self.ho = np.zeros((len(ZONE_QUOTA), len(BIN)), np.float64)
            self.t = adesso

    def decadi(self):
        g = (self.adesso - self.t).total_seconds() / 86400
        if g > 0:
            f = 0.5 ** (g / MEZZA_VITA_GIORNI)
            for a in (self.sf, self.so, self.n, self.hf, self.ho):
                a *= f
            self.t = self.adesso

    def finestra6(self, f, o):
        sit, _ = situazione(f, DURATE["R6"])
        ok = np.isfinite(f) & np.isfinite(o)
        for i in (0, 1 + SIT_PIOGGIA.index(sit) if sit else None):
            if i is None:
                continue
            self.sf[i] += np.where(ok, f, 0)
            self.so[i] += np.where(ok, o, 0)
            self.n[i] += ok

    def ora1(self, f, o):
        z = dem_comune()
        ok = np.isfinite(f) & np.isfinite(o)
        for k, (a, b) in enumerate(ZONE_QUOTA):
            m = ok & (z >= a) & (z < b) if z is not None else ok
            if m.sum() < 20:
                continue
            self.hf[k] += np.bincount(np.digitize(f[m], BIN) - 1, minlength=len(BIN))[:len(BIN)]
            self.ho[k] += np.bincount(np.digitize(o[m], BIN) - 1, minlength=len(BIN))[:len(BIN)]

    def salva(self):
        self.p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.p, sf=self.sf, so=self.so, n=self.n, hf=self.hf, ho=self.ho, t=iso(self.t))


def radar(storico, per_ora, adesso, st):
    """Confronta con il radar tutte le finestre nuove di tutte le corse archiviate (una volta sola ciascuna)."""
    arch = Path(storico) / "verifica" / "archivio"
    ultima_oss = max(per_ora) if per_ora else None
    nuovi = 0
    if ultima_oss is None:
        return 0
    cache_sit = {}
    celle = {}
    for fp in sorted(arch.glob("*.npz")):
        try:
            c = Corsa(fp)
        except Exception:  # noqa: BLE001
            continue
        mod = canonico(c.chiave)
        if not mod:
            continue
        fatti = set(st["fatti"].get(fp.stem, []))
        for met, d in DURATE.items():
            for fine, h in finestre(c, met):
                chiave = f"{met}:{fine:%Y%m%d%H}"
                if chiave in fatti or fine > ultima_oss or h > 72:
                    continue
                o = radar_somma(per_ora, fine, d["ore"])
                if o is None:
                    if adesso - fine > timedelta(days=7):
                        fatti.add(chiave)             # il radar di quel momento non c'e' piu': non riproviamo
                    continue
                fatti.add(chiave)
                f = c.somma(fine, d["ore"])
                if f is None:
                    continue
                lag = {}
                for l in d["lag"]:
                    x = c.somma(fine + timedelta(hours=l), d["ore"])
                    if x is not None:
                        lag[l] = x
                k_sit = (met, fine)
                if k_sit not in cache_sit:
                    cache_sit[k_sit] = situazione(o, d)
                sit, etich = cache_sit[k_sit]
                if sit is None:
                    continue
                if mod in CANDIDATI_CORRETTO and h <= 48:
                    if mod not in celle:
                        celle[mod] = Celle(Path(storico) / "unica", mod, adesso)
                    if met == "R6":
                        celle[mod].finestra6(f, o)
                    elif met == "R1":
                        celle[mod].ora1(f, o)
                cont = misura(f, o, lag, d)
                if cont is None:
                    continue
                b = banda_di(h)
                if b is None:
                    continue
                giorno = (fine - timedelta(hours=1)).strftime("%Y-%m-%d")
                for s in ["tutte", sit] + etich:
                    aggiungi(nodo(st, giorno, mod, b, s, met, CR), cont)
                nuovi += 1
        st["fatti"][fp.stem] = sorted(fatti)
    for c in celle.values():
        c.salva()
    return nuovi


# ------------------------------------------------------------------------------ confronto con le stazioni
def leggi_oss(percorso, adesso, giorni):
    lim = iso(adesso - timedelta(days=giorni + 1))[:13]
    oss, staz = {}, {}
    with open(percorso, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            h = r["ora"][:13]
            if h < lim:
                continue
            oss[(r["id"], h)] = (num(r.get("T")), num(r.get("Td")), num(r.get("V")), num(r.get("P")), num(r.get("C")))
            staz[r["id"]] = (num(r.get("lat")), num(r.get("lon")))
    return oss, staz


def etichette_ora(oss, situazioni):
    """Etichette di ogni ora dalle osservazioni: vento forte, calma notturna, nebbia (piu' la classe del radar)."""
    per_ora = {}
    for (sid, h), (T, Td, Vv, P, C) in oss.items():
        x = per_ora.setdefault(h, [[], []])
        if Vv is not None:
            x[0].append(Vv)
        if C is not None:
            x[1].append(int(C))
    out = {}
    for h, (vv, cc) in per_ora.items():
        et = []
        if len(vv) >= 5:
            med = float(np.median(vv))
            if med >= 22:
                et.append("vento_forte")
            hh = int(h[11:13])
            if (hh >= 18 or hh <= 5) and med <= 5 and not any(c >= 6 for c in cc):
                et.append("calma_notturna")
        if len(cc) >= 3 and sum(c in (4, 5) for c in cc) / len(cc) >= 0.2:
            et.append("nebbia")
        s = situazioni.get(h)
        if s:
            et = [s[0]] + [e for e in s[1:] if e == "orografica"] + et
        out[h] = et
    return out


WRF_MS = ("wrf", "fis_")


def file_punti(storico, adesso, giorni):
    """File delle previsioni nei punti delle stazioni (archiviate da motore/pagella.py): (modello, percorso)."""
    lim = (adesso - timedelta(days=giorni + 3)).strftime("%Y%m%dT%H")
    base = Path(storico) / "previsioni"
    for cart in sorted(base.glob("*")):
        if "@" in cart.name and (base / cart.name.split("@")[0]).exists():
            continue                    # l'area Italia serve solo ai modelli che non hanno il Lazio (ICON-EU)
        mod = canonico(cart.name)
        if not mod:
            continue
        for p in sorted(cart.glob("*.csv.gz")):
            if p.name[:11] >= lim:
                yield mod, p


def stazioni(storico, oss, adesso, st):
    """Ricalcola i confronti con le stazioni degli ultimi giorni (i giorni piu' vecchi restano come sono)."""
    giorni = GIORNI_RICALCOLO_STAZIONI
    da = (adesso - timedelta(days=giorni)).strftime("%Y-%m-%d")
    oggi = adesso.strftime("%Y-%m-%dT%H")
    etich = etichette_ora(oss, st["situazioni"])
    nuovi = {}                                    # giorno -> modello -> banda -> sit -> metodo -> contatore
    for mod, fp in file_punti(storico, adesso, giorni):
        serie = {}                                # stazione -> {valido: (ore, pioggia, passo)}
        with gzip.open(fp, "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                sid = r["id"]
                if sid.startswith("R") and sid[1:5].isdigit():
                    continue                      # punti del radar: niente stazione
                valido = r["valido"][:13]
                if valido > oggi or valido[:10] < da:
                    continue
                h = int(r["ore"])
                b = banda_di(h)
                if b is None:
                    continue
                passo = int(r.get("passo") or 1)
                pv = num(r.get("P"))
                if pv is not None and h > 0:
                    serie.setdefault(sid, {})[valido] = (h, pv, passo)
                o = oss.get((sid, valido))
                if not o:
                    continue
                sits = ["tutte"] + etich.get(valido, [])
                T, Vv = num(r.get("T")), num(r.get("V"))
                if Vv is not None and mod.startswith(WRF_MS) and r.get("uV") != "kmh":
                    Vv *= 3.6                       # le righe vecchie dei WRF avevano il vento in m/s
                nodi = nuovi.setdefault(valido[:10], {}).setdefault(mod, {}).setdefault(b, {})
                for var, p, ob in (("T", T, o[0]), ("V", Vv, o[2])):
                    if p is None or ob is None:
                        continue
                    e = p - ob
                    for s in sits:
                        c = nodi.setdefault(s, {}).setdefault(var, vuoto(CS[var]))
                        c[0] += 1
                        c[1] += abs(e)
                        c[2] += e
                # che tempo fa (formula comune): gruppo del tempo con i METAR, classe di pioggia con tutte le stazioni
                G = num(r.get("G"))
                if G is not None:
                    for s in sits:
                        c = nodi.setdefault(s, {}).setdefault("CT", vuoto(CS["CT"]))
                        if o[4] is not None:
                            c[0] += 1
                            c[1] += int(G) == GRUPPO_METAR.get(int(o[4]), -1)
                        if o[3] is not None:
                            # piove o no in quell'ora, secondo il «che tempo fa» del modello e secondo la stazione
                            bp, bo = int(G) >= 3, o[3] >= 0.3
                            if bp or bo:
                                c[2] += 1
                                c[3] += bp and bo
        # pioggia in 3 ore (finestre che finiscono a 00, 03, ... UTC), prevista contro misurata
        for sid, s in serie.items():
            for valido, (h, _, passo) in s.items():
                if int(valido[11:13]) % 3:
                    continue
                fine = datetime.strptime(valido, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
                ore3 = [(fine - timedelta(hours=k)).strftime("%Y-%m-%dT%H") for k in range(3)]
                if passo == 3:
                    pf = s[valido][1]
                elif all(x in s for x in ore3):
                    pf = sum(s[x][1] for x in ore3)
                else:
                    continue
                po = [oss.get((sid, x)) for x in ore3]
                if any(x is None or x[3] is None for x in po):
                    continue
                po = sum(x[3] for x in po)
                b = banda_di(h)
                nodi = nuovi.setdefault(valido[:10], {}).setdefault(mod, {}).setdefault(b, {})
                e = pf - po
                for x in ["tutte"] + etich.get(valido, []):
                    c = nodi.setdefault(x, {}).setdefault("PST", vuoto(CS["PST"]))
                    c[0] += 1
                    c[1] += abs(e)
                    c[2] += e
                    c[3] += pf >= 1 and po >= 1
                    c[4] += pf >= 1 > po
                    c[5] += pf < 1 <= po
                    c[6] += pf
                    c[7] += po
    # sostituisco i metodi delle stazioni nei giorni ricalcolati (il radar resta com'e')
    for giorno, mm in nuovi.items():
        g = st["giorni"].setdefault(giorno, {})
        for mod in list(g):
            for b in g[mod].values():
                for s in b.values():
                    for met in CS:
                        s.pop(met, None)
        for mod, bb in mm.items():
            for b, ss in bb.items():
                for s, met in ss.items():
                    g.setdefault(mod, {}).setdefault(b, {}).setdefault(s, {}).update(
                        {k: [round(x, 3) for x in v] for k, v in met.items()})
    return len(nuovi)


# --------------------------------------------------------------------------------------------- voti
def voti_da(c_tot):
    """Dai contatori sommati ai voti 0-100 di ogni parte (None se i casi sono troppo pochi)."""
    out = {}
    for var in ("T", "V"):
        c = c_tot.get(var)
        if c and c[0] >= 30:
            mae, bias = c[1] / c[0], c[2] / c[0]
            out[var] = {"voto": round(100 * 0.5 ** (mae / RIF_ERR[var]), 1), "errore": round(mae, 2),
                        "tendenza": round(bias, 2), "casi": int(c[0])}
    c = c_tot.get("CT")
    if c and (c[0] >= 20 or c[2] >= 20):
        parti = []
        x = {"casi": int(c[0] + c[2])}
        if c[0] >= 20:
            x["cielo"] = round(100 * c[1] / c[0], 1)
            parti.append(x["cielo"])
        if c[2] >= 20:
            x["pioggia"] = round(100 * c[3] / c[2], 1)
            parti.append(x["pioggia"])
        x["voto"] = round(float(np.mean(parti)), 1)
        out["CT"] = x
    c = c_tot.get("PST")
    if c and c[0] >= 30:
        a, b, cc = c[3], c[4], c[5]
        csi = a / (a + b + cc) if a + b + cc >= 5 else None
        q = voto_rapporto((c[6] + 1) / (c[7] + 1))
        x = {"casi": int(c[0]), "errore": round(c[1] / c[0], 2), "quantita": round(100 * ((c[6] + 1) / (c[7] + 1) - 1))}
        if csi is not None:
            x["csi"] = round(100 * csi, 1)
            x["voto"] = round(0.6 * 100 * csi + 0.4 * q, 1)
        else:
            x["voto"] = round(q, 1)
        out["PST"] = x
    for met in DURATE:
        c = c_tot.get(met)
        if not c or c[0] < 3:
            continue
        n = c[0]
        g = lambda k: c[CR.index(k)]  # noqa: E731
        x = {"voto": round(g("voto") / n, 1), "casi": int(n), "estensione": round(g("est") / n, 1),
             "distribuzione": round(g("dis") / n, 1), "intensita": round(g("int") / n, 1),
             "area": round(100 * (math.exp(g("lnA") / n) - 1)), "quantita": round(100 * ((g("pf") + 1) / (g("po") + 1) - 1))}
        if g("nt"):
            x["tempistiche"] = round(g("tem") / g("nt"), 1)
            lm = g("lag") / g("nt")
            x["ritardo_ore"] = round(lm, 1)
        if g("np") >= 3:
            sx, sy = g("dx") / g("np"), g("dy") / g("np")
            x["posizione_km"] = round(g("dd") / g("np"))
            if math.hypot(sx, sy) >= 5:
                x["direzione"] = V.direzione(sx, sy)
        out[met] = x
    acc = [out[m] for m in ("R6", "R24") if m in out]
    if acc:
        n = sum(a["casi"] for a in acc)
        out["RACC"] = {"voto": round(sum(a["voto"] * a["casi"] for a in acc) / n, 1), "casi": n}
    parti = {k: out[k]["voto"] for k in PESI if k in out}
    if parti:
        tot = sum(PESI[k] for k in parti)
        out["voto"] = round(sum(PESI[k] * v for k, v in parti.items()) / tot, 1)
        out["completo"] = len(parti) == len(PESI)
        out["parti_presenti"] = len(parti)
    return out


def somma_periodo(st, da, bande=None, sit="tutte", stagione=None):
    """modello -> {metodo: contatore sommato} sui giorni >= da (e sui mesi interi se da e' il mese o prima)."""
    tot = {}

    def entra(chiave):
        if stagione is None:
            return True
        return int(chiave[5:7]) in STAGIONI[stagione]

    sorgenti = [(g, x) for g, x in st["giorni"].items() if g >= da and entra(g)]
    sorgenti += [(m, x) for m, x in st["mesi"].items() if m >= da[:7] and entra(m + "-01")]
    for _, gg in sorgenti:
        for mod, bb in gg.items():
            for b, ss in bb.items():
                if bande and b not in bande:
                    continue
                for met, c in (ss.get(sit) or {}).items():
                    dest = tot.setdefault(mod, {}).setdefault(met, [])
                    aggiungi(dest, c)
    return tot


STAGIONI = {"inverno": (12, 1, 2), "primavera": (3, 4, 5), "estate": (6, 7, 8), "autunno": (9, 10, 11)}


def classifica(tot):
    righe = []
    for mod, cc in tot.items():
        v = voti_da(cc)
        if "voto" in v:
            righe.append(dict(v, modello=mod))
    # prima i modelli confrontati su tutto (stazioni e radar), poi gli altri: cosi' il confronto e' equo
    for r in righe:
        r["confronto_pieno"] = all(k in r for k in ("T", "V", "PST", "R1", "RACC"))
    righe.sort(key=lambda x: (not x["confronto_pieno"], -x["voto"]))
    return righe


def leggera(righe):
    """Solo i voti (per le tabelle per situazione, scadenza e stagione: file piu' leggero)."""
    out = []
    for r in righe:
        x = {"modello": r["modello"], "voto": r["voto"], "pieno": r["confronto_pieno"]}
        for k in PESI:
            if k in r:
                x[k] = r[k]["voto"]
        for k in ("R1", "R6", "R24"):
            if k in r:
                x[k + "q"] = r[k]["quantita"]
                x[k + "c"] = r[k]["casi"]
        out.append(x)
    return out


def lezioni(st, adesso, modelli):
    """Frasi automatiche su cosa abbiamo imparato: come si comportano i modelli nelle diverse situazioni."""
    out = []
    da = (adesso - timedelta(days=365)).strftime("%Y-%m-%d")
    alta = {m for m in modelli if info(m)["alta_ris"]}
    for sit in SIT_PIOGGIA[1:]:
        tot = somma_periodo(st, da, ("0-24", "24-48"), sit)
        gruppi = {"alta": [0.0, 0.0, 0], "bassa": [0.0, 0.0, 0]}
        voti = []
        for mod, cc in tot.items():
            c = cc.get("R6") or cc.get("R1")
            if not c or c[0] < 3:
                continue
            gg = gruppi["alta" if mod in alta else "bassa"]
            gg[0] += c[CR.index("pf")]
            gg[1] += c[CR.index("po")]
            gg[2] += c[0]
            voti.append((c[CR.index("voto")] / c[0], mod, c[0]))
        frase = []
        for g, nome in (("alta", "ad alta risoluzione"), ("bassa", "globali ed europei")):
            pf, po, n = gruppi[g]
            if n >= 10 and po > 0:
                d = round(100 * (pf / po - 1))
                if abs(d) >= 15:
                    frase.append(f"i modelli {nome} prevedono in media il {abs(d)}% di pioggia {'in più' if d > 0 else 'in meno'} del radar")
                else:
                    frase.append(f"i modelli {nome} prevedono quantità di pioggia vicine al radar")
        if voti:
            voti.sort(reverse=True)
            migl = [info(m)["nome"] for _, m, n in voti[:3] if n >= 5]
            if migl:
                frase.append("i più precisi finora: " + ", ".join(migl))
        if frase:
            out.append({"situazione": sit, "nome": NOMI_SIT[sit], "testo": "; ".join(frase) + "."})
    return out


def riepilogo(st, adesso):
    modelli = sorted({m for gg in list(st["giorni"].values()) + list(st["mesi"].values()) for m in gg})
    out = {"aggiornato": iso(adesso), "modelli": [info(m) for m in modelli], "pesi": PESI, "parti": NOMI_PARTI,
           "situazioni": NOMI_SIT, "bande": [b for b, _, _ in BANDE], "periodi": {}, "per_situazione": {},
           "per_banda": {}, "per_stagione": {}}
    primo = min(list(st["giorni"]) + [m + "-01" for m in st["mesi"]] or [adesso.strftime("%Y-%m-%d")])
    out["dal"] = primo
    periodi = {"7": 7, "30": 30, "90": 90, "365": 365}
    for nome_p, gg in periodi.items():
        da = (adesso - timedelta(days=gg)).strftime("%Y-%m-%d")
        out["periodi"][nome_p] = classifica(somma_periodo(st, da, ("0-24", "24-48")))
        if nome_p == "7":
            continue
        out["per_situazione"][nome_p] = {s: leggera(classifica(somma_periodo(st, da, ("0-24", "24-48"), s)))
                                         for s in SIT_PIOGGIA + ETICHETTE}
        out["per_banda"][nome_p] = {b: leggera(classifica(somma_periodo(st, da, (b,)))) for b, _, _ in BANDE}
    da = (adesso - timedelta(days=365)).strftime("%Y-%m-%d")
    out["per_stagione"] = {s: leggera(classifica(somma_periodo(st, da, ("0-24", "24-48"), stagione=s))) for s in STAGIONI}
    out["lezioni"] = lezioni(st, adesso, modelli)
    # conteggio delle situazioni viste (ore classificate dal radar, ultimi 30 giorni)
    lim = (adesso - timedelta(days=30)).strftime("%Y-%m-%dT%H")
    cont = {}
    for t, s in st["situazioni"].items():
        if t >= lim:
            for x in s:
                cont[x] = cont.get(x, 0) + 1
    out["ore_per_situazione"] = cont
    # andamento giorno per giorno del voto generale (ultimi 60 giorni, prime 48 ore)
    and_ = {}
    for g in sorted(st["giorni"]):
        tot = {}
        for mod, bb in st["giorni"][g].items():
            for b in ("0-24", "24-48"):
                for met, c in ((bb.get(b) or {}).get("tutte") or {}).items():
                    aggiungi(tot.setdefault(mod, {}).setdefault(met, []), c)
        and_[g] = {}
        for mod, cc in tot.items():
            v = voti_da({k: c for k, c in cc.items()})
            # per un giorno solo bastano meno casi: voto di ogni parte con soglie basse
            if "voto" in v:
                and_[g][mod] = v["voto"]
    out["andamento"] = and_
    return out


# --------------------------------------------------------------------------------------- storico precedente
def precedente(storico, cart, adesso):
    """Fotografia (una volta sola) delle pagelle calcolate prima della pagella unica: non si perdono."""
    p = cart / "precedente.json"
    if p.exists():
        return
    ris = {"creato": iso(adesso), "nota": "Pagelle calcolate con i metodi precedenti alla pagella unica (fino al "
           + adesso.strftime("%d/%m/%Y") + "): restano consultabili cosi' come erano."}
    for nome, f in (("generale", "pagelle.json"), ("ottica", "ottica.json"), ("stazioni", "pagella.json")):
        fp = Path(storico) / f
        if not fp.exists():
            continue
        try:
            d = json.loads(fp.read_text())
        except Exception:  # noqa: BLE001
            continue
        if nome == "generale":
            ris["generale"] = {"classifica": d.get("classifica"), "pioggia_30": (d.get("pioggia") or {}).get("30"),
                               "eventi": d.get("eventi"), "andamento": d.get("andamento")}
        elif nome == "ottica":
            ris["ottica"] = d
        else:
            ris["stazioni"] = {"classifica": d.get("classifica"), "riepilogo": d.get("riepilogo"),
                               "radar": d.get("radar"), "giorni": d.get("giorni")}
    sv = Path(storico) / "verifica" / "stato.json"
    if sv.exists():
        try:
            ris["giornaliero_radar"] = json.loads(sv.read_text()).get("somme")
        except Exception:  # noqa: BLE001
            pass
    pw = Path(storico).parent / "pagella_wrf.json"
    if pw.exists():
        try:
            d = json.loads(pw.read_text())
            ris["che_tempo_fa_wrf"] = (d.get("tempo") or {}).get("pagella")
            ris["wrf_stazioni"] = (d.get("confronto_wrf") or {}).get("classifica")
        except Exception:  # noqa: BLE001
            pass
    cart.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ris, ensure_ascii=False, separators=(",", ":")))


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storico", required=True)
    ap.add_argument("--radar", default="")
    ap.add_argument("--osservazioni", default="")
    ap.add_argument("--adesso", default="", help="solo per le prove: AAAA-MM-GGTHH")
    a = ap.parse_args()
    adesso = ora(a.adesso) if a.adesso else datetime.now(timezone.utc)
    storico = Path(a.storico)
    cart = storico / "unica"
    precedente(storico, cart, adesso)
    st = carica_stato(cart)
    if a.radar and Path(a.radar).exists():
        per_ora = V.radar_3ore(a.radar)
        # ogni ora del radar ha la sua situazione (serve anche alle stazioni)
        for t, o in per_ora.items():
            k = t.strftime("%Y-%m-%dT%H")
            if k not in st["situazioni"]:
                sit, et = situazione(o, DURATE["R1"])
                if sit:
                    st["situazioni"][k] = [sit] + et
        print("confronti nuovi con il radar:", radar(storico, per_ora, adesso, st))
        salva_stato(cart, st)
    else:
        print("radar non disponibile")
    if a.osservazioni and Path(a.osservazioni).exists():
        oss, _ = leggi_oss(a.osservazioni, adesso, GIORNI_RICALCOLO_STAZIONI + 1)
        print("giorni ricalcolati con le stazioni:", stazioni(storico, oss, adesso, st))
    compatta(st, adesso)
    salva_stato(cart, st)
    ris = riepilogo(st, adesso)
    (storico / "unica.json").write_text(json.dumps(ris, ensure_ascii=False, separators=(",", ":")))
    for r in ris["periodi"]["30"][:12]:
        print(f"  {info(r['modello'])['nome']:32s} {r['voto']:5.1f}  " +
              " ".join(f"{k}={r[k]['voto']}" for k in PESI if k in r))
    for l in ris["lezioni"]:
        print("  -", l["nome"], ":", l["testo"])


if __name__ == "__main__":
    main()
