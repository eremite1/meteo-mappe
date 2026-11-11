"""
Multimodello: media pesata dei modelli su Lazio, Italia ed Europa.
  Lazio:  ICON-2I, AROME, ECMWF, UKMO (pesi dalla pagella del Lazio)
  Italia: ICON-2I, ECMWF, UKMO, ICON-EU (pesi dalla pagella "Italia", stazioni del centro Italia)
  Europa: ECMWF, UKMO, ICON-EU (stessi pesi dell'Italia)

Ogni modello pesa in base a quanto ha sbagliato di recente (pagella delle Mappe Lazio):
peso = 1 / errore medio², calcolato per grandezza (temperatura, umidità, vento, pioggia)
e per distanza dall'inizio della corsa (0-12, 12-24, 24-48, 48-72 ore).
Dove un modello non copre il territorio (AROME a sud) la media si fa con gli altri.

Uso: python -m motore.multi <area>
Scrive in uscita/ le mappe (<area>/<parametro>_hNNN.png), indice_parziale.json e i valori nei punti
delle stazioni (punti_<area>.csv media pesata, punti_<area>_uguali.csv media semplice, per la pagella).
"""
import os
import sys
import json
import time
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone, timedelta

import numpy as np
import requests

from . import genera as G
from .catalogo import PARAMETRI, MODELLI

AREE_MULTI = {
    "lazio": dict(componenti=["icon2i", "arome", "ecmwf", "ukmo", "gfs"], massimo=72, pagella="lazio"),
    "italia": dict(componenti=["icon2i", "ecmwf", "ukmo", "iconeu", "gfs"], massimo=168, pagella="italia"),
    "europa": dict(componenti=["ecmwf", "ukmo", "iconeu", "gfs"], massimo=168, pagella="italia"),
}
AREA = "lazio"       # area in elaborazione (impostata da esegui)
PASSO = 3            # ore tra una mappa e l'altra
URL_PAGELLA = "https://raw.githubusercontent.com/eremite1/meteo-mappe/pagella/pagella.json"
NOME = "Multimodello"

# parametro -> grandezza della pagella che decide i pesi ("G" = affidabilità generale)
PESO_DA = {
    "temperatura": "T", "delta_t24": "T", "percepita": "T", "humidex": "T", "t850": "T", "zero_termico": "T",
    "rugiada": "Td", "umidita": "Td",
    "vento": "V", "raffiche": "V",
    "pioggia": "P", "neve": "P",
    "nuvole": "G", "nuvole_basse": "G", "nuvole_medie": "G", "nuvole_alte": "G",
    "pressione": "G", "cape": "G", "quota_neve": "G",
    "pioggia_isobare": "P", "z500_t500": "G",
    "anom_t850": "T", "anom_z500": "G", "anom_pressione": "G", "anom_pioggia": "P",
    "pioggia_6h": "P", "pioggia_12h": "P", "pioggia_24h": "P", "grandine": "G",
}
FASCE = [(0, 12, "0-12 ore"), (12, 24, "12-24 ore"), (24, 48, "24-48 ore"), (48, 72, "48-72 ore"),
         (72, 120, "72-120 ore"), (120, 169, "120-168 ore")]
N_MINIMO = 30        # confronti minimi perché un errore della pagella sia considerato affidabile


# ---------------------------------------------------------------------------
# Pesi dalla pagella
# ---------------------------------------------------------------------------
def carica_pagella():
    try:
        r = requests.get(URL_PAGELLA, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        G.avviso(f"pagella non disponibile, pesi uguali: {e}")
        return None


def fascia_di(ore):
    return next((f[2] for f in FASCE if f[0] <= ore < f[1]), FASCE[-1][2])


def errore(pag, modello, var, ore):
    """Errore medio del modello per la grandezza e la fascia di ore (o None se pochi dati)."""
    if not pag:
        return None
    # pagella del Lazio (sezione principale) oppure quella "italia" (griglia Italia, con ICON-EU)
    sez = pag if _M.get("sezione", "lazio") == "lazio" else (pag.get("italia") or {})
    r = (sez.get("riepilogo") or {}).get(modello, {}).get(var) or {}
    # errore che cambia GRADUALMENTE con le ore: interpolo tra i centri delle fasce che hanno dati
    # (prima i pesi cambiavano di colpo al passaggio da una fascia all'altra, es. a +12 h o +24 h)
    punti = []
    for a, b, nome in FASCE:
        s = r.get(nome)
        if s and s.get("n", 0) >= N_MINIMO and s.get("mae"):
            punti.append(((a + b) / 2, s["mae"]))
    if len(punti) >= 2:
        xs, ys = zip(*punti)
        return float(np.interp(ore, xs, ys))
    f = fascia_di(ore)
    s = r.get(f)
    if s and s.get("n", 0) >= N_MINIMO and s.get("mae"):
        return s["mae"]
    # fascia senza dati (es. le prime corse): prendo la fascia più vicina che ne ha
    nomi = [x[2] for x in FASCE]
    i = nomi.index(f)
    for d in range(1, len(nomi)):
        for j in (i - d, i + d):                      # (prima la fascia precedente, più ricca di dati)
            if 0 <= j < len(nomi):
                s = r.get(nomi[j])
                if s and s.get("n", 0) >= N_MINIMO and s.get("mae"):
                    return s["mae"]
    return None


def pesi(pag, var, presenti):
    """presenti: {modello: ore dall'inizio della sua corsa} -> {modello: peso (somma 1)}."""
    def per_var(v):
        e = {m: errore(pag, m, v, h) for m, h in presenti.items()}
        noti = [x for x in e.values() if x]
        if not noti:
            return None
        rif = float(np.median(noti))
        return {m: 1.0 / max(x or rif, 0.05) ** 2 for m, x in e.items()}

    if var == "G":
        parti = [p for p in (per_var("T"), per_var("Td"), per_var("V")) if p]
        if parti:
            w = {m: float(np.mean([p[m] / sum(p.values()) for p in parti])) for m in presenti}
        else:
            w = None
    else:
        w = per_var(var)
    if not w:
        w = {m: 1.0 for m in presenti}
    s = sum(w.values())
    return {m: v / s for m, v in w.items()}


def dissolvenza(w, t):
    """Un modello che sta per finire (es. AROME dopo circa 42-51 ore) non sparisce di colpo:
    nelle sue ultime 9 ore il suo peso scende gradualmente. Pesi rinormalizzati a somma 1."""
    fine = _M.get("fine") or {}
    x = {}
    for m, v in w.items():
        if m in fine:
            rimaste = (fine[m] - t).total_seconds() / 3600
            v = v * float(np.clip((rimaste + PASSO) / (4 * PASSO), 0.25, 1.0))
        x[m] = v
    s = sum(x.values()) or 1.0
    return {m: v / s for m, v in x.items()}


SFUMA_CELLE = 20      # ~10 km: vicino al bordo del suo territorio un modello pesa sempre meno


def _rampa(ok):
    """1 all'interno del territorio del modello, scende a 0 avvicinandosi al bordo (niente tagli netti)."""
    if ok.all():
        return 1.0
    if not ok.any():
        return 0.0
    from scipy.ndimage import distance_transform_edt
    return np.clip(distance_transform_edt(ok) / SFUMA_CELLE, 0.0, 1.0)


def media(campi, w):
    """Media pesata cella per cella; dove un modello manca, si usano gli altri.
    Vicino al bordo del territorio di un modello (es. AROME) il suo peso sfuma gradualmente."""
    num = den = None
    for m, c in campi.items():
        ok = np.isfinite(c)
        r = _rampa(ok)
        x = np.where(ok, c, 0.0) * w[m] * r
        d = ok * w[m] * r
        num = x if num is None else num + x
        den = d if den is None else den + d
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


# ---------------------------------------------------------------------------
_M = {}
PUNTI_DA = {"T": "temperatura", "Td": "rugiada", "V": "vento", "P": "pioggia"}


# ---------------------------------------------------------------------------
# Risoluzione uniforme: la "forma" della mappa segue i pesi della pagella,
# il dettaglio fine (rilievi, piccole celle) viene sempre dai modelli ad alta risoluzione.
# Cosi' la nitidezza non cambia quando cambiano i pesi.
# ---------------------------------------------------------------------------
CELLA_KM = {"lazio": 0.5, "italia": 3.0, "europa": 20.0}
SCALA_KM = 14.0          # sotto questa scala comanda il dettaglio dei modelli fini (~ maglia del modello piu' grossolano)
ALTA_RIS_KM = 3.0        # modelli considerati "ad alta risoluzione" (ICON-2I, AROME)


def _ris_km(m):
    try:
        return float(MODELLI[m]["ris"].split()[0].replace(",", "."))
    except Exception:
        return 10.0


def _passa_basso(c, sg):
    """Media mobile gaussiana che ignora le celle senza dati."""
    from scipy.ndimage import gaussian_filter
    ok = np.isfinite(c)
    den = gaussian_filter(ok.astype(np.float32), sg, mode="nearest")
    num = gaussian_filter(np.where(ok, c, 0.0).astype(np.float32), sg, mode="nearest")
    return np.where(ok & (den > 1e-3), num / np.maximum(den, 1e-3), np.nan)


def media_armonica(campi, w, moltiplicativo=False):
    """Come media(), ma con la stessa nitidezza qualunque siano i pesi."""
    sg = SCALA_KM / 2.0 / CELLA_KM.get(AREA, 3.0)
    fini = [m for m in campi if _ris_km(m) <= ALTA_RIS_KM]
    if sg < 1.0 or not fini:
        return media(campi, w)                      # griglia gia' grossolana o nessun modello fine
    basso = {m: _passa_basso(c, sg) for m, c in campi.items()}
    forma = media(basso, w)                          # la forma generale: decisa dalla pagella
    alto = media({m: campi[m] for m in fini}, {m: 1.0 for m in fini})
    alto_b = media({m: basso[m] for m in fini}, {m: 1.0 for m in fini})
    if moltiplicativo:                               # pioggia/neve: il dettaglio e' un rapporto (mai negativo)
        e = 0.3
        rapporto = np.clip((alto + e) / (alto_b + e), 0.2, 4.0)
        out = forma * np.where(np.isfinite(rapporto), rapporto, 1.0)
    else:                                            # temperatura ecc.: il dettaglio si somma
        out = forma + np.where(np.isfinite(alto - alto_b), alto - alto_b, 0.0)
    if all(np.nanmin(c) >= 0 for c in campi.values() if np.isfinite(c).any()):
        out = np.clip(out, 0, None)
    return np.where(np.isfinite(forma), out, np.nan)


def _avvia():
    from .disegno import Disegnatore
    la, lo = _M["griglia"]
    _M["dis"] = Disegnatore(AREA, la, lo, _M["ombra"])


def _punti(t, h, medie):
    """Valori della media pesata e della media semplice nei punti delle stazioni (per la pagella)."""
    st = _M.get("stazioni") or []
    if not st or AREA not in ("lazio", "italia") or "T" not in medie:
        return []
    iy = np.array([x[1] for x in st])
    ix = np.array([x[2] for x in st])
    corsa = _M["corsa"].strftime("%Y-%m-%dT%H:%MZ")
    valido = t.strftime("%Y-%m-%dT%H:%MZ")
    z = _M["terreno"]["z"][iy, ix] if (_M.get("terreno") is not None and AREA == "lazio") else None
    righe = []
    for tipo, chiave in (("pesata", AREA), ("uguali", AREA + "_uguali")):
        val = {}
        for v in ("T", "Td", "V", "P"):
            c = medie.get(v, {}).get(tipo)
            val[v] = None if c is None else c[iy, ix]
        for n, (sid, _, _) in enumerate(st):
            T = val["T"][n]
            if not np.isfinite(T):
                continue
            fmt = lambda v, k=1: None if val[v] is None or not np.isfinite(val[v][n]) else round(float(val[v][n]), k)
            righe.append([chiave, corsa, valido, h, PASSO, sid, round(float(T), 1), fmt("Td"), fmt("V"),
                          fmt("P", 2) if h > 0 else None, None if z is None else round(float(z[n]))])
    return righe


def _fai(compito):
    t, h = compito
    lavori, pag = _M["lavori"], _M["pagella"]
    dis = _M["dis"]
    data, ora_utc = G.testo_ora(t)
    risultati, errori = [], []
    ctx, presenti = {}, {}
    for m, L in lavori.items():
        if t in [x[2] for x in L.istanti(AREA)]:
            ctx[m] = G.Contesto(L, AREA, t)
            presenti[m] = int(round((t - L.corsa).total_seconds() / 3600))
    if len(ctx) < 2:
        return risultati, errori, []
    medie = {}            # grandezza della pagella -> {"pesata": campo, "uguali": campo}
    for p in PARAMETRI:
        if p["id"] not in PESO_DA or AREA not in p["aree"]:
            continue
        if (AREA, p["id"]) in _M["scale"]:
            p = dict(p, stile=dict(p["stile"], livelli=_M["scale"][(AREA, p["id"])]), stile_europa=None)
        campi, cont, fre = {}, {}, {}
        for m, c in ctx.items():
            try:
                r = p["calcolo"](c)
            except Exception as e:
                if not isinstance(e, (G.Mancante, KeyError)):
                    errori.append(f"{m}/{p['id']}/+{h}h: {type(e).__name__}: {e}")
                continue
            if isinstance(r, dict):
                campo, contorni, frecce = r["campo"], r.get("contorni"), r.get("frecce")
            else:
                campo, contorni, frecce = r, None, None
            if campo is None or not np.isfinite(campo).any():
                continue
            campi[m] = campo
            if contorni is not None:
                cont[m] = contorni
            if frecce is not None:
                fre[m] = frecce
        if len(campi) < 2:
            continue
        w = pesi(pag, PESO_DA[p["id"]], {m: presenti[m] for m in campi})
        w = dissolvenza(w, t)
        try:
            campo = media_armonica(campi, w, moltiplicativo=PESO_DA[p["id"]] == "P" and not p["id"].startswith("anom"))
            for v, pid in PUNTI_DA.items():
                if pid == p["id"]:
                    medie[v] = {"pesata": campo, "uguali": media(campi, {m: 1.0 for m in campi})}
            contorni = None
            if cont:
                wc = {m: w[m] for m in cont}
                s = sum(wc.values())
                wc = {m: v / s for m, v in wc.items()}
                primo = next(iter(cont.values()))
                contorni = (media({m: c[0] for m, c in cont.items()}, wc), primo[1], primo[2])
            frecce = None
            if fre:
                wf = {m: w[m] for m in fre}
                s = sum(wf.values())
                wf = {m: v / s for m, v in wf.items()}
                frecce = (media({m: f[0] for m, f in fre.items()}, wf), media({m: f[1] for m, f in fre.items()}, wf))
            unita = p["unita"].replace("nelle ore precedenti", f"nelle {PASSO} ore precedenti")
            quote = " · ".join(f"{MODELLI[m]['nome']} {round(w[m] * 100)}%" for m in sorted(w, key=lambda k: -w[k]))
            info = f"Media pesata secondo la pagella: {quote} · ora italiana ({ora_utc})"
            nome = f"{p['id']}_h{h:03d}.png"
            dis.disegna(os.path.join(G.USCITA, AREA, nome), campo, p, NOME, data, f"+{h}h", info,
                        contorni=contorni, frecce=frecce, unita=unita)
            risultati.append((AREA, p["id"], nome, h, t.strftime("%Y-%m-%dT%H:%MZ"), {m: round(w[m], 3) for m in w}))
        except Exception as e:
            errori.append(f"multi/{p['id']}/+{h}h: {type(e).__name__}: {e}")
    try:
        righe = _punti(t, h, medie)
    except Exception as e:
        righe = []
        errori.append(f"punti/+{h}h: {type(e).__name__}: {e}")
    for L in lavori.values():
        L.libera(t)
    return risultati, errori, righe


def esegui(area="lazio", processi=None):
    global AREA
    AREA = area
    conf = AREE_MULTI[area]
    massimo = conf["massimo"]
    t0 = time.time()
    G.log(f"=== {NOME} {area} ===")
    if os.path.exists(G.CARTELLA_DATI):
        shutil.rmtree(G.CARTELLA_DATI)
    griglia = G.griglia_area(AREA)
    terreno = None
    if area == "lazio":
        dem = G.carica_dem()
        terreno = G.prepara_terreno(dem, *griglia) if dem is not None else None

    lavori = {}
    for m in conf["componenti"]:
        if AREA not in MODELLI[m]["aree"]:
            continue
        try:
            L = G.Lavoro(m, [AREA])
        except Exception as e:
            G.avviso(f"{m}: escluso dal multimodello ({e})")
            continue
        # nel multimodello ogni modello si usa ogni 3 ore, con la pioggia delle 3 ore precedenti
        L.cfg = dict(L.cfg, aree={AREA: (PASSO, min(massimo, MODELLI[m]["aree"][AREA][1]))})
        L.passo[AREA] = PASSO
        L.terreno = terreno
        G.log(f"{m}: corsa {L.corsa:%Y-%m-%d %H} UTC, scarico i dati...")
        L.scarica()
        L.prepara_cumulate()
        lavori[m] = L
    if len(lavori) < 2:
        raise RuntimeError("servono almeno due modelli per il multimodello")

    # la corsa di riferimento è la più recente; le ore partono da lì
    ref = max(L.corsa for L in lavori.values())
    tempi = sorted({x[2] for L in lavori.values() for x in L.istanti(AREA)})
    compiti = []
    for t in tempi:
        h = int(round((t - ref).total_seconds() / 3600))
        if 0 <= h <= massimo and h % PASSO == 0:
            if sum(t in [x[2] for x in L.istanti(AREA)] for L in lavori.values()) >= 2:
                compiti.append((t, h))
    if os.environ.get("MAX_ISTANTI"):
        compiti = compiti[:int(os.environ["MAX_ISTANTI"])]
    G.log(f"Istanti da elaborare: {len(compiti)} · modelli: {', '.join(lavori)}")

    # scale delle temperature: quelle del primo modello disponibile
    scale = {}
    for m in conf["componenti"]:
        if m in lavori:
            lavori[m].calcola_scale()
            scale = lavori[m].scale_corsa
            break

    if os.path.exists(G.USCITA):
        shutil.rmtree(G.USCITA)
    os.makedirs(os.path.join(G.USCITA, AREA), exist_ok=True)

    pag = carica_pagella()
    stazioni = G.carica_stazioni(*griglia) if AREA in ("lazio", "italia") else []
    fine = {}
    for m, L in lavori.items():
        ist = [x[2] for x in L.istanti(AREA)]
        if ist:
            fine[m] = max(ist)
    # un modello che arriva fino all'ultima ora elaborata non "finisce": niente dissolvenza
    ultimo = max(t for t, h in compiti) if compiti else None
    fine = {m: f for m, f in fine.items() if ultimo is None or f < ultimo}
    _M.update(fine=fine)
    _M.update(lavori=lavori, pagella=pag, corsa=ref, griglia=griglia, scale=scale, sezione=conf["pagella"],
              stazioni=stazioni, terreno=terreno, ombra=terreno["ombra"] if terreno is not None else None)
    _avvia()
    tutti, errori, punti = [], [], []
    nproc = processi or max(1, min(4, os.cpu_count() or 1))
    if nproc == 1:
        for c in compiti:
            r, e, pu = _fai(c)
            tutti += r
            errori += e
            punti += pu
    else:
        with ProcessPoolExecutor(nproc, mp_context=mp.get_context("fork")) as ex:
            for n, (r, e, pu) in enumerate(ex.map(_fai, compiti)):
                tutti += r
                errori += e
                punti += pu
                if (n + 1) % 5 == 0:
                    G.log(f"  istanti completati: {n + 1}/{len(compiti)}")
    G.log(f"Mappe disegnate: {len(tutti)}")
    if errori:
        G.avviso(f"[{NOME} {area}] {len(errori)} mappe con errori, es.: " + " | ".join(errori[:4]))
    G.scrivi_punti(punti)

    corsa_txt = ref.strftime("%Y-%m-%dT%H:%MZ")
    indice = dict(modello="multi", nome=NOME, risoluzione="media pesata",
                  corsa=corsa_txt, corse={AREA: corsa_txt},
                  componenti={AREA: {m: L.corsa.strftime("%Y-%m-%dT%H:%MZ") for m, L in lavori.items()}},
                  pagella=(pag or {}).get("aggiornato"),
                  aggiornato_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), aree={})
    for a, pid, nome, h, v, w in sorted(tutti, key=lambda x: (x[0], x[1], x[3])):
        indice["aree"].setdefault(a, {}).setdefault(pid, []).append(
            dict(file=f"{a}/{nome}", scadenza=h, valido_utc=v, pesi=w))
    with open(os.path.join(G.USCITA, "indice_parziale.json"), "w") as f:
        json.dump(indice, f, separators=(",", ":"))
    G.log(f"=== Fine {NOME} {area}: {len(tutti)} mappe in {time.time() - t0:.0f} s ===")
    return len(tutti)


def gia_fatto(area):
    """True se il multimodello pubblicato per l'area usa già le corse più recenti di tutti i modelli."""
    from .controlla import corsa_disponibile
    try:
        r = requests.get("https://raw.githubusercontent.com/eremite1/meteo-mappe/mappe-multi/indice.json", timeout=30)
        if r.status_code != 200:
            return False
        usate = (r.json().get("componenti") or {}).get(area) or {}
    except Exception:
        return False
    if not usate:
        return False
    for m in AREE_MULTI[area]["componenti"]:
        if area not in MODELLI[m]["aree"]:
            continue
        c = corsa_disponibile(m)
        if c and c[:13] != (usate.get(m) or "")[:13]:
            return False
    return True


if __name__ == "__main__":
    area = sys.argv[1] if len(sys.argv) > 1 else "lazio"
    if os.environ.get("FORZA", "false").lower() != "true" and gia_fatto(area):
        G.log(f"Multimodello {area} già aggiornato con le ultime corse, niente da fare.")
        sys.exit(0)
    if esegui(area) == 0:
        sys.exit(1)
