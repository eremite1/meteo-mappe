#!/usr/bin/env python3
"""PAGELLA UNICA: la pioggia prevista da TUTTI i modelli confrontata con quella vista dal radar, come mappe intere.

Ogni modello (Mappe Lazio, WRF Nerola 3 km e 1 km, e qualsiasi modello aggiunto in futuro) pubblica la sua pioggia
cumulata su una GRIGLIA COMUNE del Lazio (0,025 gradi, circa 2-3 km). Il radar della Protezione Civile fa lo stesso
ora per ora (nowcasting-nerola/radar_punti.py). Qui, ogni 3 ore:

  1. ARCHIVIO: la corsa piu' recente di ogni modello viene salvata (pioggia caduta ogni 3 ore, fino a +72 h).
  2. CONFRONTO: ogni finestra di 3 ore gia' osservata dal radar viene confrontata con la previsione di ogni corsa:
     - VOTO (Fractions Skill Score, Roberts e Lean 2008): quanto coincidono le zone di pioggia (almeno 1 mm in 3 ore)
       guardando quadrati di circa 35 km (e 7 km per il dettaglio). 100 = identiche, 0 = niente in comune.
     - ERRORE DI TEMPO: la stessa previsione spostata di 3 e 6 ore prima e dopo: lo spostamento che la fa coincidere
       meglio con il radar dice se il modello e' in anticipo o in ritardo.
     - ERRORE DI POSIZIONE: lo spostamento (km e direzione) che sovrappone meglio la mappa prevista a quella del radar
       (correlazione incrociata), come nella componente "posizione" del metodo SAL (Wernli e altri 2008).
     - ERRORE DI QUANTITA': pioggia totale prevista / pioggia totale vista dal radar sulla stessa area.
     - TIPO DI TEMPO, dal radar: "temporali" (piogge intense e a chiazze), "perturbazione" (pioggia estesa),
       "piogge sparse" (il resto). Le finestre asciutte sia nella previsione sia nella realta' non contano.
  3. RIEPILOGO (pagelle.json): classifica, errori per tipo di tempo e per scadenza, ultimi eventi, piu' i confronti con
     le stazioni (temperatura, vento) presi da pagella.json. Un modello nuovo entra da solo: basta che pubblichi la
     griglia (motore/genera.py lo fa per ogni modello delle Mappe Lazio).

  python -m motore.verifica --storico DIR --radar radar_orario.npz --griglie chiave=file.npz ... [--wrf d01 d02]
"""
import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------- griglia comune
LON0, LAT0, PASSO, NX, NY = 11.30, 41.05, 0.025, 118, 86          # riga 0 = sud
LONS = LON0 + PASSO * (np.arange(NX) + 0.5)
LATS = LAT0 + PASSO * (np.arange(NY) + 0.5)
KM_X = PASSO * 111.32 * math.cos(math.radians(42.1))
KM_Y = PASSO * 110.57
VUOTO = 65535
SOGLIA = 1.0              # mm in 3 ore
SCALE = {"fss": 15, "fss_fine": 3}                  # celle: circa 35 km e 7 km
RITARDI = (-6, -3, 0, 3, 6)
BANDE = (("0-24", 0, 24), ("24-48", 24, 48), ("48-72", 48, 72))
TIPI = ("perturbazione", "temporali", "sparse")
GIORNI_TENUTI = 120
MAX_SPOST = 25            # celle (circa 50-70 km)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%MZ")


def ora(s):
    s = s.replace("_", "T")
    return datetime.strptime(s[:13], "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)


def su_griglia(lat, lon, valori, massima_km=None):
    """Valori (nx o 2D) dal reticolo del modello alla griglia comune: punto piu' vicino; fuori copertura = NaN."""
    from scipy.spatial import cKDTree
    lat, lon = np.asarray(lat, dtype="float64"), np.asarray(lon, dtype="float64")
    n_val = np.asarray(valori[0]).size if len(valori) else 0
    if lat.ndim == 1 and lon.ndim == 1 and not (lat.size == lon.size == n_val):
        lon, lat = np.meshgrid(lon, lat)          # assi di una griglia regolare
    # altrimenti: elenco di punti sparsi (griglie ridotte), stesso numero dei valori
    c = math.cos(math.radians(42.1))
    albero = cKDTree(np.c_[lon.ravel() * c, lat.ravel()])
    GX, GY = np.meshgrid(LONS * c, LATS)
    d, i = albero.query(np.c_[GX.ravel(), GY.ravel()])
    passo_mod = np.median(albero.query(albero.data[:200], k=2)[0][:, 1]) if albero.n > 2 else 0.1
    lim = max(passo_mod * 1.5, PASSO * 0.8) if massima_km is None else massima_km / 111.0
    out = []
    for v in valori:
        g = np.asarray(v, dtype="float64").ravel()[i]
        g[d > lim] = np.nan
        out.append(g.reshape(NY, NX))
    return out


def salva_griglia(percorso, corsa, tempi, cumulate):
    """cumulate: pioggia cumulata dall'inizio della corsa, gia' sulla griglia comune, una per istante."""
    a = np.stack([np.where(np.isfinite(c), np.clip(np.round(c * 10), 0, VUOTO - 1), VUOTO) for c in cumulate]).astype("<u2")
    np.savez_compressed(percorso, corsa=iso(corsa), tempi=np.array([iso(t) for t in tempi]), cum=a)


def leggi_griglia(percorso):
    z = np.load(percorso)
    cum = z["cum"].astype("float64")
    cum = np.where(cum == VUOTO, np.nan, cum / 10.0)
    return ora(str(z["corsa"])), [ora(str(t)) for t in z["tempi"]], cum


# --------------------------------------------------------------------------------------------- WRF Nerola

def wrf_griglie(d01, d02, cartella):
    """WRF 3 km (d01) e WRF 1 km (d02, con il 3 km intorno, come esce davvero dal modello annidato) sulla griglia
    comune. Ritorna {chiave: percorso}."""
    out = {}
    z1 = np.load(d01)
    tempi = [ora(str(t)) for t in z1["tempi"]]
    cum1 = z1["RAINC"] + z1["RAINNC"]
    g1 = su_griglia(z1["lat"], z1["lon"], cum1)
    p = Path(cartella) / "wrf3km.npz"
    salva_griglia(p, tempi[0], tempi, g1)
    out["wrf3km"] = p
    if d02 and Path(d02).exists():
        z2 = np.load(d02)
        t2 = [ora(str(t)) for t in z2["tempi"]]
        if t2 == tempi:
            g2 = su_griglia(z2["lat"], z2["lon"], z2["RAINC"] + z2["RAINNC"])
            g12 = [np.where(np.isfinite(b), b, a) for a, b in zip(g1, g2)]
            p = Path(cartella) / "wrf1km.npz"
            salva_griglia(p, tempi[0], tempi, g12)
            out["wrf1km"] = p
    return out


def wrf_griglia(f, chiave, cartella):
    """Un solo dominio di un'altra catena WRF (GFS, 333 m) sulla griglia comune."""
    z = np.load(f)
    tempi = [ora(str(t)) for t in z["tempi"]]
    p = Path(cartella) / f"{chiave}.npz"
    salva_griglia(p, tempi[0], tempi, su_griglia(z["lat"], z["lon"], z["RAINC"] + z["RAINNC"]))
    return p


def wrf_punti(d01, d02, stazioni, cartella, altri=()):
    """Previsioni del WRF nei punti delle stazioni e del radar, nel formato dei punti delle Mappe Lazio
    (corsa, valido, ore, passo, id, T, Td, V, P, h): T e Td in gradi, vento in m/s, pioggia in 3 ore (passo 3)."""
    import csv
    righe_out = {}
    for chiave, f in [("wrf3km", d01), ("wrf1km", d02)] + list(altri):
        if not f or not Path(f).exists():
            continue
        z = np.load(f)
        lat, lon = z["lat"], z["lon"]
        tempi = [ora(str(t)) for t in z["tempi"]]
        corsa = tempi[0]
        T = z["T2"] - (273.15 if np.nanmean(z["T2"]) > 150 else 0)
        e = z["Q2"] * z["PSFC"] / (0.622 + z["Q2"]) / 100.0
        lne = np.log(np.maximum(e, 1e-3) / 6.112)
        Td = 243.5 * lne / (17.67 - lne)
        V = np.hypot(z["U10"], z["V10"])
        cum = z["RAINC"] + z["RAINNC"]
        c = math.cos(math.radians(42.1))
        righe = []
        for sid, la, lo in stazioni:
            d2 = ((lon - lo) * c) ** 2 + (lat - la) ** 2
            iy, ix = np.unravel_index(int(np.argmin(d2)), d2.shape)
            if not (0 < iy < lat.shape[0] - 1 and 0 < ix < lat.shape[1] - 1) or d2[iy, ix] > (0.03 ** 2):
                continue
            for k, t in enumerate(tempi):
                h = int(round((t - corsa).total_seconds() / 3600))
                p = ""
                if h > 0 and h % 3 == 0:
                    p = round(float(cum[k, iy, ix] - cum[k - 3, iy, ix]), 2)
                righe.append([iso(corsa), iso(t), h, 3, sid, round(float(T[k, iy, ix]), 1), round(float(Td[k, iy, ix]), 1),
                              round(float(V[k, iy, ix]), 1), p, round(float(z["hgt"][iy, ix]))])
        p = Path(cartella) / f"{chiave}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["corsa", "valido", "ore", "passo", "id", "T", "Td", "V", "P", "h"])
            w.writerows(righe)
        righe_out[chiave] = p
    return righe_out


# ------------------------------------------------------------------------------------ codifica compatta

def comprimi(mm):
    """0 = niente, 1-254 = mm in scala logaritmica (0,1 - 300 mm), 255 = nessun dato."""
    a = np.asarray(mm, dtype="float64")
    q = np.where(a >= 0.1, np.clip(np.round(np.log10(np.maximum(a, 0.1) / 0.1) / math.log10(3000) * 253) + 1, 1, 254), 0)
    return np.where(np.isfinite(a), q, 255).astype(np.uint8)


def espandi(q):
    q = np.asarray(q)
    v = np.where(q > 0, 0.1 * 10 ** ((q.astype("float64") - 1) / 253 * math.log10(3000)), 0.0)
    return np.where(q == 255, np.nan, v)


# -------------------------------------------------------------------------------------------- misure

def fss(f, o, scala):
    from scipy import ndimage
    m = np.isfinite(f) & np.isfinite(o)
    if m.sum() < 500:
        return None
    pf = ((np.nan_to_num(f) >= SOGLIA) & m).astype("float64")
    po = ((np.nan_to_num(o) >= SOGLIA) & m).astype("float64")
    n = ndimage.uniform_filter(m.astype("float64"), scala, mode="constant")
    ok = (n > 0.5) & m
    ff = ndimage.uniform_filter(pf, scala, mode="constant")[ok] / n[ok]
    fo = ndimage.uniform_filter(po, scala, mode="constant")[ok] / n[ok]
    rif = np.mean(ff ** 2) + np.mean(fo ** 2)
    if rif <= 0:
        return None
    return float(1 - np.mean((ff - fo) ** 2) / rif)


def spostamento(f, o):
    """Spostamento (km verso est, km verso nord) di cui va mossa l'osservazione per sovrapporsi alla previsione:
    positivo = il modello mette la pioggia piu' a est / piu' a nord del vero."""
    m = np.isfinite(f) & np.isfinite(o)
    a = np.where(m, np.log1p(np.nan_to_num(f)), 0.0)
    b = np.where(m, np.log1p(np.nan_to_num(o)), 0.0)
    if a.sum() <= 0 or b.sum() <= 0:
        return None
    a, b = a - a[m].mean() * m, b - b[m].mean() * m
    ny, nx = a.shape
    A = np.fft.rfft2(a, s=(2 * ny, 2 * nx))
    B = np.fft.rfft2(b, s=(2 * ny, 2 * nx))
    cc = np.fft.irfft2(A * np.conj(B), s=(2 * ny, 2 * nx))
    cc = np.fft.fftshift(cc)
    cy, cx = ny, nx
    fin = cc[cy - MAX_SPOST:cy + MAX_SPOST + 1, cx - MAX_SPOST:cx + MAX_SPOST + 1]
    iy, ix = np.unravel_index(int(np.argmax(fin)), fin.shape)
    dy, dx = iy - MAX_SPOST, ix - MAX_SPOST
    if abs(dy) == MAX_SPOST or abs(dx) == MAX_SPOST:
        return None                       # al limite della ricerca: non affidabile
    return dx * KM_X, dy * KM_Y


def tipo_di_tempo(ore_radar):
    """Dalle 3 mappe orarie del radar: perturbazione, temporali, sparse; None se e' asciutto."""
    s = np.nansum(ore_radar, axis=0)
    ok = np.isfinite(ore_radar).all(axis=0)
    if ok.sum() < 500:
        return None
    area = float((s[ok] >= SOGLIA).mean())
    if area < 0.01:
        return None
    intensa = float(np.percentile(np.max(ore_radar, axis=0)[ok], 99.5))
    if area >= 0.25 and intensa < 25:
        return "perturbazione"
    if intensa >= 12:
        return "temporali"
    return "perturbazione" if area >= 0.25 else "sparse"


# ------------------------------------------------------------------------------------------ archivio

def archivia(cartella, chiave, percorso):
    """Salva la corsa (pioggia ogni 3 ore) se non c'e' gia'. Ritorna il nome del file archiviato o None."""
    corsa, tempi, cum = leggi_griglia(percorso)
    nome = Path(cartella) / f"{chiave}__{corsa:%Y%m%d%H}.npz"
    if nome.exists():
        return None
    pos = {t: k for k, t in enumerate(tempi)}
    fini, campi = [], []
    for h in range(3, 73, 3):
        t, t0 = corsa + timedelta(hours=h), corsa + timedelta(hours=h - 3)
        if t in pos and (t0 in pos or h == 3 and corsa in pos):
            prima = cum[pos[t0]] if t0 in pos else np.zeros_like(cum[0])
            fini.append(iso(t))
            campi.append(comprimi(np.clip(cum[pos[t]] - prima, 0, None)))
    if not campi:
        return None
    np.savez_compressed(nome, chiave=chiave, corsa=iso(corsa), fini=np.array(fini), mm=np.stack(campi))
    return nome.name


def radar_3ore(percorso):
    z = np.load(percorso)
    ore = [ora(str(t)) for t in z["ore"]]
    mm = z["mm"].astype("float64")
    mm = np.where(mm == VUOTO, np.nan, mm / 10.0)
    per = {t: mm[k] for k, t in enumerate(ore)}
    return per


def finestra_oss(per_ora, fine):
    tre = [per_ora.get(fine - timedelta(hours=k)) for k in range(3)]
    if any(x is None for x in tre):
        return None, None
    st = np.stack(tre)
    return st.sum(axis=0), st


# ---------------------------------------------------------------------------------------- confronto

def nuovo_contatore():
    return {"n": 0, "fss": 0.0, "fss_fine": 0.0, "nf": 0, "pf": 0.0, "po": 0.0,
            "rit": {str(r): [0.0, 0] for r in RITARDI}, "ns": 0, "sx": 0.0, "sy": 0.0, "sd": 0.0}


def confronta(storico, per_ora, adesso):
    arch = Path(storico) / "verifica" / "archivio"
    stato_p = Path(storico) / "verifica" / "stato.json"
    stato = json.loads(stato_p.read_text()) if stato_p.exists() else {}
    fatti = stato.setdefault("fatti", {})
    somme = stato.setdefault("somme", {})          # modello -> giorno -> banda -> tipo -> contatore
    eventi = stato.setdefault("eventi", {})        # giorno -> {tipo: {..conteggi}, modelli: {m: contatore}}
    nuovi = 0
    for f in sorted(arch.glob("*.npz")):
        try:
            z = np.load(f)
            chiave, corsa = str(z["chiave"]), ora(str(z["corsa"]))
            fini = [ora(str(t)) for t in z["fini"]]
        except Exception:  # noqa: BLE001
            f.unlink(missing_ok=True)
            continue
        if adesso - corsa > timedelta(hours=72 + 30):
            f.unlink(missing_ok=True)               # tutte le finestre ormai sono passate
            fatti.pop(f.stem, None)
            continue
        campi = {t: espandi(z["mm"][k]) for k, t in enumerate(fini)}
        gia = set(fatti.setdefault(f.stem, []))
        for t in fini:
            if iso(t) in gia:
                continue
            o, ore3 = finestra_oss(per_ora, t)
            if o is None:
                continue
            gia.add(iso(t))
            fatti[f.stem].append(iso(t))
            prev = campi[t]
            tipo = tipo_di_tempo(ore3)
            area_f = float((np.nan_to_num(prev)[np.isfinite(o)] >= SOGLIA).mean()) if np.isfinite(o).any() else 0
            if tipo is None:
                if area_f < 0.01:
                    continue                         # asciutto per tutti: non dice nulla
                tipo = "sparse"                      # il modello vede pioggia che non c'e'
            h = int(round((t - corsa).total_seconds() / 3600))
            banda = next((b for b, a, c in BANDE if a < h <= c), None)
            if banda is None:
                continue
            giorno = t.strftime("%Y-%m-%d")
            cont = somme.setdefault(chiave, {}).setdefault(giorno, {}).setdefault(banda, {}).setdefault(tipo, nuovo_contatore())
            ev = eventi.setdefault(giorno, {"tipi": {}, "modelli": {}})
            ev["tipi"][tipo] = ev["tipi"].get(tipo, 0) + 1
            ce = ev["modelli"].setdefault(chiave, nuovo_contatore()) if banda != "48-72" else None
            s = fss(prev, o, SCALE["fss"])
            if s is None:
                continue
            sf = fss(prev, o, SCALE["fss_fine"])
            m = np.isfinite(prev) & np.isfinite(o)
            for c in (cont, ce):
                if c is None:
                    continue
                c["n"] += 1
                c["fss"] += s
                if sf is not None:
                    c["fss_fine"] += sf
                    c["nf"] += 1
                c["pf"] += float(np.nan_to_num(prev)[m].sum())
                c["po"] += float(np.nan_to_num(o)[m].sum())
            migliore, v_migl = 0, s
            for r in RITARDI:
                tr = t + timedelta(hours=r)
                if tr in campi:
                    v = fss(campi[tr], o, SCALE["fss"])
                    if v is not None and v > v_migl + 0.02:
                        migliore, v_migl = r, v
                    if v is not None:
                        for c in (cont, ce):
                            if c is not None:
                                c["rit"][str(r)][0] += v
                                c["rit"][str(r)][1] += 1
            # posizione: si confronta la mappa del momento che coincide meglio (cosi' l'errore di tempo non si
            # confonde con quello di posizione)
            pr = campi[t + timedelta(hours=migliore)]
            area_r = float((np.nan_to_num(pr)[np.isfinite(o)] >= SOGLIA).mean()) if np.isfinite(o).any() else 0
            if area_r >= 0.01 and float((o[np.isfinite(o)] >= SOGLIA).mean()) >= 0.01:
                sp = spostamento(pr, o)
                if sp is not None:
                    for c in (cont, ce):
                        if c is not None:
                            c["ns"] += 1
                            c["sx"] += sp[0]
                            c["sy"] += sp[1]
                            c["sd"] += math.hypot(*sp)
            nuovi += 1
    limite = (adesso - timedelta(days=GIORNI_TENUTI)).strftime("%Y-%m-%d")
    for m in somme.values():
        for g in [g for g in m if g < limite]:
            del m[g]
    for g in [g for g in eventi if g < limite]:
        del eventi[g]
    stato["aggiornato"] = iso(adesso)
    stato_p.write_text(json.dumps(stato, separators=(",", ":")))
    return nuovi, stato


# ------------------------------------------------------------------------------------------- riepilogo

def somma(conts):
    t = nuovo_contatore()
    for c in conts:
        for k in ("n", "fss", "fss_fine", "nf", "pf", "po", "ns", "sx", "sy", "sd"):
            t[k] += c[k]
        for r, (v, n) in c["rit"].items():
            t["rit"][r][0] += v
            t["rit"][r][1] += n
    return t


def misure(c):
    if c["n"] < 3:
        return None
    out = {"voto": round(100 * c["fss"] / c["n"], 1), "casi": c["n"]}
    if c["nf"]:
        out["voto_fine"] = round(100 * c["fss_fine"] / c["nf"], 1)
    if c["po"] > 0:
        out["quantita"] = round(100 * (c["pf"] / c["po"] - 1))
    curva = {r: round(100 * v / n, 1) for r, (v, n) in c["rit"].items() if n >= 3}
    if len(curva) >= 3 and "0" in curva:
        out["ritardi"] = curva
        best = max(curva, key=lambda r: curva[r])
        # conta solo se migliora davvero (almeno 2 punti rispetto a "nessuno spostamento")
        out["tempo"] = int(best) if curva[best] >= curva["0"] + 2 else 0
    if c["ns"] >= 3:
        sx, sy = c["sx"] / c["ns"], c["sy"] / c["ns"]
        out["posizione"] = {"km": round(c["sd"] / c["ns"]), "verso_km": round(math.hypot(sx, sy)),
                            "direzione": direzione(sx, sy) if math.hypot(sx, sy) >= 5 else None}
    return out


def direzione(sx, sy):
    nomi = ["est", "nord-est", "nord", "nord-ovest", "ovest", "sud-ovest", "sud", "sud-est"]
    return "troppo a " + nomi[int(round(math.degrees(math.atan2(sy, sx)) / 45)) % 8]


def voto_punti(riep):
    """Temperatura e vento alle stazioni (prime 48 ore) trasformati in un voto 0-100 semplice:
    temperatura 100 - 20 x errore medio (°C); vento 100 - 15 x errore medio (m/s)."""
    out = {}
    for v, k in (("T", 20), ("V", 15)):
        r = [x for f, x in (riep.get(v) or {}).items() if f in ("0-12 ore", "12-24 ore", "24-48 ore") and x and x.get("n")]
        if not r:
            continue
        n = sum(x["n"] for x in r)
        mae = sum(x["mae"] * x["n"] for x in r) / n
        bias = sum(x["bias"] * x["n"] for x in r) / n
        out[v] = {"errore": round(mae, 2), "tendenza": round(bias, 2), "voto": round(max(0.0, 100 - k * mae), 1), "casi": n}
    return out


def riepilogo(stato, pagella, nomi, adesso):
    somme = stato.get("somme", {})
    modelli = sorted(set(somme) | set(nomi))
    periodi = {}
    for giorni in (7, 30, 90):
        da = (adesso - timedelta(days=giorni)).strftime("%Y-%m-%d")
        P = {}
        for m in modelli:
            gg = {g: v for g, v in somme.get(m, {}).items() if g >= da}
            voce = {}
            for tipo in ("tutti",) + TIPI:
                for banda in ("tutte",) + tuple(b for b, _, _ in BANDE):
                    cc = [c for g in gg.values() for b, tt in g.items() if banda in ("tutte", b)
                          for ti, c in tt.items() if tipo in ("tutti", ti)]
                    ms = misure(somma(cc)) if cc else None
                    if ms:
                        voce.setdefault(tipo, {})[banda] = ms
            if voce:
                P[m] = voce
        periodi[str(giorni)] = P
    # ultimi eventi di pioggia (giorni con almeno 2 finestre bagnate)
    eventi = []
    for g in sorted(stato.get("eventi", {}), reverse=True)[:12]:
        e = stato["eventi"][g]
        if sum(e["tipi"].values()) < 2:
            continue
        mm = {m: misure(c) for m, c in e["modelli"].items()}
        mm = {m: v for m, v in mm.items() if v}
        if mm:
            eventi.append({"giorno": g, "tipo": max(e["tipi"], key=e["tipi"].get), "modelli": mm})
    # stazioni (da pagella.json)
    punti, fisica = {}, {}
    for m, riep in (pagella.get("riepilogo") or {}).items():
        v = voto_punti(riep)
        if v:
            (fisica if m.startswith("fis_") else punti)[m] = v
    # classifica generale: media dei voti disponibili (pioggia sulle mappe 30 giorni prime 48 ore, temperatura, vento)
    gen = []
    for m in [x for x in modelli if not x.startswith("fis_")]:
        p = (periodi["30"].get(m, {}).get("tutti", {}))
        pioggia = None
        cc = [x for b, x in p.items() if b in ("0-24", "24-48")]
        if cc:
            n = sum(x["casi"] for x in cc)
            pioggia = round(sum(x["voto"] * x["casi"] for x in cc) / n, 1)
        t = punti.get(m, {}).get("T", {}).get("voto")
        v = punti.get(m, {}).get("V", {}).get("voto")
        parti = {"pioggia": pioggia, "temperatura": t, "vento": v}
        pesi = {"pioggia": 0.5, "temperatura": 0.3, "vento": 0.2}
        dis = {k: x for k, x in parti.items() if x is not None}
        if not dis:
            continue
        tot = sum(pesi[k] for k in dis)
        gen.append({"modello": m, "voto": round(sum(x * pesi[k] for k, x in dis.items()) / tot, 1), "parti": parti,
                    "completo": len(dis) == 3})
    gen.sort(key=lambda x: (not x["completo"], -x["voto"]))
    return {"aggiornato": iso(adesso), "modelli": [{"id": m, **nomi.get(m, {"nome": m, "ris": ""})} for m in modelli],
            "classifica": gen, "pioggia": periodi, "eventi": eventi, "stazioni": punti,
            "fisica": {m: dict(v, scadenze={k: x for k, x in ((pagella.get("riepilogo") or {}).get(m, {}).get("T") or {}).items()})
                       for m, v in fisica.items()},
            "stazioni_giorni": pagella.get("giorni"), "bande": [b for b, _, _ in BANDE], "tipi": list(TIPI)}


def grafici(ris, pagella, storico, adesso):
    """Dati per i grafici delle pagelle:
    - scadenze: errore medio alle stazioni per fascia di scadenza (Lazio fino a 72 ore, Italia oltre);
    - nerola: ultimi giorni alla stazione piu' vicina a Nerola, osservato e tutti i modelli;
    - andamento: errore medio (ultimi 7 giorni, prime 48 ore) giorno per giorno, per vedere come cambia nel tempo."""
    sc = {}
    for fonte in (pagella.get("riepilogo") or {}, (pagella.get("italia") or {}).get("riepilogo") or {}):
        for m, vv in fonte.items():
            if m.startswith("fis_"):
                continue
            for v in ("T", "V"):
                for f, x in (vv.get(v) or {}).items():
                    if not x or not x.get("n") or x["n"] < 30:
                        continue
                    da = int(f.split("-")[0])
                    voci = sc.setdefault(m, {}).setdefault(v, {})
                    if fonte is pagella.get("riepilogo") or da >= 72:
                        voci.setdefault(f, {"fascia": f, "da": da, "errore": x["mae"], "tendenza": x["bias"], "casi": x["n"]})
    ris["scadenze"] = {m: {v: sorted(ff.values(), key=lambda x: x["da"]) for v, ff in vv.items()} for m, vv in sc.items()}
    vicine = [st for st in pagella.get("stazioni") or [] if st.get("serie") and st.get("dist_km", 99) < 5]
    if vicine:
        st = vicine[0]
        ris["nerola"] = {"stazione": st.get("nome") or st["id"], "dist_km": st.get("dist_km"), "tempi": st["serie"]["tempi"],
                         "oss": {v: st["serie"]["oss"].get(v) for v in ("T", "V")},
                         "modelli": {m: {v: x.get(v) for v in ("T", "V")} for m, x in st["serie"]["modelli"].items()
                                     if not m.startswith("fis_")}}
    sp = storico / "verifica" / "andamento.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    storia = json.loads(sp.read_text()) if sp.exists() else {}
    oggi = adesso.strftime("%Y-%m-%d")
    storia[oggi] = {m: {v: x[v]["errore"] for v in ("T", "V") if v in x} for m, x in (ris.get("stazioni") or {}).items()}
    limite = (adesso - timedelta(days=120)).strftime("%Y-%m-%d")
    storia = {g: v for g, v in sorted(storia.items()) if g >= limite}
    sp.write_text(json.dumps(storia, separators=(",", ":")))
    ris["andamento"] = storia


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storico", required=True)
    ap.add_argument("--radar", default="")
    ap.add_argument("--griglie", nargs="*", default=[])
    ap.add_argument("--wrf", nargs="*", default=[], help="d01_tutto.npz [d02_tutto.npz]")
    ap.add_argument("--nomi", default="", help="json {chiave: {nome, ris}}")
    ap.add_argument("--wrf-altri", nargs="*", default=[], help="chiave=dominio_tutto.npz delle altre catene WRF")
    a = ap.parse_args()
    adesso = datetime.now(timezone.utc)
    arch = Path(a.storico) / "verifica" / "archivio"
    arch.mkdir(parents=True, exist_ok=True)
    griglie = dict(x.split("=", 1) for x in a.griglie)
    if a.wrf:
        try:
            tmp = Path(a.storico).parent / "griglie_wrf"
            tmp.mkdir(exist_ok=True)
            griglie.update({k: str(v) for k, v in wrf_griglie(a.wrf[0], a.wrf[1] if len(a.wrf) > 1 else "", tmp).items()})
        except Exception as e:  # noqa: BLE001
            print("ATTENZIONE: griglie WRF non pronte:", e)
    for voce in a.wrf_altri:
        try:
            k, f = voce.split("=", 1)
            tmp = Path(a.storico).parent / "griglie_wrf"
            tmp.mkdir(exist_ok=True)
            griglie[k] = str(wrf_griglia(f, k, tmp))
        except Exception as e:  # noqa: BLE001
            print("ATTENZIONE: griglia", voce, "non pronta:", e)
    for k, p in griglie.items():
        try:
            n = archivia(arch, k, p)
            print(f"{k}: " + (f"archiviata {n}" if n else "corsa gia' in archivio"))
        except Exception as e:  # noqa: BLE001
            print(f"ATTENZIONE {k}: {e}")
    stato = {}
    if a.radar and Path(a.radar).exists():
        nuovi, stato = confronta(a.storico, radar_3ore(a.radar), adesso)
        print(f"confronti nuovi con il radar: {nuovi}")
        try:   # tavola giornaliera: pioggia del radar contro tutti i modelli, a occhio
            from motore import tavola
            print("tavole della pioggia disegnate:", tavola.aggiorna(a.storico, radar_3ore(a.radar), adesso))
        except Exception as e:  # noqa: BLE001
            print("ATTENZIONE: tavola della pioggia non pronta:", e)
    else:
        sp = Path(a.storico) / "verifica" / "stato.json"
        stato = json.loads(sp.read_text()) if sp.exists() else {}
        print("radar orario non disponibile: nessun confronto nuovo")
    pag_p = Path(a.storico) / "pagella.json"
    pagella = json.loads(pag_p.read_text()) if pag_p.exists() else {}
    nomi = {m["id"]: {"nome": m["nome"], "ris": m.get("ris", "")} for m in pagella.get("modelli", [])}
    if a.nomi:
        nomi.update(json.loads(a.nomi))
    ris = riepilogo(stato, pagella, nomi, adesso)
    try:
        grafici(ris, pagella, Path(a.storico), adesso)
    except Exception as e:  # noqa: BLE001
        print("ATTENZIONE: dati dei grafici non pronti:", e)
    (Path(a.storico) / "pagelle.json").write_text(json.dumps(ris, ensure_ascii=False, separators=(",", ":")))
    print("classifica:", ", ".join(f"{x['modello']} {x['voto']}" for x in ris["classifica"]))


if __name__ == "__main__":
    main()
