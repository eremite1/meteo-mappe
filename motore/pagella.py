"""
Pagella delle Mappe Lazio: la previsione di ogni modello confrontata con le stazioni meteo reali.

Ogni corsa delle mappe salva nel ramo del modello il file punti_lazio.csv: i valori previsti (gli stessi
delle mappe, temperatura col dettaglio del terreno compresa) nei punti delle stazioni del Lazio.
Le osservazioni sono quelle che la pagella di WRF Nerola raccoglie già (aeroporti e Protezione Civile).

Questo programma:
  1. aggiunge allo storico i punti_lazio.csv nuovi (una cartella per modello, un file per corsa);
  2. confronta previsto e osservato degli ultimi giorni;
  3. scrive pagella.json, che la pagina del sito mostra.

Uso: python -m motore.pagella --storico CARTELLA --osservazioni osservazioni.csv --punti modello=file.csv ...
"""
import argparse
import csv
import gzip
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .catalogo import MODELLI

GIORNI = 7                 # finestra del confronto
GIORNI_STORICO = 12        # quanto tengo le previsioni salvate (file compressi)
GIORNI_SERIE = 4           # giorni mostrati nei grafici delle stazioni
STAZIONI_SERIE = 30        # grafici solo per le stazioni più vicine a Nerola (file leggero)
NEROLA = (42.1606, 12.7856)
FASCE = [(0, 12, "0-12 ore"), (12, 24, "12-24 ore"), (24, 48, "24-48 ore"), (48, 73, "48-72 ore")]
VARIABILI = ["T", "Td", "V", "P"]
SOGLIA_PIOGGIA = 0.2       # mm: sotto questa soglia "non piove"


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def ora(s):
    return datetime.strptime(s[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:00Z")


# ---------------------------------------------------------------------------
def archivia(storico, modello, file_punti):
    """Divide punti_lazio.csv per corsa e lo salva in previsioni/<modello>/<corsa>.csv."""
    righe = list(csv.DictReader(open(file_punti, newline="")))
    if not righe:
        return 0
    cart = storico / "previsioni" / modello
    cart.mkdir(parents=True, exist_ok=True)
    per_corsa = {}
    for r in righe:
        per_corsa.setdefault(r["corsa"], []).append(r)
    for corsa, rr in per_corsa.items():
        nome = corsa.replace("-", "").replace(":", "")[:11]          # 20261103T12
        with gzip.open(cart / f"{nome}.csv.gz", "wt", newline="") as f:
            w = csv.DictWriter(f, list(rr[0].keys()))
            w.writeheader()
            w.writerows(rr)
    # pulizia delle corse troppo vecchie
    limite = (datetime.now(timezone.utc) - timedelta(days=GIORNI_STORICO)).strftime("%Y%m%dT%H")
    for p in cart.glob("*.csv.gz"):
        if p.name[:11] < limite:
            p.unlink()
    return len(per_corsa)


def leggi_previsioni(storico, giorni):
    limite = (datetime.now(timezone.utc) - timedelta(days=giorni + 3)).strftime("%Y%m%dT%H")
    out = {}
    for cart in sorted((storico / "previsioni").glob("*")):
        rr = []
        for p in sorted(cart.glob("*.csv.gz")):
            if p.name[:11] >= limite:
                with gzip.open(p, "rt", newline="") as f:
                    rr += list(csv.DictReader(f))
        out[cart.name] = rr
    return out


def leggi_osservazioni(percorso):
    oss, stazioni = {}, {}
    for r in csv.DictReader(open(percorso, newline="", encoding="utf-8")):
        h = r["ora"][:13] + ":00Z"
        oss[(r["id"], h)] = {v: num(r.get(v)) for v in VARIABILI}
        stazioni[r["id"]] = {"id": r["id"], "nome": r["nome"], "lat": float(r["lat"]), "lon": float(r["lon"]),
                             "quota": num(r["quota"])}
    return oss, stazioni


def osservato(oss, sid, valido, v, passo):
    """Valore osservato confrontabile con la previsione; la pioggia è la somma delle ore del passo."""
    if v != "P":
        o = oss.get((sid, valido))
        return o[v] if o else None
    t = ora(valido)
    somma = 0.0
    for k in range(passo):
        o = oss.get((sid, iso(t - timedelta(hours=k))))
        if not o or o["P"] is None:
            return None
        somma += o["P"]
    return somma


def statistiche(coppie, pioggia=False):
    if len(coppie) < 1:
        return None
    p = np.array([c[0] for c in coppie])
    o = np.array([c[1] for c in coppie])
    e = p - o
    s = {"mae": round(float(np.mean(np.abs(e))), 2), "bias": round(float(np.mean(e)), 2), "n": int(len(e))}
    if pioggia:
        # quante volte ha azzeccato "piove / non piove"
        s["azzecca"] = round(float(np.mean((p >= SOGLIA_PIOGGIA) == (o >= SOGLIA_PIOGGIA))) * 100, 1)
    return s


def dist_nerola(st):
    return round(math.hypot(st["lat"] - NEROLA[0], (st["lon"] - NEROLA[1]) * math.cos(math.radians(NEROLA[0]))) * 111.0, 1)


# ---------------------------------------------------------------------------
def calcola(oss, stazioni, previsioni, giorni=GIORNI):
    adesso = datetime.now(timezone.utc)
    limite = iso(adesso - timedelta(days=giorni))
    limite_serie = iso(adesso - timedelta(days=GIORNI_SERIE))
    modelli = [m for m in MODELLI if previsioni.get(m)]
    riep = {m: {v: {f[2]: [] for f in FASCE} for v in VARIABILI} for m in modelli}
    per_staz = {}
    quota_mod = {}
    ultime = {}                                  # (modello, stazione, ora) -> riga della corsa più recente
    for m in modelli:
        for r in previsioni[m]:
            sid, valido = r["id"], r["valido"][:13] + ":00Z"
            if r.get("h") not in (None, ""):
                quota_mod.setdefault(sid, {}).setdefault(m, num(r["h"]))
            if valido < limite or valido > iso(adesso):
                continue
            k = (m, sid, valido)
            if k not in ultime or r["corsa"] > ultime[k]["corsa"]:
                ultime[k] = r
            ore = int(r["ore"])
            fascia = next((f[2] for f in FASCE if f[0] <= ore < f[1]), None)
            if fascia is None:
                continue
            passo = int(r.get("passo") or 1)
            for v in VARIABILI:
                p = num(r.get(v))
                if p is None:
                    continue
                o = osservato(oss, sid, valido, v, passo)
                if o is None:
                    continue
                riep[m][v][fascia].append((p, o))
                if ore < 24:
                    per_staz.setdefault(sid, {}).setdefault(m, {}).setdefault(v, []).append((p, o))

    riepilogo = {m: {v: {f: statistiche(c, v == "P") for f, c in ff.items()} for v, ff in vv.items()} for m, vv in riep.items()}
    # classifica: errore medio nelle prime 48 ore, per variabile
    classifica = {}
    for v in VARIABILI:
        voci = []
        for m in modelli:
            tutte = [c for f in FASCE[:3] for c in riep[m][v][f[2]]]
            s = statistiche(tutte, v == "P")
            if s and s["n"] >= 20:
                voci.append(dict(s, modello=m))
        classifica[v] = sorted(voci, key=lambda x: x["mae"])

    elenco = []
    for sid, st in stazioni.items():
        if sid not in quota_mod:
            continue                               # fuori dal Lazio delle mappe
        voce = dict(st)
        voce["dist_km"] = dist_nerola(st)
        voce["quota_modello"] = quota_mod.get(sid, {})
        voce["errori"] = {m: {v: statistiche(c, v == "P") for v, c in vv.items()} for m, vv in per_staz.get(sid, {}).items()}
        if not voce["errori"]:
            continue
        elenco.append(voce)
    elenco.sort(key=lambda v: v["dist_km"])
    for voce in elenco[:STAZIONI_SERIE]:
        sid = voce["id"]
        tempi = sorted({k[2] for k in ultime if k[1] == sid and k[2] >= limite_serie} |
                       {k[1] for k in oss if k[0] == sid and limite_serie <= k[1] <= iso(adesso)})
        voce["serie"] = {
            "tempi": tempi,
            "oss": {v: [(oss.get((sid, t)) or {}).get(v) for t in tempi] for v in ("T", "Td", "V", "P")},
            "modelli": {m: {v: [num(ultime[(m, sid, t)].get(v)) if (m, sid, t) in ultime else None for t in tempi]
                            for v in ("T", "Td", "V")} for m in modelli},
        }
    for voce in elenco[STAZIONI_SERIE:]:
        voce["serie"] = None
    ha_pioggia = any(o["P"] is not None for o in oss.values())
    return {
        "aggiornato": adesso.strftime("%Y-%m-%dT%H:%MZ"),
        "giorni": giorni,
        "fonte": "Osservazioni: aeroporti (METAR) e stazioni della Protezione Civile e di MeteoNetwork, raccolte dalla pagella di WRF Nerola",
        "pioggia_osservata": ha_pioggia,
        "modelli": [{"id": m, "nome": MODELLI[m]["nome"], "ris": MODELLI[m]["ris"]} for m in modelli],
        "fasce": [f[2] for f in FASCE],
        "riepilogo": riepilogo,
        "classifica": classifica,
        "stazioni": elenco,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storico", required=True)
    ap.add_argument("--osservazioni", required=True)
    ap.add_argument("--punti", nargs="*", default=[], help="modello=percorso di punti_lazio.csv")
    a = ap.parse_args()
    storico = Path(a.storico)
    storico.mkdir(parents=True, exist_ok=True)
    for voce in a.punti:
        m, p = voce.split("=", 1)
        if Path(p).exists() and Path(p).stat().st_size > 0:
            print(f"{m}: corse archiviate {archivia(storico, m, p)}")
    oss, stazioni = leggi_osservazioni(a.osservazioni)
    pag = calcola(oss, stazioni, leggi_previsioni(storico, GIORNI))
    (storico / "pagella.json").write_text(json.dumps(pag, ensure_ascii=False, separators=(",", ":")))
    for v, cl in pag["classifica"].items():
        if cl:
            print(f"{v}: " + ", ".join(f"{x['modello']} {x['mae']} (n={x['n']})" for x in cl))
    print(f"pagella scritta: {len(pag['stazioni'])} stazioni, modelli {[m['id'] for m in pag['modelli']]}")


if __name__ == "__main__":
    main()
