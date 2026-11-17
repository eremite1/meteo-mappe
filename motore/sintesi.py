#!/usr/bin/env python3
"""Multimodello e media semplice ricostruiti ORA PER ORA sulla griglia della pagella unica (circa 2,5 km), con gli
stessi pesi e lo stesso modo di unire i modelli del Multimodello delle mappe (motore/multi.py, bordo di AROME
compreso). Servono perche' le mappe del Multimodello escono ogni 3 ore: cosi' anche lui viene confrontato con il
radar ora per ora, come tutti gli altri modelli.

  python -m motore.sintesi --pagella pagella.json --uscita CARTELLA chiave=griglia.npz ...
"""
import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np

from motore import verifica as V

COMPONENTI = ("icon2i", "arome", "ecmwf", "ukmo", "gfs")
CELLA_KM = 2.5
ORE = 72


def incrementi(percorso):
    """{istante: pioggia dell'ora precedente} dalle cumulate di una corsa (solo dove i dati sono orari)."""
    corsa, tempi, cum = V.leggi_griglia(percorso)
    out = {}
    for k in range(1, len(tempi)):
        if tempi[k] - tempi[k - 1] == timedelta(hours=1):
            out[tempi[k]] = np.clip(cum[k] - cum[k - 1], 0, None)
    return corsa, out


def ricostruisci(griglie, pag, uscita):
    from motore import multi as M
    M.AREA = "lazio"
    dati = {}
    for m in COMPONENTI:
        if m in griglie and Path(griglie[m]).exists():
            try:
                dati[m] = incrementi(griglie[m])
            except Exception as e:  # noqa: BLE001
                print(f"{m}: griglia non letta ({e})")
    if len(dati) < 2:
        print("servono almeno due modelli: multimodello non ricostruito")
        return {}
    ref = max(c for c, _ in dati.values())
    tempi, pes, ugu = [ref], [], []
    for h in range(1, ORE + 1):
        t = ref + timedelta(hours=h)
        campi = {m: inc[t] for m, (c, inc) in dati.items() if t in inc}
        if len(campi) < 2:
            break
        presenti = {m: int(round((t - dati[m][0]).total_seconds() / 3600)) for m in campi}
        w = M.pesi(pag, "P", presenti)
        pes.append(M.media(campi, w, moltiplicativo=True, cella_km=CELLA_KM))
        ugu.append(M.media(campi, {m: 1.0 for m in campi}, moltiplicativo=True, cella_km=CELLA_KM))
        tempi.append(t)
    if len(tempi) < 13:
        print("troppe poche ore in comune: multimodello non ricostruito")
        return {}
    out = {}
    for chiave, serie in (("multi", pes), ("media", ugu)):
        cum = [np.zeros((V.NY, V.NX))]
        for x in serie:
            cum.append(cum[-1] + np.nan_to_num(x))
        cum = [np.where(np.isfinite(serie[0]), c, np.nan) for c in cum]
        p = Path(uscita) / f"{chiave}.npz"
        V.salva_griglia(p, ref, tempi, cum)
        out[chiave] = p
    print(f"multimodello ricostruito: corsa {ref:%Y-%m-%d %H} UTC, {len(tempi) - 1} ore, modelli {', '.join(dati)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pagella", default="")
    ap.add_argument("--uscita", required=True)
    ap.add_argument("griglie", nargs="*")
    a = ap.parse_args()
    pag = json.loads(Path(a.pagella).read_text()) if a.pagella and Path(a.pagella).exists() else None
    Path(a.uscita).mkdir(parents=True, exist_ok=True)
    ricostruisci(dict(x.split("=", 1) for x in a.griglie), pag, a.uscita)


if __name__ == "__main__":
    main()
