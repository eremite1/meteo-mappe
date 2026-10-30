"""
Controlla quali modelli hanno una corsa nuova rispetto a quella già pubblicata.

Uso:  python -m motore.controlla [auto|tutti|icon2i,ecmwf,...] [forza]
Stampa una riga per GitHub Actions:  modelli=["icon2i","ecmwf"]
"""
import os
import sys
import json
import requests

from .catalogo import MODELLI, fonti_di
from . import fonti

from datetime import datetime, timezone


def fonti_ora(s):
    s = s.replace("Z", "")
    return datetime.fromisoformat(s[:16]).replace(tzinfo=timezone.utc).timestamp()


REPO = os.environ.get("GITHUB_REPOSITORY", "eremite1/meteo-mappe")


def corsa_pubblicata(modello):
    try:
        r = requests.get(f"https://raw.githubusercontent.com/{REPO}/mappe-{modello}/indice.json",
                         timeout=20)
        if r.status_code == 200:
            return r.json().get("corsa")
    except Exception:
        pass
    return None


def corsa_disponibile(modello):
    """Corsa completa disponibile per tutti i dataset del modello (o None)."""
    cfg = MODELLI[modello]
    dataset = sorted({d for a in cfg["aree"] for d in fonti_di(modello, a)})
    corse = set()
    for d in dataset:
        j = fonti.ultimo(d)
        if j is None or not j.get("completed", True):
            return None
        vt = j["valid_times"]
        ore = (fonti_ora(vt[-1]) - fonti_ora(vt[0])) / 3600
        if ore < cfg["min_ore"]:
            return None                     # corsa breve intermedia: si aspetta quella lunga
        corse.add(j["reference_time"])
    if len(corse) != 1:
        return None
    rt = next(iter(corse))
    return rt[:16] + "Z" if len(rt) > 16 else rt


def da_aggiornare(scelta="auto", forza=False):
    if scelta in ("auto", "", "tutti"):
        candidati = list(MODELLI)
    else:
        candidati = [m.strip() for m in scelta.split(",") if m.strip() in MODELLI]
    out = []
    for m in candidati:
        nuova = corsa_disponibile(m)
        vecchia = corsa_pubblicata(m)
        print(f"{m}: disponibile {nuova} · pubblicata {vecchia}", file=sys.stderr)
        if nuova and (forza or scelta not in ("auto", "") or nuova != vecchia):
            out.append(m)
    return out


if __name__ == "__main__":
    scelta = sys.argv[1] if len(sys.argv) > 1 else "auto"
    forza = len(sys.argv) > 2 and sys.argv[2].lower() in ("true", "1", "forza", "si")
    print("modelli=" + json.dumps(da_aggiornare(scelta, forza)))
