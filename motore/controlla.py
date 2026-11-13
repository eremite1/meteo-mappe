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


_indici = {}


def corsa_pubblicata(modello, area=None):
    if modello not in _indici:
        _indici[modello] = None
        try:
            r = requests.get(f"https://raw.githubusercontent.com/{REPO}/mappe-{modello}/indice.json",
                             timeout=20)
            if r.status_code == 200:
                _indici[modello] = r.json()
        except Exception:
            pass
    ind = _indici[modello]
    if not ind:
        return None
    if area:
        return (ind.get("corse") or {}).get(area, ind.get("corsa") if area in ind.get("aree", {}) else None)
    return ind.get("corsa")


def _ore(vt):
    return (fonti_ora(vt[-1]) - fonti_ora(vt[0])) / 3600 if vt else 0


_scelte = {}


def scegli_corsa(modello, area=None):
    """La corsa più recente, completa e abbastanza lunga, disponibile per tutti i dataset del modello
    (o della sola area indicata: un'area può chiedere corse più lunghe, es. ECMWF Europa).
    Restituisce {dataset: info} oppure None."""
    chiave = (modello, area)
    if chiave in _scelte:
        return _scelte[chiave]
    from datetime import timedelta
    cfg = MODELLI[modello]
    aree = [area] if area else list(cfg["aree"])
    dataset = sorted({d for a in aree for d in fonti_di(modello, a)})
    min_ore = max(cfg.get("min_ore_aree", {}).get(a, cfg["min_ore"]) for a in aree)
    basi = {d: fonti.ultimo(d) for d in dataset}
    if any(b is None for b in basi.values()):
        _scelte[chiave] = None
        return None
    # candidati: dalla più vecchia "ultima corsa" dei dataset, indietro di 3 ore alla volta per 36 ore
    t0 = min(datetime.fromisoformat(b["reference_time"][:16]) for b in basi.values())
    for k in range(13):
        c = (t0 - timedelta(hours=3 * k)).strftime("%Y-%m-%dT%H:%M")
        infos = {}
        for d in dataset:
            j = fonti.info_corsa(d, c, basi[d])
            if j is None or _ore(j["valid_times"]) < min_ore:
                infos = None
                break
            infos[d] = j
        if infos:
            _scelte[chiave] = infos
            return infos
    _scelte[chiave] = None
    return None


def corsa_disponibile(modello, area=None):
    infos = scegli_corsa(modello, area)
    if not infos:
        return None
    rt = next(iter(infos.values()))["reference_time"]
    return rt[:16] + "Z"


def da_aggiornare(scelta="auto", forza=False):
    if scelta in ("auto", "", "tutti"):
        candidati = list(MODELLI)
    else:
        candidati = [m.strip() for m in scelta.split(",") if m.strip() in MODELLI]
    out = []
    for m in candidati:
        for a in MODELLI[m]["aree"]:
            nuova = corsa_disponibile(m, a)
            vecchia = corsa_pubblicata(m, a)
            print(f"{m}/{a}: disponibile {nuova} · pubblicata {vecchia}", file=sys.stderr)
            if nuova and (forza or scelta not in ("auto", "") or nuova != vecchia):
                out.append({"modello": m, "area": a})
    return out


if __name__ == "__main__":
    scelta = sys.argv[1] if len(sys.argv) > 1 else "auto"
    forza = len(sys.argv) > 2 and sys.argv[2].lower() in ("true", "1", "forza", "si")
    lavori = da_aggiornare(scelta, forza)
    print("lavori=" + json.dumps(lavori))
    print("modelli=" + json.dumps(sorted({l["modello"] for l in lavori})))
