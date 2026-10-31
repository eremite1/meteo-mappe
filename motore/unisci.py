"""
Unisce le mappe appena prodotte (una cartella per area) con quelle già pubblicate del modello.

Uso: python -m motore.unisci <cartella_ramo> <cartella_nuova> [<cartella_nuova> ...]
  - cartella_ramo: contenuto attuale del ramo mappe-<modello> (può essere vuota)
  - cartella_nuova: uscita di un lavoro (contiene <area>/ e indice_parziale.json)
"""
import os
import sys
import json
import shutil
from datetime import datetime, timezone

from .catalogo import MODELLI


def main(ramo, nuove):
    pi = os.path.join(ramo, "indice.json")
    indice = json.load(open(pi)) if os.path.exists(pi) else {}
    indice.setdefault("aree", {})
    indice.setdefault("corse", {})
    for cartella in nuove:
        pp = os.path.join(cartella, "indice_parziale.json")
        if not os.path.exists(pp):
            continue
        parz = json.load(open(pp))
        for chiave in ("modello", "nome", "risoluzione"):
            indice[chiave] = parz[chiave]
        for area, params in parz["aree"].items():
            dest = os.path.join(ramo, area)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(os.path.join(cartella, area), dest)
            indice["aree"][area] = params
            indice["corse"][area] = parz["corsa"]
        indice.setdefault("diagnostica", {}).update(parz.get("diagnostica", {}))
    # tengo solo le aree previste per il modello
    previste = set(MODELLI[indice["modello"]]["aree"])
    for area in list(indice["aree"]):
        if area not in previste:
            indice["aree"].pop(area)
            indice["corse"].pop(area, None)
            shutil.rmtree(os.path.join(ramo, area), ignore_errors=True)
    indice["corsa"] = max(indice["corse"].values()) if indice["corse"] else None
    indice["aggiornato_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    with open(pi, "w") as f:
        json.dump(indice, f, separators=(",", ":"))
    print(f"{indice['modello']}: aree {sorted(indice['aree'])}, corse {indice['corse']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
