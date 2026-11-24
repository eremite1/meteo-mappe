"""
CARTA SINOTTICA SUL SATELLITE: l'analisi automatica (fronti, centri, getto, linee temporalesche...) dell'ora
piu' vicina ad adesso, calcolata con l'ultima corsa ECMWF, disegnata sopra l'immagine Meteosat piu' recente
"RGB Airmass" di EUMETSAT (EUMETView, servizio WMS pubblico): in quella immagine le masse d'aria hanno colori
diversi (rosso/arancio = aria secca scesa dall'alto, tipica dietro le saccature; verde = aria tropicale calda e
umida; blu/viola = aria fredda polare o artica; bianco = nubi alte e spesse).

Uscita: sat/sinottica_sat.jpg e sat/sinottica_sat.json. Si avvia con:  python -m motore.sinottica_sat
"""
import io
import json
import os
import shutil
import sys
import urllib.request
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from .genera import Lavoro, Contesto, CARTELLA_DATI, testo_ora, log
from .catalogo import _sinottica
from .analisi import riassunto
from .disegno import Disegnatore
from .disegno_sinottica import disegna_sinottica

WMS = "https://view.eumetsat.int/geoserver/wms"
STRATI = ["mtg_fd:rgb_airmass", "msg_fes:rgb_airmass", "mtg_fd:ir105_hrfi"]
BBOX = [-28.0, 48.0, 28.0, 72.0]                     # lon0, lon1, lat0, lat1 (piu' largo della carta)
USCITA = "sat"


def satellite():
    """(immagine, strato, ora) dell'ultima immagine disponibile; la prima che funziona tra quelle in STRATI."""
    for strato in STRATI:
        url = (f"{WMS}?service=WMS&version=1.1.1&request=GetMap&layers={strato}&styles=&srs=EPSG:4326"
               f"&bbox={BBOX[0]},{BBOX[2]},{BBOX[1]},{BBOX[3]}&width=2000&height={int(2000 * (BBOX[3] - BBOX[2]) / (BBOX[1] - BBOX[0]))}"
               f"&format=image/png&transparent=false")
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "meteo-mappe (www.meteonerola.it)"}), timeout=180)
            tipo = r.headers.get("Content-Type", "")
            dati = r.read()
            if "image" not in tipo:
                log(f"{strato}: risposta non valida ({tipo}): {dati[:200]!r}")
                continue
            img = np.asarray(Image.open(io.BytesIO(dati)).convert("RGB"))
            if img.mean() < 3:
                log(f"{strato}: immagine vuota")
                continue
            return img, strato
        except Exception as e:  # noqa: BLE001
            log(f"{strato}: {e}")
    return None, None


def main():
    adesso = datetime.now(timezone.utc)
    L = Lavoro("ecmwf", ["europa"])
    ist = L.istanti("europa")
    sel = min(ist, key=lambda x: abs((x[2] - adesso).total_seconds()))
    k, h, t = sel
    log(f"Corsa ECMWF {L.corsa:%Y-%m-%d %H} UTC, istante +{h}h ({t:%d/%m %H} UTC)")
    L.cfg = dict(L.cfg, aree={"europa": (L.cfg["aree"]["europa"][0], max(h, 3))})
    L.istanti = lambda area: [sel]
    if os.path.exists(CARTELLA_DATI):
        shutil.rmtree(CARTELLA_DATI)
    L.scarica()
    try:
        L.prepara_cumulate()
    except Exception as e:  # noqa: BLE001
        log(f"cumulate non preparate: {e}")
    img, strato = satellite()
    if img is None:
        sys.exit("nessuna immagine del satellite")
    la, lo = L.griglie["europa"]
    dis = Disegnatore("europa", la, lo)
    r = _sinottica(Contesto(L, "europa", t))
    os.makedirs(USCITA, exist_ok=True)
    data, ora_utc = testo_ora(t)
    nome_strato = "RGB Airmass" if "airmass" in strato else "infrarosso"
    info = f"Satellite Meteosat ({nome_strato}) © EUMETSAT · analisi su ECMWF corsa {L.corsa:%d/%m %H} UTC"
    disegna_sinottica(dis, os.path.join(USCITA, "sinottica_sat.png"), r["analisi"],
                      f"Satellite: {testo_ora(adesso)[0]} circa · analisi: {data}", None, info,
                      titolo="Carta sinottica sul satellite", sfondo=(img, BBOX))
    out = riassunto(r["griglia"], r["analisi"])
    out.update(satellite=strato, ora_satellite=adesso.strftime("%Y-%m-%dT%H:%MZ"), analisi=t.strftime("%Y-%m-%dT%H:%MZ"),
               corsa=L.corsa.strftime("%Y-%m-%dT%H:%MZ"), scadenza=h)
    with open(os.path.join(USCITA, "sinottica_sat.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False)
    log("Fatto: " + ", ".join(sorted(os.listdir(USCITA))))


if __name__ == "__main__":
    main()
