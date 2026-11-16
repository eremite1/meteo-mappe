#!/usr/bin/env python3
"""CONFRONTO OTTICO GIORNALIERO: per ogni giorno, la pioggia vista dal radar e quella prevista da OGNI corsa di OGNI
modello (Mappe dei modelli e WRF, varianti di fisica comprese) che copre quel giorno, a blocchi di 3 ore e sulle 24 ore.
Per ogni confronto il voto (FSS: quanto coincidono le zone di pioggia, quadrati di circa 35 km e 7 km), la quantita'
(pioggia prevista / pioggia vera sulla stessa zona) e lo spostamento della pioggia.

La pagina del sito disegna le mappe da sola: qui si preparano dati compatti.
  confronto/indice.json                       giorni disponibili
  confronto/AAAAMMGG/indice.json              finestre, modelli, corse e voti
  confronto/AAAAMMGG/radar.bin                9 mappe (24 ore + 8 blocchi di 3 ore), 1 byte per cella, compresse gzip
  confronto/AAAAMMGG/<modello>.bin            le stesse 9 mappe per ogni corsa, una dopo l'altra
Codifica di ogni cella come motore/verifica.comprimi: 0 = niente, 1-254 = mm in scala logaritmica, 255 = nessun dato.
Il giorno va da mezzanotte a mezzanotte UTC (dall'01 all'01 ora italiana d'inverno).
"""
import gzip
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from motore import verifica as V

GIORNI_TENUTI = 10
ORE_PRIMA = 72              # corse partite fino a 3 giorni prima
NOMI = {"multi": "Multimodello", "icon2i": "ICON-2I", "arome": "AROME", "ecmwf": "ECMWF IFS", "ukmo": "UKMO", "gfs": "GFS",
        "iconeu": "ICON-EU", "wrf3km": "WRF 3 km · ECMWF", "wrf1km": "WRF 1 km · ECMWF", "wrf333m": "WRF 333 m · ECMWF",
        "wrfgfs3km": "WRF 3 km · GFS", "wrfgfs1km": "WRF 1 km · GFS", "wrficoneu3km": "WRF 3 km · ICON-EU",
        "wrficoneu1km": "WRF 1 km · ICON-EU", "wrficon2i3km": "WRF 3 km · ICON-2I", "wrficon2i1km": "WRF 1 km · ICON-2I"}
BASI = {"gfs": "GFS", "ecmwf": "ECMWF", "iconeu": "ICON-EU", "icon2i": "ICON-2I"}


def nome(k):
    if k in NOMI:
        return NOMI[k]
    if k.startswith("wrffis_"):                       # wrffis_gfs_f3 -> WRF 3 km · GFS · fisica 3
        _, base, f = k.split("_", 2)
        return f"WRF 3 km · {BASI.get(base, base)} · fisica {f.lstrip('f')}"
    return k


def gruppo(k):
    return "fisica" if k.startswith("wrffis_") else ("wrf" if k.startswith("wrf") else "modelli")


SOGLIE = {"tot": (1, 10, 30), "3h": (1, 5, 15)}      # pioggia debole, moderata, forte (mm)


def fss(f, o, m, soglia, scala):
    from scipy import ndimage
    pf = ((np.nan_to_num(f) >= soglia) & m).astype("float64")
    po = ((np.nan_to_num(o) >= soglia) & m).astype("float64")
    if pf.sum() + po.sum() == 0:
        return None                                    # nessuno dei due arriva a questa soglia: non conta
    n = ndimage.uniform_filter(m.astype("float64"), scala, mode="constant")
    ok = (n > 0.5) & m
    ff = ndimage.uniform_filter(pf, scala, mode="constant")[ok] / n[ok]
    fo = ndimage.uniform_filter(po, scala, mode="constant")[ok] / n[ok]
    rif = np.mean(ff ** 2) + np.mean(fo ** 2)
    return None if rif <= 0 else float(1 - np.mean((ff - fo) ** 2) / rif)


def voto(f, o, m, soglie, scala):
    """Media del FSS alle tre soglie: con la sola soglia di 1 mm, quando piove ovunque, tutti prenderebbero 100."""
    v = [x for x in (fss(f, o, m, s, scala) for s in soglie) if x is not None]
    return None if not v else sum(v) / len(v)


def misure(f, o, tipo="3h"):
    m = np.isfinite(f) & np.isfinite(o)
    if m.sum() < 300:
        return None
    s35, s7 = voto(f, o, m, SOGLIE[tipo], V.SCALE["fss"]), voto(f, o, m, SOGLIE[tipo], V.SCALE["fss_fine"])
    pf, po = float(np.nansum(np.where(m, f, 0))), float(np.nansum(np.where(m, o, 0)))
    secco_f, secco_o = not (np.nan_to_num(f)[m] >= V.SOGLIA).any(), not (np.nan_to_num(o)[m] >= V.SOGLIA).any()
    r = {"q": None if po < 0.5 * m.sum() * 0.01 else round(pf / po, 2),
         "mf": round(pf / m.sum(), 1), "mo": round(po / m.sum(), 1), "cop": round(100 * float(m.mean()))}
    if secco_f and secco_o:
        r["v"] = r["v7"] = None
        r["secco"] = True                              # asciutto nel modello e nella realta': indovinato, ma non fa voto
    else:
        r["v"] = None if s35 is None else round(100 * s35)
        r["v7"] = None if s7 is None else round(100 * s7)
    sp = V.spostamento(f, o) if not (secco_f or secco_o) else None
    if sp:
        r["sp"] = [round(sp[0]), round(sp[1])]
    return r


def scrivi_bin(percorso, campi):
    vuoto = np.full((V.NY, V.NX), 255, np.uint8)
    dati = b"".join((V.comprimi(c) if c is not None else vuoto).tobytes() for c in campi)
    with gzip.open(percorso, "wb", compresslevel=9) as g:
        g.write(dati)


def giorno(storico, per_ora, g, adesso):
    """Prepara la cartella del giorno g (mezzanotte UTC). Ritorna il riassunto per l'indice generale o None."""
    arch = Path(storico) / "verifica" / "archivio"
    fini = [g + timedelta(hours=3 * (w + 1)) for w in range(8)]
    radar = []
    for t in fini:
        ore = [per_ora.get(t - timedelta(hours=k)) for k in range(3)]
        radar.append(None if any(x is None for x in ore) else np.stack(ore).sum(axis=0))
    disp = [w for w in range(8) if radar[w] is not None]
    if not disp:
        return None
    tot_r = np.sum([radar[w] for w in disp], axis=0)
    cart = Path(storico) / "confronto" / f"{g:%Y%m%d}"
    tmp = cart.with_name(cart.name + ".nuovo")
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    scrivi_bin(tmp / "radar.bin", [tot_r] + radar)
    ok = np.isfinite(tot_r)
    ix, iy = int((12.786 - V.LON0) / V.PASSO), int((42.163 - V.LAT0) / V.PASSO)
    indice = {"giorno": f"{g:%Y-%m-%d}", "aggiornato": V.iso(adesso), "finestre": [V.iso(t) for t in fini], "radar_finestre": disp,
              "completo": len(disp) == 8, "radar": {"media": round(float(np.nanmean(tot_r)), 1), "max": round(float(np.nanmax(tot_r)), 1),
              "area": round(100 * float((tot_r[ok] >= V.SOGLIA).mean())), "nerola": round(float(tot_r[iy, ix]), 1)},
              "griglia": {"lon0": V.LON0, "lat0": V.LAT0, "passo": V.PASSO, "nx": V.NX, "ny": V.NY}, "modelli": {}}
    fin_iso = {V.iso(t): w for w, t in enumerate(fini)}
    per_mod = {}
    for f in sorted(arch.glob("*.npz")):
        try:
            k, cs = f.stem.split("__")
            corsa = datetime.strptime(cs, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if corsa >= g + timedelta(hours=24) or corsa < g - timedelta(hours=ORE_PRIMA):
            continue
        z = np.load(f)
        campi = [None] * 8
        for j, t in enumerate(z["fini"]):
            w = fin_iso.get(str(t))
            if w is not None:
                campi[w] = V.espandi(z["mm"][j])
        if all(c is None for c in campi):
            continue
        per_mod.setdefault(k, []).append((corsa, campi))
    for k, corse in per_mod.items():
        corse.sort(key=lambda x: x[0], reverse=True)
        voci, tutti = [], []
        for corsa, campi in corse:
            completo = all(campi[w] is not None for w in disp)
            tot_f = np.sum([campi[w] for w in disp], axis=0) if completo else None
            voti = {"tot": misure(tot_f, tot_r, "tot") if completo else None}
            for w in disp:
                if campi[w] is not None:
                    voti[str(w)] = misure(campi[w], radar[w])
            n = tot_f[iy, ix] if completo else np.nan
            voci.append({"corsa": V.iso(corsa), "ore": int((g - corsa).total_seconds() // 3600), "completo": completo,
                         "prima": corsa <= g, "voti": voti, "nerola": None if not np.isfinite(n) else round(float(n), 1)})
            tutti += [tot_f] + campi
        scrivi_bin(tmp / f"{k}.bin", tutti)
        indice["modelli"][k] = {"nome": nome(k), "gruppo": gruppo(k), "corse": voci}
    (tmp / "indice.json").write_text(json.dumps(indice, ensure_ascii=False, separators=(",", ":")))
    shutil.rmtree(cart, ignore_errors=True); tmp.rename(cart)
    migliore = max(((m["nome"], c["voti"]["tot"]["v"]) for m in indice["modelli"].values() for c in m["corse"]
                    if c["prima"] and c["voti"].get("tot") and c["voti"]["tot"].get("v") is not None), key=lambda x: x[1], default=None)
    return {"giorno": indice["giorno"], "completo": indice["completo"], "radar": indice["radar"], "modelli": len(indice["modelli"]),
            "migliore": migliore}


def confini(percorso):
    """Coste, confini regionali e capoluoghi sulla griglia comune, PNG trasparente da sovrapporre alle mappe (una volta sola)."""
    if percorso.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from motore.tavola import _confini, CITTA
    k = 6
    fig = plt.figure(figsize=(V.NX * k / 100, V.NY * k / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    _confini(ax)
    for lw, c in ((1.6, "#222"),):
        for l in ax.lines:
            l.set_color(c); l.set_linewidth(lw * (1 if l.get_linewidth() > .6 else .6))
    for c, (x, y) in CITTA.items():
        ax.plot(x, y, "o", ms=4 if c == "Nerola" else 3, color="#d0002a" if c == "Nerola" else "#111")
        ax.text(x + .03, y + .03, c, fontsize=9 if c == "Nerola" else 8, weight="bold", color="#d0002a" if c == "Nerola" else "#111")
    ax.set_xlim(V.LON0, V.LON0 + V.PASSO * V.NX); ax.set_ylim(V.LAT0, V.LAT0 + V.PASSO * V.NY)
    fig.savefig(percorso, transparent=True)
    plt.close(fig)


def aggiorna(storico, per_ora, adesso):
    """Rifa oggi e ieri (arrivano corse e ore di radar nuove); i giorni piu' vecchi solo se mancano."""
    base = Path(storico) / "confronto"
    base.mkdir(parents=True, exist_ok=True)
    try:
        confini(base / "confini.png")
    except Exception as e:  # noqa: BLE001
        print("confini non disegnati:", e)
    ip = base / "indice.json"
    vecchio = {d["giorno"]: d for d in json.loads(ip.read_text()).get("giorni", [])} if ip.exists() else {}
    oggi = adesso.replace(hour=0, minute=0, second=0, microsecond=0)
    ore = sorted(per_ora)
    if not ore:
        return
    primo = ore[0].replace(hour=0, minute=0, second=0, microsecond=0)
    g = max(primo, oggi - timedelta(days=GIORNI_TENUTI - 1))
    while g <= oggi:
        chiave = f"{g:%Y-%m-%d}"
        if g >= oggi - timedelta(days=1) or chiave not in vecchio or not (base / f"{g:%Y%m%d}").exists():
            try:
                r = giorno(storico, per_ora, g, adesso)
                if r:
                    vecchio[chiave] = r
                    print(f"confronto {chiave}: {r['modelli']} modelli, radar {r['radar']}, migliore {r['migliore']}")
            except Exception as e:  # noqa: BLE001
                print(f"confronto {chiave}: errore {e}")
        g += timedelta(days=1)
    limite = f"{oggi - timedelta(days=GIORNI_TENUTI - 1):%Y-%m-%d}"
    for d in list(vecchio):
        if d < limite:
            vecchio.pop(d)
            shutil.rmtree(base / d.replace("-", ""), ignore_errors=True)
    ip.write_text(json.dumps({"aggiornato": V.iso(adesso), "giorni": sorted(vecchio.values(), key=lambda x: x["giorno"], reverse=True)},
                             ensure_ascii=False, indent=1))
