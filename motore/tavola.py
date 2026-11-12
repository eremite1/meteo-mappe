#!/usr/bin/env python3
"""TAVOLA GIORNALIERA: la pioggia di un giorno intero vista dal radar, accanto a quella prevista da ogni modello
con la corsa del giorno prima. Confronto "a occhio": stesse mappe, stessa scala di colori, stessa cornice.

Si usa l'archivio della pagella unica (verifica/archivio: pioggia ogni 3 ore di ogni corsa sulla griglia comune
del Lazio) e il radar orario sulla stessa griglia. Il giorno va da mezzanotte a mezzanotte UTC (dall'01 all'01
ora italiana d'inverno), perche' i modelli danno la pioggia a blocchi di 3 ore che partono da mezzanotte UTC.

Per ogni modello si prende la corsa piu' recente partita PRIMA dell'inizio del giorno (la previsione "del giorno
prima"): nessun modello vede in anticipo la pioggia che deve prevedere.
Uscite: tavole/pioggia_AAAAMMGG.png e tavole/indice.json (ultimi 14 giorni).
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from motore import verifica as V

GIORNI_TENUTI = 14
ORE_RADAR_MINIME = 22          # su 24: il radar a volte salta un'ora
LIVELLI = [1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
COLORI = ["#c8e6ff", "#96cdff", "#5aaaf5", "#1e78dc", "#00b45a", "#5ad200",
          "#c8e600", "#ffdc00", "#ffa000", "#ff3c00", "#c80078", "#800080"]
NOMI = {"multi": "Multimodello", "media": "Media semplice", "icon2i": "ICON-2I", "arome": "AROME", "ecmwf": "ECMWF IFS",
        "ukmo": "UKMO", "gfs": "GFS", "iconeu": "ICON-EU", "wrf3km": "WRF 3 km · ECMWF", "wrf1km": "WRF 1 km · ECMWF",
        "wrf333m": "WRF 333 m", "wrfgfs3km": "WRF 3 km · GFS", "wrfgfs1km": "WRF 1 km · GFS",
        "wrficoneu3km": "WRF 3 km · ICON-EU", "wrficoneu1km": "WRF 1 km · ICON-EU",
        "wrficon2i3km": "WRF 3 km · ICON-2I", "wrficon2i1km": "WRF 1 km · ICON-2I"}
ORDINE = list(NOMI)
CITTA = {"RM": (12.4964, 41.9028), "LT": (12.9043, 41.4676), "FR": (13.3441, 41.6390), "RI": (12.8634, 42.4036),
         "VT": (12.1067, 42.4207), "Nerola": (12.7856, 42.1606)}


def previsione_del_giorno(arch, giorno):
    """chiave -> (corsa, mm 24 h) dalla corsa piu' recente partita prima del giorno che lo copre tutto."""
    fini_giorno = [V.iso(giorno + timedelta(hours=h)) for h in range(3, 25, 3)]
    migliori = {}
    for f in arch.glob("*.npz"):
        try:
            chiave, corsa_s = f.stem.split("__")
            corsa = datetime.strptime(corsa_s, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if chiave.startswith("wrffis_"):
            continue                                   # varianti di fisica: sono nella classifica ottica, non nella tavola
        if corsa >= giorno or corsa < giorno - timedelta(hours=48):
            continue
        if chiave in migliori and migliori[chiave][0] >= corsa:
            continue
        z = np.load(f)
        pos = {str(t): k for k, t in enumerate(z["fini"])}
        if not all(t in pos for t in fini_giorno):
            continue
        mm = sum(V.espandi(z["mm"][pos[t]]) for t in fini_giorno)
        migliori[chiave] = (corsa, mm)
    return migliori


def radar_del_giorno(per_ora, giorno):
    ore = [per_ora.get(giorno + timedelta(hours=h)) for h in range(1, 25)]
    presenti = [o for o in ore if o is not None]
    if len(presenti) < ORE_RADAR_MINIME:
        return None, len(presenti)
    st = np.stack(presenti)
    tot = np.nansum(st, axis=0) * 24 / len(presenti)      # se manca un'ora, la stima la compensa
    tot[~np.isfinite(st).any(axis=0)] = np.nan
    return tot, len(presenti)


def _confini(ax):
    """Coste e confini regionali, se cartopy e' disponibile (sul server delle pagelle si'); altrimenti niente."""
    try:
        import cartopy.io.shapereader as shp
        for nome, col, lw in (("coastline", "#333", .7), ("admin_1_states_provinces_lines", "#555", .45)):
            cat = "physical" if nome == "coastline" else "cultural"
            for g in shp.Reader(shp.natural_earth("10m", cat, nome)).geometries():
                for parte in getattr(g, "geoms", [g]):
                    x, y = parte.xy if hasattr(parte, "xy") else parte.exterior.xy
                    ax.plot(x, y, color=col, lw=lw)
    except Exception:  # noqa: BLE001
        pass


def disegna(percorso, giorno, radar, ore_radar, prev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(COLORI)
    cmap.set_over("#4b004b")
    cmap.set_under((1, 1, 1, 0))
    norm = BoundaryNorm(LIVELLI, len(COLORI))
    chiavi = sorted(prev, key=lambda k: (ORDINE.index(k) if k in ORDINE else 99, k))
    pannelli = [("Radar (pioggia caduta davvero)", radar, None)] + [(NOMI.get(k, k), prev[k][1], prev[k][0]) for k in chiavi]
    nc = 4
    nr = (len(pannelli) + nc - 1) // nc
    fig, assi = plt.subplots(nr, nc, figsize=(nc * 3.3, nr * 3.55 + 1.3), squeeze=False)
    fig.subplots_adjust(left=.01, right=.99, top=1 - .95 / (nr * 3.55 + 1.3), bottom=.75 / (nr * 3.55 + 1.3), wspace=.04, hspace=.22)
    lon = np.concatenate([[V.LON0], V.LON0 + V.PASSO * (np.arange(V.NX) + 1)])
    lat = np.concatenate([[V.LAT0], V.LAT0 + V.PASSO * (np.arange(V.NY) + 1)])
    rif = np.nanmean(radar)
    for n, ax in enumerate(assi.flat):
        ax.set_xticks([]); ax.set_yticks([])
        if n >= len(pannelli):
            ax.axis("off")
            continue
        titolo, campo, corsa = pannelli[n]
        ax.set_facecolor("#f4f6f8")
        ax.pcolormesh(lon, lat, np.ma.masked_invalid(campo), cmap=cmap, norm=norm, shading="flat")
        _confini(ax)
        for c, (x, y) in CITTA.items():
            ax.plot(x, y, "o", ms=2.5 if c != "Nerola" else 3.5, color="k")
            ax.text(x + .03, y + .03, c, fontsize=6.5 if c != "Nerola" else 7.5, weight="bold" if c == "Nerola" else None)
        ax.set_xlim(lon[0], lon[-1]); ax.set_ylim(lat[0], lat[-1]); ax.set_aspect(1 / np.cos(np.radians(42.1)))
        ok = np.isfinite(campo) & np.isfinite(radar)
        media = np.nanmean(np.where(ok, campo, np.nan)) if ok.any() else np.nan
        rif_q = np.nanmean(np.where(ok, radar, np.nan)) if ok.any() else np.nan     # radar sulla stessa zona
        sotto = (f"radar: {ore_radar}/24 ore" if corsa is None
                 else f"corsa {corsa:%d/%m %H} UTC · media {media:.1f} mm"
                 + (f" ({media / rif_q * 100:.0f}% del radar)" if rif_q > .2 and np.isfinite(media) else ""))
        if corsa is None:
            sotto = f"media {rif:.1f} mm · " + sotto
        ax.set_title(titolo, fontsize=10, weight="bold", loc="left", pad=14)
        ax.text(0, 1.015, sotto, transform=ax.transAxes, fontsize=7.2, color="#444")
        if n == 0:
            for s in ax.spines.values():
                s.set_edgecolor("#111"); s.set_linewidth(2.2)
    fine = giorno + timedelta(days=1)
    H = nr * 3.55 + 1.3
    fig.suptitle(f"Pioggia di {['lunedì', 'martedì', 'mercoledì', 'giovedì', 'venerdì', 'sabato', 'domenica'][giorno.weekday()]} "
                 f"{giorno:%d/%m/%Y}: radar contro modelli (previsione del giorno prima)", fontsize=13, weight="bold", x=.01, ha="left", y=1 - .15 / H)
    fig.text(.01, 1 - .5 / H, f"24 ore dalle 01 del {giorno:%d/%m} alle 01 del {fine:%d/%m} (ora italiana) · stessa scala di colori per tutti · "
             "www.meteonerola.it", fontsize=8, color="#555")
    cax = fig.add_axes([.2, .42 / H, .6, .14 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max", ticks=LIVELLI)
    cb.set_label("mm in 24 ore", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.savefig(percorso, dpi=80)
    plt.close(fig)


def aggiorna(storico, per_ora, adesso):
    """Disegna le tavole dei giorni completi che mancano (al massimo gli ultimi GIORNI_TENUTI)."""
    storico = Path(storico)
    arch = storico / "verifica" / "archivio"
    cart = storico / "tavole"
    cart.mkdir(parents=True, exist_ok=True)
    ip = cart / "indice.json"
    indice = json.loads(ip.read_text()) if ip.exists() else {"giorni": {}}
    oggi = adesso.replace(hour=0, minute=0, second=0, microsecond=0)
    fatti = 0
    for k in range(GIORNI_TENUTI, 0, -1):
        giorno = oggi - timedelta(days=k)
        nome = f"pioggia_{giorno:%Y%m%d}.png"
        g = giorno.strftime("%Y-%m-%d")
        if g in indice["giorni"] and indice["giorni"][g].get("completa"):
            continue
        radar, ore_radar = radar_del_giorno(per_ora, giorno)
        if radar is None:
            continue
        prev = previsione_del_giorno(arch, giorno)
        if not prev:
            continue
        disegna(cart / nome, giorno, radar, ore_radar, prev)
        ok = np.isfinite(radar)
        indice["giorni"][g] = {
            "file": f"tavole/{nome}", "ore_radar": ore_radar, "radar_mm": round(float(np.nanmean(radar)), 1),
            "modelli": {m: {"nome": NOMI.get(m, m), "corsa": V.iso(c), "mm": round(float(np.nanmean(np.where(ok, x, np.nan))), 1)}
                        for m, (c, x) in prev.items()},
            # completa: il radar ha tutte le ore (altrimenti la si ridisegna se arrivano le ore mancanti)
            "completa": ore_radar == 24}
        fatti += 1
    limite = (oggi - timedelta(days=GIORNI_TENUTI)).strftime("%Y-%m-%d")
    for g in [g for g in indice["giorni"] if g < limite]:
        (cart / Path(indice["giorni"][g]["file"]).name).unlink(missing_ok=True)
        del indice["giorni"][g]
    indice["aggiornato"] = V.iso(adesso)
    ip.write_text(json.dumps(indice, ensure_ascii=False, separators=(",", ":")))
    return fatti
