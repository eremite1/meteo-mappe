#!/usr/bin/env python3
"""CONTROLLO ADESSO: la pioggia delle ultime ore vista dal radar confrontata con TUTTI i modelli (Mappe Lazio e WRF),
ciascuno con la sua corsa piu' recente che copre quelle ore. Stesse misure della pagella unica (motore/verifica.py):
voto FSS a 35 km e 7 km, pioggia prevista / pioggia vera, spostamento della pioggia. Scrive una tabella e un mosaico.

  python -m motore.adesso --radar radar_orario.npz --griglie k=f.npz ... --wrf k=f.npz ... --uscita DIR
"""
import argparse
import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np

from motore import verifica as V

NEROLA = (12.786, 42.163)


def campo(tempi, cum, t0, t1):
    pos = {t: k for k, t in enumerate(tempi)}
    if t1 not in pos or t0 not in pos:
        return None
    return np.clip(cum[pos[t1]] - cum[pos[t0]], 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radar", required=True)
    ap.add_argument("--griglie", nargs="*", default=[])
    ap.add_argument("--wrf", nargs="*", default=[])
    ap.add_argument("--ore", type=int, default=3)
    ap.add_argument("--uscita", default="controllo")
    a = ap.parse_args()
    out = Path(a.uscita); out.mkdir(parents=True, exist_ok=True)

    per = V.radar_3ore(a.radar)
    ore = sorted(per)
    fine = ore[-1]
    fin = [fine - timedelta(hours=k) for k in range(a.ore)]
    if any(t not in per for t in fin):
        raise SystemExit("radar incompleto nelle ultime ore")
    oss = np.stack([per[t] for t in fin]).sum(axis=0)
    inizio = fine - timedelta(hours=a.ore)
    ix = int((NEROLA[0] - V.LON0) / V.PASSO); iy = int((NEROLA[1] - V.LAT0) / V.PASSO)
    ok = np.isfinite(oss)
    print(f"Radar: pioggia da {inizio:%d/%m %H:%M} a {fine:%H:%M} UTC; area bagnata (>=1 mm) {100 * (oss[ok] >= 1).mean():.0f}% del Lazio, "
          f"massimo {np.nanmax(oss):.1f} mm, Nerola {oss[iy, ix]:.1f} mm")

    modelli = {}
    for kv in a.griglie:
        k, f = kv.split("=", 1)
        try:
            c, t, cum = V.leggi_griglia(f)
            modelli[k] = (c, t, cum)
        except Exception as e:
            print(f"{k}: illeggibile ({e})")
    for kv in a.wrf:
        k, f = kv.split("=", 1)
        try:
            z = np.load(f)
            t = [V.ora(str(x)) for x in z["tempi"]]
            cum = np.stack(V.su_griglia(z["lat"], z["lon"], z["RAINC"] + z["RAINNC"]))
            modelli[k] = (t[0], t, cum)
        except Exception as e:
            print(f"{k}: illeggibile ({e})")

    righe, campi = [], {}
    po = float(np.nansum(np.where(ok, oss, 0)))
    for k, (corsa, t, cum) in sorted(modelli.items()):
        f = campo(t, cum, inizio, fine)
        if f is None:
            righe.append({"modello": k, "corsa": f"{corsa:%d/%m %H}", "nota": "non copre queste ore"})
            continue
        m = ok & np.isfinite(f)
        if m.sum() < 500:
            righe.append({"modello": k, "corsa": f"{corsa:%d/%m %H}", "nota": "fuori dal Lazio"})
            continue
        s35, s7 = V.fss(f, oss, V.SCALE["fss"]), V.fss(f, oss, V.SCALE["fss_fine"])
        pf = float(np.nansum(np.where(m, f, 0))); pom = float(np.nansum(np.where(m, oss, 0)))
        sp = V.spostamento(f, oss)
        campi[k] = f
        righe.append({"modello": k, "corsa": f"{corsa:%d/%m %H}", "scadenza": int((fine - corsa).total_seconds() // 3600),
                      "voto35": None if s35 is None else round(100 * s35), "voto7": None if s7 is None else round(100 * s7),
                      "quantita": None if pom <= 0 else round(pf / pom, 2),
                      "area": round(100 * float((f[m] >= 1).mean())),
                      "sposta_km": None if sp is None else [round(sp[0]), round(sp[1])],
                      "nerola": None if not np.isfinite(f[iy, ix]) else round(float(f[iy, ix]), 1)})
    righe.sort(key=lambda r: -(r.get("voto35") if r.get("voto35") is not None else -1))
    print(f"\n{'modello':<22}{'corsa':>9}{'+h':>4}{'voto35':>8}{'voto7':>7}{'prev/vera':>10}{'area%':>7}{'Nerola':>8}  spostamento")
    for r in righe:
        if "nota" in r:
            print(f"{r['modello']:<22}{r['corsa']:>9}   {r['nota']}"); continue
        sp = r["sposta_km"]
        sps = "-" if sp is None else f"{abs(sp[0])} km {'E' if sp[0] >= 0 else 'O'}, {abs(sp[1])} km {'N' if sp[1] >= 0 else 'S'}"
        print(f"{r['modello']:<22}{r['corsa']:>9}{r['scadenza']:>4}{str(r['voto35']):>8}{str(r['voto7']):>7}{str(r['quantita']):>10}"
              f"{r['area']:>7}{str(r['nerola']):>8}  {sps}")
    (out / "adesso.json").write_text(json.dumps({"inizio": V.iso(inizio), "fine": V.iso(fine),
        "radar": {"area": round(100 * float((oss[ok] >= 1).mean())), "max": round(float(np.nanmax(oss)), 1), "nerola": round(float(oss[iy, ix]), 1)},
        "modelli": righe}, ensure_ascii=False, indent=1))

    # mosaico: radar + i modelli in ordine di voto
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    liv = [0.2, 1, 2, 5, 10, 20, 40, 80]
    cm = ListedColormap(["#a6d8ff", "#3ba3ff", "#1f6fe0", "#19c37d", "#f1e11c", "#ff8a1a", "#e8172c"]); cm.set_under((0, 0, 0, 0))
    norm = BoundaryNorm(liv, cm.N)
    elenco = [("RADAR", oss, "")] + [(r["modello"], campi[r["modello"]], f"voto {r['voto35']} · ×{r['quantita']}") for r in righe if r["modello"] in campi]
    n = len(elenco); nc = 6; nr = math.ceil(n / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(nc * 3.2, nr * 2.75), facecolor="#0b1220")
    for k, ax in enumerate(np.ravel(axs)):
        ax.set_facecolor("#1b2433"); ax.set_xticks([]); ax.set_yticks([])
        if k >= n:
            ax.axis("off"); continue
        nome, f, sotto = elenco[k]
        ax.imshow(np.where(np.isfinite(f), f, np.nan), origin="lower", cmap=cm, norm=norm,
                  extent=[V.LON0, V.LON0 + V.PASSO * V.NX, V.LAT0, V.LAT0 + V.PASSO * V.NY], interpolation="nearest")
        ax.contour(V.LONS, V.LATS, np.nan_to_num(oss), levels=[1], colors="white", linewidths=0.6)
        ax.plot(*NEROLA, "o", ms=3, color="#ff5a6a")
        ax.set_title(nome + ("\n" + sotto if sotto else f"\n{inizio:%H}-{fine:%H} UTC"), color="white" if k else "#6fe3ff", fontsize=8.5)
    fig.suptitle(f"Pioggia {inizio:%d/%m %H:%M}-{fine:%H:%M} UTC: radar contro modelli (bordo bianco = pioggia vera >= 1 mm)", color="white", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "adesso.png", dpi=90, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
