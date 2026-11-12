#!/usr/bin/env python3
"""CLASSIFICA OTTICA CONTRO IL RADAR: ogni modello e ogni WRF (varianti di fisica comprese) sovrapposto al radar,
mappa contro mappa, per la pioggia oraria ("istantanea") e per gli accumuli di 3, 6 e 24 ore.

Per ogni durata si confrontano le mappe a tre soglie (pioggia debole, moderata, forte) con il Fractions Skill Score
(FSS): in ogni quadrato di circa 15 km si confronta la frazione di territorio con pioggia sopra soglia prevista e
osservata. 100 = mappa identica a quella del radar, 0 = nessuna somiglianza. Il voto di una durata e' la media
delle tre soglie (pesata sul numero di casi); il voto totale e' la media delle quattro durate.
Si usano le previsioni fino a 48 ore. Ogni confronto si fa una volta sola (stato in verifica/ottica_stato.json).
Uscita: ottica.json.
"""
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from motore import verifica as V

DURATE = {"1h": 1, "3h": 3, "6h": 6, "24h": 24}
NOMI_DURATE = {"1h": "Pioggia oraria", "3h": "Accumulo 3 ore", "6h": "Accumulo 6 ore", "24h": "Accumulo 24 ore"}
SOGLIE = {"1h": (0.5, 2, 5), "3h": (1, 5, 15), "6h": (1, 10, 25), "24h": (1, 10, 30)}
SCALA = 7                       # celle della griglia comune (circa 15 km)
ORE_MAX = 48
GIORNI = 30
PERIODI = (7, 30)
CASI_MINIMI = 4                 # per soglia


def fss(f, o, soglia):
    m = np.isfinite(f) & np.isfinite(o)
    if m.sum() < 200:
        return None
    bf = ((np.nan_to_num(f) >= soglia) & m).astype("float32")
    bo = ((np.nan_to_num(o) >= soglia) & m).astype("float32")
    if bf.sum() == 0 and bo.sum() == 0:
        return None                                   # nessuno ha pioggia sopra soglia: non dice nulla
    pm = uniform_filter(m.astype("float32"), SCALA, mode="constant")
    ff = uniform_filter(bf, SCALA, mode="constant") / np.maximum(pm, 1e-6)
    fo = uniform_filter(bo, SCALA, mode="constant") / np.maximum(pm, 1e-6)
    ok = pm > 0.5
    num = np.mean((ff[ok] - fo[ok]) ** 2)
    den = np.mean(ff[ok] ** 2) + np.mean(fo[ok] ** 2)
    return float(1 - num / den) if den > 0 else None


def _radar(per_ora, fine, ore):
    x = [per_ora.get(fine - timedelta(hours=k)) for k in range(ore)]
    if any(v is None for v in x):
        return None
    st = np.stack(x)
    s = st.sum(axis=0)
    s[~np.isfinite(st).all(axis=0)] = np.nan
    return s


def _previsione(z, corsa, fine, ore):
    """Accumulo previsto che finisce a 'fine' lungo 'ore': dalle finestre orarie (mm1) o da quelle di 3 ore (mm)."""
    if ore == 1:
        if "fini1" not in z.files:
            return None
        pos = {str(t): k for k, t in enumerate(z["fini1"])}
        k = pos.get(V.iso(fine))
        return None if k is None else V.espandi(z["mm1"][k])
    if ore % 3:
        return None
    pos = {str(t): k for k, t in enumerate(z["fini"])}
    pezzi = [pos.get(V.iso(fine - timedelta(hours=3 * j))) for j in range(ore // 3)]
    if any(p is None for p in pezzi):
        return None
    return sum(V.espandi(z["mm"][p]) for p in pezzi)


def aggiorna(storico, per_ora, adesso):
    storico = Path(storico)
    arch = storico / "verifica" / "archivio"
    sp = storico / "verifica" / "ottica_stato.json"
    st = json.loads(sp.read_text()) if sp.exists() else {}
    fatti = st.setdefault("fatti", {})
    somme = st.setdefault("somme", {})               # modello -> giorno -> durata -> {"s": {soglia: [n, fss]}, "pf", "po", "n"}
    nuovi = 0
    for f in sorted(arch.glob("*.npz")):
        try:
            z = np.load(f)
            chiave, corsa = str(z["chiave"]), V.ora(str(z["corsa"]))
        except Exception:  # noqa: BLE001
            continue
        gia = fatti.setdefault(f.stem, {})
        for d, ore in DURATE.items():
            fatte = set(gia.setdefault(d, []))
            passo = 1 if ore == 1 else min(ore, 6)
            for h in range(ore, ORE_MAX + 1, passo):
                fine = corsa + timedelta(hours=h)
                if ore == 24 and fine.hour != 0:
                    continue                         # giorni interi, da mezzanotte UTC
                if ore == 6 and fine.hour % 6:
                    continue
                if V.iso(fine) in fatte or fine > adesso:
                    continue
                o = _radar(per_ora, fine, ore)
                if o is None:
                    continue
                p = _previsione(z, corsa, fine, ore)
                if p is None:
                    fatte.add(V.iso(fine))           # questo modello non ha questa durata: inutile riprovare
                    continue
                fatte.add(V.iso(fine))
                m = np.isfinite(p) & np.isfinite(o)
                if m.sum() < 200:
                    continue
                if (np.nan_to_num(p)[m] >= SOGLIE[d][0]).mean() < 0.005 and (o[m] >= SOGLIE[d][0]).mean() < 0.005:
                    continue                         # asciutto per tutti e due
                c = somme.setdefault(chiave, {}).setdefault(fine.strftime("%Y-%m-%d"), {}).setdefault(
                    d, {"s": {}, "pf": 0.0, "po": 0.0, "n": 0})
                c["n"] += 1
                c["pf"] += float(np.nan_to_num(p)[m].sum())
                c["po"] += float(o[m].sum())
                for s in SOGLIE[d]:
                    v = fss(p, o, s)
                    if v is not None:
                        x = c["s"].setdefault(str(s), [0, 0.0])
                        x[0] += 1
                        x[1] += v
                nuovi += 1
            gia[d] = sorted(fatte)
    # pulizia
    nomi_arch = {f.stem for f in arch.glob("*.npz")}
    for k in [k for k in fatti if k not in nomi_arch]:
        del fatti[k]
    limite = (adesso - timedelta(days=GIORNI)).strftime("%Y-%m-%d")
    for m in somme.values():
        for g in [g for g in m if g < limite]:
            del m[g]
    st["aggiornato"] = V.iso(adesso)
    sp.write_text(json.dumps(st, separators=(",", ":")))
    return nuovi, st


def riepilogo(st, nomi, adesso):
    out = {"aggiornato": V.iso(adesso), "durate": NOMI_DURATE, "soglie": {d: list(s) for d, s in SOGLIE.items()},
           "scala_km": round(SCALA * V.KM_X), "ore_max": ORE_MAX, "periodi": {}}
    for per in PERIODI:
        limite = (adesso - timedelta(days=per)).strftime("%Y-%m-%d")
        righe = []
        for m, giorni in st.get("somme", {}).items():
            r = {"modello": m, "nome": nomi.get(m, m), "durate": {}}
            for d in DURATE:
                tot = {"s": {}, "pf": 0.0, "po": 0.0, "n": 0}
                for g, x in giorni.items():
                    if g < limite or d not in x:
                        continue
                    c = x[d]
                    tot["n"] += c["n"]; tot["pf"] += c["pf"]; tot["po"] += c["po"]
                    for s, (n, v) in c["s"].items():
                        a = tot["s"].setdefault(s, [0, 0.0]); a[0] += n; a[1] += v
                soglie = {s: (round(v / n * 100, 1), n) for s, (n, v) in tot["s"].items() if n >= CASI_MINIMI}
                if not soglie:
                    continue
                pesi = sum(n for _, n in soglie.values())
                r["durate"][d] = {"voto": round(sum(v * n for v, n in soglie.values()) / pesi, 1), "casi": tot["n"],
                                  "soglie": {s: v for s, (v, n) in soglie.items()},
                                  "quantita": round((tot["pf"] / tot["po"] - 1) * 100) if tot["po"] > 0 else None}
            if r["durate"]:
                r["voto"] = round(sum(x["voto"] for x in r["durate"].values()) / len(r["durate"]), 1)
                r["completo"] = len(r["durate"]) >= 3
                righe.append(r)
        righe.sort(key=lambda r: (not r["completo"], -r["voto"]))
        out["periodi"][str(per)] = righe
    return out


def nome_variante(chiave):
    """wrffis_gfs_f2 -> 'WRF 3 km · GFS · F2'."""
    forz = {"ecmwf": "ECMWF", "gfs": "GFS", "icon2i": "ICON-2I", "iconeu": "ICON-EU"}
    try:
        _, k, f = chiave.split("_")
        return f"WRF 3 km · {forz.get(k, k)} · {f.upper()}"
    except ValueError:
        return chiave
