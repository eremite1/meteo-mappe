#!/usr/bin/env python3
"""WRF CORRETTO: il WRF che va meglio nella pagella unica, corretto con gli errori del passato su TUTTO il territorio.

Si parte dal WRF con il voto piu' alto (ultimi 30 giorni) e si correggono, cella per cella:

  TEMPERATURA, PUNTO DI RUGIADA, VENTO: l'errore medio delle ultime due settimane di ogni stazione (circa 200
    stazioni: aeroporti, Protezione Civile, MeteoNetwork, Nerola), per fascia di 3 ore del giorno, viene «spalmato»
    sul territorio: ogni punto prende soprattutto la correzione delle stazioni vicine (entro 20-30 km) e, lontano da
    tutte, quella media della regione. Vento: rapporto tra misurato e previsto.

  PIOGGIA, con il RADAR (zona per zona, non uguale dappertutto):
    1. tempistiche: se il modello e' in media in anticipo o in ritardo (pagella ora per ora), la pioggia viene
       spostata nel tempo;
    2. posizione: se il modello mette la pioggia sistematicamente troppo a est, nord... (accumuli di 6 ore), la mappa
       viene spostata;
    3. intensita' ed estensione: la distribuzione delle intensita' orarie del modello viene resa uguale a quella del
       radar (stesse probabilita', «quantile mapping»), separatamente per pianura e costa, colline, montagna;
    4. distribuzione: dove il modello sbaglia in piu' o in meno in una certa zona (es. troppa pioggia sui monti,
       poca sulla costa), la pioggia viene moltiplicata per il rapporto radar/modello di quella zona, imparato in
       ogni cella separatamente per tipo di tempo previsto (perturbazione estesa, temporali isolati...).
    Gli errori vecchi contano sempre meno (mezza vita 30 giorni): la correzione si aggiorna da sola giorno per giorno.

Usa solo errori di PRIMA della partenza della corsa: e' una vera previsione. Il risultato entra nella pagella unica
come «WRF corretto», con le stesse regole di tutti gli altri modelli.

  python -m motore.corretto --storico DIR --osservazioni osservazioni.csv chiave=dominio_tutto.npz ...
"""
import argparse
import csv
import gzip
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

from motore import unica as U
from motore import verifica as V

GIORNI_STAZIONI = 14
L_KM = 25.0                     # raggio d'influenza di una stazione
PESO_REGIONE = 0.3              # quanto pesa la correzione media regionale (lontano dalle stazioni comanda lei)
MIN_CASI_STAZIONE = 6
C = math.cos(math.radians(42.1))


def ora_t(s):
    return datetime.strptime(s[:13], "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)


def scegli_base(storico, presenti):
    """Il WRF con il voto piu' alto negli ultimi 30 giorni tra quelli disponibili adesso."""
    p = Path(storico) / "unica.json"
    if p.exists():
        cl = json.loads(p.read_text()).get("periodi", {}).get("30", [])
        for r in cl:
            if r["modello"] in presenti and r.get("confronto_pieno"):
                return r["modello"], r["voto"]
        for r in cl:
            if r["modello"] in presenti:
                return r["modello"], r["voto"]
    for m in ("wrf1km", "wrf3km"):
        if m in presenti:
            return m, None
    return next(iter(presenti)), None


# ------------------------------------------------------------------------------------- stazioni
def errori_stazioni(storico, base, oss, staz, inizio):
    """{fascia di 3 ore: {stazione: (bias T, bias Td, rapporto vento, casi)}} prima di 'inizio'."""
    da = (inizio - timedelta(days=GIORNI_STAZIONI)).strftime("%Y-%m-%dT%H")
    fino = inizio.strftime("%Y-%m-%dT%H")
    acc = {}
    cart = Path(storico) / "previsioni" / base
    lim = (inizio - timedelta(days=GIORNI_STAZIONI + 3)).strftime("%Y%m%dT%H")
    for p in sorted(cart.glob("*.csv.gz")) if cart.exists() else []:
        if p.name[:11] < lim:
            continue
        with gzip.open(p, "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                valido = r["valido"][:13]
                if not (da <= valido < fino) or int(r["ore"]) > 72:
                    continue
                o = oss.get((r["id"], valido))
                if not o:
                    continue
                T, Td, Vv = U.num(r.get("T")), U.num(r.get("Td")), U.num(r.get("V"))
                if Vv is not None and r.get("uV") != "kmh":
                    Vv *= 3.6
                a = acc.setdefault(int(valido[11:13]) // 3, {}).setdefault(r["id"], [0.0, 0, 0.0, 0, 0.0, 0.0])
                if T is not None and o[0] is not None:
                    a[0] += T - o[0]
                    a[1] += 1
                if Td is not None and o[1] is not None:
                    a[2] += Td - o[1]
                    a[3] += 1
                if Vv is not None and o[2] is not None:
                    a[4] += Vv
                    a[5] += o[2]
    out = {}
    for fascia, ss in acc.items():
        out[fascia] = {}
        for sid, a in ss.items():
            if a[1] < MIN_CASI_STAZIONE or sid not in staz:
                continue
            bt = a[0] / a[1]
            btd = a[2] / a[3] if a[3] >= MIN_CASI_STAZIONE else bt
            rv = float(np.clip((a[5] + 3) / (a[4] + 3), 0.4, 1.6)) if a[4] + a[5] > 0 else 1.0
            out[fascia][sid] = (bt, btd, rv, a[1])
    return out


def spalma(valori, staz, lat, lon):
    """Valori delle stazioni -> campo sulla griglia del modello (media pesata con le distanze + media regionale)."""
    if not valori:
        return None
    ids = list(valori)
    sl = np.array([staz[i][0] for i in ids])
    so = np.array([staz[i][1] for i in ids])
    v = np.array([valori[i] for i in ids], dtype="float64")
    reg = float(np.median(v))
    num = np.full(lat.shape, PESO_REGIONE * reg)
    den = np.full(lat.shape, PESO_REGIONE)
    for k in range(len(ids)):
        d2 = (((lon - so[k]) * C) ** 2 + (lat - sl[k]) ** 2) * 111.0 ** 2
        w = np.exp(-d2 / (2 * L_KM ** 2))
        num += w * v[k]
        den += w
    return num / den


# ------------------------------------------------------------------------------------- pioggia
def mappa_quantili(hf, ho):
    """Funzione intensita' prevista -> intensita' «come il radar» (stesse probabilita'), o None se pochi dati."""
    if hf.sum() < 500 or ho.sum() < 500:
        return None
    cf = np.cumsum(hf) / hf.sum()
    co = np.cumsum(ho) / ho.sum()
    centri = np.concatenate([[0.0], np.sqrt(U.BIN[1:-1] * U.BIN[2:]), [U.BIN[-1]]])

    def f(x):
        p = np.interp(x, centri, cf)
        y = np.interp(p, co, centri)
        y = np.where(x < 0.05, 0.0, y)
        return np.minimum(y, np.minimum(3 * x + 2, 120.0))
    return f


def errori_radar(storico, base, adesso):
    """Ritardo medio (ore) e spostamento medio (km est, km nord) del modello negli ultimi 30 giorni."""
    st = U.carica_stato(Path(storico) / "unica")
    da = (adesso - timedelta(days=30)).strftime("%Y-%m-%d")
    tot = U.somma_periodo(st, da, ("0-24", "24-48")).get(base, {})
    lag, dx, dy, n_lag, n_pos = 0.0, 0.0, 0.0, 0, 0
    c = tot.get("R1")
    if c and c[U.CR.index("nt")] >= 10:
        n_lag = c[U.CR.index("nt")]
        lag = c[U.CR.index("lag")] / n_lag
    c = tot.get("R6")
    if c and c[U.CR.index("np")] >= 8:
        n_pos = c[U.CR.index("np")]
        dx, dy = c[U.CR.index("dx")] / n_pos, c[U.CR.index("dy")] / n_pos
    return {"ritardo_ore": round(lag, 2), "casi_tempo": int(n_lag), "est_km": round(dx, 1), "nord_km": round(dy, 1),
            "casi_posizione": int(n_pos)}


def correggi_pioggia(ora_p, lat, lon, hgt, cel):
    """Intensita' (quantile mapping per fascia di quota) e distribuzione zona per zona (fattore radar/modello per tipo
    di tempo previsto). ora_p: pioggia oraria (tempo, y, x) su qualsiasi griglia con lat, lon, quota."""
    info = {}
    if cel is None:
        return ora_p, info
    ora_p = ora_p.copy()
    iy = np.clip(np.floor((lat - V.LAT0) / V.PASSO).astype(int), 0, V.NY - 1)
    ix = np.clip(np.floor((lon - V.LON0) / V.PASSO).astype(int), 0, V.NX - 1)
    dentro = (lat >= V.LAT0) & (lat < V.LAT0 + V.PASSO * V.NY) & (lon >= V.LON0) & (lon < V.LON0 + V.PASSO * V.NX)
    stessa = lat.shape == (V.NY, V.NX)
    qm_usati = 0
    for kz, (a, b) in enumerate(U.ZONE_QUOTA):
        f = mappa_quantili(cel.hf[kz], cel.ho[kz])
        if f is None:
            continue
        m = (hgt >= a) & (hgt < b)
        x = ora_p[:, m]
        y = np.maximum(f(x), x * 0.5)
        y = np.exp((1 - FORZA_QM) * np.log(np.maximum(x, 1e-3)) + FORZA_QM * np.log(np.maximum(y, 1e-3)))
        ora_p[:, m] = np.where(x >= SOGLIA_QM, y, x)
        qm_usati += 1
    info["intensita_zone_quota"] = qm_usati
    glob_f = (float(cel.so[0].sum()) + 50) / (float(cel.sf[0].sum()) + 50)
    usati = {}
    for k0 in range(1, ora_p.shape[0], 6):
        fin = min(k0 + 6, ora_p.shape[0])
        acc = ora_p[k0:fin].sum(axis=0)
        acc_c = acc if stessa else V.su_griglia(lat, lon, [acc])[0]
        sit, _ = U.situazione(acc_c, U.DURATE["R6"])
        i = 1 + U.SIT_PIOGGIA.index(sit) if sit else 0
        if float(np.nanmax(cel.n[i])) < 6:
            i = 0
        n = cel.n[i]
        kk = 1.0 * n
        with np.errstate(invalid="ignore", divide="ignore"):
            fat = (cel.so[i] + kk) / (cel.sf[i] + kk)
            # il volume complessivo l'ha gia' sistemato il quantile mapping: qui solo come si distribuisce
            fat = fat / ((float(cel.so[i].sum()) + 50) / (float(cel.sf[i].sum()) + 50)) if qm_usati else fat
        fat = np.where(n >= 3, fat, 1.0)
        fat = 1 + np.clip(n / 12.0, 0, 1) * (fat - 1)
        fat = ndimage.gaussian_filter(np.clip(np.nan_to_num(fat, nan=1.0), 0.4, 2.5), 2.0, mode="nearest")
        fn = np.where(dentro, fat[iy, ix], 1.0)
        ora_p[k0:fin] *= fn[None]
        nome = U.SIT_PIOGGIA[i - 1] if i else "tutte"
        usati[nome] = usati.get(nome, 0) + 1
    info["fattori_zona"] = usati
    info["fattore_globale_radar"] = round(glob_f, 2)
    return ora_p, info


SOGLIA_QM = 0.2      # mm/h: si corregge l'intensita' della pioggia vera del modello, non si crea pioggia dal nulla
FORZA_QM = 0.6       # quanto ci si fida del radar (0 = niente correzione, 1 = intensita' identiche al radar)


def correggi(base, npz, storico, oss, staz, adesso):
    z = np.load(npz)
    lat, lon = z["lat"].astype("float64"), z["lon"].astype("float64")
    tempi = [V.ora(str(t)) for t in z["tempi"]]
    inizio = tempi[0]
    T2 = z["T2"].astype("float64")
    Q2, PS = z["Q2"].astype("float64"), z["PSFC"].astype("float64")
    U10, V10 = z["U10"].astype("float64"), z["V10"].astype("float64")
    cum = (z["RAINC"] + z["RAINNC"]).astype("float64")
    nuv = z["NUVT"].astype("float32") if "NUVT" in z.files else None
    rapporto = {"base": base, "corsa": V.iso(inizio), "creato": V.iso(adesso)}

    # --- temperatura, rugiada, vento
    err = errori_stazioni(storico, base, oss, staz, inizio)
    campi_f = {}
    for fascia, ss in err.items():
        campi_f[fascia] = tuple(spalma({k: v[i] for k, v in ss.items()}, staz, lat, lon) for i in range(3))
    tk = T2 - 273.15 if np.nanmean(T2) > 150 else T2.copy()
    e = Q2 * PS / (0.622 + Q2) / 100.0
    lne = np.log(np.maximum(e, 1e-3) / 6.112)
    Td = 243.5 * lne / (17.67 - lne)
    for k, t in enumerate(tempi):
        cf = campi_f.get(t.hour // 3)
        if not cf or cf[0] is None:
            continue
        tk[k] -= cf[0]
        Td[k] = np.minimum(Td[k] - cf[1], tk[k])
        U10[k] *= cf[2]
        V10[k] *= cf[2]
    ev = 6.112 * np.exp(17.67 * Td / (Td + 243.5))
    Q2c = 0.622 * ev / np.maximum(PS / 100.0 - 0.378 * ev, 1.0)
    tutte = [v for ss in err.values() for v in ss.values()]
    rapporto["stazioni"] = len({sid for ss in err.values() for sid in ss})
    if tutte:
        rapporto["correzione_media_T"] = round(-float(np.mean([v[0] for v in tutte])), 2)
        rapporto["fattore_medio_vento"] = round(float(np.mean([v[2] for v in tutte])), 2)

    # --- pioggia
    ora_p = np.clip(np.diff(cum, axis=0, prepend=cum[:1]), 0, None)          # pioggia dell'ora precedente
    er = errori_radar(storico, base, adesso)
    rapporto["radar"] = er
    lag = int(round(er["ritardo_ore"])) if abs(er["ritardo_ore"]) >= 0.75 else 0
    if lag:
        idx = np.clip(np.arange(len(tempi)) + lag, 1, len(tempi) - 1)
        ora_p = np.concatenate([ora_p[:1], ora_p[idx[1:]]])
    rapporto["spostamento_ore"] = -lag
    # spostamento: la pioggia va mossa al contrario dell'errore medio
    passo_km_x = float(np.nanmedian(np.abs(np.diff(lon, axis=1)))) * 111.32 * C
    passo_km_y = float(np.nanmedian(np.abs(np.diff(lat, axis=0)))) * 110.57
    sx = -er["est_km"] / passo_km_x if math.hypot(er["est_km"], er["nord_km"]) >= 8 else 0.0
    sy = -er["nord_km"] / passo_km_y if math.hypot(er["est_km"], er["nord_km"]) >= 8 else 0.0
    verso_nord = lat[-1, 0] > lat[0, 0]
    if sx or sy:
        for k in range(len(tempi)):
            ora_p[k] = ndimage.shift(ora_p[k], (sy if verso_nord else -sy, sx), order=1, mode="nearest")
    rapporto["spostamento_km"] = [round(-er["est_km"], 1), round(-er["nord_km"], 1)] if (sx or sy) else [0, 0]
    # memoria zona per zona
    cp = Path(storico) / "unica" / "celle" / f"{base}.npz"
    cel = U.Celle(Path(storico) / "unica", base, adesso) if cp.exists() else None
    # quota di ogni punto del modello -> fascia di quota; cella della griglia comune di ogni punto del modello
    hgt = z["hgt"].astype("float64")
    vol_prima = float(ora_p[1:49].sum())
    ora_p, info_p = correggi_pioggia(ora_p, lat, lon, hgt, cel)
    rapporto.update(info_p)
    if cel is None:
        rapporto["nota"] = "memoria del radar per questo modello non ancora pronta: pioggia corretta solo nei tempi e nella posizione"
    vol_dopo = float(ora_p[1:49].sum())
    rapporto["pioggia_48h_prima_dopo"] = [round(vol_prima / max(ora_p[0].size, 1), 2), round(vol_dopo / max(ora_p[0].size, 1), 2)]
    ora_p[0] = 0
    cumc = np.cumsum(ora_p, axis=0)
    out = dict(lat=lat.astype("float32"), lon=lon.astype("float32"), hgt=hgt.astype("float32"),
               T2=(tk + 273.15).astype("float32"), Q2=Q2c.astype("float32"), PSFC=PS.astype("float32"),
               U10=U10.astype("float32"), V10=V10.astype("float32"), RAINC=np.zeros_like(cumc, dtype="float32"),
               RAINNC=cumc.astype("float32"), tempi=z["tempi"])
    if nuv is not None:
        out["NUVT"] = nuv
    return out, rapporto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storico", required=True)
    ap.add_argument("--osservazioni", required=True)
    ap.add_argument("--adesso", default="")
    ap.add_argument("wrf", nargs="*", help="chiave=dominio_tutto.npz dei WRF disponibili")
    a = ap.parse_args()
    adesso = V.ora(a.adesso) if a.adesso else datetime.now(timezone.utc)
    storico = Path(a.storico)
    presenti = {k: f for k, f in (x.split("=", 1) for x in a.wrf) if Path(f).exists() and k in U.CANDIDATI_CORRETTO}
    if not presenti:
        print("nessun WRF disponibile: WRF corretto non calcolato")
        return
    base, voto = scegli_base(storico, presenti)
    z = np.load(presenti[base])
    corsa = str(V.ora(str(z["tempi"][0])))
    cart = storico / "corretto"
    info_p = cart / "corretto.json"
    vecchio = json.loads(info_p.read_text()) if info_p.exists() else {}
    if vecchio.get("base") == base and vecchio.get("corsa") == V.iso(V.ora(str(z["tempi"][0]))) and (cart / "corretto_tutto.npz").exists():
        print(f"WRF corretto gia' calcolato per {base} corsa {corsa}")
    else:
        oss, staz = U.leggi_oss(a.osservazioni, adesso, GIORNI_STAZIONI + 4)
        staz = {k: v for k, v in staz.items() if v[0] is not None and v[1] is not None}
        out, rapporto = correggi(base, presenti[base], storico, oss, staz, adesso)
        rapporto["voto_base"] = voto
        cart.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cart / "corretto_tutto.npz", **out)
        info_p.write_text(json.dumps(rapporto, ensure_ascii=False, indent=1))
        print("WRF corretto:", json.dumps(rapporto, ensure_ascii=False))
    # entra nella pagella come tutti gli altri: punti delle stazioni e griglia comune
    from motore import pagella as P
    oss_staz = []
    with open(a.osservazioni, newline="", encoding="utf-8") as fh:
        visti = set()
        for r in csv.DictReader(fh):
            if r["id"] not in visti:
                visti.add(r["id"])
                oss_staz.append((r["id"], float(r["lat"]), float(r["lon"])))
    tmp = storico.parent / "corretto_tmp"
    tmp.mkdir(exist_ok=True)
    punti = V.wrf_punti(None, None, oss_staz, tmp, [("wrfcorretto", str(cart / "corretto_tutto.npz"))])
    if "wrfcorretto" in punti:
        print("punti archiviati:", P.archivia(storico, "wrfcorretto", punti["wrfcorretto"]))
    g = V.wrf_griglia(cart / "corretto_tutto.npz", "wrfcorretto", tmp)
    print("griglia:", V.archivia(storico / "verifica" / "archivio", "wrfcorretto", g) or "gia' in archivio")


if __name__ == "__main__":
    main()
