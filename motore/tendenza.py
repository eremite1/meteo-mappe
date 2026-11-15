"""
Tendenza per settimane e decadi a Nerola, calcolata SOLO dagli indici (teleconnessioni, stratosfera, oceani, MJO)
e da come, nella storia dal 1950 a oggi, il tempo sull'Italia centrale ha risposto a quegli stessi indici.

Come funziona, in breve
  1. Indici giornalieri (NAO, AO, PNA della NOAA CPC; MJO del Bureau of Meteorology; vento a 10 hPa e 60° N, cioe'
     il vortice polare, dalla rianalisi NCEP/NCAR) e mensili (ENSO, QBO, AMO, EA, SCA, EA/WR, POL: gli stessi del
     modello stagionale, vedi stagionale.py).
  2. Temperatura a 2 m e pioggia giornaliere della rianalisi NCEP/NCAR sull'Italia centrale dal 1950, trasformate in
     anomalie rispetto al clima 1991-2020 di quel giorno dell'anno.
  3. Per ogni periodo da prevedere (6 settimane, 18 decadi) si prendono, in ogni anno dal 1950, gli indici dello stesso
     giorno di oggi (e di 10 giorni prima e dopo, per avere piu' esempi) e il tempo che e' venuto nel periodo
     corrispondente: una regressione "ridge" (lineare, con freno contro l'eccesso di adattamento) impara il legame.
  4. L'affidabilita' si misura rifacendo le previsioni di ogni anno senza far vedere al modello quell'anno.
  5. Con gli indici di oggi si calcola la tendenza: anomalia di temperatura, di pioggia, probabilita' che il periodo
     sia piu' caldo / piu' piovoso del normale e gli indici che contano di piu'.

Uso: python -m motore.tendenza --out uscita_tendenza [--archivio cartella_del_ramo_pubblicato]
"""
import argparse
import json
import math
import os
import re
from datetime import date, datetime, timedelta, timezone

import numpy as np

from motore import stagionale as st

DATASET = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/"
BOX = dict(lat=(41.0, 43.5), lon=(11.0, 14.5))          # Italia centrale (Nerola e' dentro)
G0 = date(1950, 1, 1)
DAILY = {
    "NAO": "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.cdas.z500.19500101_current.csv",
    "AO": "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.ao.cdas.z1000.19500101_current.csv",
    "PNA": "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.pna.cdas.z500.19500101_current.csv",
}
MJO = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"
MENSILI = ["ENSO", "QBO", "AMO", "EA", "SCA", "EAWR", "POL"]
NOMI = {
    "NAO10": "NAO degli ultimi 10 giorni", "NAO30": "NAO dell'ultimo mese", "AO10": "AO degli ultimi 10 giorni",
    "AO30": "AO dell'ultimo mese", "PNA30": "PNA dell'ultimo mese", "MJO1": "MJO (fasi 4-5)", "MJO2": "MJO (fasi 6-7)",
    "VORT10": "vortice polare", "VORTD": "tendenza del vortice polare", "ENSO": "El Niño / La Niña", "QBO": "QBO",
    "AMO": "Atlantico (AMO)", "EA": "East Atlantic (EA)", "SCA": "Scandinavo (SCA)", "EAWR": "East Atlantic / West Russia",
    "POL": "Polare-Eurasiatico", "PERST": "caldo o freddo dell'ultimo mese", "PERSP": "pioggia dell'ultimo mese",
    "TREND": "riscaldamento globale",
}
LAMBDA = [1.0, 3.0, 10.0, 30.0, 100.0]


def log(*a):
    print(*a, flush=True)


def gi(d):
    return (d - G0).days


# ------------------------------------------------------------------------------------------------ dati giornalieri
def serie_giornaliera_cpc(url, n):
    s = np.full(n, np.nan)
    t = st._get(url)
    if not t:
        log("  non disponibile:", url)
        return s
    for riga in t.splitlines():
        p = re.split(r"[,\s]+", riga.strip())
        if len(p) >= 4 and re.fullmatch(r"\d{4}", p[0]):
            try:
                k = gi(date(int(p[0]), int(p[1]), int(p[2])))
                x = float(p[3])
            except ValueError:
                continue
            if 0 <= k < n and abs(x) < 90:
                s[k] = x
    return s


def serie_mjo(n):
    r1, r2 = np.full(n, np.nan), np.full(n, np.nan)
    t = st._get(MJO)
    if not t:
        log("  MJO non disponibile")
        return r1, r2
    for riga in t.splitlines():
        p = riga.split()
        if len(p) >= 5 and re.fullmatch(r"\d{4}", p[0]):
            try:
                k = gi(date(int(p[0]), int(p[1]), int(p[2])))
                a, b = float(p[3]), float(p[4])
            except ValueError:
                continue
            if 0 <= k < n and abs(a) < 90:
                r1[k], r2[k] = a, b
    return r1, r2


def rianalisi_anno(anno):
    """Media sul riquadro dell'Italia centrale di temperatura (°C) e pioggia (mm/giorno), e vento a 10 hPa / 60° N."""
    import xarray as xr
    out = {}
    for chiave, (f, var) in {"T": (f"surface_gauss/air.2m.gauss.{anno}.nc", "air"),
                             "P": (f"surface_gauss/prate.sfc.gauss.{anno}.nc", "prate")}.items():
        ds = xr.open_dataset(DATASET + f)
        da = ds[var]
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon").sortby("lat")
        v = da.sel(lat=slice(*BOX["lat"]), lon=slice(*BOX["lon"])).mean(("lat", "lon")).load()
        x = v.values.astype("float64")
        if chiave == "T" and np.nanmean(x) > 100:
            x = x - 273.15                                    # kelvin -> gradi
        if chiave == "P" and np.nanmean(x) < 0.01:
            x = x * 86400.0                                   # kg/m2/s -> mm/giorno (alcuni anni sono gia' in mm)
        out[chiave] = (v["time"].values, x)
    ds = xr.open_dataset(DATASET + f"pressure/uwnd.{anno}.nc")
    u = ds["uwnd"].sel(level=10, lat=60.0).mean("lon").load()
    out["U10"] = (u["time"].values, u.values.astype("float64"))
    return out


def rianalisi(n, archivio):
    """Serie giornaliere dal 1950; gli anni gia' scaricati si riprendono dall'archivio pubblicato."""
    serie = {k: np.full(n, np.nan) for k in ("T", "P", "U10")}
    vecchio = os.path.join(archivio or "", "rianalisi.npz")
    if archivio and os.path.exists(vecchio):
        z = np.load(vecchio)
        for k in serie:
            m = min(n, len(z[k]))
            serie[k][:m] = z[k][:m]
        serie["T"][serie["T"] > 100] = np.nan
        for anno in range(1950, datetime.now(timezone.utc).year + 1):      # pioggia convertita due volte per errore
            k0, k1 = gi(date(anno, 1, 1)), min(n, gi(date(anno, 12, 31)) + 1)
            if k0 < n and np.nanmean(serie["P"][k0:k1]) > 50:
                serie["P"][k0:k1] /= 86400.0
    oggi = datetime.now(timezone.utc).date()
    for anno in range(1950, oggi.year + 1):
        k0, k1 = gi(date(anno, 1, 1)), min(n, gi(date(anno, 12, 31)) + 1)
        completo = all(np.isfinite(serie[k][k0:k1]).mean() > 0.97 for k in serie)
        if completo and anno < oggi.year - 1:
            continue
        try:
            r = rianalisi_anno(anno)
        except Exception as e:  # noqa: BLE001
            log(f"  rianalisi {anno} non letta: {e}")
            continue
        for k, (tempi, v) in r.items():
            for t, x in zip(tempi, v):
                d = np.datetime64(t, "D").astype(object)
                i = gi(d)
                if 0 <= i < n:
                    serie[k][i] = x
        log(f"  rianalisi {anno}: ok")
    return serie


def completa_con_era5(r, n):
    """La rianalisi NCEP/NCAR non viene piu' aggiornata (si ferma a marzo 2026): i giorni mancanti si completano con
    ERA5 (Open-Meteo), riportato alla stessa scala della rianalisi con i giorni in cui ci sono entrambe."""
    import requests
    oggi = G0 + timedelta(days=n - 1)
    punti = [(41.6, 11.9), (41.6, 13.6), (42.9, 11.9), (42.9, 13.6), (42.2, 12.8)]
    T, P = [], []
    u = ("https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s&start_date=2023-01-01&end_date=%s"
         "&daily=temperature_2m_mean,precipitation_sum&timezone=GMT"
         % (",".join(str(p_[0]) for p_ in punti), ",".join(str(p_[1]) for p_ in punti), oggi.isoformat()))
    risposta = None
    for _ in range(4):
        try:
            risposta = requests.get(u, timeout=180, headers=st.UA).json()
            break
        except Exception as e:  # noqa: BLE001
            log("  ERA5, riprovo:", e)
    if risposta is None:
        return
    for j in (risposta if isinstance(risposta, list) else [risposta]):
        d_ = j["daily"]
        T.append(np.array([np.nan if v is None else v for v in d_["temperature_2m_mean"]], dtype="float64"))
        P.append(np.array([np.nan if v is None else v for v in d_["precipitation_sum"]], dtype="float64"))
        tempi = d_["time"]
    T, P = np.nanmean(T, axis=0), np.nanmean(P, axis=0)
    ks = np.array([gi(date.fromisoformat(t)) for t in tempi])
    ok = (ks < n) & (ks >= 0)
    ks, T, P = ks[ok], T[ok], P[ok]
    for nome, v, molt in (("T", T, False), ("P", P, True)):
        s = r[nome]
        insieme = np.isfinite(s[ks]) & np.isfinite(v)
        if insieme.sum() < 60:
            continue
        if molt:
            f = np.nansum(s[ks][insieme]) / max(np.nansum(v[insieme]), 1e-6)
            nuovo = v * f
        else:
            nuovo = v + np.nanmean(s[ks][insieme] - v[insieme])
        manca = ~np.isfinite(s[ks]) & np.isfinite(nuovo)
        s[ks[manca]] = nuovo[manca]
        log(f"  {nome}: completati con ERA5 {manca.sum()} giorni")


def anomalie(s, n):
    """Anomalia rispetto al clima 1991-2020 del giorno dell'anno (lisciato su 31 giorni)."""
    doy = np.array([(G0 + timedelta(days=k)).timetuple().tm_yday - 1 for k in range(n)])
    anni = np.array([(G0 + timedelta(days=k)).year for k in range(n)])
    sel = (anni >= 1991) & (anni <= 2020) & np.isfinite(s)
    clima = np.array([np.nanmean(s[sel & (doy == d)]) if (sel & (doy == d)).any() else np.nan for d in range(366)])
    clima[365] = clima[364] if not np.isfinite(clima[365]) else clima[365]
    pad = np.concatenate([clima[-15:], clima, clima[:15]])
    lis = np.convolve(pad, np.ones(31) / 31, mode="same")[15:-15]
    return s - lis[doy], lis, doy


# ------------------------------------------------------------------------------------------------ predittori
def media(s, k, giorni):
    v = s[max(0, k - giorni + 1):k + 1]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan


def predittori(k, d, ind_m, aT, aP):
    """Valori degli indici al giorno k (ultimo giorno noto)."""
    g = G0 + timedelta(days=k)
    im = st.idx(g.year, g.month) - 1                        # ultimo mese completo
    x = {
        "NAO10": media(d["NAO"], k, 10), "NAO30": media(d["NAO"], k, 30), "AO10": media(d["AO"], k, 10),
        "AO30": media(d["AO"], k, 30), "PNA30": media(d["PNA"], k, 30),
        "MJO1": media(d["RMM1"], k, 5), "MJO2": media(d["RMM2"], k, 5),
        "VORT10": media(d["aU10"], k, 10), "VORTD": media(d["aU10"], k, 10) - media(d["aU10"], k - 10, 10),
        "PERST": media(aT, k, 30), "PERSP": media(aP, k, 30), "TREND": (g.year - 1990) / 30.0,
    }
    for nome in MENSILI:
        s = ind_m.get(nome)
        v = np.nan
        for j in range(im, im - 3, -1):                    # ultimo valore noto (fino a 3 mesi prima)
            if s is not None and 0 <= j < len(s) and np.isfinite(s[j]):
                v = float(s[j])
                break
        x[nome] = v
    return x


def ridge(X, y, lam):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def allena(Xs, ys, anni):
    """Sceglie il freno (lambda) con la validazione incrociata per anni; ritorna coefficienti, abilita' e errore."""
    meglio = None
    for lam in LAMBDA:
        prev = np.zeros_like(ys)
        for a in np.unique(anni):
            tr = anni != a
            b = ridge(Xs[tr], ys[tr], lam)
            prev[~tr] = Xs[~tr] @ b
        r = np.corrcoef(prev, ys)[0, 1] if np.std(prev) > 0 else 0
        if meglio is None or r > meglio[1]:
            meglio = (lam, r, prev)
    lam, r, prev = meglio
    b = ridge(Xs, ys, lam)
    sigma = float(np.std(ys - prev))
    giusti = float(np.mean(np.sign(prev) == np.sign(ys)))
    return b, float(r), sigma, giusti


def periodi(oggi):
    """(nome, tipo, inizio, fine) per 6 settimane (da lunedi' prossimo) e 18 decadi (dalla prossima)."""
    out = []
    lun = oggi + timedelta(days=(7 - oggi.weekday()) % 7 or 7)
    for s in range(6):
        a = lun + timedelta(days=7 * s)
        out.append(("settimana", a, a + timedelta(days=6)))
    g = oggi.day
    a = date(oggi.year, oggi.month, 11 if g <= 10 else 21) if g <= 20 else (date(oggi.year + (oggi.month == 12), oggi.month % 12 + 1, 1))
    for _ in range(18):
        if a.day == 1:
            b = date(a.year, a.month, 10)
        elif a.day == 11:
            b = date(a.year, a.month, 20)
        else:
            b = (date(a.year + (a.month == 12), a.month % 12 + 1, 1) - timedelta(days=1))
        out.append(("decade", a, b))
        a = b + timedelta(days=1)
    return out


# ------------------------------------------------------------------------------------------------ principale
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--archivio", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    oggi = datetime.now(timezone.utc).date()
    n = gi(oggi) + 1
    log("Indici mensili...")
    ind_m = st.scarica_indici(oggi)
    log("Indici giornalieri...")
    d = {k: serie_giornaliera_cpc(u, n) for k, u in DAILY.items()}
    d["RMM1"], d["RMM2"] = serie_mjo(n)
    log("Rianalisi giornaliera (Italia centrale e vortice polare)...")
    r = rianalisi(n, a.archivio)
    np.savez_compressed(os.path.join(a.out, "rianalisi.npz"), **r)      # solo NCEP/NCAR (ERA5 si riprende ogni volta)
    completa_con_era5(r, n)
    aT, climaT, doy = anomalie(r["T"], n)
    aP, climaP, _ = anomalie(r["P"], n)
    d["aU10"], _, _ = anomalie(r["U10"], n)
    # ultimo giorno con gli indici giornalieri principali: le previsioni partono da li'
    ultimo = max(k for k in range(n) if np.isfinite(r["T"][k])) if np.isfinite(r["T"]).any() else n - 1
    k_ora = min(n - 1, max([k for k in range(n - 1, n - 20, -1) if np.isfinite(d["NAO"][k])] or [n - 1]))
    x_ora = predittori(k_ora, d, ind_m, aT, aP)
    nomi = list(x_ora)
    log("Predittori di oggi:", {k: round(v, 2) for k, v in x_ora.items() if np.isfinite(v)})

    risultati = []
    oggi_d = G0 + timedelta(days=k_ora)
    for tipo, ini, fin in periodi(oggi):
        off0, lung = (ini - oggi_d).days, (fin - ini).days + 1
        righe, yT, yP, anni = [], [], [], []
        for anno in range(1951, oggi.year):
            for sposta in (-10, 0, 10):
                try:
                    g = date(anno, oggi_d.month, min(oggi_d.day, 28)) + timedelta(days=sposta)
                except ValueError:
                    continue
                k = gi(g)
                k1, k2 = k + off0, k + off0 + lung
                if k2 > ultimo:
                    continue
                t = aT[k1:k2]
                p = aP[k1:k2]
                if np.isfinite(t).mean() < 0.8:
                    continue
                righe.append(predittori(k, d, ind_m, aT, aP))
                yT.append(np.nanmean(t))
                yP.append(np.nanmean(p))
                anni.append(anno)
        if len(righe) < 60:
            log(f"  {tipo} {ini}: pochi esempi ({len(righe)})")
            continue
        X = np.array([[x[k] for k in nomi] for x in righe], dtype="float64")
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        usa = np.isfinite(mu) & (np.isfinite(X).mean(axis=0) > 0.5)
        Xs = np.where(np.isfinite(X), (X - mu) / sd, 0.0)[:, usa]
        xo = np.array([x_ora[k] for k in nomi])
        xs = np.where(np.isfinite(xo), (xo - mu) / sd, 0.0)[usa]
        anni = np.array(anni)
        voce = {"tipo": tipo, "dal": ini.isoformat(), "al": fin.isoformat(), "esempi": len(righe)}
        nomi_u = [k for k, u in zip(nomi, usa) if u]
        for var, y, clima in (("t", np.array(yT), climaT), ("p", np.array(yP), climaP)):
            ym = y.mean()
            b, rcv, sigma, giusti = allena(Xs, y - ym, anni)
            prev = float(xs @ b + ym)
            contrib = sorted(((float(b[j] * xs[j]), nomi_u[j], float(xo[nomi.index(nomi_u[j])])) for j in range(len(b))),
                             key=lambda z: -abs(z[0]))
            prob = 0.5 * (1 + math.erf((prev) / (max(sigma, 1e-6) * math.sqrt(2))))
            dd = [(ini + timedelta(days=j)).timetuple().tm_yday - 1 for j in range(lung)]
            norm = float(np.mean(clima[dd]))
            voce[var] = {"anomalia": round(prev, 2), "prob_sopra": round(prob, 2), "abilita": round(rcv, 2),
                         "giusti": round(giusti, 2), "clima": round(norm, 2),
                         "perche": [{"indice": c[1], "nome": NOMI.get(c[1], c[1]), "valore": round(c[2], 2), "effetto": round(c[0], 3)}
                                    for c in contrib[:3] if abs(c[0]) > 0.02]}
            if var == "p":
                voce[var]["percento"] = round(100 * prev / max(norm, 0.3))
        risultati.append(voce)
        log(f"  {tipo} {ini}..{fin}: T {voce['t']['anomalia']:+.2f} (r={voce['t']['abilita']:.2f}), "
            f"P {voce['p']['percento']:+d}% (r={voce['p']['abilita']:.2f})")
    uscita = {"emessa": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "indici_al": oggi_d.isoformat(),
              "indici": {k: (round(v, 2) if np.isfinite(v) else None) for k, v in x_ora.items()},
              "nomi": NOMI, "periodi": risultati,
              "fonti": ["NOAA CPC (NAO, AO, PNA giornalieri; EA, SCA, EA/WR, POL)", "NOAA PSL (ENSO, QBO, AMO)",
                        "Bureau of Meteorology (MJO)", "Rianalisi NCEP/NCAR (temperatura, pioggia, vortice polare)"]}
    with open(os.path.join(a.out, "tendenza.json"), "w") as f:
        json.dump(uscita, f, ensure_ascii=False, separators=(",", ":"))
    log("Scritto tendenza.json con", len(risultati), "periodi")


if __name__ == "__main__":
    main()
