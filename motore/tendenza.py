"""
Tendenza per settimane, decadi e mesi sull'Europa, calcolata SOLO dagli indici (teleconnessioni, stratosfera, oceani)
e da come, dal 1950 a oggi, il tempo ha risposto a quegli stessi indici.

Come funziona, in breve
  1. Indici giornalieri (NAO, AO, PNA della NOAA CPC; vento a 10 hPa e 60° N, cioe' il vortice polare, dalla
     rianalisi NCEP/NCAR) e mensili (ENSO, QBO, AMO, EA, SCA, EA/WR, POL: gli stessi del modello stagionale).
  2. Temperatura a 2 m e pioggia giornaliere della rianalisi NCEP/NCAR sull'Europa dal 1950, come anomalie rispetto al
     clima 1991-2020 di quel giorno dell'anno.
  3. Per ogni periodo (6 settimane, 18 decadi, 3 mesi) si prendono, in ogni anno dal 1951, gli indici dello stesso giorno
     di oggi (e di 10 giorni prima e dopo, per avere piu' esempi) e il tempo venuto nel periodo corrispondente: una
     regressione "ridge" (lineare, con freno contro l'eccesso di adattamento) impara il legame, punto per punto.
  4. L'affidabilita' si misura rifacendo le previsioni di ogni anno senza far vedere al modello quell'anno.
  5. Con gli indici di oggi: mappe di anomalia di temperatura e di pioggia, probabilita' e indici che contano di piu'.
Un solo motore per settimane, decadi e mesi: le tre viste sono coerenti tra loro.

Uso: python -m motore.tendenza --out uscita_tendenza --cache cartella_memoria
"""
import argparse
import json
import os
import re
import shutil
from datetime import date, datetime, timedelta, timezone

import numpy as np

from motore import stagionale as st

DATASET = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/"
G0 = date(1950, 1, 1)
DAILY = {
    "NAO": "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.cdas.z500.19500101_current.csv",
    "AO": "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.ao.cdas.z1000.19500101_current.csv",
    "PNA": "https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.pna.cdas.z500.19500101_current.csv",
}
MENSILI = ["ENSO", "QBO", "AMO", "EA", "SCA", "EAWR", "POL"]
NOMI = {
    "NAO10": "NAO degli ultimi 10 giorni", "NAO30": "NAO dell'ultimo mese", "AO10": "AO degli ultimi 10 giorni",
    "AO30": "AO dell'ultimo mese", "PNA30": "PNA dell'ultimo mese", "VORT10": "vortice polare",
    "VORTD": "tendenza del vortice polare", "ENSO": "El Niño / La Niña", "QBO": "QBO", "AMO": "Atlantico (AMO)",
    "EA": "East Atlantic (EA)", "SCA": "Scandinavo (SCA)", "EAWR": "East Atlantic / West Russia",
    "POL": "Polare-Eurasiatico", "TREND": "riscaldamento globale",
}
LAMBDA = [1.0, 3.0, 10.0, 30.0, 100.0]
MESI = st.MESI


def log(*a):
    print(*a, flush=True)


def gi(d):
    return (d - G0).days


# ------------------------------------------------------------------------------------------------ dati
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


def rianalisi_anno(anno):
    """Temperatura (°C) e pioggia (mm/giorno) sull'Europa e vento a 10 hPa / 60° N di un anno."""
    import xarray as xr
    out = {}
    for chiave, (f, var) in {"T": (f"surface_gauss/air.2m.gauss.{anno}.nc", "air"),
                             "P": (f"surface_gauss/prate.sfc.gauss.{anno}.nc", "prate")}.items():
        da = xr.open_dataset(DATASET + f)[var]
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon").sortby("lat")
        da = da.sel(lat=slice(*st.LAT), lon=slice(*st.LON)).load()
        x = da.values.astype("float64")
        if chiave == "T" and np.nanmean(x) > 100:
            x = x - 273.15
        if chiave == "P" and np.nanmean(x) < 0.01:
            x = x * 86400.0
        out[chiave] = (da["time"].values, x)
        out["lat"], out["lon"] = da.lat.values, da.lon.values
    u = xr.open_dataset(DATASET + f"pressure/uwnd.{anno}.nc")["uwnd"].sel(level=10, lat=60.0).mean("lon").load()
    out["U10"] = (u["time"].values, u.values.astype("float64"))
    return out


def rianalisi(n, cache):
    """Serie giornaliere dal 1950, anno per anno, memorizzate in `cache` (gli anni chiusi non si riscaricano)."""
    os.makedirs(cache, exist_ok=True)
    oggi = datetime.now(timezone.utc).date()
    T = P = lat = lon = None
    U = np.full(n, np.nan)
    for anno in range(1950, oggi.year + 1):
        f = os.path.join(cache, f"r_{anno}.npz")
        chiuso = anno < oggi.year - 1
        z = None
        if os.path.exists(f) and chiuso:
            z = dict(np.load(f))
        else:
            try:
                r = rianalisi_anno(anno)
                z = {"k": np.array([gi(np.datetime64(t, "D").astype(object)) for t in r["T"][0]]),
                     "T": r["T"][1].astype("float32"), "P": r["P"][1].astype("float32"),
                     "kU": np.array([gi(np.datetime64(t, "D").astype(object)) for t in r["U10"][0]]), "U": r["U10"][1],
                     "lat": r["lat"], "lon": r["lon"]}
                np.savez_compressed(f, **z)
                log(f"  rianalisi {anno}: scaricata ({len(z['k'])} giorni)")
            except Exception as e:  # noqa: BLE001
                log(f"  rianalisi {anno} non letta: {e}")
                if os.path.exists(f):
                    z = dict(np.load(f))
        if z is None:
            continue
        if T is None:
            lat, lon = z["lat"], z["lon"]
            T = np.full((n, len(lat), len(lon)), np.nan, dtype="float32")
            P = np.full((n, len(lat), len(lon)), np.nan, dtype="float32")
        ok = (z["k"] >= 0) & (z["k"] < n)
        T[z["k"][ok]] = z["T"][ok]
        P[z["k"][ok]] = z["P"][ok]
        okU = (z["kU"] >= 0) & (z["kU"] < n)
        U[z["kU"][okU]] = z["U"][okU]
    return T, P, U, lat, lon


def anomalie(s):
    """Anomalia rispetto al clima 1991-2020 del giorno dell'anno (lisciato su 31 giorni). s: [giorni, ...]."""
    n = s.shape[0]
    giorni = [G0 + timedelta(days=k) for k in range(n)]
    doy = np.array([min(g.timetuple().tm_yday, 365) - 1 for g in giorni])
    anni = np.array([g.year for g in giorni])
    sel = (anni >= 1991) & (anni <= 2020)
    clima = np.stack([np.nanmean(s[sel & (doy == d)], axis=0) for d in range(365)])
    pad = np.concatenate([clima[-15:], clima, clima[:15]])
    c = np.cumsum(np.concatenate([np.zeros((1,) + pad.shape[1:]), pad]), axis=0)
    lis = (c[31:] - c[:-31]) / 31
    return (s - lis[doy]).astype("float32"), lis, doy


def media(s, k, giorni):
    v = s[max(0, k - giorni + 1):k + 1]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan


def predittori(k, d, ind_m):
    g = G0 + timedelta(days=k)
    im = st.idx(g.year, g.month) - 1
    x = {"NAO10": media(d["NAO"], k, 10), "NAO30": media(d["NAO"], k, 30), "AO10": media(d["AO"], k, 10),
         "AO30": media(d["AO"], k, 30), "PNA30": media(d["PNA"], k, 30),
         "VORT10": media(d["aU"], k, 10), "VORTD": media(d["aU"], k, 10) - media(d["aU"], k - 10, 10),
         "TREND": (g.year - 1990) / 30.0}
    for nome in MENSILI:
        s = ind_m.get(nome)
        v = np.nan
        for j in range(im, im - 3, -1):
            if s is not None and 0 <= j < len(s) and np.isfinite(s[j]):
                v = float(s[j])
                break
        x[nome] = v
    return x


def periodi(oggi):
    out = []
    lun = oggi + timedelta(days=(7 - oggi.weekday()) % 7 or 7)
    for s in range(6):
        a = lun + timedelta(days=7 * s)
        out.append(("settimana", a, a + timedelta(days=6)))
    g = oggi.day
    a = date(oggi.year, oggi.month, 11 if g <= 10 else 21) if g <= 20 else date(oggi.year + (oggi.month == 12), oggi.month % 12 + 1, 1)
    for _ in range(18):
        b = date(a.year, a.month, 10) if a.day == 1 else date(a.year, a.month, 20) if a.day == 11 else \
            date(a.year + (a.month == 12), a.month % 12 + 1, 1) - timedelta(days=1)
        out.append(("decade", a, b))
        a = b + timedelta(days=1)
    a = date(oggi.year + (oggi.month == 12), oggi.month % 12 + 1, 1)
    for _ in range(3):
        b = date(a.year + (a.month == 12), a.month % 12 + 1, 1) - timedelta(days=1)
        out.append(("mese", a, b))
        a = b + timedelta(days=1)
    return out


def allena(Xs, Y, anni):
    """Ridge punto per punto con il freno scelto per ogni punto dalla validazione incrociata per anni.
    Ritorna (coefficienti [p, punti], correlazione, sigma dell'errore)."""
    folds = [anni == a for a in np.unique(anni)]
    migliore_r = np.full(Y.shape[1], -9.0)
    B = np.zeros((Xs.shape[1], Y.shape[1]))
    sig = np.ones(Y.shape[1])
    ok = np.isfinite(Y).mean(axis=0) > 0.8          # pochi buchi nei dati: si riempiono con la media (anomalia 0)
    Yc = np.nan_to_num(Y)
    for lam in LAMBDA:
        prev = np.zeros_like(Yc)
        for te in folds:
            tr = ~te
            A = Xs[tr].T @ Xs[tr] + lam * np.eye(Xs.shape[1])
            prev[te] = Xs[te] @ np.linalg.solve(A, Xs[tr].T @ Yc[tr])
        pc, yc = prev - prev.mean(0), Yc - Yc.mean(0)
        r = (pc * yc).sum(0) / np.maximum(np.sqrt((pc ** 2).sum(0) * (yc ** 2).sum(0)), 1e-12)
        meglio = (r > migliore_r) & ok
        if meglio.any():
            b = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ Yc)
            B[:, meglio] = b[:, meglio]
            sig[meglio] = np.std(Yc - prev, axis=0)[meglio]
            migliore_r[meglio] = r[meglio]
    return B, np.where(ok, migliore_r, np.nan), sig


def nome_periodo(tipo, a, b):
    if tipo == "mese":
        return f"{MESI[a.month - 1]} {a.year}"
    if tipo == "decade":
        return f"{['1ª', '2ª', '3ª'][min((a.day - 1) // 10, 2)]} decade di {MESI[a.month - 1]}"
    return f"{a.day} {MESI[a.month - 1][:3]} – {b.day} {MESI[b.month - 1][:3]}"


# ------------------------------------------------------------------------------------------------ principale
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="cache_tendenza")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    oggi = datetime.now(timezone.utc).date()
    n = gi(oggi) + 1
    log("Indici mensili...")
    ind_m = st.scarica_indici(oggi)
    log("Indici giornalieri...")
    d = {k: serie_giornaliera_cpc(u, n) for k, u in DAILY.items()}
    log("Rianalisi giornaliera...")
    T, P, U, lat, lon = rianalisi(n, a.cache)
    aT, climaT, doy = anomalie(T)
    aP, climaP, _ = anomalie(P)
    aU, _, _ = anomalie(U)
    d["aU"] = aU
    ultimo = int(np.where(np.isfinite(T[:, 0, 0]))[0].max())
    log("Rianalisi fino al", G0 + timedelta(days=ultimo))
    k_ora = max([k for k in range(n - 1, n - 20, -1) if np.isfinite(d["NAO"][k])] or [n - 1])
    oggi_d = G0 + timedelta(days=k_ora)
    x_ora = predittori(k_ora, d, ind_m)
    nomi = list(x_ora)
    log("Indici di oggi:", {k: round(v, 2) for k, v in x_ora.items() if np.isfinite(v)})
    ny, nx = len(lat), len(lon)
    risultati = []
    for num, (tipo, ini, fin) in enumerate(periodi(oggi)):
        off0, lung = (ini - oggi_d).days, (fin - ini).days + 1
        righe, yT, yP, anni = [], [], [], []
        for anno in range(1951, oggi.year):
            for sposta in (-10, 0, 10):
                g = date(anno, oggi_d.month, min(oggi_d.day, 28)) + timedelta(days=sposta)
                k = gi(g)
                k1, k2 = k + off0, k + off0 + lung
                if k2 > ultimo:
                    continue
                righe.append(predittori(k, d, ind_m))
                yT.append(np.nanmean(aT[k1:k2], axis=0).ravel())
                yP.append(np.nanmean(aP[k1:k2], axis=0).ravel())
                anni.append(anno)
        if len(righe) < 60:
            log(f"  {tipo} {ini}: pochi esempi ({len(righe)})")
            continue
        X = np.array([[x[k] for k in nomi] for x in righe], dtype="float64")
        mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        xo = np.array([x_ora[k] for k in nomi])
        usa = np.isfinite(mu) & (np.isfinite(X).mean(axis=0) > 0.5) & np.isfinite(xo)
        usa[nomi.index("TREND")] = True
        Xs = np.where(np.isfinite(X), (X - mu) / sd, 0.0)[:, usa]
        xs = np.where(np.isfinite(xo), (xo - mu) / sd, 0.0)[usa]
        nomi_u = [k for k, u in zip(nomi, usa) if u]
        anni = np.array(anni)
        dd = [min((ini + timedelta(days=j)).timetuple().tm_yday, 365) - 1 for j in range(lung)]
        clP = np.mean(climaP[dd], axis=0)
        voce = {"tipo": tipo, "nome": nome_periodo(tipo, ini, fin), "dal": ini.isoformat(), "al": fin.isoformat(),
                "esempi": len(righe), "immagini": {}}
        sotto = f"Tendenza MeteoNerola dagli indici al {oggi_d:%d/%m/%Y} · puntinato: affidabilità storica bassa (correlazione < 0,2)"
        for var, Y in (("t", np.array(yT)), ("p", np.array(yP))):
            # gli anni che mancano nell'archivio (download non riuscito) sono righe intere vuote: si tolgono
            riga_ok = np.isfinite(Y).mean(axis=1) > 0.5
            Y = Y[riga_ok]
            ym = np.nanmean(Y, axis=0)
            log(f"    {var}: {riga_ok.sum()} esempi su {len(riga_ok)}, punti con dati completi {np.isfinite(Y).all(axis=0).mean():.0%}, buchi {np.isnan(Y).mean():.1%}")
            B, r, sig = allena(Xs[riga_ok], Y - ym, anni[riga_ok])
            prev = (xs @ B + ym).reshape(ny, nx)
            contrib = (xs[:, None] * B).reshape(len(xs), ny, nx)
            prob = st.prob_sopra(prev, sig.reshape(ny, nx))
            r = r.reshape(ny, nx)
            poco = (r < 0.2).astype(float)
            if var == "t":
                st.disegna(lat, lon, prev, f"Temperatura · {voce['nome']}", sotto, os.path.join(a.out, f"t_{num}.png"),
                           np.arange(-2.5, 2.51, 0.25), "RdBu_r", "°C rispetto alla media 1991-2020", poco)
                voce["immagini"]["temperatura"] = f"t_{num}.png"
                box = {"anomalia": round(st.media_box(prev, lat, lon, st.CENTRO_ITALIA), 2)}
            else:
                perc = 100 * prev / np.maximum(clP, 0.2)
                st.disegna(lat, lon, np.clip(perc, -60, 60), f"Pioggia · {voce['nome']}", sotto,
                           os.path.join(a.out, f"p_{num}.png"), np.arange(-50, 51, 10), "BrBG",
                           "% rispetto alla pioggia normale 1991-2020", poco)
                voce["immagini"]["pioggia"] = f"p_{num}.png"
                box = {"percento": int(round(st.media_box(perc, lat, lon, st.CENTRO_ITALIA))),
                       "anomalia": round(st.media_box(prev, lat, lon, st.CENTRO_ITALIA), 2)}
            box["prob_sopra"] = int(round(st.media_box(prob, lat, lon, st.CENTRO_ITALIA)))
            box["abilita"] = round(st.media_box(r, lat, lon, st.CENTRO_ITALIA), 2)
            ct = [(st.media_box(contrib[j], lat, lon, st.CENTRO_ITALIA), nomi_u[j]) for j in range(len(nomi_u))]
            ct.sort(key=lambda z: -abs(z[0]))
            box["perche"] = [{"indice": c[1], "nome": NOMI.get(c[1], c[1]), "valore": round(x_ora[c[1]], 2),
                              "effetto": round(c[0], 3)} for c in ct[:3] if abs(c[0]) > 0.02]
            voce[var] = box
        risultati.append(voce)
        log(f"  {voce['nome']}: T {voce['t']['anomalia']:+.2f} ({voce['t']['prob_sopra']}%, r={voce['t']['abilita']}), "
            f"P {voce['p']['percento']:+d}% ({voce['p']['prob_sopra']}%, r={voce['p']['abilita']})")
    uscita = {"emessa": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "indici_al": oggi_d.isoformat(),
              "rianalisi_al": (G0 + timedelta(days=ultimo)).isoformat(),
              "indici": {k: (round(v, 2) if np.isfinite(v) else None) for k, v in x_ora.items()},
              "nomi": NOMI, "periodi": risultati,
              "fonti": ["NOAA CPC (NAO, AO, PNA giornalieri; EA, SCA, EA/WR, POL)", "NOAA PSL (ENSO, QBO, AMO)",
                        "Rianalisi NCEP/NCAR (temperatura, pioggia, vortice polare)"]}
    with open(os.path.join(a.out, "tendenza.json"), "w") as f:
        json.dump(uscita, f, ensure_ascii=False, separators=(",", ":"))
    pagina = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tendenza.html")
    if os.path.exists(pagina):
        shutil.copy(pagina, os.path.join(a.out, "index.html"))
    log("Scritto tendenza.json con", len(risultati), "periodi")


if __name__ == "__main__":
    main()
