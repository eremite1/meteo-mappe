"""
Modello stagionale MeteoNerola (Europa): previsione delle anomalie di temperatura e di pioggia per i
prossimi 3 mesi partendo dagli indici di teleconnessione, della stratosfera e degli oceani.

Come funziona, in breve
  1. Scarica la storia mensile (dal 1950) degli indici: NAO, AO, EA, SCA, EA/WR, POL, PNA (NOAA CPC),
     El Niño 3.4, QBO, AMO (NOAA PSL), vortice polare (vento a 10 hPa e 60° N, rianalisi NCEP/NCAR),
     più la tendenza di lungo periodo (riscaldamento globale) e la persistenza dell'anomalia locale.
  2. Scarica temperatura a 2 m e pioggia mensili della rianalisi NCEP/NCAR sull'Europa (dal 1950).
  3. Per ogni punto e per ogni mese di anticipo (1, 2, 3) impara con una regressione "ridge"
     (lineare, con freno contro l'eccesso di adattamento) come gli indici di oggi influenzano il tempo
     dei mesi successivi, nella stessa stagione dell'anno. Gli effetti degli indici si sommano tra loro.
  4. L'affidabilità si misura rifacendo le previsioni del passato anno per anno, senza far vedere al
     modello l'anno da prevedere (validazione incrociata): quella è la "pagella storica".
  5. Ogni previsione emessa viene archiviata e, quando il mese è passato, confrontata con quanto
     è successo (pagella reale).

Uso: python -m motore.stagionale --out uscita_stagionale [--archivio cartella_del_ramo_pubblicato]
"""
import argparse
import io
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import requests

UA = {"User-Agent": "MeteoNerola-stagionale/1.0"}
CPC = "https://ftp.cpc.ncep.noaa.gov/wd52dg/data/indices/tele_index.nh"
AO = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii"
PSL = "https://psl.noaa.gov/data/correlation/"
OPENDAP = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Monthlies/"
ANNO0 = 1950
LAT = (30.0, 72.5)
LON = (-25.0, 45.0)
ITALIA = dict(lat=(37.0, 47.0), lon=(7.0, 18.5))
CENTRO_ITALIA = dict(lat=(41.0, 43.5), lon=(11.0, 14.5))

INDICI = {
    # sigla: (nome, spiegazione breve della fase positiva)
    "NAO": ("Oscillazione Nord Atlantica", "anticiclone delle Azzorre forte e depressione d'Islanda profonda: correnti atlantiche verso il nord Europa"),
    "AO": ("Oscillazione Artica", "aria fredda chiusa sul Polo, vortice compatto"),
    "EA": ("Atlantico Orientale", "alta pressione spostata a sud sull'Atlantico orientale"),
    "SCA": ("Scandinavo", "anticiclone sulla Scandinavia"),
    "EAWR": ("Atlantico Est / Russia Ovest", "alta sull'Europa occidentale e bassa sulla Russia occidentale"),
    "POL": ("Polare / Eurasiatico", "vortice polare intenso sull'Eurasia"),
    "PNA": ("Pacifico / Nord America", "onda lunga sul Pacifico e sul Nord America"),
    "ENSO": ("El Niño (Niño 3.4)", "acque del Pacifico equatoriale più calde del normale (El Niño)"),
    "QBO": ("Oscillazione Quasi Biennale", "venti occidentali nella stratosfera equatoriale (30 hPa)"),
    "AMO": ("Oscillazione Multidecennale Atlantica", "Atlantico del nord più caldo del normale"),
    "VORTICE": ("Vortice polare stratosferico", "vortice polare più forte del normale (vento a 10 hPa, 60° N)"),
    "TREND": ("Tendenza di lungo periodo", "riscaldamento degli ultimi decenni"),
    "PERS": ("Persistenza", "anomalia del mese appena passato nello stesso punto"),
}
MESI = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto",
        "settembre", "ottobre", "novembre", "dicembre"]


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------- indici
def _get(url, tentativi=3):
    for k in range(tentativi):
        try:
            r = requests.get(url, timeout=60, headers=UA)
            if r.status_code == 200 and len(r.content) > 200:
                return r.text
        except Exception as e:
            log("  errore", url, e)
    return None


def serie_vuota(n):
    return np.full(n, np.nan)


def n_mesi(fino):
    return (fino.year - ANNO0) * 12 + fino.month


def idx(anno, mese):
    return (anno - ANNO0) * 12 + mese - 1


def leggi_cpc(n):
    out = {k: serie_vuota(n) for k in ("NAO", "EA", "WP", "EPNP", "PNA", "EAWR", "SCA", "TNH", "POL", "PT")}
    t = _get(CPC)
    if not t:
        return out
    for riga in t.splitlines():
        m = re.match(r"\s*(\d{4})\s+(\d{1,2})\s(.*)$", riga)
        if not m:
            continue
        valori = re.findall(r"-?\d+\.\d+", m.group(3))      # colonne a larghezza fissa: -99.90 puo' essere attaccato
        if len(valori) < 10:
            continue
        i = idx(int(m.group(1)), int(m.group(2)))
        if 0 <= i < n:
            for k, v in zip(out, valori[:10]):
                x = float(v)
                out[k][i] = np.nan if x < -90 else x
    return out


def leggi_anno_mese_valore(url, n):
    s = serie_vuota(n)
    t = _get(url)
    if t:
        for riga in t.splitlines():
            p = riga.split()
            if len(p) == 3 and re.fullmatch(r"\d{4}", p[0]):
                i = idx(int(p[0]), int(p[1]))
                if 0 <= i < n:
                    x = float(p[2])
                    s[i] = np.nan if abs(x) > 90 else x
    return s


def leggi_psl(nome, n, mancante=-99.0):
    """Formato PSL: prima riga 'anno_inizio anno_fine', poi 'anno v1 ... v12'."""
    s = serie_vuota(n)
    t = _get(PSL + nome)
    if t:
        for riga in t.splitlines()[1:]:
            p = riga.split()
            if len(p) == 13 and re.fullmatch(r"\d{4}", p[0]):
                for m, v in enumerate(p[1:], 1):
                    i = idx(int(p[0]), m)
                    try:
                        x = float(v)
                    except ValueError:
                        continue
                    if 0 <= i < n and x > mancante + 1 and abs(x) < 900:
                        s[i] = x
    return s


def leggi_qbo_cpc(n):
    s = serie_vuota(n)
    t = _get("https://www.cpc.ncep.noaa.gov/data/indices/qbo.u30.index")
    if t:
        for riga in t.splitlines():
            p = riga.split()
            if len(p) >= 2 and re.fullmatch(r"\d{4}", p[0]):
                for m, v in enumerate(p[1:13], 1):
                    try:
                        x = float(v)
                    except ValueError:
                        continue
                    i = idx(int(p[0]), m)
                    if 0 <= i < n and abs(x) < 900:
                        s[i] = x
            elif riga.strip().upper().startswith("ANOMALY") or "STANDARDIZED" in riga.upper():
                break                                   # dopo i valori originali il file riporta anomalie: non servono
    return s


def leggi_vortice(n):
    """Vento zonale medio a 10 hPa e 60° N (rianalisi NCEP/NCAR), m/s."""
    s = serie_vuota(n)
    try:
        import xarray as xr
        ds = xr.open_dataset(OPENDAP + "pressure/uwnd.mon.mean.nc")
        u = ds["uwnd"].sel(level=10, lat=60.0).mean("lon").load()
        for t, v in zip(u["time"].values, u.values):
            d = np.datetime64(t, "M").astype(object)
            i = idx(d.year, d.month)
            if 0 <= i < n:
                s[i] = float(v)
    except Exception as e:
        log("  vortice non disponibile:", e)
    return s


def anomalia_mensile(s, n):
    """Toglie la media 1991-2020 di ogni mese (per QBO e vortice, che non sono già anomalie)."""
    out = s.copy()
    for m in range(12):
        sel = [idx(a, m + 1) for a in range(1991, 2021) if idx(a, m + 1) < n]
        mu = np.nanmean(s[sel]) if sel else np.nan
        out[m::12] = s[m::12] - mu
    return out


def scarica_indici(fino):
    n = n_mesi(fino)
    log("Scarico gli indici...")
    cpc = leggi_cpc(n)
    ind = {k: cpc[k] for k in ("NAO", "EA", "SCA", "EAWR", "POL", "PNA")}
    ind["AO"] = leggi_anno_mese_valore(AO, n)
    ind["ENSO"] = leggi_psl("nina34.anom.data", n)
    qbo = leggi_psl("qbo.data", n)
    recente = leggi_qbo_cpc(n)                      # CPC (rianalisi CORe) e' aggiornato ogni mese
    qbo = np.where(np.isfinite(recente), recente, qbo)
    ind["QBO"] = anomalia_mensile(qbo, n)
    ind["AMO"] = leggi_psl("amon.us.data", n)
    ind["VORTICE"] = anomalia_mensile(leggi_vortice(n), n)
    for k, v in ind.items():
        ok = np.isfinite(v)
        ultimo = np.where(ok)[0].max() if ok.any() else -1
        log(f"  {k}: {ok.sum()} mesi, ultimo {ANNO0 + ultimo // 12}-{ultimo % 12 + 1:02d}" if ultimo >= 0 else f"  {k}: assente")
    return ind


# ---------------------------------------------------------------------------- rianalisi
def scarica_campo(nome):
    """Temperatura a 2 m (°C) o pioggia (mm/giorno) mensili NCEP/NCAR sull'Europa: (tempi, lat, lon, valori)."""
    import xarray as xr
    f, var = {"t2m": ("surface_gauss/air.2m.mon.mean.nc", "air"),
              "pioggia": ("surface_gauss/prate.sfc.mon.mean.nc", "prate")}[nome]
    ds = xr.open_dataset(OPENDAP + f)
    da = ds[var]
    da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon").sortby("lat")
    da = da.sel(lat=slice(*LAT), lon=slice(*LON), time=slice(f"{ANNO0}-01-01", None)).load()
    v = da.values.astype("float64")
    if nome == "t2m" and np.nanmean(v) > 100:
        v -= 273.15
    if nome == "pioggia":
        v *= 86400.0
    tempi = [np.datetime64(t, "M").astype(object) for t in da["time"].values]
    log(f"  {nome}: {len(tempi)} mesi fino a {tempi[-1]:%Y-%m}, griglia {v.shape[1]}x{v.shape[2]}")
    return tempi, da.lat.values, da.lon.values, v


def in_serie(tempi, v, n):
    out = np.full((n,) + v.shape[1:], np.nan)
    for t, x in zip(tempi, v):
        i = idx(t.year, t.month)
        if 0 <= i < n:
            out[i] = x
    return out


def clima_e_anomalie(campo):
    n = campo.shape[0]
    clima = np.zeros((12,) + campo.shape[1:])
    anom = np.full_like(campo, np.nan)
    for m in range(12):
        sel = [idx(a, m + 1) for a in range(1991, 2021) if idx(a, m + 1) < n]
        clima[m] = np.nanmean(campo[sel], axis=0)
        anom[m::12] = campo[m::12] - clima[m]
    return clima, anom


# ---------------------------------------------------------------------------- modello
NOMI_P = ["NAO", "AO", "EA", "SCA", "EAWR", "POL", "PNA", "ENSO", "QBO", "AMO", "VORTICE", "TREND", "PERS"]
LAMBDA = [0.5, 2.0, 5.0, 10.0, 25.0, 60.0]


def predittori(ind, anom, i0, n):
    """Matrice dei predittori al mese i0: indici + tendenza (+ persistenza per punto, a parte)."""
    x = [ind[k][i0] if i0 < len(ind[k]) else np.nan for k in NOMI_P[:-2]]
    x.append((i0 / 12.0 + ANNO0 - 1990) / 30.0)
    return np.array(x)


def ridge(X, y, lam):
    """Coefficienti della regressione ridge (X già standardizzato, con colonna di 1 non penalizzata)."""
    n, p = X.shape
    P = np.eye(p) * lam
    P[0, 0] = 0
    return np.linalg.solve(X.T @ X + P, X.T @ y)


def allena(ind, anom, i_emissione, anticipo, n_obs):
    """Per ogni punto: coefficienti, previsione, abilità storica (validazione incrociata), errore tipico."""
    m_bersaglio = (i_emissione + anticipo) % 12
    righe = []
    for i0 in range(12, n_obs):
        i1 = i0 + anticipo
        if i1 >= n_obs or i0 >= i_emissione:
            continue
        d = (i1 % 12 - m_bersaglio) % 12
        if d not in (0, 1, 11):                       # stessa stagione: mese bersaglio ±1
            continue
        righe.append((i0, i1))
    X0 = np.array([predittori(ind, anom, i0, n_obs) for i0, _ in righe])
    ok = np.all(np.isfinite(X0), axis=1)
    # i predittori con troppi buchi (es. vortice prima del 1948 o assente) vengono tolti
    usati = [j for j in range(X0.shape[1]) if np.isfinite(X0[:, j]).mean() > 0.8]
    X0 = X0[:, usati]
    ok = np.all(np.isfinite(X0), axis=1)
    righe = [r for r, k in zip(righe, ok) if k]
    X0 = X0[ok]
    mu, sd = X0.mean(0), X0.std(0) + 1e-9
    Xs = (X0 - mu) / sd
    anni = np.array([ANNO0 + i0 // 12 for i0, _ in righe])
    x_now = predittori(ind, anom, i_emissione, n_obs)[usati]
    ny, nx = anom.shape[1:]
    sd_pers = np.nanstd(np.array([anom[i0] for i0, _ in righe]), axis=0) + 1e-9
    risultati = dict(prev=np.full((ny, nx), np.nan), abilita=np.full((ny, nx), np.nan),
                     sigma=np.full((ny, nx), np.nan), coef=np.full((len(usati) + 1, ny, nx), np.nan),
                     contrib=np.full((len(usati) + 1, ny, nx), np.nan))
    # scelta del freno (lambda) una volta per tutte, con un sottoinsieme di punti
    rng = np.random.default_rng(0)
    punti = [(j, i) for j in range(ny) for i in range(nx)]
    prova = [punti[k] for k in rng.choice(len(punti), size=min(60, len(punti)), replace=False)]
    anni_u = np.unique(anni)
    gruppi = np.array_split(anni_u, 10)

    def matrice(j, i):
        pers = np.array([anom[i0, j, i] for i0, _ in righe])
        y = np.array([anom[i1, j, i] for _, i1 in righe])
        Xp = np.column_stack([np.ones(len(y)), Xs, (pers - np.nanmean(pers)) / sd_pers[j, i]])
        return Xp, y, np.nanmean(pers)

    def cv(Xp, y, lam):
        pred = np.full(len(y), np.nan)
        for g in gruppi:
            fuori = np.isin(anni, g)
            ok = ~fuori & np.isfinite(y) & np.isfinite(Xp[:, -1])
            if ok.sum() < 20:
                continue
            b = ridge(Xp[ok], y[ok], lam)
            tst = fuori & np.isfinite(Xp[:, -1])
            pred[tst] = Xp[tst] @ b
        return pred

    punteggi = []
    for lam in LAMBDA:
        r = []
        for j, i in prova:
            Xp, y, _ = matrice(j, i)
            p = cv(Xp, y, lam)
            ok = np.isfinite(p) & np.isfinite(y)
            if ok.sum() > 30:
                r.append(1 - np.mean((p[ok] - y[ok]) ** 2) / np.var(y[ok]))
        punteggi.append(np.mean(r) if r else -9)
    lam = LAMBDA[int(np.argmax(punteggi))]
    pers_now = anom[i_emissione]
    for j in range(ny):
        for i in range(nx):
            Xp, y, pers_m = matrice(j, i)
            ok = np.isfinite(y) & np.isfinite(Xp[:, -1])
            if ok.sum() < 40:
                continue
            p = cv(Xp, y, lam)
            okp = ok & np.isfinite(p)
            if okp.sum() > 30 and np.std(p[okp]) > 0:
                risultati["abilita"][j, i] = np.corrcoef(p[okp], y[okp])[0, 1]
                risultati["sigma"][j, i] = np.sqrt(np.mean((p[okp] - y[okp]) ** 2))
            b = ridge(Xp[ok], y[ok], lam)
            pv = pers_now[j, i] if np.isfinite(pers_now[j, i]) else pers_m
            xn = np.concatenate([[1.0], (x_now - mu) / sd, [(pv - pers_m) / sd_pers[j, i]]])
            risultati["prev"][j, i] = xn @ b
            risultati["coef"][:, j, i] = b[1:]
            risultati["contrib"][:, j, i] = b[1:] * xn[1:]
    risultati["nomi"] = [NOMI_P[k] for k in usati] + ["PERS"]
    risultati["lambda"] = lam
    risultati["x_now"] = dict(zip([NOMI_P[k] for k in usati], [float(v) for v in x_now]))
    risultati["n_casi"] = len(righe)
    return risultati


# ---------------------------------------------------------------------------- uscite
def media_box(campo, lat, lon, box):
    j = (lat >= box["lat"][0]) & (lat <= box["lat"][1])
    i = (lon >= box["lon"][0]) & (lon <= box["lon"][1])
    sub = campo[np.ix_(j, i)]
    w = np.cos(np.radians(lat[j]))[:, None] * np.ones(i.sum())[None, :]
    ok = np.isfinite(sub)
    return float(np.sum(sub[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")


def prob_sopra(prev, sigma):
    from scipy.stats import norm
    return 100 * norm.cdf(prev / np.maximum(sigma, 1e-6))


def disegna(lat, lon, campo, titolo, sotto, file, livelli, cmap, unita, punti_bassi=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    fig = plt.figure(figsize=(9, 7.2))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=12, central_latitude=50))
    ax.set_extent([-15, 40, 33, 68], crs=ccrs.PlateCarree())
    cf = ax.contourf(lon, lat, campo, levels=livelli, cmap=cmap, extend="both", transform=ccrs.PlateCarree(), alpha=0.92)
    if punti_bassi is not None:
        ax.contourf(lon, lat, punti_bassi, levels=[0.5, 1.5], colors="none", hatches=["...."], transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, alpha=0.7)
    ax.plot(12.7856, 42.1606, "k*", ms=8, transform=ccrs.PlateCarree())
    cb = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.03, fraction=0.05, aspect=40)
    cb.set_label(unita)
    ax.set_title(titolo, fontsize=13, fontweight="bold", loc="left")
    fig.text(0.02, 0.015, sotto, fontsize=8, color="#555")
    plt.savefig(file, dpi=110, bbox_inches="tight")
    plt.close(fig)


def spiegazione(nomi, contrib_t, contrib_p, x_now):
    """Frasi automatiche: quali indici spingono il tempo dell'Italia centrale, e in che direzione."""
    frasi = []
    ordine = sorted(range(len(nomi)), key=lambda k: -abs(contrib_t[k]))
    for k in ordine[:5]:
        n = nomi[k]
        ct, cp = contrib_t[k], contrib_p[k]
        if abs(ct) < 0.05 and abs(cp) < 3:
            continue
        nome = INDICI[n][0]
        if n in x_now:
            v = x_now[n]
            fase = "positiva" if v > 0 else "negativa"
            if n == "ENSO":
                fase = "El Niño" if v > 0.5 else "La Niña" if v < -0.5 else "neutra"
            elif n == "QBO":
                fase = "occidentale" if v > 0 else "orientale"
            elif n == "VORTICE":
                fase = "forte" if v > 0 else "debole"
            testa = f"{nome} {fase} ({v:+.1f})"
        elif n == "TREND":
            testa = nome
        else:
            testa = "Persistenza dell'anomalia del mese passato"
        effetto_t = ("più caldo" if ct > 0 else "più freddo") + f" ({ct:+.1f} °C)"
        effetto_p = ("più piovoso" if cp > 0 else "più secco") + f" ({cp:+.0f}%)"
        frasi.append(f"{testa}: spinge verso un tempo {effetto_t} e {effetto_p} sull'Italia centrale.")
    return frasi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--archivio", default=None, help="cartella con le previsioni già emesse (ramo pubblicato)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    oggi = datetime.now(timezone.utc)
    ind = scarica_indici(oggi)
    n = n_mesi(oggi)
    log("Scarico temperatura e pioggia mensili (NCEP/NCAR)...")
    tT, lat, lon, vT = scarica_campo("t2m")
    tP, _, _, vP = scarica_campo("pioggia")
    campoT, campoP = in_serie(tT, vT, n), in_serie(tP, vP, n)
    climaT, anomT = clima_e_anomalie(campoT)
    climaP, anomP = clima_e_anomalie(campoP)
    n_obs = idx(tT[-1].year, tT[-1].month) + 1

    # mese di emissione: l'ultimo mese con quasi tutti gli indici principali disponibili
    principali = ["NAO", "AO", "EA", "SCA", "ENSO"]
    i_em = None
    for i in range(n - 1, n - 8, -1):
        if sum(np.isfinite(ind[k][i]) for k in principali) >= 4:
            i_em = i
            break
    if i_em is None:
        raise SystemExit("indici recenti non disponibili")
    # indici mancanti nel mese di emissione: ultimo valore noto (al massimo 3 mesi prima), altrimenti 0 = neutro
    for k, v in ind.items():
        if not np.isfinite(v[i_em]):
            prec = [v[i] for i in range(i_em - 1, i_em - 4, -1) if i >= 0 and np.isfinite(v[i])]
            v[i_em] = prec[0] if prec else 0.0
            log(f"  {k}: valore del mese di emissione non ancora uscito, uso {v[i_em]:+.2f}")
    em_anno, em_mese = ANNO0 + i_em // 12, i_em % 12 + 1
    log(f"Mese di emissione: {MESI[em_mese - 1]} {em_anno} (rianalisi fino a {tT[-1]:%Y-%m})")
    # la persistenza usa l'ultima anomalia osservata disponibile
    anomT_e, anomP_e = anomT.copy(), anomP.copy()
    for A in (anomT_e, anomP_e):
        if not np.isfinite(A[i_em]).any():
            A[i_em] = 0.0

    uscita = dict(emesso_utc=oggi.strftime("%Y-%m-%dT%H:%MZ"), mese_emissione=f"{em_anno}-{em_mese:02d}",
                  rianalisi_fino=f"{tT[-1]:%Y-%m}", indici={}, mesi=[], fonti=[
                      "Indici di teleconnessione: NOAA Climate Prediction Center",
                      "Niño 3.4, QBO, AMO: NOAA Physical Sciences Laboratory",
                      "Rianalisi NCEP/NCAR R1 (NOAA PSL): vortice polare, temperatura e pioggia dal 1950"])
    for k in NOMI_P[:-2]:
        storia = ind[k][max(0, i_em - 11):i_em + 1]
        uscita["indici"][k] = dict(nome=INDICI[k][0], positivo=INDICI[k][1], valore=round(float(ind[k][i_em]), 2),
                                   ultimi_12=[None if not np.isfinite(x) else round(float(x), 2) for x in storia])
    previsioni_salvate = {}
    for L in (1, 2, 3):
        i_t = i_em + L
        anno_t, mese_t = ANNO0 + i_t // 12, i_t % 12 + 1
        log(f"Anticipo {L}: {MESI[mese_t - 1]} {anno_t}")
        rT = allena(ind, anomT_e, i_em, L, n_obs)
        rP = allena(ind, anomP_e, i_em, L, n_obs)
        clP = climaP[mese_t - 1]
        percP = 100 * rP["prev"] / np.maximum(clP, 0.2)
        probT = prob_sopra(rT["prev"], rT["sigma"])
        probP = prob_sopra(rP["prev"], rP["sigma"])
        poco = (rT["abilita"] < 0.2).astype(float)
        tag = f"{anno_t}{mese_t:02d}"
        sotto = (f"Modello stagionale MeteoNerola · emesso con gli indici di {MESI[em_mese - 1]} {em_anno} · "
                 f"puntinato: affidabilità storica bassa (correlazione < 0,2)")
        disegna(lat, lon, rT["prev"], f"Temperatura · {MESI[mese_t - 1]} {anno_t}", sotto, os.path.join(a.out, f"temperatura_{L}.png"),
                np.arange(-2.5, 2.51, 0.25), "RdBu_r", "°C rispetto alla media 1991-2020", poco)
        disegna(lat, lon, np.clip(percP, -60, 60), f"Pioggia · {MESI[mese_t - 1]} {anno_t}",
                sotto.replace("(correlazione", "della temperatura (correlazione"), os.path.join(a.out, f"pioggia_{L}.png"),
                np.arange(-50, 51, 10), "BrBG", "% rispetto alla pioggia normale 1991-2020", (rP["abilita"] < 0.2).astype(float))
        disegna(lat, lon, rT["abilita"], f"Affidabilità storica temperatura · {L} mes{'e' if L == 1 else 'i'} prima",
                "Correlazione tra previsioni rifatte sul passato (dal 1950, senza vedere l'anno previsto) e realtà: 0 = inutile, 1 = perfetta",
                os.path.join(a.out, f"abilita_t_{L}.png"), np.arange(-0.1, 0.81, 0.1), "viridis", "correlazione")
        disegna(lat, lon, rP["abilita"], f"Affidabilità storica pioggia · {L} mes{'e' if L == 1 else 'i'} prima",
                "Correlazione tra previsioni rifatte sul passato e realtà: la pioggia è molto meno prevedibile della temperatura",
                os.path.join(a.out, f"abilita_p_{L}.png"), np.arange(-0.1, 0.81, 0.1), "viridis", "correlazione")
        ctT = [media_box(rT["contrib"][k], lat, lon, CENTRO_ITALIA) for k in range(len(rT["nomi"]))]
        ctP = [media_box(100 * rP["contrib"][k] / np.maximum(clP, 0.2), lat, lon, CENTRO_ITALIA) for k in range(len(rP["nomi"]))]
        voce = dict(anticipo=L, mese=f"{anno_t}-{mese_t:02d}", nome_mese=f"{MESI[mese_t - 1]} {anno_t}",
                    immagini=dict(temperatura=f"temperatura_{L}.png", pioggia=f"pioggia_{L}.png",
                                  abilita_t=f"abilita_t_{L}.png", abilita_p=f"abilita_p_{L}.png"))
        for nome_box, box in (("centro_italia", CENTRO_ITALIA), ("italia", ITALIA)):
            voce[nome_box] = dict(
                t=round(media_box(rT["prev"], lat, lon, box), 2),
                prob_caldo=round(media_box(probT, lat, lon, box)),
                p_perc=round(media_box(percP, lat, lon, box)),
                prob_piovoso=round(media_box(probP, lat, lon, box)),
                abilita_t=round(media_box(rT["abilita"], lat, lon, box), 2),
                abilita_p=round(media_box(rP["abilita"], lat, lon, box), 2))
        voce["contributi"] = [dict(indice=k, nome=INDICI[k][0], t=round(ct, 2), p=round(cp, 1))
                              for k, ct, cp in zip(rT["nomi"], ctT, ctP)]
        voce["spiegazione"] = spiegazione(rT["nomi"], ctT, ctP, rT["x_now"])
        voce["casi_storici"] = rT["n_casi"]
        uscita["mesi"].append(voce)
        previsioni_salvate[f"t_{L}"] = rT["prev"].astype("float32")
        previsioni_salvate[f"p_{L}"] = rP["prev"].astype("float32")
        log(f"  centro Italia: {voce['centro_italia']}")
    # archivio delle previsioni emesse (per la pagella reale)
    arch = os.path.join(a.out, "archivio")
    os.makedirs(arch, exist_ok=True)
    if a.archivio and os.path.isdir(os.path.join(a.archivio, "archivio")):
        import shutil
        for f in os.listdir(os.path.join(a.archivio, "archivio")):
            shutil.copy(os.path.join(a.archivio, "archivio", f), arch)
    np.savez_compressed(os.path.join(arch, f"prev_{em_anno}{em_mese:02d}.npz"), lat=lat, lon=lon,
                        mesi=np.array([v["mese"] for v in uscita["mesi"]]), **previsioni_salvate)
    uscita["verifica"] = verifica(arch, lat, lon, anomT, anomP, n_obs)
    with open(os.path.join(a.out, "stagionale.json"), "w") as f:
        json.dump(uscita, f, ensure_ascii=False, indent=1)
    import shutil
    shutil.copy(os.path.join(os.path.dirname(__file__), "stagionale.html"), os.path.join(a.out, "index.html"))
    open(os.path.join(a.out, ".nojekyll"), "w").close()
    log("Fatto.")


def verifica(arch, lat, lon, anomT, anomP, n_obs):
    """Confronta le previsioni archiviate con i mesi già osservati nella rianalisi."""
    righe = []
    for f in sorted(os.listdir(arch)):
        if not f.startswith("prev_"):
            continue
        z = np.load(os.path.join(arch, f))
        for L, mese in enumerate(z["mesi"], 1):
            a, m = map(int, str(mese).split("-"))
            i = idx(a, m)
            if i >= n_obs or not np.isfinite(anomT[i]).any():
                continue
            for var, oss in (("t", anomT[i]), ("p", anomP[i])):
                pr = z[f"{var}_{L}"]
                ok = np.isfinite(pr) & np.isfinite(oss)
                if ok.sum() < 20:
                    continue
                segno = float(np.mean(np.sign(pr[ok]) == np.sign(oss[ok])) * 100)
                corr = float(np.corrcoef(pr[ok], oss[ok])[0, 1])
                righe.append(dict(emessa=f[5:11], mese=str(mese), anticipo=L, variabile=var,
                                  segno_giusto=round(segno), correlazione=round(corr, 2),
                                  prev_centro=round(media_box(pr, lat, lon, CENTRO_ITALIA), 2),
                                  oss_centro=round(media_box(oss, lat, lon, CENTRO_ITALIA), 2)))
    return righe


if __name__ == "__main__":
    main()
