"""
ANALISI SINOTTICA AUTOMATICA (carta sinottica di meteonerola.it, area Europa).

Dai campi del modello ricava le strutture che un previsore disegna su una carta sinottica:
  - FRONTI con il metodo del "parametro termico di fronte" (TFP, Renard e Clarke 1965; Hewson 1998) sulla
    temperatura potenziale equivalente a 850 hPa (theta-e) lisciata: il fronte sta sul lato caldo della zona
    di forte gradiente, dove il gradiente "inizia" (massimo del TFP lungo la direzione del gradiente).
    Il TIPO viene dal vento a 850 hPa che attraversa il fronte: verso l'aria calda = fronte FREDDO, verso
    l'aria fredda = fronte CALDO, quasi parallelo = STAZIONARIO. Vicino a una depressione profonda, dove fronte
    caldo e freddo si uniscono e l'aria calda e' sollevata (lingua di theta-e che si avvolge), il fronte diventa
    OCCLUSO (regola empirica, dichiarata come tale).
  - LINEE DI INSTABILITA' (fronti temporaleschi): zone allungate con CAPE alto e convergenza del vento a 850 hPa.
  - RICHIAMI CALDI: forte avvezione calda a 850 hPa con venti meridionali (spesso davanti ai fronti freddi).
  - MASSE D'ARIA: classi da theta-e e temperatura a 850 hPa e umidita' (artica, polare continentale/marittima,
    temperata, subtropicale marittima, subtropicale continentale/africana).
  - CORRENTE A GETTO: vento a 300 hPa oltre 110 km/h.
  - SACCATURE e PROMONTORI: assi dal geopotenziale a 500 hPa.
  - Centri di ALTA e BASSA pressione, isobare, velatura di nuvole e pioggia.
E' un'analisi oggettiva automatica: un'approssimazione di quella di un previsore, non la sostituisce.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, label

R_TERRA = 6371e3

MASSE = [  # (id, nome, colore)
    ("A", "Artica", "#6f8fd8"),
    ("Pc", "Polare continentale", "#8fb2e6"),
    ("Pm", "Polare marittima", "#86c5d8"),
    ("T", "Temperata (mista)", "#b9d6b0"),
    ("Sm", "Subtropicale marittima", "#f0c987"),
    ("Sc", "Subtropicale continentale (africana)", "#e59a6b"),
]


def theta_e(t_c, rh):
    """Temperatura potenziale equivalente (K) a 850 hPa da temperatura (gradi C) e umidita' relativa (%) - Bolton 1980."""
    t = t_c + 273.15
    rh = np.clip(rh, 1, 100)
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    e = rh / 100.0 * es
    r = 0.622 * e / (850.0 - e)                                   # rapporto di mescolanza (kg/kg)
    td = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112)) + 273.15
    tl = 1.0 / (1.0 / (td - 56.0) + np.log(t / td) / 800.0) + 56.0
    th = t * (1000.0 / 850.0) ** (0.2854 * (1 - 0.28 * r))
    return th * np.exp((3.376 / tl - 0.00254) * r * 1000 * (1 + 0.81 * r))


class Griglia:
    def __init__(self, lat, lon):
        self.lat, self.lon = np.asarray(lat, float), np.asarray(lon, float)
        self.dy = R_TERRA * np.deg2rad(abs(self.lat[1] - self.lat[0]))
        self.dx = R_TERRA * np.deg2rad(abs(self.lon[1] - self.lon[0])) * np.cos(np.deg2rad(self.lat))[:, None]
        self.passo = abs(self.lat[1] - self.lat[0])

    def grad(self, f):
        gy, gx = np.gradient(f)
        return gx / self.dx, gy / self.dy                          # per metro (est, nord)

    def a_lonlat(self, ii, jj):
        return (np.interp(jj, np.arange(len(self.lon)), self.lon), np.interp(ii, np.arange(len(self.lat)), self.lat))


def _linee_zero(campo, maschera, minimo_pt):
    """Linee dove 'campo' vale zero, tenute solo dove 'maschera' e' vera, in coordinate di indice (i, j)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    cs = plt.contour(np.where(np.isfinite(campo), campo, 0), levels=[0])
    segs = cs.allsegs[0]
    plt.close(fig)
    out = []
    for s in segs:
        pezzo = []
        for x, y in s:                                             # x = j, y = i
            i, j = int(round(y)), int(round(x))
            if 0 <= i < maschera.shape[0] and 0 <= j < maschera.shape[1] and maschera[i, j]:
                pezzo.append((y, x))
            else:
                if len(pezzo) >= minimo_pt:
                    out.append(np.array(pezzo))
                pezzo = []
        if len(pezzo) >= minimo_pt:
            out.append(np.array(pezzo))
    return out


def _liscia_linea(p, n=3):
    if len(p) < 2 * n + 1:
        return p
    k = np.ones(2 * n + 1) / (2 * n + 1)
    q = p.copy()
    q[n:-n, 0] = np.convolve(p[:, 0], k, "valid")
    q[n:-n, 1] = np.convolve(p[:, 1], k, "valid")
    return q


def _campiona(campo, p):
    i = np.clip(np.round(p[:, 0]).astype(int), 0, campo.shape[0] - 1)
    j = np.clip(np.round(p[:, 1]).astype(int), 0, campo.shape[1] - 1)
    return campo[i, j]


def _lunghezza_km(G, p):
    lo, la = G.a_lonlat(p[:, 0], p[:, 1])
    dx = np.diff(lo) * 111.2 * np.cos(np.deg2rad(la[:-1]))
    dy = np.diff(la) * 111.2
    return float(np.hypot(dx, dy).sum())


def centri_pressione(G, p, extent):
    k = max(5, int(round(9.0 / G.passo)))
    pp = np.where(np.isfinite(p), p, np.nanmean(p))
    centri = []
    for tipo, filtro, ok in (("A", maximum_filter, lambda v: v >= 1016), ("B", minimum_filter, lambda v: v <= 1010)):
        f = filtro(pp, size=k, mode="nearest")
        for i, j in zip(*np.where(pp == f)):
            lo, la = float(G.lon[j]), float(G.lat[i])
            if not (extent[0] + 1 <= lo <= extent[1] - 1 and extent[2] + 1 <= la <= extent[3] - 1) or not ok(pp[i, j]):
                continue
            if any(abs(c["i"] - i) < k // 2 and abs(c["j"] - j) < k // 2 for c in centri):
                continue
            centri.append(dict(tipo=tipo, i=int(i), j=int(j), lon=round(lo, 2), lat=round(la, 2), hpa=int(round(float(pp[i, j])))))
    return centri


def analizza(G, t850, rh850, u850, v850, mslp, z500, vgetto, cape=None, prec=None, nuvole=None, extent=None):
    """Tutte le strutture. Venti in km/h, z500 in m, mslp in hPa. Ritorna un dizionario pronto per il disegno."""
    out = {}
    sig = max(1.0, 0.9 / G.passo)                                # lisciatura sinottica (~100 km)
    te = gaussian_filter(np.nan_to_num(theta_e(t850, rh850), nan=290.0), sig)
    us, vs = gaussian_filter(np.nan_to_num(u850), sig), gaussian_filter(np.nan_to_num(v850), sig)

    # ---------- fronti (TFP) ----------
    gx, gy = G.grad(te)
    mod = np.hypot(gx, gy) * 1e5                                  # K per 100 km
    nx, ny = gx * 1e5 / np.maximum(mod, 1e-6), gy * 1e5 / np.maximum(mod, 1e-6)   # versore verso l'aria calda
    mgx, mgy = G.grad(mod)
    tfp = -(mgx * nx + mgy * ny) * 1e5                           # K per (100 km)^2
    tx, ty = G.grad(tfp)
    loc = (tx * nx + ty * ny) * 1e5                              # zero = massimo del TFP lungo il gradiente
    mod_med = gaussian_filter(mod, sig)
    maschera = (tfp > 0.15) & (mod_med > 1.6)                    # zona frontale vera (soglie alla Hewson, tarate su 850 hPa)
    linee = _linee_zero(loc, maschera, max(6, int(2.0 / G.passo)))
    vn = (us * nx + vs * ny)                                     # km/h: >0 il vento spinge verso l'aria calda
    fronti = []
    bassi = [c for c in (centri_pressione(G, mslp, extent) if extent else []) if c["tipo"] == "B" and c["hpa"] <= 1000]
    for p in linee:
        p = _liscia_linea(p)
        if _lunghezza_km(G, p) < 350:
            continue
        v = gaussian_filter(_campiona(vn, p).astype(float), 3)
        tipo = np.where(v > 6, "freddo", np.where(v < -6, "caldo", "stazionario"))
        # occluso: vicino (< 450 km) a una depressione profonda, sul lato dove il fronte si avvolge
        lo, la = G.a_lonlat(p[:, 0], p[:, 1])
        for b in bassi:
            d = np.hypot((lo - b["lon"]) * 111 * np.cos(np.deg2rad(b["lat"])), (la - b["lat"]) * 111)
            tipo = np.where((d < 450) & (tipo != "stazionario"), "occluso", tipo)
        # spezza in tratti dello stesso tipo (tratti di almeno ~150 km)
        inizio = 0
        for k in range(1, len(p) + 1):
            if k == len(p) or tipo[k] != tipo[inizio]:
                tratto = p[max(0, inizio - 1):k]
                if len(tratto) >= 3 and _lunghezza_km(G, tratto) >= 150:
                    fronti.append(dict(tipo=str(tipo[inizio]), i=tratto[:, 0].tolist(), j=tratto[:, 1].tolist(),
                                       nx=_campiona(nx, tratto).tolist(), ny=_campiona(ny, tratto).tolist()))
                inizio = k
    out["fronti"] = fronti

    # ---------- linee di instabilita' ----------
    linee_inst = []
    if cape is not None:
        dux, _ = G.grad(us / 3.6)
        _, dvy = G.grad(vs / 3.6)
        conv = -(dux + dvy) * 1e5                                # convergenza (1e-5 /s)
        z = (gaussian_filter(np.nan_to_num(cape), 1.5) > 600) & (gaussian_filter(conv, 1.5) > 1.0)
        lab, n = label(z)
        for k in range(1, n + 1):
            ii, jj = np.where(lab == k)
            if len(ii) < 12:
                continue
            pts = np.c_[ii, jj].astype(float)
            c0 = pts.mean(0)
            w, vv = np.linalg.eigh(np.cov((pts - c0).T))
            asse = vv[:, -1]
            s = (pts - c0) @ asse
            a, b = c0 + asse * s.min(), c0 + asse * s.max()
            seg = np.array([a, b])
            if _lunghezza_km(G, seg) >= 150 and w[-1] > 3 * max(w[0], 1e-6):
                linee_inst.append(dict(i=seg[:, 0].tolist(), j=seg[:, 1].tolist()))
    out["instabilita"] = linee_inst

    # ---------- richiami caldi ----------
    tl = gaussian_filter(np.nan_to_num(t850), sig)
    tgx, tgy = G.grad(tl)
    adv = -(us / 3.6 * tgx + vs / 3.6 * tgy) * 3600 * 6         # gradi in 6 ore
    adv = gaussian_filter(adv, sig)
    k = max(5, int(round(4.0 / G.passo)))
    picchi = (adv == maximum_filter(adv, size=k)) & (adv > 3.0) & (vs > 20)
    richiami = []
    for i, j in sorted(zip(*np.where(picchi)), key=lambda q: -adv[q])[:7]:
        richiami.append(dict(i=int(i), j=int(j), u=float(us[i, j]), v=float(vs[i, j]), forza=float(adv[i, j])))
    out["richiami"] = richiami

    # ---------- masse d'aria ----------
    te_r = theta_e(t850, rh850)
    cls = np.full(te_r.shape, 3, dtype=float)                    # temperata
    secco = rh850 < 55
    cls = np.where(te_r < 279, 0, cls)                           # artica
    cls = np.where((te_r >= 279) & (te_r < 290) & secco, 1, cls)  # polare continentale
    cls = np.where((te_r >= 279) & (te_r < 290) & ~secco, 2, cls)  # polare marittima
    cls = np.where((te_r >= 306) & ~((t850 > 12) & secco), 4, cls)  # subtropicale marittima
    cls = np.where((te_r >= 306) & (t850 > 12) & secco, 5, cls)    # subtropicale continentale (africana)
    cls = np.where(np.isfinite(te_r), cls, np.nan)
    out["masse"] = cls
    out["theta_e"] = te

    # ---------- getto, assi di saccatura e promontorio ----------
    out["getto"] = vgetto
    zl = gaussian_filter(np.nan_to_num(z500, nan=5500), sig * 1.5)
    zx = np.gradient(zl, axis=1)
    zxx = np.gradient(zx, axis=1)
    anom = zl - zl.mean(axis=1, keepdims=True)
    curv = gaussian_filter(zxx, sig)
    minpt = max(6, int(3.0 / G.passo))
    sacc = [p for p in _linee_zero(zx, (curv > 0.4) & (anom < -40), minpt) if _lunghezza_km(G, p) > 500]
    prom = [p for p in _linee_zero(zx, (curv < -0.4) & (anom > 40), minpt) if _lunghezza_km(G, p) > 500]
    out["saccature"] = [dict(i=p[:, 0].tolist(), j=p[:, 1].tolist()) for p in sacc]
    out["promontori"] = [dict(i=p[:, 0].tolist(), j=p[:, 1].tolist()) for p in prom]
    out["prec"] = prec
    out["nuvole"] = nuvole
    out["centri"] = centri_pressione(G, mslp, extent) if extent else []
    return out


# ---------------------------------------------------------------------------
# Testo "in breve" per la pagina del sito
# ---------------------------------------------------------------------------
ZONE = [
    ("sulle isole Britanniche", -3, 54), ("sul mare del Nord", 3, 56), ("sulla Scandinavia", 15, 62),
    ("sull'Atlantico", -15, 50), ("sulla penisola Iberica", -4, 40), ("sulla Francia", 2, 47), ("sulla Germania", 10, 51),
    ("sulla Polonia", 19, 52), ("sulle Alpi", 10, 46.5), ("sui Balcani", 20, 43), ("sull'Europa orientale", 30, 50),
    ("sul golfo del Leone", 4.5, 42.5), ("sul nord Italia", 10.5, 45.3), ("sul Tirreno", 11.5, 40.5),
    ("sull'Adriatico", 15.5, 43), ("sul sud Italia", 16, 39.5), ("sulla Sardegna", 9, 40), ("sulla Corsica", 9, 42.2),
    ("sul centro Italia", 12.8, 42.5), ("sullo Ionio", 18.5, 37.5), ("sulla Grecia", 22, 39), ("sul nord Africa", 5, 35),
    ("sul Mediterraneo occidentale", 5, 39), ("sul Mediterraneo orientale", 28, 35), ("sul mar Nero", 34, 43.5),
]


def dove(lon, lat):
    return min(ZONE, key=lambda z: ((z[1] - lon) * np.cos(np.deg2rad(lat))) ** 2 + (z[2] - lat) ** 2)[0]


def riassunto(G, a, ner=(12.79, 42.16)):
    """Strutture in forma compatta per il file .json (il sito ne fa il testo)."""
    out = dict(fronti=[], masse_nerola=None, getto_italia=False, instabilita=[], richiami=[], centri=[])
    for f in a["fronti"]:
        lo, la = G.a_lonlat(np.array(f["i"]), np.array(f["j"]))
        d = np.hypot((lo - ner[0]) * 82, (la - ner[1]) * 111)
        k = int(np.argmin(d))
        out["fronti"].append(dict(tipo=f["tipo"], da=dove(lo[0], la[0]), a=dove(lo[-1], la[-1]), vicino=dove(lo[k], la[k]),
                                  km_nerola=int(d[k])))
    i = int(np.argmin(np.abs(G.lat - ner[1]))); j = int(np.argmin(np.abs(G.lon - ner[0])))
    if np.isfinite(a["masse"][i, j]):
        out["masse_nerola"] = MASSE[int(a["masse"][i, j])][1]
    g = a["getto"]
    if g is not None:
        box = g[(G.lat[:, None] > 37) & (G.lat[:, None] < 47) & (G.lon[None, :] > 6) & (G.lon[None, :] < 19)] if g.ndim == 2 else np.array([])
        out["getto_italia"] = bool(box.size and np.nanmax(box) > 150)
    for l in a["instabilita"]:
        lo, la = G.a_lonlat(np.array(l["i"]), np.array(l["j"]))
        out["instabilita"].append(dove(lo.mean(), la.mean()))
    for r in a["richiami"]:
        lo, la = G.a_lonlat(np.array([r["i"]]), np.array([r["j"]]))
        out["richiami"].append(dove(lo[0], la[0]))
    out["centri"] = [dict(tipo=c["tipo"], hpa=c["hpa"], lon=c["lon"], lat=c["lat"], dove=dove(c["lon"], c["lat"])) for c in a["centri"]]
    return out
