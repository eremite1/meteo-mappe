"""
Disegno della CARTA SINOTTICA ANALIZZATA (stile scuro, simboli classici dei fronti).
Usa la proiezione e le geometrie gia' pronte del Disegnatore dell'area Europa. Se viene dato uno sfondo
(immagine satellitare in lat/lon), l'analisi viene disegnata sopra il satellite.
"""
import io
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon, FancyArrow
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from .analisi import MASSE

FONDO = "#0d1520"
TERRA = "#1b2723"
MARE = "#101c29"
COL = {"freddo": "#3d8bff", "caldo": "#ff4d4d", "occluso": "#b45cff", "stazionario": "#ff4d4d"}
FIRMA = "www.meteonerola.it"


def _proietta(dis, lon, lat):
    p = dis.proj.transform_points(dis.dati_crs, np.asarray(lon), np.asarray(lat))
    return p[:, 0], p[:, 1]


def _a_lonlat(dis, ii, jj):
    return (np.interp(jj, np.arange(len(dis.lon)), dis.lon), np.interp(ii, np.arange(len(dis.lat)), dis.lat))


def _fronte(ax, dis, f, scala):
    """Linea del fronte con triangoli / semicerchi dalla parte giusta."""
    lo, la = _a_lonlat(dis, np.array(f["i"]), np.array(f["j"]))
    x, y = _proietta(dis, lo, la)
    nx, ny = np.array(f["nx"]), np.array(f["ny"])
    # direzione "verso l'aria calda" nella proiezione: si sposta un poco il punto in quella direzione
    d = 0.3
    x2, y2 = _proietta(dis, lo + nx * d / np.cos(np.deg2rad(la)), la + ny * d)
    wx, wy = x2 - x, y2 - y
    n = np.hypot(wx, wy) + 1e-9
    wx, wy = wx / n, wy / n
    tipo = f["tipo"]
    alone = [pe.Stroke(linewidth=5.2, foreground=FONDO, alpha=0.6), pe.Normal()]
    if tipo == "stazionario":
        ax.plot(x, y, color=COL["caldo"], lw=2.6, zorder=8, solid_capstyle="round", path_effects=alone)
        ax.plot(x, y, color=COL["freddo"], lw=2.6, zorder=8.1, dashes=(6, 6))
    else:
        ax.plot(x, y, color=COL[tipo], lw=2.6, zorder=8, solid_capstyle="round", path_effects=alone)
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    if s[-1] < scala * 2:
        return
    r = scala * 0.42
    pos = np.arange(scala * 0.8, s[-1] - scala * 0.5, scala)
    for k, sp in enumerate(pos):
        cx, cy = np.interp(sp, s, x), np.interp(sp, s, y)
        ax_, ay_ = np.interp(sp - r, s, x), np.interp(sp - r, s, y)
        bx_, by_ = np.interp(sp + r, s, x), np.interp(sp + r, s, y)
        tx, ty = bx_ - ax_, by_ - ay_
        tn = np.hypot(tx, ty) + 1e-9
        tx, ty = tx / tn, ty / tn
        vx, vy = np.interp(sp, s, wx), np.interp(sp, s, wy)
        # normale alla linea dalla parte dell'aria calda
        px, py = -ty, tx
        if px * vx + py * vy < 0:
            px, py = -px, -py
        if tipo == "freddo":
            forme = [("tri", 1, COL["freddo"])]
        elif tipo == "caldo":
            forme = [("semi", -1, COL["caldo"])]
        elif tipo == "occluso":
            forme = [("tri" if k % 2 == 0 else "semi", 1, COL["occluso"])]
        else:
            forme = [("tri", 1, COL["freddo"])] if k % 2 == 0 else [("semi", -1, COL["caldo"])]
        for forma, verso, colore in forme:
            qx, qy = px * verso, py * verso
            if forma == "tri":
                poly = [(ax_, ay_), (bx_, by_), (cx + qx * r * 1.25, cy + qy * r * 1.25)]
            else:
                ang = np.linspace(0, np.pi, 14)
                poly = [(cx + (np.cos(a) * tx + np.sin(a) * qx) * r, cy + (np.cos(a) * ty + np.sin(a) * qy) * r) for a in ang]
            ax.add_patch(Polygon(poly, closed=True, facecolor=colore, edgecolor=FONDO, lw=0.6, zorder=8.5,
                                 transform=dis.proj))


def disegna_sinottica(dis, file, a, data, scadenza, info, titolo="Carta sinottica analizzata", sfondo=None):
    """a = risultato di analisi.analizza(); sfondo = (immagine RGB, [lon0, lon1, lat0, lat1]) per la versione satellite."""
    pc = dis.dati_crs
    fig = plt.figure(figsize=(13.6, 10.4), facecolor=FONDO)
    ax = fig.add_axes([0.02, 0.15, 0.96, 0.77], projection=dis.proj)
    ax.set_extent(dis.a["extent"], crs=pc)
    ax.set_facecolor(FONDO)
    try:
        ax.spines["geo"].set_edgecolor("#2c3c4d")
    except Exception:
        pass
    X, Y = dis.X, dis.Y

    if sfondo is not None:
        img, est = sfondo
        ax.imshow(img, origin="upper", extent=est, transform=pc, zorder=0.5, interpolation="bilinear")
        ax.add_geometries(dis.coste, dis.crs_geo, facecolor="none", edgecolor="#e8eef5", linewidth=0.5, alpha=0.7, zorder=3)
        ax.add_geometries(dis.stati, dis.crs_geo, facecolor="none", edgecolor="#e8eef5", linewidth=0.35, alpha=0.45,
                          linestyle=(0, (3, 2)), zorder=3)
        alfa_masse = 0.0
    else:
        ax.add_geometries(dis.mare, dis.crs_geo, facecolor=MARE, edgecolor="none", zorder=0.4)
        ax.add_geometries(dis.terra, dis.crs_geo, facecolor=TERRA, edgecolor="none", zorder=0.4)
        alfa_masse = 0.5
        # masse d'aria (campitura discreta e morbida)
        m = a["masse"]
        cmap = ListedColormap([c for _, _, c in MASSE])
        ax.contourf(X, Y, m, levels=np.arange(-0.5, len(MASSE)), cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, len(MASSE)), cmap.N),
                    alpha=alfa_masse, transform=dis.crs_disegno, zorder=1, antialiased=True)
        ax.add_geometries(dis.coste, dis.crs_geo, facecolor="none", edgecolor="#8fa3b5", linewidth=0.45, zorder=3)
        ax.add_geometries(dis.stati, dis.crs_geo, facecolor="none", edgecolor="#6c8094", linewidth=0.3,
                          linestyle=(0, (3, 2)), zorder=3)
        # velatura di pioggia
        if a.get("prec") is not None:
            ax.contourf(X, Y, np.nan_to_num(a["prec"]), levels=[0.5, 3, 1e4], colors=["#ffffff", "#ffffff"],
                        alpha=0.13, transform=dis.crs_disegno, zorder=2)
            matplotlib.rcParams["hatch.color"] = "#e8f1ff"
            matplotlib.rcParams["hatch.linewidth"] = 0.6
            ax.contourf(X, Y, np.nan_to_num(a["prec"]), levels=[0.5, 3, 1e4], colors="none", hatches=["...", "ooo"],
                        transform=dis.crs_disegno, zorder=2.1)

    # corrente a getto (nastri viola)
    g = a.get("getto")
    if g is not None:
        ax.contourf(X, Y, np.nan_to_num(g), levels=[150, 200, 260, 1e4], colors=["#8f6bff", "#b07dff", "#d49bff"],
                    alpha=0.22 if sfondo is None else 0.25, transform=dis.crs_disegno, zorder=4)
        ax.contour(X, Y, np.nan_to_num(g), levels=[150, 220], colors=["#c9b3ff"], linewidths=[0.8, 1.1], alpha=0.85,
                   transform=dis.crs_disegno, zorder=4)

    # isobare
    if a.get("mslp") is not None:
        p = a["mslp"]
        v = p[np.isfinite(p)]
        lv = np.arange(np.floor(v.min() / 4) * 4, np.ceil(v.max() / 4) * 4 + 4, 4)
        cs = ax.contour(X, Y, p, levels=lv, colors="#f4f7fb", linewidths=[1.5 if l % 20 == 0 else 0.85 for l in lv],
                        alpha=0.9, transform=dis.crs_disegno, zorder=5)
        try:
            for t in ax.clabel(cs, cs.levels, fmt="%d", fontsize=7, inline=True, colors="#f4f7fb"):
                t.set_path_effects([pe.withStroke(linewidth=2, foreground=FONDO)])
        except Exception:
            pass

    # assi di saccatura e promontorio (dal 500 hPa)
    for l in a.get("saccature", []):
        x, y = _proietta(dis, *_a_lonlat(dis, np.array(l["i"]), np.array(l["j"])))
        ax.plot(x, y, color="#f0a040", lw=2.4, dashes=(7, 4), zorder=6, transform=dis.proj,
                path_effects=[pe.Stroke(linewidth=4.5, foreground=FONDO, alpha=0.5), pe.Normal()])
    for l in a.get("promontori", []):
        x, y = _proietta(dis, *_a_lonlat(dis, np.array(l["i"]), np.array(l["j"])))
        ax.plot(x, y, color="#f7d84a", lw=2.0, dashes=(1.5, 3), zorder=6, transform=dis.proj)

    # richiami caldi
    lim = dis.proj.transform_points(pc, np.array(dis.a["extent"][:2]), np.array([45, 45]))
    larg = abs(lim[1, 0] - lim[0, 0])
    for r in a.get("richiami", []):
        lo, la = _a_lonlat(dis, np.array([r["i"]]), np.array([r["j"]]))
        x, y = _proietta(dis, lo, la)
        u, v = r["u"], r["v"]
        n = np.hypot(u, v) + 1e-9
        L = larg * (0.045 + min(r["forza"], 10) * 0.004)
        ax.add_patch(FancyArrow(x[0] - u / n * L / 2, y[0] - v / n * L / 2, u / n * L, v / n * L, width=larg * 0.012,
                                head_width=larg * 0.03, head_length=larg * 0.022, length_includes_head=True,
                                facecolor="#ff7a45", edgecolor=FONDO, lw=0.6, alpha=0.9, zorder=7, transform=dis.proj))

    # fronti
    scala = larg * 0.026
    for f in a.get("fronti", []):
        _fronte(ax, dis, f, scala)
    # linee di instabilita' (fronti temporaleschi): linea magenta tratto-punto-punto
    for l in a.get("instabilita", []):
        x, y = _proietta(dis, *_a_lonlat(dis, np.array(l["i"]), np.array(l["j"])))
        ax.plot(x, y, color="#ff4fd8", lw=2.8, dashes=(8, 3, 1.5, 3, 1.5, 3), zorder=8.6, transform=dis.proj,
                path_effects=[pe.Stroke(linewidth=5, foreground=FONDO, alpha=0.6), pe.Normal()])

    # centri di alta e bassa pressione
    for c in a.get("centri", []):
        col = "#5aa9ff" if c["tipo"] == "A" else "#ff5a5a"
        alone = [pe.withStroke(linewidth=4, foreground=FONDO)]
        ax.text(c["lon"], c["lat"], c["tipo"], transform=pc, fontsize=24, fontweight="bold", color=col,
                ha="center", va="center", zorder=9, path_effects=alone, clip_on=True)
        ax.text(c["lon"], c["lat"] - 1.35, str(c["hpa"]), transform=pc, fontsize=8.5, fontweight="bold",
                color="#f4f7fb", ha="center", va="top", zorder=9, path_effects=alone, clip_on=True)

    # citta'
    for nome, (lo, la) in dis.a["citta"].items():
        ax.plot(lo, la, "o", ms=2.6, color="#ffffff", transform=pc, zorder=9)
        ax.text(lo, la, " " + nome, transform=pc, fontsize=7, color="#d7e1ec", va="center", zorder=9,
                path_effects=[pe.withStroke(linewidth=2, foreground=FONDO)])
    ax.plot(12.79, 42.16, "*", ms=11, color="#ffd34d", markeredgecolor=FONDO, transform=pc, zorder=9.5)

    # testata
    fig.text(0.02, 0.965, titolo, color="#f4f7fb", fontsize=19, fontweight="bold", va="center")
    fig.text(0.02, 0.935, data + ("" if scadenza is None else "  ·  " + scadenza), color="#9fb3c8", fontsize=11.5, va="center")
    fig.text(0.98, 0.965, "meteonerola.it", color="#ffd34d", fontsize=14, fontweight="bold", va="center", ha="right")
    fig.text(0.98, 0.935, info, color="#7f93a8", fontsize=8.5, va="center", ha="right")

    # leggenda
    voci = [Line2D([], [], color=COL["freddo"], lw=2.6, marker=">", ms=7, label="Fronte freddo"),
            Line2D([], [], color=COL["caldo"], lw=2.6, marker="o", ms=6, label="Fronte caldo"),
            Line2D([], [], color=COL["occluso"], lw=2.6, marker="^", ms=6, label="Fronte occluso"),
            Line2D([], [], color=COL["caldo"], lw=2.6, ls=(0, (6, 6)), label="Fronte stazionario"),
            Line2D([], [], color="#ff4fd8", lw=2.6, ls=(0, (8, 3, 1.5, 3, 1.5, 3)), label="Linea temporalesca"),
            Line2D([], [], color="#ff7a45", lw=0, marker=r"$\rightarrow$", ms=14, label="Richiamo caldo"),
            Line2D([], [], color="#f0a040", lw=2.4, ls=(0, (7, 4)), label="Saccatura (500 hPa)"),
            Line2D([], [], color="#f7d84a", lw=2, ls=(0, (1.5, 3)), label="Promontorio (500 hPa)"),
            Patch(facecolor="#b07dff", alpha=0.5, label="Corrente a getto (>150 km/h)"),
            Line2D([], [], color="#f4f7fb", lw=1, label="Isobare (hPa)")]
    if sfondo is None:
        voci.append(Patch(facecolor="#2a3a4a", edgecolor="#e8f1ff", hatch="...", label="Precipitazioni"))
        voci += [Patch(facecolor=c, alpha=0.75, label=n) for _, n, c in MASSE]
    leg = fig.legend(handles=voci, loc="lower center", ncol=5 if sfondo is None else 4, frameon=False, fontsize=8.6,
                     labelcolor="#d7e1ec", handlelength=2.6, columnspacing=1.6, bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, 0.135, "Analisi automatica MeteoNerola (metodo del parametro termico di fronte): un'approssimazione di quella di un previsore",
             color="#6f8397", fontsize=7.5, ha="center")

    buf = io.BytesIO()
    fig.savefig(buf, dpi=130, facecolor=FONDO)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if sfondo is None:
        img = img.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
        img.save(file, optimize=True)
    else:
        img.save(os.path.splitext(file)[0] + ".jpg", quality=86, optimize=True)
