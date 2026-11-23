"""
Disegno delle mappe (una classe per area, che prepara una volta sola coste e confini).
"""
import os
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image

from .catalogo import AREE

COLORE_MARE = "#e6ebf0"
FIRMA = "www.meteonerola.it"


class Disegnatore:
    def __init__(self, area, lat, lon, ombra=None):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cartopy.io.shapereader as shpreader
        self.ccrs = ccrs
        self.a = AREE[area]
        self.area = area
        self.lat, self.lon = lat, lon
        self.ombra = ombra
        self.dati_crs = ccrs.PlateCarree()
        if self.a["proiezione"] == "lambert":
            lo, la = self.a["centro"]
            e = self.a["extent"]
            self.proj = ccrs.LambertConformal(central_longitude=lo, central_latitude=la,
                                              standard_parallels=(e[2] + 5, e[3] - 5))
        else:
            self.proj = ccrs.PlateCarree()

        # geometrie ritagliate una volta sola
        from shapely.geometry import box
        e = self.a["extent"]
        m = 3.0 if area == "europa" else 0.6
        zona = box(e[0] - m, e[2] - m, e[1] + m, e[3] + m)
        ext = [e[0] - m, e[1] + m, e[2] - m, e[3] + m]
        sc = self.a["scala_coste"]

        def rit(geoms):
            out = []
            for g in geoms:
                try:
                    if g.intersects(zona):
                        g = g.intersection(zona)
                        if not g.is_empty:
                            out.append(g)
                except Exception:
                    pass
            return out

        self.mare = rit(cfeature.OCEAN.with_scale(sc).intersecting_geometries(ext))
        self.terra = rit(cfeature.LAND.with_scale(sc).intersecting_geometries(ext))
        self.laghi = rit(cfeature.LAKES.with_scale(sc).intersecting_geometries(ext))
        self.coste = rit(cfeature.COASTLINE.with_scale(sc).intersecting_geometries(ext))
        self.stati = rit(cfeature.BORDERS.with_scale(sc).intersecting_geometries(ext))
        self.interni = []
        try:
            if self.a["confini"] in ("province", "regioni"):
                if self.a["confini"] == "province":
                    shp = shpreader.natural_earth("10m", "cultural", "admin_1_states_provinces")
                    geo = [r.geometry for r in shpreader.Reader(shp).records()
                           if r.attributes.get("admin") == "Italy"]
                else:
                    shp = shpreader.natural_earth("10m", "cultural", "admin_1_states_provinces_lines")
                    geo = [r.geometry for r in shpreader.Reader(shp).records()
                           if r.attributes.get("adm0_name") == "Italy"]
                self.interni = rit(geo)
        except Exception:
            self.interni = []

        # --- calcoli fatti una volta sola per tutte le mappe dell'area -----------------
        # coordinate della griglia già nella proiezione della mappa (niente riproiezione a ogni mappa)
        LO, LA = np.meshgrid(lon, lat)
        if self.a["proiezione"] == "lambert":
            pts = self.proj.transform_points(self.dati_crs, LO, LA)
            self.X, self.Y = pts[..., 0], pts[..., 1]
            self.crs_disegno = self.proj

            def proietta(geoms):
                out = []
                for g in geoms:
                    try:
                        gp = self.proj.project_geometry(g, self.dati_crs)
                        if not gp.is_empty:
                            out.append(gp)
                    except Exception:
                        pass
                return out
            for nome in ("mare", "terra", "laghi", "coste", "stati", "interni"):
                setattr(self, nome, proietta(getattr(self, nome)))
            self.crs_geo = self.proj
        else:
            self.X, self.Y = LO, LA
            self.crs_disegno = self.dati_crs
            self.crs_geo = self.dati_crs

    # ------------------------------------------------------------------
    def _centri(self, ax, p, file, z500):
        """Carta sinottica: lettere A (alta pressione) e B (bassa) sui massimi e minimi delle isobare, e un
        file .json accanto alla mappa con i centri e i valori su Nerola (la pagina del sito ne fa un testo)."""
        import json
        import matplotlib.patheffects as pe
        from scipy.ndimage import maximum_filter, minimum_filter
        passo = abs(float(self.lat[1] - self.lat[0]))
        k = max(5, int(round(9.0 / passo)))                 # un centro ogni ~9 gradi al massimo
        pp = np.where(np.isfinite(p), p, np.nanmean(p))
        bordo = max(2, int(round(1.5 / passo)))
        centri = []
        for tipo, filtro, ok in (("A", maximum_filter, lambda v: v >= 1016), ("B", minimum_filter, lambda v: v <= 1010)):
            f = filtro(pp, size=k, mode="nearest")
            ii, jj = np.where(pp == f)
            for i, j in zip(ii, jj):
                if i < bordo or j < bordo or i >= pp.shape[0] - bordo or j >= pp.shape[1] - bordo or not ok(pp[i, j]):
                    continue
                if any(abs(c["i"] - i) < k // 2 and abs(c["j"] - j) < k // 2 for c in centri):
                    continue
                centri.append(dict(tipo=tipo, i=int(i), j=int(j), lon=round(float(self.lon[j]), 2),
                                   lat=round(float(self.lat[i]), 2), hpa=int(round(float(pp[i, j])))))
        pc = self.dati_crs
        for c in centri:
            col = "#1d4ed8" if c["tipo"] == "A" else "#c0262d"
            alone = [pe.withStroke(linewidth=4, foreground="white")]
            ax.text(c["lon"], c["lat"], c["tipo"], transform=pc, fontsize=22, fontweight="bold", color=col,
                    ha="center", va="center", zorder=7, path_effects=alone)
            ax.text(c["lon"], c["lat"] - 1.3, str(c["hpa"]), transform=pc, fontsize=8, fontweight="bold",
                    color=col, ha="center", va="top", zorder=7, path_effects=alone)
        # valori su Nerola
        jn = int(np.argmin(np.abs(np.asarray(self.lon) - 12.79))); i_n = int(np.argmin(np.abs(np.asarray(self.lat) - 42.16)))
        dati = dict(centri=[{x: c[x] for x in ("tipo", "lon", "lat", "hpa")} for c in centri],
                    nerola=dict(hpa=round(float(p[i_n, jn]), 1) if np.isfinite(p[i_n, jn]) else None,
                                z500=round(float(z500[i_n, jn]), 1) if np.isfinite(z500[i_n, jn]) else None))
        try:
            with open(os.path.splitext(file)[0] + ".json", "w") as f:
                json.dump(dati, f)
        except OSError:
            pass

    # ------------------------------------------------------------------
    def disegna(self, file, campo, p, titolo_modello, data, scadenza, info,
                contorni=None, frecce=None, unita=None, extra=None):
        extra = extra or {}
        st = p.get("stile_europa") if (self.area == "europa" and p.get("stile_europa")) else p["stile"]
        pc = self.dati_crs
        fig = plt.figure(figsize=self.a["figsize"], facecolor="white")
        ax = plt.axes(projection=self.proj)
        ax.set_extent(self.a["extent"], crs=pc)
        ax.set_facecolor("#efefef")              # grigio = nessun dato del modello

        ombra = st.get("ombra", False)
        dentro = np.where(np.isfinite(campo), 1.0, np.nan)
        if np.isfinite(dentro).any():
            ax.contourf(self.X, self.Y, dentro, levels=[0.5, 1.5], colors=["white"],
                        transform=self.crs_disegno, zorder=0)
        if ombra and self.ombra is not None:
            o = np.where(np.isfinite(campo), self.ombra, np.nan)
            ax.imshow(o, extent=[self.lon[0], self.lon[-1], self.lat[0], self.lat[-1]],
                      origin="lower", cmap="gray", vmin=0.2, vmax=1.0, alpha=0.5,
                      transform=pc, zorder=0.5)

        livelli = st["livelli"]
        if "colori" in st:
            cmap = ListedColormap(st["colori"])
            if st.get("inverso"):
                cmap = ListedColormap(st["colori"][::-1])
            norm = BoundaryNorm(livelli, cmap.N)
        else:
            cmap = st["cmap"].reversed() if st.get("inverso") else st["cmap"]
            norm = None
        campo_d = campo
        if st.get("buco_centrale"):
            # valori vicini a zero lasciati trasparenti
            k = len(livelli) // 2
            campo_d = np.where((campo > livelli[k - 1]) & (campo < livelli[k]), np.nan, campo)
        alpha = st.get("alpha", 0.85 if ombra else 0.92)
        mesh = ax.contourf(self.X, self.Y, campo_d, levels=livelli, cmap=cmap, norm=norm,
                           extend=st["extend"], alpha=alpha, transform=self.crs_disegno, zorder=1)

        if contorni is not None:
            cf, cl, passo = contorni
            if cl is None:
                v = cf[np.isfinite(cf)]
                if v.size:
                    cl = np.arange(np.floor(v.min() / passo) * passo, np.ceil(v.max() / passo) * passo + passo, passo)
            if cl is not None and len(cl) > 1:
                cs = ax.contour(self.X, self.Y, cf, levels=cl, colors=extra.get("colore_contorni", "#1a1a1a"),
                                linewidths=extra.get("spessore_contorni", 0.7), transform=self.crs_disegno, zorder=4)
                try:
                    ax.clabel(cs, cs.levels, fmt="%d", fontsize=extra.get("testo_contorni", 7), inline=True,
                              colors=extra.get("colore_etichette"))
                except Exception:
                    pass

        if frecce is not None:
            u, v = frecce
            k = max(1, int(round(self.a["frecce_ogni"] / (self.lat[1] - self.lat[0]))))
            uu, vv = u[k // 2::k, k // 2::k], v[k // 2::k, k // 2::k]
            mod = np.hypot(uu, vv)
            mod = np.where(mod < 0.5, np.nan, mod)
            lo2, la2 = np.meshgrid(self.lon[k // 2::k], self.lat[k // 2::k])
            un, vn = uu / mod, vv / mod
            if self.crs_disegno is not pc:
                # direzione delle frecce ruotata nella proiezione della mappa
                un, vn = self.proj.transform_vectors(pc, lo2, la2, np.nan_to_num(un), np.nan_to_num(vn))
                m2 = np.hypot(un, vn)
                m2 = np.where(m2 == 0, np.nan, m2)
                un, vn = un / m2, vn / m2
                un = np.where(np.isfinite(mod), un, np.nan)
            X2, Y2 = self.X[k // 2::k, k // 2::k], self.Y[k // 2::k, k // 2::k]
            ax.quiver(X2, Y2, un, vn, transform=self.crs_disegno, zorder=4, color="#1a1a1a",
                      alpha=0.75, scale=38, width=0.0022, headwidth=4, headlength=4.5,
                      headaxislength=4)

        if st.get("solo_mare"):
            ax.add_geometries(self.terra, self.crs_geo, facecolor="#e9e4da", edgecolor="none", zorder=2)
        else:
            # mare sotto i colori: così pioggia, vento, nuvole... si vedono anche sul mare
            ax.add_geometries(self.mare, self.crs_geo, facecolor=COLORE_MARE, edgecolor="none",
                              zorder=2 if st.get("solo_terra") else 0.6)
        ax.add_geometries(self.laghi, self.crs_geo, facecolor=COLORE_MARE, edgecolor="#5a7a99",
                          linewidth=0.4, zorder=2)
        ax.add_geometries(self.coste, self.crs_geo, facecolor="none", edgecolor="#333333",
                          linewidth=0.8 if self.area != "europa" else 0.6, zorder=3)
        ax.add_geometries(self.stati, self.crs_geo, facecolor="none", edgecolor="#333333",
                          linewidth=0.7, zorder=3)
        if self.interni:
            ax.add_geometries(self.interni, self.crs_geo, facecolor="none", edgecolor="#555555",
                              linewidth=0.4, zorder=3)

        if extra.get("centri") is not None:
            self._centri(ax, extra["centri"], file, campo)

        for nome, (lo, la) in self.a["citta"].items():
            ax.plot(lo, la, marker="o", color="black", markersize=3.2, transform=pc, zorder=5)
            ax.text(lo, la, " " + nome, transform=pc, fontsize=7.5, fontweight="bold",
                    va="center", ha="left", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="white", alpha=0.75,
                              edgecolor="none"))

        cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.025, shrink=0.86,
                            aspect=30, extend=st["extend"])
        cbar.set_label(unita or p["unita"], fontsize=8.5, fontweight="bold")
        cbar.ax.tick_params(labelsize=7.5)
        if st.get("etichette"):
            # scala a categorie: scritte al posto dei numeri
            centri = [(livelli[i] + livelli[i + 1]) / 2 for i in range(len(livelli) - 1)]
            cbar.set_ticks(centri)
            cbar.set_ticklabels(st["etichette"], fontsize=8.5, fontweight="bold")
            cbar.ax.tick_params(length=0)

        ax.set_title(p["titolo"], loc="left", fontsize=14, fontweight="bold", pad=24)
        ax.set_title(titolo_modello, loc="right", fontsize=11.5, fontweight="bold",
                     color="#1f4e79", pad=24)
        ax.text(0, 1.012, data, transform=ax.transAxes, fontsize=10.5, color="#333333",
                ha="left", va="bottom")
        ax.text(1, 1.012, scadenza, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
                color="#555555", ha="right", va="bottom")
        ax.text(0.012, 0.018, info, transform=ax.transAxes, fontsize=6.3, color="#555555",
                ha="left", va="bottom", zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"))
        ax.text(0.988, 0.018, FIRMA, transform=ax.transAxes, fontsize=7.5, fontweight="bold",
                ha="right", va="bottom", zorder=10,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#aaaaaa"))

        buf = io.BytesIO()
        fig.savefig(buf, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        # immagine a 256 colori: stessa resa, file 3-4 volte più leggero
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE) \
           .save(file, optimize=True)
