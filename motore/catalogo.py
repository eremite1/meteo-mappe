"""
Catalogo: modelli, aree geografiche e parametri (con colori e modo di calcolo).
Per aggiungere un parametro basta aggiungere una voce a PARAMETRI.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import calcoli as C


# ---------------------------------------------------------------------------
# MODELLI
# fonti: per ogni area, i dataset dell'archivio Open-Meteo da cui leggere (in ordine di
#        preferenza: se un parametro manca nel primo si cerca nel secondo)
# aree:  passo tra le mappe (ore) e scadenza massima (ore)
# sigma: quanto lisciare il terreno per imitare quello "visto" dal modello (celle da 500 m)
# min_ore: si elaborano solo le corse lunghe almeno così (scarta le corse brevi intermedie)
# ---------------------------------------------------------------------------
MODELLI = {
    "icon2i": dict(nome="ICON-2I", ris="2,2 km", sigma=4, min_ore=40,
                   fonti={"*": ["italia_meteo_arpae_icon_2i"]},
                   aree={"lazio": (1, 72), "italia": (3, 168)}),
    "arome": dict(nome="AROME", ris="1,3 km", sigma=3, min_ore=40,
                  fonti={"*": ["meteofrance_arome_france_hd"]},
                  aree={"lazio": (1, 72)}),
    "ecmwf": dict(nome="ECMWF IFS", ris="9 km", sigma=12, min_ore=90,
                  fonti={"lazio": ["ecmwf_ifs", "ecmwf_ifs025"],
                         "italia": ["ecmwf_ifs", "ecmwf_ifs025"],
                         "europa": ["ecmwf_ifs025"]},
                  aree={"lazio": (3, 72), "italia": (3, 168), "europa": (3, 168)}),
    "ukmo": dict(nome="UKMO", ris="10 km", sigma=14, min_ore=90,
                 fonti={"*": ["ukmo_global_deterministic_10km"]},
                 aree={"lazio": (3, 72), "italia": (3, 168), "europa": (3, 168)}),
    "iconeu": dict(nome="ICON-EU", ris="7 km", sigma=10, min_ore=90,
                   fonti={"*": ["dwd_icon_eu"]},
                   aree={"italia": (3, 168), "europa": (3, 168)}),
}


def fonti_di(modello, area):
    f = MODELLI[modello]["fonti"]
    return f.get(area) or f["*"]


# ---------------------------------------------------------------------------
# AREE GEOGRAFICHE
# ---------------------------------------------------------------------------
AREE = {
    "lazio": dict(nome="Lazio", extent=[11.3, 14.1, 41.0, 42.8], passo=0.005,
                  proiezione="piatta", terreno=True, scala_coste="10m",
                  confini="province", frecce_ogni=0.12, figsize=(10, 7.6),
                  citta={"RM": (12.4964, 41.9028), "LT": (12.9043, 41.4676),
                         "FR": (13.3441, 41.6390), "RI": (12.8634, 42.4036),
                         "VT": (12.1081, 42.4204), "Nerola": (12.7856, 42.1606)}),
    "italia": dict(nome="Italia", extent=[6.0, 19.0, 36.0, 47.4], passo=0.03,
                   dati=[3.0, 22.0, 34.5, 48.9],
                   proiezione="lambert", centro=(12.5, 42.0), scala_coste="10m",
                   confini="regioni", frecce_ogni=0.55, figsize=(9.2, 9.4),
                   citta={"Roma": (12.50, 41.90), "Milano": (9.19, 45.46),
                          "Napoli": (14.27, 40.85), "Torino": (7.69, 45.07),
                          "Palermo": (13.36, 38.12), "Bari": (16.87, 41.12),
                          "Firenze": (11.26, 43.77), "Bologna": (11.34, 44.49),
                          "Venezia": (12.33, 45.44), "Cagliari": (9.11, 39.22),
                          "Reggio C.": (15.65, 38.11), "Pescara": (14.21, 42.46),
                          "Ancona": (13.52, 43.62), "Genova": (8.93, 44.41)}),
    "europa": dict(nome="Europa", extent=[-20.0, 40.0, 33.0, 68.0], passo=0.2,
                   dati=[-52.0, 78.0, 24.0, 80.0],     # più ampia: la proiezione mostra anche gli angoli
                   proiezione="lambert", centro=(10.0, 50.0), scala_coste="50m",
                   confini="stati", frecce_ogni=2.2, figsize=(10, 7.8),
                   citta={"Roma": (12.50, 41.90), "Parigi": (2.35, 48.86),
                          "Londra": (-0.13, 51.51), "Berlino": (13.40, 52.52),
                          "Madrid": (-3.70, 40.42), "Vienna": (16.37, 48.21),
                          "Varsavia": (21.01, 52.23), "Atene": (23.73, 37.98),
                          "Oslo": (10.75, 59.91), "Kiev": (30.52, 50.45),
                          "Istanbul": (28.98, 41.01), "Lisbona": (-9.14, 38.72)}),
}


# ---------------------------------------------------------------------------
# COLORI
# ---------------------------------------------------------------------------
def cm(colori, nome):
    return LinearSegmentedColormap.from_list(nome, colori)


CMAP_TEMP = cm(["#2c0a4a", "#5b2a8c", "#3f4fbf", "#2f8fd8", "#5cc8e8", "#a6e6d8",
                "#e9f7c7", "#fff3a6", "#ffd27a", "#ff9d4d", "#f45a2c", "#c41e3a",
                "#7a0b2a"], "temp")
CMAP_PIOGGIA = ["#c8e6ff", "#96cdff", "#5aaaf5", "#1e78dc", "#00b45a", "#5ad200",
                "#c8e600", "#ffdc00", "#ffa000", "#ff3c00", "#c80078", "#800080"]
CMAP_NEVE = ["#e8f4ff", "#c6e2ff", "#a0c8f0", "#78aee6", "#5590d2", "#6a6ad2",
             "#8c55c8", "#aa3cb4", "#c8288c", "#dc1e5a"]
CMAP_VENTO = cm(["#f7fbff", "#c6e9c2", "#7fd0a8", "#3fb0c8", "#2b7fd8", "#7b5cd6",
                 "#c53fb3", "#e8355f", "#f08a1f", "#ffe14d"], "vento")
CMAP_NUVOLE = cm(["#eef2f6", "#c3cbd3", "#8e99a6", "#56606c"], "nuvole")
CMAP_INSTAB = cm(["#fffbd0", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02",
                  "#b3001b", "#7a0177", "#49006a"], "instab")
CMAP_UMIDO = cm(["#f6e8c3", "#dfc27d", "#f5f5f5", "#c7eae5", "#80cdc1", "#35978f",
                 "#01665e", "#003c30"], "umido")


def livelli(da, a, passo):
    return list(np.round(np.arange(da, a + passo / 2, passo), 3))


S_TEMP = dict(livelli=livelli(-30, 45, 1), cmap=CMAP_TEMP, extend="both")
S_TEMP_EU = dict(livelli=livelli(-36, 44, 2), cmap=CMAP_TEMP, extend="both")
S_PIOGGIA = dict(livelli=[0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150],
                 colori=CMAP_PIOGGIA, extend="max", ombra=True)
S_PIOGGIA_TOT = dict(livelli=[1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200],
                     colori=CMAP_PIOGGIA, extend="max", ombra=True)
S_NEVE = dict(livelli=[0.2, 0.5, 1, 2, 3, 5, 10, 15, 20, 30, 50],
              colori=CMAP_NEVE, extend="max", ombra=True)
S_NEVE_SUOLO = dict(livelli=[1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200],
                    colori=CMAP_NEVE, extend="max", ombra=True)


# ---------------------------------------------------------------------------
# PARAMETRI
# Ogni parametro: id, titolo, categoria, unità, stile, aree, funzione di calcolo.
# La funzione riceve "c" (contesto) e restituisce una mappa oppure un dizionario
# con chiavi: campo, contorni=(campo, livelli), frecce=(u, v).
# Se un dato necessario manca, il contesto solleva un'eccezione e la mappa viene saltata.
# ---------------------------------------------------------------------------
TUTTE = {"lazio", "italia", "europa"}
LOC = {"lazio", "italia"}
PARAMETRI = []


def P(id, titolo, categoria, unita, stile, aree, calcolo, **extra):
    PARAMETRI.append(dict(id=id, titolo=titolo, categoria=categoria, unita=unita,
                          stile=stile, aree=aree, calcolo=calcolo, **extra))


def _isobare(c, liscia):
    p = c.v("pressure_msl")
    return c.liscia(p, liscia)


def _liscia_isobare(c):
    return {"lazio": 16, "italia": 6, "europa": 1.5}[c.area]


def _passo_isobare(c):
    return {"lazio": 1, "italia": 2, "europa": 4}[c.area]


def _livelli_da(campo, passo):
    v = campo[np.isfinite(campo)]
    if v.size == 0:
        return [0, 1]
    return list(np.arange(np.floor(v.min() / passo) * passo, np.ceil(v.max() / passo) * passo + passo, passo))


# --- Temperatura ------------------------------------------------------------
P("temperatura", "Temperatura a 2 m", "Temperatura", "°C", S_TEMP, TUTTE,
  lambda c: c.t2m(), stile_europa=S_TEMP_EU)
P("percepita", "Temperatura percepita", "Temperatura", "°C (vento e umidità)", S_TEMP, LOC,
  lambda c: C.temperatura_percepita(c.t2m(), c.ur2m(), c.vel("10m")))
P("rugiada", "Punto di rugiada", "Temperatura", "°C",
  dict(livelli=livelli(-24, 28, 2), cmap=cm(["#7f3b08", "#e08214", "#fee0b6", "#f7f7f7",
                                               "#d8daeb", "#8073ac", "#2d004b"], "td"),
       extend="both"), TUTTE, lambda c: c.td2m())
P("bulbo_umido", "Temperatura di bulbo umido", "Temperatura", "°C", S_TEMP, LOC,
  lambda c: C.bulbo_umido(c.t2m(), c.ur2m()))
P("umidita", "Umidità relativa", "Temperatura", "%",
  dict(livelli=livelli(0, 100, 10), cmap=CMAP_UMIDO, extend="neither"), LOC,
  lambda c: c.ur2m())
P("gelate", "Rischio gelate e brina", "Temperatura", "0 nessuno · 1 basso · 2 moderato · 3 alto · 4 molto alto",
  dict(livelli=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.01],
       colori=["#d6f0ff", "#a8d8f0", "#7cbde6", "#4f9ad6", "#2f6fc0", "#3d3da8", "#5b1f8f"],
       extend="neither", ombra=True), LOC,
  lambda c: C.rischio_gelate(c.t2m(), c.ur2m(), c.nuvole(), c.vel("10m")))
P("zero_termico", "Zero termico", "Temperatura", "metri di quota",
  dict(livelli=livelli(0, 4500, 250), cmap=plt.get_cmap("Spectral_r"), extend="max"), TUTTE,
  lambda c: c.zero_termico())
P("quota_neve", "Quota neve", "Temperatura", "metri di quota",
  dict(livelli=livelli(0, 3000, 200), cmap=plt.get_cmap("Spectral_r"), extend="max"), LOC,
  lambda c: c.v("snowfall_height", minimo=0))

# --- Precipitazioni -----------------------------------------------------------
P("pioggia", "Precipitazioni", "Precipitazioni", "mm nelle ore precedenti", S_PIOGGIA, TUTTE,
  lambda c: c.acc("precipitation"))
P("pioggia_isobare", "Precipitazioni e isobare", "Precipitazioni", "mm nelle ore precedenti · isobare in hPa",
  S_PIOGGIA, {"italia", "europa"},
  lambda c: dict(campo=c.acc("precipitation"),
                 contorni=(_isobare(c, _liscia_isobare(c)), None, _passo_isobare(c))))
P("pioggia_totale", "Precipitazioni totali", "Precipitazioni", "mm dall'inizio della previsione",
  S_PIOGGIA_TOT, TUTTE, lambda c: c.acc_totale("precipitation"))
P("rovesci", "Rovesci e temporali", "Precipitazioni", "mm di pioggia convettiva nelle ore precedenti",
  S_PIOGGIA, LOC, lambda c: c.acc("showers"))
P("neve", "Neve fresca", "Precipitazioni", "cm nelle ore precedenti", S_NEVE, TUTTE,
  lambda c: c.neve_acc())
P("neve_totale", "Neve fresca totale", "Precipitazioni", "cm dall'inizio della previsione",
  S_NEVE_SUOLO, LOC, lambda c: c.neve_acc(totale=True))
P("neve_suolo", "Neve al suolo", "Precipitazioni", "cm", S_NEVE_SUOLO, TUTTE,
  lambda c: c.neve_suolo())

# --- Vento ------------------------------------------------------------------
P("vento", "Vento a 10 m", "Vento", "km/h (le frecce indicano la direzione)",
  dict(livelli=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110], cmap=CMAP_VENTO, extend="max"),
  TUTTE, lambda c: dict(campo=c.vel("10m"), frecce=c.uv("10m")))
P("raffiche", "Raffiche di vento", "Vento", "km/h",
  dict(livelli=[20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140], cmap=plt.get_cmap("YlOrRd"),
       extend="max", ombra=True), TUTTE, lambda c: c.v("wind_gusts_10m", minimo=0, unita="vento"))
P("vento_850", "Vento a 850 hPa (~1500 m)", "Vento", "km/h",
  dict(livelli=[10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140], cmap=CMAP_VENTO, extend="max"),
  TUTTE, lambda c: dict(campo=c.vel("850hPa"), frecce=c.uv("850hPa")))
P("getto", "Corrente a getto (300 hPa)", "Vento", "km/h a circa 9000 m",
  dict(livelli=[80, 100, 120, 140, 160, 180, 200, 220, 250, 280, 320], cmap=CMAP_VENTO, extend="max"),
  {"italia", "europa"}, lambda c: dict(campo=c.vel(c.livello_getto()), frecce=c.uv(c.livello_getto())))

# --- Nuvole, pressione, visibilità -----------------------------------------
S_NUV = dict(livelli=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], cmap=CMAP_NUVOLE, extend="neither", ombra=True)
P("nuvole", "Nuvolosità totale", "Nuvole e pressione", "% di cielo coperto", S_NUV, TUTTE,
  lambda c: c.nuvole())
P("nuvole_basse", "Nuvole basse", "Nuvole e pressione", "% (sotto i 2000 m)", S_NUV, TUTTE,
  lambda c: c.v("cloud_cover_low", 0, 100))
P("nuvole_medie", "Nuvole medie", "Nuvole e pressione", "% (2000-6000 m)", S_NUV, TUTTE,
  lambda c: c.v("cloud_cover_mid", 0, 100))
P("nuvole_alte", "Nuvole alte", "Nuvole e pressione", "% (sopra i 6000 m)", S_NUV, TUTTE,
  lambda c: c.v("cloud_cover_high", 0, 100))
P("pressione", "Pressione al livello del mare", "Nuvole e pressione", "hPa",
  dict(livelli=livelli(960, 1050, 2), cmap=plt.get_cmap("RdBu_r"), extend="both", alpha=0.55),
  TUTTE, lambda c: dict(campo=_isobare(c, _liscia_isobare(c)),
                        contorni=(_isobare(c, _liscia_isobare(c)), None, _passo_isobare(c))))
P("visibilita", "Visibilità e nebbia", "Nuvole e pressione", "km",
  dict(livelli=[0.2, 0.5, 1, 2, 5, 10], colori=["#7a7a7a", "#9a9a9a", "#b8b8b8", "#d4d4d4", "#ececec"],
       extend="min", inverso=True), LOC, lambda c: c.visibilita())
P("radiazione", "Radiazione solare", "Nuvole e pressione", "W/m²",
  dict(livelli=livelli(50, 1000, 50), cmap=plt.get_cmap("inferno"), extend="max"), LOC,
  lambda c: c.v("shortwave_radiation", minimo=0))

# --- Quota (carte sinottiche) -------------------------------------------------
P("z500_t500", "Geopotenziale e temperatura a 500 hPa", "Quota", "°C a 500 hPa · isoipse in decametri",
  dict(livelli=livelli(-44, -4, 2), cmap=CMAP_TEMP, extend="both"), {"italia", "europa"},
  lambda c: dict(campo=c.v("temperature_500hPa"),
                 contorni=(c.liscia(c.v("geopotential_height_500hPa"), 1) / 10, None,
                           4 if c.area == "europa" else 2)))
P("t850", "Temperatura a 850 hPa e geopotenziale", "Quota", "°C a circa 1500 m · isoipse in decametri",
  dict(livelli=livelli(-28, 30, 2), cmap=CMAP_TEMP, extend="both"), TUTTE,
  lambda c: dict(campo=c.v("temperature_850hPa"),
                 contorni=(c.liscia(c.v("geopotential_height_850hPa"), 1) / 10, None,
                           4 if c.area == "europa" else 2)))
P("umidita_700", "Umidità relativa a 700 hPa", "Quota", "% a circa 3000 m",
  dict(livelli=livelli(10, 100, 10), cmap=CMAP_UMIDO, extend="neither"), TUTTE,
  lambda c: c.v("relative_humidity_700hPa", 0, 100))
P("moti_verticali", "Moti verticali a 700 hPa", "Quota", "cm/s (positivo = aria che sale)",
  dict(livelli=[-60, -40, -25, -15, -8, -3, 3, 8, 15, 25, 40, 60], cmap=plt.get_cmap("RdBu"),
       extend="both", buco_centrale=True), TUTTE, lambda c: c.ascesa("700hPa"))
P("theta_e_850", "Theta-e a 850 hPa", "Quota", "°C (aria calda e umida = più energia)",
  dict(livelli=livelli(0, 80, 4), cmap=CMAP_TEMP, extend="both"), TUTTE,
  lambda c: C.theta_e(c.v("temperature_850hPa"), c.td_livello("850hPa"), 850.0))
P("acqua_precipitabile", "Acqua precipitabile", "Quota", "mm di vapore nella colonna d'aria",
  dict(livelli=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60],
       cmap=cm(["#fff7d6", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#253494",
                "#5e2a84", "#9e1f86"], "pw"), extend="max"), TUTTE,
  lambda c: c.v("total_column_integrated_water_vapour", minimo=0))
P("temperatura_mare", "Temperatura del mare", "Quota", "°C in superficie",
  dict(livelli=livelli(6, 30, 1), cmap=CMAP_TEMP, extend="both", solo_mare=True),
  {"italia", "europa"}, lambda c: c.v("sea_surface_temperature"))
P("strato_limite", "Altezza dello strato rimescolato", "Quota", "metri",
  dict(livelli=[100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000],
       cmap=plt.get_cmap("YlGnBu"), extend="max"), LOC,
  lambda c: c.v("boundary_layer_height", minimo=0))

# --- Temporali e instabilità --------------------------------------------------
P("cape", "CAPE (energia per i temporali)", "Temporali", "J/kg",
  dict(livelli=[50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000], cmap=CMAP_INSTAB,
       extend="max", ombra=True), TUTTE, lambda c: c.v("cape", minimo=0))
P("cin", "CIN (inibizione convettiva)", "Temporali", "J/kg (più alto = temporali più 'bloccati')",
  dict(livelli=[5, 25, 50, 100, 150, 200, 300, 400, 500], cmap=plt.get_cmap("Blues"),
       extend="max", ombra=True), LOC, lambda c: np.abs(c.v("convective_inhibition")))
P("fulminazioni", "Potenziale di fulminazione", "Temporali", "J/kg (Lightning Potential Index)",
  dict(livelli=[0.5, 1, 2, 5, 10, 20, 35, 50, 75, 100], cmap=CMAP_INSTAB, extend="max", ombra=True),
  LOC, lambda c: c.v("lightning_potential", minimo=0))
P("k_index", "K-Index", "Temporali", "sopra 30: temporali probabili · sopra 35: diffusi",
  dict(livelli=[15, 20, 24, 28, 30, 32, 34, 36, 38, 40, 44], cmap=CMAP_INSTAB, extend="both"), TUTTE,
  lambda c: C.k_index(c.v("temperature_850hPa"), c.td_livello("850hPa"),
                      c.v("temperature_700hPa"), c.td_livello("700hPa"), c.v("temperature_500hPa")))
P("total_totals", "Total Totals", "Temporali", "sopra 48: temporali · sopra 55: forti",
  dict(livelli=[40, 44, 46, 48, 50, 52, 54, 56, 58, 60, 64], cmap=CMAP_INSTAB, extend="both"), TUTTE,
  lambda c: C.total_totals(c.v("temperature_850hPa"), c.td_livello("850hPa"), c.v("temperature_500hPa")))
P("showalter", "Showalter Index", "Temporali", "sotto 0: instabile · sotto -4: molto instabile",
  dict(livelli=[-9, -7, -5, -4, -3, -2, -1, 0, 1, 2, 4], cmap=CMAP_INSTAB, extend="both", inverso=True),
  TUTTE, lambda c: C.showalter(c.v("temperature_850hPa"), c.td_livello("850hPa"), c.v("temperature_500hPa")))
P("lifted_index", "Lifted Index (dal suolo)", "Temporali", "sotto 0: instabile · sotto -4: molto instabile",
  dict(livelli=[-9, -7, -5, -4, -3, -2, -1, 0, 1, 2, 4], cmap=CMAP_INSTAB, extend="both", inverso=True),
  {"lazio"}, lambda c: C.lifted_index(c.t2m(), c.td2m(), c.pressione_suolo(), c.v("temperature_500hPa")))
P("shear", "Wind shear 0-6 km", "Temporali", "m/s (sopra 15-20: temporali organizzati)",
  dict(livelli=[5, 10, 12.5, 15, 17.5, 20, 25, 30, 40], cmap=plt.get_cmap("YlGnBu"), extend="max"), TUTTE,
  lambda c: dict(campo=C.shear(*c.uv("10m"), *c.uv("500hPa")) / 3.6,
                 frecce=tuple(a - b for a, b in zip(c.uv("500hPa"), c.uv("10m")))))
P("sweat", "SWEAT Index", "Temporali", "sopra 300: temporali forti · sopra 400: possibili tornado",
  dict(livelli=[100, 150, 200, 250, 300, 350, 400, 500, 600], cmap=CMAP_INSTAB, extend="max"), TUTTE,
  lambda c: c.sweat())

# --- Altri indici ---------------------------------------------------------------
P("incendi", "Pericolo incendi (Fosberg)", "Altri indici", "0-100 (sopra 50: alto)",
  dict(livelli=[10, 20, 30, 40, 50, 60, 70, 80],
       cmap=cm(["#fff7bc", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#8c2d04"], "fuoco"),
       extend="max"), LOC, lambda c: C.fosberg(c.t2m(), c.ur2m(), c.vel("10m")))


# --- Parametri aggiuntivi -------------------------------------------------------
P("humidex", "Humidex (afa)", "Temperatura", "°C percepiti col caldo umido (sopra 40: disagio forte)",
  dict(livelli=livelli(20, 48, 2), cmap=cm(["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
                                               "#fc4e2a", "#e31a1c", "#b10026", "#67001f"], "afa"),
       extend="both"), LOC, lambda c: C.humidex(c.t2m(), c.td2m()))
P("inversione", "Inversione termica", "Temperatura",
  "°C: temperatura a 1500 m meno quella al suolo (positiva = aria più calda in quota)",
  dict(livelli=[-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10], cmap=plt.get_cmap("RdBu_r"),
       extend="both"), {"lazio"}, lambda c: c.v("temperature_850hPa") - c.t2m())
P("t925", "Temperatura a 925 hPa (~800 m)", "Temperatura", "°C",
  dict(livelli=livelli(-24, 34, 2), cmap=CMAP_TEMP, extend="both"), LOC,
  lambda c: c.v("temperature_925hPa"))
P("tipo_precipitazione", "Pioggia, neve o mista", "Precipitazioni", "tipo di precipitazione nelle ore precedenti",
  dict(livelli=[0.5, 1.5, 2.5, 3.5], colori=["#3a8fd9", "#b36ad9", "#e8f4ff"], extend="neither",
       ombra=True, etichette=["pioggia", "mista", "neve"]), LOC,
  lambda c: C.tipo_precipitazione(c.acc("precipitation"), C.bulbo_umido(c.t2m(), c.ur2m())))
P("vento_925", "Vento a 925 hPa (~800 m)", "Vento", "km/h",
  dict(livelli=[10, 20, 30, 40, 50, 60, 70, 80, 100, 120], cmap=CMAP_VENTO, extend="max"), LOC,
  lambda c: dict(campo=c.vel("925hPa"), frecce=c.uv("925hPa")))
P("vento_700", "Vento a 700 hPa (~3000 m)", "Vento", "km/h",
  dict(livelli=[10, 20, 30, 40, 50, 60, 80, 100, 120, 150], cmap=CMAP_VENTO, extend="max"), TUTTE,
  lambda c: dict(campo=c.vel("700hPa"), frecce=c.uv("700hPa")))
P("vento_500", "Vento a 500 hPa (~5500 m)", "Vento", "km/h",
  dict(livelli=[20, 40, 60, 80, 100, 120, 140, 160, 200, 240], cmap=CMAP_VENTO, extend="max"), TUTTE,
  lambda c: dict(campo=c.vel("500hPa"), frecce=c.uv("500hPa")))
P("base_nubi", "Base delle nubi cumuliformi", "Nuvole e visibilità", "metri sopra il suolo",
  dict(livelli=[100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000], cmap=plt.get_cmap("GnBu_r"),
       extend="max"), LOC, lambda c: C.base_nubi(c.t2m(), c.td2m()))
P("nebbia", "Rischio nebbia", "Nuvole e visibilità", "",
  dict(livelli=[0.5, 1.5, 2.5, 3.5], colori=["#d9d9d9", "#a6a6a6", "#6e6e6e"], extend="neither",
       ombra=True, etichette=["basso", "moderato", "alto"]), LOC,
  lambda c: C.rischio_nebbia(c.t2m(), c.td2m(), c.vel("10m"),
                             c.v("cloud_cover_low", 0, 100) if c.ha("cloud_cover_low") else None))
P("umidita_850", "Umidità relativa a 850 hPa", "Pressione e quota", "% a circa 1500 m",
  dict(livelli=livelli(10, 100, 10), cmap=CMAP_UMIDO, extend="neither"), TUTTE,
  lambda c: c.v("relative_humidity_850hPa", 0, 100))
P("gradiente_850_500", "Gradiente termico 850-500 hPa", "Pressione e quota",
  "°C per km (sopra 7: atmosfera instabile)",
  dict(livelli=[4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9], cmap=CMAP_INSTAB, extend="both"), TUTTE,
  lambda c: C.gradiente_verticale(c.v("temperature_850hPa"), c.v("temperature_500hPa"),
                                  c.v("geopotential_height_850hPa"), c.v("geopotential_height_500hPa")))
P("cape_shear", "Temporali organizzati (CAPE × shear)", "Temporali",
  "sopra 1: possibili temporali organizzati · sopra 2: supercelle",
  dict(livelli=[0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4], cmap=CMAP_INSTAB, extend="max", ombra=True), TUTTE,
  lambda c: (c.v("cape", minimo=0) / 1000.0) * (C.shear(*c.uv("10m"), *c.uv("500hPa")) / 3.6 / 20.0))
P("instabilita_potenziale", "Instabilità potenziale", "Temporali",
  "°C: theta-e a 850 meno theta-e a 500 hPa (positivo = instabile se l'aria viene sollevata)",
  dict(livelli=[-12, -8, -4, -2, 0, 2, 4, 6, 8, 12, 16], cmap=CMAP_INSTAB, extend="both"), TUTTE,
  lambda c: C.theta_e(c.v("temperature_850hPa"), c.td_livello("850hPa"), 850.0)
            - C.theta_e(c.v("temperature_500hPa"), c.td_livello("500hPa"), 500.0))
P("temperatura_suolo", "Temperatura del terreno in superficie", "Suolo e agricoltura", "°C",
  dict(livelli=livelli(-10, 40, 2), cmap=CMAP_TEMP, extend="both"), LOC,
  lambda c: c.primo("soil_temperature_0cm", "soil_temperature_0_to_7cm"))
P("umidita_suolo", "Umidità del terreno (primi cm)", "Suolo e agricoltura", "% di acqua nel terreno",
  dict(livelli=[5, 10, 15, 20, 25, 30, 35, 40, 45], cmap=cm(["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5",
                                                              "#5ab4ac", "#01665e"], "suolo"), extend="both"),
  LOC, lambda c: c.primo("soil_moisture_0_to_1cm", "soil_moisture_0_to_7cm", minimo=0) * 100)
P("evapotraspirazione", "Evapotraspirazione (ET0)", "Suolo e agricoltura",
  "mm/ora d'acqua persa da un prato (riferimento per irrigare)",
  dict(livelli=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
       cmap=cm(["#f7fcb9", "#addd8e", "#41ab5d", "#006837", "#004529"], "et0"), extend="max"), LOC,
  lambda c: C.et0_oraria(c.t2m(), c.ur2m(), c.vel("10m"), c.v("shortwave_radiation"), c.quota_suolo()))

# --- Macrocategorie (ordine e raggruppamento sulla pagina) -------------------------
CATEGORIA_DI = {
    "Temperatura": ["temperatura", "percepita", "humidex", "rugiada", "bulbo_umido", "umidita",
                    "inversione", "t925", "zero_termico", "quota_neve"],
    "Precipitazioni": ["pioggia", "tipo_precipitazione", "pioggia_isobare", "pioggia_totale", "rovesci",
                       "neve", "neve_totale", "neve_suolo"],
    "Vento": ["vento", "raffiche", "vento_925", "vento_850", "vento_700", "vento_500", "getto"],
    "Nuvole e visibilità": ["nuvole", "nuvole_basse", "nuvole_medie", "nuvole_alte", "base_nubi",
                            "nebbia", "visibilita", "radiazione"],
    "Pressione e quota": ["pressione", "z500_t500", "t850", "umidita_850", "umidita_700",
                          "moti_verticali", "theta_e_850", "gradiente_850_500", "acqua_precipitabile",
                          "strato_limite", "temperatura_mare"],
    "Temporali": ["cape", "cin", "fulminazioni", "cape_shear", "instabilita_potenziale", "k_index",
                  "total_totals", "showalter", "lifted_index", "shear", "sweat"],
    "Suolo e agricoltura": ["gelate", "temperatura_suolo", "umidita_suolo", "evapotraspirazione", "incendi"],
}
for _cat, _ids in CATEGORIA_DI.items():
    for _p in PARAMETRI:
        if _p["id"] in _ids:
            _p["categoria"] = _cat
CATEGORIE = list(CATEGORIA_DI)
_ordine = [i for ids in CATEGORIA_DI.values() for i in ids]
PARAMETRI.sort(key=lambda p: _ordine.index(p["id"]) if p["id"] in _ordine else 999)
