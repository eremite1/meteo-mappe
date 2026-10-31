"""
Calcoli meteorologici: grandezze derivate e indici di instabilità.

Tutte le funzioni lavorano su array numpy (una mappa intera alla volta).
Unità: temperature in °C, pressioni in hPa, umidità relativa in %, vento in km/h
(salvo dove indicato), quote in metri.
"""
import numpy as np

RD = 287.04        # costante dei gas per l'aria secca (J/kg/K)
CP = 1005.7        # calore specifico a pressione costante (J/kg/K)
LV = 2.501e6       # calore latente di vaporizzazione (J/kg)
EPS = 0.622
K0 = 273.15


# ---------------------------------------------------------------------------
# Umidità
# ---------------------------------------------------------------------------
def pressione_vapore_sat(t):
    """Pressione di vapore saturo (hPa) a temperatura t (°C), formula di Bolton."""
    return 6.112 * np.exp(17.67 * t / (t + 243.5))


def punto_rugiada(t, ur):
    """Punto di rugiada (°C) da temperatura (°C) e umidità relativa (%)."""
    ur = np.clip(ur, 1, 100)
    g = np.log(ur / 100.0) + 17.67 * t / (t + 243.5)
    return 243.5 * g / (17.67 - g)


def umidita_relativa(t, td):
    return np.clip(100.0 * pressione_vapore_sat(td) / pressione_vapore_sat(t), 0, 100)


def bulbo_umido(t, ur):
    """Temperatura di bulbo umido (°C), formula di Stull (2011), valida a ~1000 hPa."""
    ur = np.clip(ur, 5, 99)
    return (t * np.arctan(0.151977 * np.sqrt(ur + 8.313659))
            + np.arctan(t + ur) - np.arctan(ur - 1.676331)
            + 0.00391838 * ur ** 1.5 * np.arctan(0.023101 * ur) - 4.686035)


def rapporto_mescolanza(td, p):
    """Rapporto di mescolanza (kg/kg) da punto di rugiada (°C) e pressione (hPa)."""
    e = pressione_vapore_sat(td)
    return EPS * e / (p - e)


# ---------------------------------------------------------------------------
# Temperature potenziali e sollevamento di una particella
# ---------------------------------------------------------------------------
def theta_e(t, td, p):
    """Temperatura potenziale equivalente (°C) secondo Bolton (1980)."""
    tk = t + K0
    r = rapporto_mescolanza(td, p)
    tl = 1.0 / (1.0 / (td + K0 - 56.0) + np.log(tk / (td + K0)) / 800.0) + 56.0
    th = tk * (1000.0 / p) ** (0.2854 * (1 - 0.28 * r))
    return th * np.exp((3.376 / tl - 0.00254) * r * 1000 * (1 + 0.81 * r)) - K0


def _gradiente_umido(tk, p):
    """dT/dp (K/hPa) lungo l'adiabatica satura (pseudo-adiabatica)."""
    es = 6.112 * np.exp(17.67 * (tk - K0) / (tk - K0 + 243.5))
    rs = EPS * es / (p - es)
    num = (RD * tk + LV * rs) / p
    den = CP + (LV * LV * rs * EPS) / (RD * tk * tk)
    return num / den


def solleva(t, td, p_da, p_a=500.0, passi=40):
    """Temperatura (°C) che raggiunge una particella sollevata da p_da a p_a hPa:
    adiabatica secca fino al livello di condensazione, poi adiabatica satura."""
    tk = t + K0
    tdk = td + K0
    # temperatura e pressione al livello di condensazione (Bolton)
    t_lcl = 1.0 / (1.0 / (tdk - 56.0) + np.log(tk / tdk) / 800.0) + 56.0
    p_lcl = p_da * (t_lcl / tk) ** (CP / RD)
    p_lcl = np.minimum(p_lcl, p_da)
    # sopra p_a già al livello di condensazione: solo adiabatica secca
    secco = p_lcl <= p_a
    t_fin_secco = tk * (p_a / p_da) ** (RD / CP)
    # adiabatica satura da p_lcl a p_a, integrata a passi
    p = np.where(secco, p_a, p_lcl).astype(float)
    tt = np.where(secco, t_fin_secco, t_lcl).astype(float)
    dp = (p_a - p) / passi
    for _ in range(passi):
        k1 = _gradiente_umido(tt, p)
        k2 = _gradiente_umido(tt + 0.5 * dp * k1, p + 0.5 * dp)
        tt = tt + dp * k2
        p = p + dp
    return np.where(secco, t_fin_secco, tt) - K0


# ---------------------------------------------------------------------------
# Indici di instabilità
# ---------------------------------------------------------------------------
def k_index(t850, td850, t700, td700, t500):
    return (t850 - t500) + td850 - (t700 - td700)


def total_totals(t850, td850, t500):
    return t850 + td850 - 2.0 * t500


def showalter(t850, td850, t500):
    """Showalter: T500 dell'ambiente meno T500 di una particella sollevata da 850 hPa."""
    return t500 - solleva(t850, td850, 850.0, 500.0)


def lifted_index(t, td, p_suolo, t500):
    """Lifted Index da particella al suolo (pressione al suolo in hPa)."""
    return t500 - solleva(t, td, p_suolo, 500.0)


def sweat(td850, tt, v850_kt, v500_kt, dir850, dir500):
    """SWEAT index (Miller 1972). Velocità in nodi, direzioni in gradi (da cui spira)."""
    a = 12.0 * np.maximum(td850, 0)
    b = 20.0 * np.maximum(tt - 49.0, 0)
    c = 2.0 * v850_kt + v500_kt
    s = np.sin(np.radians(dir500 - dir850))
    d = 125.0 * (s + 0.2)
    cond = ((dir850 >= 130) & (dir850 <= 250) & (dir500 >= 210) & (dir500 <= 310)
            & ((dir500 - dir850) > 0) & (v850_kt >= 15) & (v500_kt >= 15))
    d = np.where(cond, np.maximum(d, 0), 0)
    return a + b + c + d


def shear(u_basso, v_basso, u_alto, v_alto):
    """Differenza vettoriale di vento (stesse unità dell'input)."""
    return np.hypot(u_alto - u_basso, v_alto - v_basso)


# ---------------------------------------------------------------------------
# Vento
# ---------------------------------------------------------------------------
def uv_da_vel_dir(vel, direzione):
    rad = np.radians(direzione)
    return -vel * np.sin(rad), -vel * np.cos(rad)


def vel_dir_da_uv(u, v):
    vel = np.hypot(u, v)
    direzione = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    return vel, direzione


# ---------------------------------------------------------------------------
# Indici "pratici"
# ---------------------------------------------------------------------------
def temperatura_percepita(t, ur, vento_kmh):
    """Wind chill (freddo e vento) o Heat Index (caldo e umido), altrimenti la temperatura."""
    v = np.maximum(vento_kmh, 0)
    wc = 13.12 + 0.6215 * t - 11.37 * v ** 0.16 + 0.3965 * t * v ** 0.16
    tf = t * 9 / 5 + 32
    hi_f = (-42.379 + 2.04901523 * tf + 10.14333127 * ur - 0.22475541 * tf * ur
            - 6.83783e-3 * tf * tf - 5.481717e-2 * ur * ur + 1.22874e-3 * tf * tf * ur
            + 8.5282e-4 * tf * ur * ur - 1.99e-6 * tf * tf * ur * ur)
    hi = (hi_f - 32) * 5 / 9
    out = np.array(t, dtype=float, copy=True)
    freddo = (t <= 10) & (v > 4.8)
    caldo = (t >= 27) & (ur >= 40)
    out[freddo] = wc[freddo]
    out[caldo] = hi[caldo]
    return out


def rischio_gelate(t, ur, nuvole, vento_kmh):
    """Indice 0-4 di rischio gelo/brina.
    Parte dalla temperatura di bulbo umido (più bassa della temperatura quando l'aria è secca)
    e aumenta il rischio con cielo sereno e vento debole (forte raffreddamento notturno)."""
    tw = bulbo_umido(t, ur)
    base = np.select([tw <= -2, tw <= 0, tw <= 1.5, tw <= 3], [3.0, 2.0, 1.0, 0.5], 0.0)
    sereno = np.clip((60 - nuvole) / 60, 0, 1)
    calmo = np.clip((15 - vento_kmh) / 15, 0, 1)
    bonus = np.where(tw <= 4, sereno * calmo, 0)
    return np.clip(base + bonus, 0, 4)


def fosberg(t, ur, vento_kmh):
    """Fosberg Fire Weather Index (0-100): pericolo meteorologico di incendi
    (senza memoria dei giorni precedenti)."""
    tf = t * 9 / 5 + 32
    h = np.clip(ur, 0, 100)
    m = np.where(h < 10, 0.03229 + 0.281073 * h - 0.000578 * h * tf,
         np.where(h <= 50, 2.22749 + 0.160107 * h - 0.01478 * tf,
                  21.0606 + 0.005565 * h * h - 0.00035 * h * tf - 0.483199 * h))
    eta = 1 - 2 * (m / 30) + 1.5 * (m / 30) ** 2 - 0.5 * (m / 30) ** 3
    u_mph = vento_kmh / 1.609
    return np.clip(eta * np.sqrt(1 + u_mph ** 2) / 0.3002, 0, 100)


# ---------------------------------------------------------------------------
# Zero termico dai livelli di pressione (quando il modello non lo fornisce)
# ---------------------------------------------------------------------------
def zero_termico_da_livelli(temps, quote, t2m=None, z_suolo=None):
    """temps, quote: liste di mappe dal basso verso l'alto (°C, m).
    Restituisce la quota (m) a cui la temperatura scende sotto 0 °C la prima volta."""
    out = np.full(temps[0].shape, np.nan)
    if t2m is not None and z_suolo is not None:
        temps = [t2m] + list(temps)
        quote = [z_suolo] + list(quote)
    for k in range(len(temps) - 1):
        ta, tb, za, zb = temps[k], temps[k + 1], quote[k], quote[k + 1]
        attraversa = np.isnan(out) & (ta > 0) & (tb <= 0)
        f = ta / np.where(ta - tb == 0, np.nan, ta - tb)
        out = np.where(attraversa, za + f * (zb - za), out)
    # già sotto zero al livello più basso: zero termico "al suolo"
    gelo = np.isnan(out) & (temps[0] <= 0)
    out = np.where(gelo, quote[0], out)
    return np.maximum(out, 0)


# ---------------------------------------------------------------------------
# Parametri aggiuntivi
# ---------------------------------------------------------------------------
def humidex(t, td):
    """Humidex (indice di afa, canadese)."""
    e = 6.11 * np.exp(5417.7530 * (1 / 273.16 - 1 / (td + K0)))
    return t + 0.5555 * (e - 10.0)


def tipo_precipitazione(pioggia_mm, tw):
    """1 = pioggia, 2 = mista (pioggia e neve), 3 = neve; NaN dove non precipita.
    Si basa sulla temperatura di bulbo umido vicino al suolo."""
    tipo = np.where(tw <= 0.5, 3.0, np.where(tw <= 1.5, 2.0, 1.0))
    # 0 = non precipita (lasciato trasparente); NaN solo fuori dal territorio del modello
    return np.where(pioggia_mm >= 0.2, tipo, np.where(np.isfinite(pioggia_mm), 0.0, np.nan))


def base_nubi(t, td):
    """Quota approssimata della base delle nubi cumuliformi sopra il suolo (m): 125 m per grado di scarto."""
    return np.clip(125.0 * (t - td), 0, None)


def rischio_nebbia(t, td, vento_kmh, nuvole_basse=None):
    """0 nessuno · 1 basso · 2 moderato · 3 alto. Aria quasi satura e vento debole."""
    scarto = np.clip((2.5 - (t - td)) / 2.5, 0, 1)
    calma = np.clip((15 - vento_kmh) / 12, 0, 1)
    r = 3.0 * scarto * calma
    if nuvole_basse is not None:
        r = np.where(nuvole_basse > 80, np.maximum(r, 1.5 * scarto * 2), r)
    return np.clip(np.round(r), 0, 3)


def et0_oraria(t, ur, vento10_kmh, rad_wm2, quota_m=0.0):
    """Evapotraspirazione di riferimento FAO-56 su base oraria (mm/ora): quanta acqua
    perde un prato ben irrigato. Utile per orto e irrigazione."""
    u2 = np.maximum(vento10_kmh / 3.6 * 0.748, 0.1)
    p = 101.3 * ((293 - 0.0065 * quota_m) / 293) ** 5.26
    gam = 0.000665 * p
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    ea = es * np.clip(ur, 0, 100) / 100
    delta = 4098 * es / (t + 237.3) ** 2
    rs = np.clip(rad_wm2, 0, None) * 0.0036
    rn = 0.77 * rs - 0.07
    g = np.where(rs > 0.01, 0.1 * rn, 0.5 * rn)
    et = (0.408 * delta * (rn - g) + gam * (37 / (t + 273)) * u2 * (es - ea)) / (delta + gam * (1 + 0.34 * u2))
    return np.clip(et, 0, None)


def gradiente_verticale(t_basso, t_alto, z_basso, z_alto):
    """Diminuzione di temperatura con la quota (°C per km) tra due livelli."""
    dz = np.where(np.abs(z_alto - z_basso) < 1, np.nan, (z_alto - z_basso) / 1000.0)
    return (t_basso - t_alto) / dz
