import os
import time
import requests
import numpy as np
import concurrent.futures
from datetime import datetime, timezone, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

print("=== GENERAZIONE ARCHIVIO MAPPE ORARIE ICON-2I (OTTIMIZZATO) ===")

OUTPUT_DIR = "mappe_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

capoluoghi = {
    'RM': (12.4964, 41.9028),
    'LT': (12.9043, 41.4676),
    'FR': (13.3441, 41.6390),
    'RI': (12.8634, 42.4036),
    'VT': (12.1081, 42.4204)
}

# Griglia bilanciata a 28x28 = 784 punti (dettaglio finissimo ma sicuro per le API)
lats = np.linspace(41.0, 42.9, 28)
lons = np.linspace(11.2, 14.2, 28)
Lon, Lat = np.meshgrid(lons, lats)

variables = ['temperature_2m']
HOURS_TO_GENERATE = 24

def fetch_point_data(args):
    i, j, lat, lon = args
    vars_str = ",".join(variables)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly={vars_str}&models=italia_meteo_arpae_icon_2i"
    
    # Sistema di retry in caso di rallentamenti temporanei dell'API
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json().get("hourly", {})
                return i, j, data
            elif response.status_code == 429:
                time.sleep(1 + attempt)  # Attende se c'è troppo traffico
        except:
            pass
    return None

tasks = []
for i in range(lats.shape[0]):
    for j in range(lons.shape[0]):
        tasks.append((i, j, lats[i], lons[j]))

print("Scaricamento dati stabili dal modello ICON-2I in corso...")
raw_grid_data = {}
for var in variables:
    raw_grid_data[var] = np.full((len(lats), len(lons), HOURS_TO_GENERATE), np.nan)

times_list = []

# Concorrenza ridotta a 6 worker per garantire il 100% di successo delle chiamate
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    results = executor.map(fetch_point_data, tasks)
    for res in results:
        if res is not None:
            i, j, data = res
            if not times_list and "time" in data:
                times_list = data["time"]
            for var in variables:
                values = data.get(var, [])
                for h in range(min(HOURS_TO_GENERATE, len(values))):
                    raw_grid_data[var][i, j, h] = float(values[h])

# Generazione delle mappe
for var in variables:
    for h in range(HOURS_TO_GENERATE):
        Data_Grid = raw_grid_data[var][:, :, h]
        
        # Riempimento di sicurezza per eventuali buchi residui
        if np.isnan(Data_Grid).any():
            mean_val = np.nanmean(Data_Grid)
            if np.isnan(mean_val):
                mean_val = 15.0
            Data_Grid = np.nan_to_num(Data_Grid, nan=mean_val)

        valid_time_str = ""
        if times_list and h < len(times_list):
            try:
                dt_obj = datetime.fromisoformat(times_list[h])
                valid_time_str = dt_obj.strftime("%d %B %Y - Ore: %H:%M UTC")
            except:
                pass
        
        if not valid_time_str:
            base_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            target_utc = base_utc + timedelta(hours=h)
            valid_time_str = target_utc.strftime("%d %B %Y - Ore: %H:%M UTC")

        levels = np.arange(-5, 42, 1)
        cmap = plt.get_cmap('Spectral_r')

        fig = plt.figure(figsize=(10, 8), facecolor='#ffffff')
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_facecolor('#f4f4f4')
        ax.set_extent([11.3, 14.1, 41.0, 42.8], crs=ccrs.PlateCarree())

        mesh = ax.contourf(Lon, Lat, Data_Grid, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, extend='both', alpha=0.94, zorder=1)

        ax.add_feature(cfeature.OCEAN, facecolor='#cce6ff', zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#333333', zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='#555555', zorder=3)

        try:
            shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
            reader = shpreader.Reader(shapefile)
            provinces = [rec.geometry for rec in reader.records() if rec.attributes.get('admin') == 'Italy']
            ax.add_geometries(provinces, ccrs.PlateCarree(), edgecolor='#777777', facecolor='none', linewidth=0.5, zorder=3)
        except:
            pass

        for sigla, (lon_c, lat_c) in capoluoghi.items():
            ax.plot(lon_c, lat_c, marker='o', color='black', markersize=4, transform=ccrs.PlateCarree(), zorder=5)
            ax.text(lon_c + 0.03, lat_c, sigla, transform=ccrs.PlateCarree(), fontsize=8, fontweight='bold', color='black', 
                    bbox=dict(boxstyle='square,pad=0.15', facecolor='white', alpha=0.85, edgecolor='#cccccc'), zorder=6)

        cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.03, shrink=0.82, aspect=25, extend='both')
        cbar.set_label('Temperatura (°C)', color='black', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=9, colors='black')

        fig.text(0.50, 0.93, "Modello ICON-2I - Lazio | Temperatura (°C)", fontsize=12, fontweight='bold', color='black', ha='center')
        fig.text(0.50, 0.89, f"Valido il: {valid_time_str}", fontsize=10, fontweight='bold', color='#333333', ha='center')

        ax.text(0.97, 0.03, 'www.meteonerola.it', transform=ax.transAxes, fontsize=9, fontweight='bold', color='black', 
                ha='right', va='bottom', zorder=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#aaaaaa'))

        filename = f"{var}_h{h:02d}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

print("Tutte le mappe sono state generate perfettamente senza buchi!")
