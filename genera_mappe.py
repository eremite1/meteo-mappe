import os
import requests
import numpy as np
import concurrent.futures
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

print("=== GENERAZIONE ARCHIVIO MAPPE ORARIE ICON-2I ===")

OUTPUT_DIR = "mappe_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

capoluoghi = {
    'RM': (12.4964, 41.9028),
    'LT': (12.9043, 41.4676),
    'FR': (13.3441, 41.6390),
    'RI': (12.8634, 42.4036),
    'VT': (12.1081, 42.4204)
}

# Griglia ottimizzata a 20x20 = 400 punti per coprire il Lazio
lats = np.linspace(41.0, 42.9, 20)
lons = np.linspace(11.2, 14.2, 20)
Lon, Lat = np.meshgrid(lons, lats)

variables = ['temperature_2m']
HOURS_TO_GENERATE = 24  # Prime 24 ore

def fetch_point_data(args):
    i, j, lat, lon = args
    vars_str = ",".join(variables)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly={vars_str}&models=italia_meteo_arpae_icon_2i"
    try:
        response = requests.get(url, timeout=8)
        if response.status_code == 200:
            data = response.json().get("hourly", {})
            return i, j, data
    except:
        pass
    return None

tasks = []
for i in range(lats.shape[0]):
    for j in range(lons.shape[0]):
        tasks.append((i, j, lats[i], lons[j]))

print("Scaricamento dati completi dal modello ICON-2I in corso...")
raw_grid_data = {}
for var in variables:
    raw_grid_data[var] = np.full((len(lats), len(lons), HOURS_TO_GENERATE), np.nan)

# Usiamo max_workers=8 per non sovraccaricare l'API ed evitare richieste rifiutate/NaN
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    results = executor.map(fetch_point_data, tasks)
    for res in results:
        if res is not None:
            i, j, data = res
            for var in variables:
                values = data.get(var, [])
                for h in range(min(HOURS_TO_GENERATE, len(values))):
                    raw_grid_data[var][i, j, h] = float(values[h])

# Generazione delle mappe per ogni variabile e per ogni ora
for var in variables:
    for h in range(HOURS_TO_GENERATE):
        Data_Grid = raw_grid_data[var][:, :, h]
        
        # Se ci sono valori NaN (buchi neri), li interpoliamo usando i valori vicini
        if np.isnan(Data_Grid).any():
            mask = np.isnan(Data_Grid)
            try:
                Data_Grid[mask] = ndimage.generic_filter(
                    Data_Grid, 
                    lambda x: np.nanmean(x[~np.isnan(x)]) if np.any(~np.isnan(x)) else 0, 
                    size=3, 
                    mode='nearest'
                )[mask]
            except:
                pass

        if np.isnan(Data_Grid).all():
            continue

        levels = np.arange(-5, 42, 1)
        cmap = plt.get_cmap('Spectral_r')

        fig = plt.figure(figsize=(10, 9), facecolor='#1a1a1a')
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_facecolor('#1a1a1a')
        ax.set_extent([11.3, 14.1, 41.0, 42.8], crs=ccrs.PlateCarree())

        mesh = ax.contourf(Lon, Lat, Data_Grid, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, extend='both', alpha=0.96, zorder=1)

        ax.add_feature(cfeature.OCEAN, facecolor='#122b39', zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#111111', zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='#222222', zorder=3)

        try:
            shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
            reader = shpreader.Reader(shapefile)
            provinces = [rec.geometry for rec in reader.records() if rec.attributes.get('admin') == 'Italy']
            ax.add_geometries(provinces, ccrs.PlateCarree(), edgecolor='#222222', facecolor='none', linewidth=0.6, zorder=3)
        except:
            pass

        for sigla, (lon_c, lat_c) in capoluoghi.items():
            ax.plot(lon_c, lat_c, marker='o', color='black', markersize=5, transform=ccrs.PlateCarree(), zorder=5)
            ax.text(lon_c + 0.03, lat_c, sigla, transform=ccrs.PlateCarree(), fontsize=9, fontweight='bold', color='black', 
                    bbox=dict(boxstyle='square,pad=0.15', facecolor='white', alpha=0.9, edgecolor='none'), zorder=6)

        cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.04, shrink=0.85, extend='both')
        cbar.set_label('Temperatura a 2m (°C)', color='white', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=9, colors='white')

        fig.text(0.5, 0.92, f"Modello ICON-2I - Lazio | Temperatura a 2m (+{h}h)", fontsize=13, fontweight='bold', color='white', ha='center')
        ax.text(0.97, 0.03, 'www.meteonerola.it', transform=ax.transAxes, fontsize=9, fontweight='bold', color='white', 
                ha='right', va='bottom', zorder=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='#222222', alpha=0.9, edgecolor='#555555'))

        filename = f"{var}_h{h:02d}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

print("Tutte le mappe orarie sono state generate con successo nella cartella mappe_output/")
