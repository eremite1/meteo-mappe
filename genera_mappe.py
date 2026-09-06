import os
import requests
import numpy as np
import concurrent.futures
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

print("=== GENERAZIONE MAPPA LAZIO ICON-2I (GRIGLIA OTTIMIZZATA) ===")

os.makedirs("mappe_output", exist_ok=True)

capoluoghi = {
    'RM': (12.4964, 41.9028),
    'LT': (12.9043, 41.4676),
    'FR': (13.3441, 41.6390),
    'RI': (12.8634, 42.4036),
    'VT': (12.1081, 42.4204)
}

# Griglia a 20x20 = 400 punti (sotto il limite dei 1000 dell'API) per coprire perfettamente il Lazio
lats = np.linspace(41.0, 42.9, 20)
lons = np.linspace(11.2, 14.2, 20)
Lon, Lat = np.meshgrid(lons, lats)
Data_Grid = np.full_like(Lon, 20.0)

def fetch_point(args):
    i, j, lat, lon = args
    # Usiamo il nome modello ufficiale corretto
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m&models=italia_meteo_arpae_icon_2i"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            hourly_data = response.json().get("hourly", {}).get("temperature_2m", [])
            if hourly_data:
                return i, j, float(hourly_data[0])
    except:
        pass
    return None

tasks = []
for i in range(lats.shape[0]):
    for j in range(lons.shape[0]):
        tasks.append((i, j, lats[i], lons[j]))

print("Scaricamento dati paralleli dal modello ICON-2I...")
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(fetch_point, tasks)
    for res in results:
        if res is not None:
            i, j, val = res
            Data_Grid[i, j] = val

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
except Exception as e:
    print(f"Nota confini: {e}")

for sigla, (lon_c, lat_c) in capoluoghi.items():
    ax.plot(lon_c, lat_c, marker='o', color='black', markersize=5, transform=ccrs.PlateCarree(), zorder=5)
    ax.text(lon_c + 0.03, lat_c, sigla, transform=ccrs.PlateCarree(), fontsize=9, fontweight='bold', color='black', 
            bbox=dict(boxstyle='square,pad=0.15', facecolor='white', alpha=0.9, edgecolor='none'), zorder=6)

cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.04, shrink=0.85, extend='both')
cbar.set_label('Temperatura a 2m (°C)', color='white', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=9, colors='white')

fig.text(0.5, 0.92, "Modello ICON-2I - Lazio | Temperatura a 2m", fontsize=13, fontweight='bold', color='white', ha='center')
ax.text(0.97, 0.03, 'www.meteonerola.it', transform=ax.transAxes, fontsize=9, fontweight='bold', color='white', 
        ha='right', va='bottom', zorder=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='#222222', alpha=0.9, edgecolor='#555555'))

output_path = "mappe_output/mappa_lazio_cartopy.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)

print("Mappa ICON-2I generata con successo!")
