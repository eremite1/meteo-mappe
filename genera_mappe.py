import os
import requests
import numpy as np
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

print("=== GENERAZIONE MAPPA LAZIO HD ===")

os.makedirs("mappe_output", exist_ok=True)

capoluoghi = {
    'RM': (12.4964, 41.9028),
    'LT': (12.9043, 41.4676),
    'FR': (13.3441, 41.6390),
    'RI': (12.8634, 42.4036),
    'VT': (12.1081, 42.4204)
}

# Griglia ad alta densità (25x25 = 625 punti)
lats = np.linspace(41.0, 43.0, 25)
lons = np.linspace(11.5, 14.0, 25)
Lon, Lat = np.meshgrid(lons, lats)
Data_Grid = np.zeros_like(Lon)

def fetch_point(args):
    i, j, lat, lon = args
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m&models=icon_seamless"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            val = response.json().get("current", {}).get("temperature_2m", 20)
            return i, j, val
    except:
        pass
    return i, j, 20

print("Scaricamento parallelo griglia HD in corso...")
tasks = []
for i in range(lats.shape[0]):
    for j in range(lons.shape[0]):
        tasks.append((i, j, lats[i], lons[j]))

# Esecuzione in parallelo per massima velocità
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    results = executor.map(fetch_point, tasks)
    for i, j, val in results:
        Data_Grid[i, j] = val

levels = np.arange(0, 42, 1) # Livelli più fitti per sfumature morbide
cmap = plt.get_cmap('Spectral_r')

print("Disegno della mappa in alta definizione...")
fig = plt.figure(figsize=(12, 11), facecolor='#ffffff')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_facecolor('#eef6fc')
ax.set_extent([11.2, 14.2, 40.8, 43.0], crs=ccrs.PlateCarree())

# Usiamo shading gouraud/antialiasing indiretto tramite contourf fitto
mesh = ax.contourf(Lon, Lat, Data_Grid, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, extend='both', alpha=0.90, zorder=1)

ax.add_feature(cfeature.OCEAN, facecolor='#b4e6ff', zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#222222', zorder=3)
ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='#444444', zorder=3)

try:
    shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
    reader = shpreader.Reader(shapefile)
    provinces = [rec.geometry for rec in reader.records() if rec.attributes.get('admin') == 'Italy']
    ax.add_geometries(provinces, ccrs.PlateCarree(), edgecolor='#555555', facecolor='none', linewidth=0.5, zorder=3)
except Exception as e:
    print(f"Nota confini: {e}")

for sigla, (lon_c, lat_c) in capoluoghi.items():
    ax.plot(lon_c, lat_c, marker='o', color='black', markersize=5, transform=ccrs.PlateCarree(), zorder=5)
    ax.text(lon_c + 0.03, lat_c, sigla, transform=ccrs.PlateCarree(), fontsize=9, fontweight='bold', color='black', 
            bbox=dict(boxstyle='square,pad=0.15', facecolor='white', alpha=0.85, edgecolor='none'), zorder=6)

cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.03, shrink=0.75, extend='both')
cbar.set_label('Temperatura a 2m [°C]', color='#000000', fontsize=11, fontweight='bold', labelpad=10)
cbar.ax.tick_params(labelsize=10)

fig.text(0.5, 0.91, "Modello ICON - Lazio | Temperatura a 2m (HD)", fontsize=14, fontweight='bold', color='#111111', ha='center')
ax.text(0.97, 0.03, 'www.meteonerola.it', transform=ax.transAxes, fontsize=10, fontweight='bold', color='#111111', 
        ha='right', va='bottom', zorder=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffff', alpha=0.9, edgecolor='#666666'))

output_path = "mappe_output/mappa_lazio_cartopy.png"
# Salvataggio in alta definizione (DPI 300)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)

print(f"Mappa HD creata con successo in {output_path}!")
