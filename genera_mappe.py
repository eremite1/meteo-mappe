import os
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

print("=== INIZIO GENERAZIONE MAPPA LAZIO CON CARTOPY ===")

os.makedirs("mappe_output", exist_ok=True)

capoluoghi = {
    'RM': (12.4964, 41.9028),
    'LT': (12.9043, 41.4676),
    'FR': (13.3441, 41.6390),
    'RI': (12.8634, 42.4036),
    'VT': (12.1081, 42.4204)
}

# Griglia ottimizzata per essere velocissima ma dettagliata
lats = np.linspace(41.0, 43.0, 12)
lons = np.linspace(11.5, 14.0, 12)
Lon, Lat = np.meshgrid(lons, lats)
Data_Grid = np.zeros_like(Lon)

print("Scaricamento dati in corso...")
for i in range(lats.shape[0]):
    for j in range(lons.shape[0]):
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lats[i]}&longitude={lons[j]}&current=temperature_2m&models=icon_seamless"
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                Data_Grid[i, j] = response.json().get("current", {}).get("temperature_2m", 20)
            else:
                Data_Grid[i, j] = 20
        except:
            Data_Grid[i, j] = 20

levels = np.arange(0, 42, 2)
cmap = plt.get_cmap('Spectral_r')

print("Disegno della mappa geografica...")
fig = plt.figure(figsize=(10, 9), facecolor='#ffffff')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_facecolor('#eef6fc')
ax.set_extent([11.2, 14.2, 40.8, 43.0], crs=ccrs.PlateCarree())

mesh = ax.contourf(Lon, Lat, Data_Grid, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, extend='both', alpha=0.85, zorder=1)

ax.add_feature(cfeature.OCEAN, facecolor='#b4e6ff', zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#222222', zorder=3)
ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='#444444', zorder=3)

try:
    shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
    reader = shpreader.Reader(shapefile)
    provinces = [rec.geometry for rec in reader.records() if rec.attributes.get('admin') == 'Italy']
    ax.add_geometries(provinces, ccrs.PlateCarree(), edgecolor='#666666', facecolor='none', linewidth=0.4, zorder=3)
except Exception as e:
    print(f"Nota confini: {e}")

for sigla, (lon_c, lat_c) in capoluoghi.items():
    ax.plot(lon_c, lat_c, marker='o', color='black', markersize=4, transform=ccrs.PlateCarree(), zorder=5)
    ax.text(lon_c + 0.03, lat_c, sigla, transform=ccrs.PlateCarree(), fontsize=8, fontweight='bold', color='black', 
            bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'), zorder=6)

cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.03, shrink=0.75, extend='both')
cbar.set_label('Temperatura a 2m [°C]', color='#000000', fontsize=10, fontweight='bold', labelpad=10)

fig.text(0.5, 0.90, "Modello ICON - Lazio | Temperatura a 2m", fontsize=12, fontweight='bold', color='#111111', ha='center')
ax.text(0.97, 0.03, 'www.meteonerola.it', transform=ax.transAxes, fontsize=9, fontweight='bold', color='#111111', 
        ha='right', va='bottom', zorder=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff', alpha=0.9, edgecolor='#666666'))

output_path = "mappe_output/mappa_lazio_cartopy.png"
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)

print(f"Mappa creata con successo in {output_path}!")
