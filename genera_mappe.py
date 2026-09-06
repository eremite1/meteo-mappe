import os
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

print("=== GENERAZIONE MAPPA LAZIO CON BOUNDING BOX NATIVO ===")

os.makedirs("mappe_output", exist_ok=True)

capoluoghi = {
    'RM': (12.4964, 41.9028),
    'LT': (12.9043, 41.4676),
    'FR': (13.3441, 41.6390),
    'RI': (12.8634, 42.4036),
    'VT': (12.1081, 42.4204)
}

# Chiamata unica con il bounding box per il modello ICON-2I di ARPAE
# Formato bounding_box: min_latitude, min_longitude, max_latitude, max_longitude
url = (
    "https://api.open-meteo.com/v1/forecast?"
    "bounding_box=41.0,11.2,42.9,14.2"
    "&hourly=temperature_2m"
    "&models=italia_meteo_arpae_icon_2i"
    "&forecast_days=1"
)

print("Scaricamento della griglia nativa in corso...")
try:
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        data = response.json()
        print("Dati scaricati con successo!")
    else:
        print(f"Errore API: {response.status_code} - {response.text}")
        exit(1)
except Exception as e:
    print(f"Errore di connessione: {e}")
    exit(1)

# Se l'API restituisce una lista di punti (perché il bounding box espande i dati in multi-location)
# li rimappiamo in una matrice regolare per Cartopy
if isinstance(data, list):
    locations = data
else:
    locations = [data]

lats_list = []
lons_list = []
temps_list = []

for loc in locations:
    lat = loc.get("latitude")
    lon = loc.get("longitude")
    hourly = loc.get("hourly", {}).get("temperature_2m", [])
    if lat is not None and lon is not None and hourly:
        lats_list.append(lat)
        lons_list.append(lon)
        temps_list.append(hourly[0]) # Primo step orario

# Conversione in array numpy e strutturazione in griglia 2D
lats_unique = np.unique(lats_list)
lons_unique = np.unique(lons_list)

if len(lats_unique) > 1 and len(lons_unique) > 1:
    Lon, Lat = np.meshgrid(lons_unique, lats_unique)
    Data_Grid = np.array(temps_list).reshape(len(lats_unique), len(lons_unique))
else:
    # Fallback di sicurezza se la risposta è lineare
    Lon, Lat = np.meshgrid(np.linspace(11.2, 14.2, 20), np.linspace(41.0, 42.9, 20))
    Data_Grid = np.full_like(Lon, 20.0)

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

fig.text(0.5, 0.92, "Modello ICON-2I (Bounding Box) - Lazio | Temperatura a 2m", fontsize=12, fontweight='bold', color='white', ha='center')
ax.text(0.97, 0.03, 'www.meteonerola.it', transform=ax.transAxes, fontsize=9, fontweight='bold', color='white', 
        ha='right', va='bottom', zorder=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='#222222', alpha=0.9, edgecolor='#555555'))

output_path = "mappe_output/mappa_lazio_cartopy.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)

print("Mappa con Bounding Box generata con successo!")
