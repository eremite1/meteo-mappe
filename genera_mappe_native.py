import os
import time
import requests
import numpy as np
import concurrent.futures
from datetime import datetime, timezone, timedelta
from scipy.ndimage import zoom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

print("=== GENERAZIONE MAPPE NATIVE DARK THEME (ICON-2I) ===")

OUTPUT_DIR = "mappe_native"
os.makedirs(OUTPUT_DIR, exist_ok=True)

capoluoghi = {
    'RM': (12.4964, 41.9028),
    'LT': (12.9043, 41.4676),
    'FR': (13.3441, 41.6390),
    'RI': (12.8634, 42.4036),
    'VT': (12.1081, 42.4204)
}

lats = np.linspace(41.0, 42.9, 30)
lons = np.linspace(11.2, 14.2, 30)
Lon, Lat = np.meshgrid(lons, lats)

variables = ['temperature_2m']
HOURS_TO_GENERATE = 24

def fetch_point_data(args):
    i, j, lat, lon = args
    vars_str = ",".join(variables)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly={vars_str}&models=italia_meteo_arpae_icon_2i"
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json().get("hourly", {})
                return i, j, data
            elif response.status_code == 429:
                time.sleep(1 + attempt)
        except:
            pass
    return None

tasks = []
for i in range(lats.shape[0]):
    for j in range(lons.shape[0]):
        tasks.append((i, j, lats[i], lons[j]))

print("Scaricamento dati reali dal modello in corso...")
raw_grid_data = {var: np.full((len(lats), len(lons), HOURS_TO_GENERATE), np.nan) for var in variables}
times_list = []

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
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

for var in variables:
    for h in range(HOURS_TO_GENERATE):
        Data_Grid = raw_grid_data[var][:, :, h]
        
        if np.isnan(Data_Grid).any():
            mean_val = np.nanmean(Data_Grid)
            if np.isnan(mean_val):
                mean_val = 15.0
            Data_Grid = np.nan_to_num(Data_Grid, nan=mean_val)

        zoom_factor = 4
        HighRes_Grid = zoom(Data_Grid, zoom_factor, order=3)
        
        hires_lats = np.linspace(lats.min(), lats.max(), lats.shape[0] * zoom_factor)
        hires_lons = np.linspace(lons.min(), lons.max(), lons.shape[0] * zoom_factor)
        HiRes_Lon, HiRes_Lat = np.meshgrid(hires_lons, hires_lats)

        valid_time_str = ""
        run_str = "Run ICON-2I"
        if times_list and h < len(times_list):
            try:
                dt_obj = datetime.fromisoformat(times_list[h])
                valid_time_str = dt_obj.strftime("%A %d %B %Y - Ore: %H:%M UTC")
            except:
                pass
        
        if not valid_time_str:
            base_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            target_utc = base_utc + timedelta(hours=h)
            valid_time_str = target_utc.strftime("%A %d %B %Y - Ore: %H:%M UTC")

        # Scala termica estesa stile Meteocloud (da -24°C a +44°C)
        levels = np.arange(-24, 46, 2)
        cmap = plt.get_cmap('Spectral_r')

        # Stile Dark Theme identico al riferimento
        fig = plt.figure(figsize=(9, 9), facecolor='#121212')
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_facecolor('#1a1a1a')
        ax.set_extent([11.3, 14.1, 41.0, 42.8], crs=ccrs.PlateCarree())

        mesh = ax.contourf(HiRes_Lon, HiRes_Lat, HighRes_Grid, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, extend='both', alpha=0.92, zorder=1)

        ax.add_feature(cfeature.OCEAN, facecolor='#16222a', zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#555555', zorder=3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='#444444', zorder=3)

        try:
            shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
            reader = shpreader.Reader(shapefile)
            provinces = [rec.geometry for rec in reader.records() if rec.attributes.get('admin') == 'Italy']
            ax.add_geometries(provinces, ccrs.PlateCarree(), edgecolor='#444444', facecolor='none', linewidth=0.4, zorder=3)
        except:
            pass

        for sigla, (lon_c, lat_c) in capoluoghi.items():
            ax.plot(lon_c, lat_c, marker='o', color='white', markersize=3.5, transform=ccrs.PlateCarree(), zorder=5)
            ax.text(lon_c + 0.03, lat_c, sigla, transform=ccrs.PlateCarree(), fontsize=7, fontweight='bold', color='black', 
                    bbox=dict(boxstyle='square,pad=0.15', facecolor='white', alpha=0.9, edgecolor='none'), zorder=6)

        # Header box in alto a sinistra stile Meteocloud
        header_text = f"{valid_time_str} (+{h}h)\nTemperatura a 2 m a risoluzione nativa (°C)"
        ax.text(0.03, 0.95, header_text, transform=ax.transAxes, fontsize=9, fontweight='bold', color='white',
                va='top', ha='left', bbox=dict(boxstyle='square,pad=0.4', facecolor='#8B0000', alpha=0.85, edgecolor='none'), zorder=10)

        # Colorbar orizzontale in basso
        cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.04, fraction=0.04, aspect=35, extend='both')
        cbar.ax.tick_params(labelsize=8, colors='white')
        cbar.ax.xaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')

        # Logo / Footer in basso a sinistra
        ax.text(0.03, 0.03, 'METEO NEROLA', transform=ax.transAxes, fontsize=9, fontweight='bold', color='white', 
                ha='left', va='bottom', zorder=10, bbox=dict(boxstyle='square,pad=0.3', facecolor='#121212', alpha=0.7, edgecolor='none'))

        filename = f"temperature_2m_h{h:02d}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

print("Mappe Dark Theme ad alta definizione generate con successo in mappe_native/")
