import os
import xarray as xr
import numpy as np
from datetime import datetime, timezone, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

print("=== GENERAZIONE MAPPE NATIVE ICON-2I ===")

OUTPUT_DIR = "mappe_native"
os.makedirs(OUTPUT_DIR, exist_ok=True)

capoluoghi = {
    'RM': (12.4964, 41.9028),
    'LT': (12.9043, 41.4676),
    'FR': (13.3441, 41.6390),
    'RI': (12.8634, 42.4036),
    'VT': (12.1081, 42.4204)
}

HOURS_TO_GENERATE = 24

for h in range(HOURS_TO_GENERATE):
    base_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    target_utc = base_utc + timedelta(hours=h)
    valid_time_str = target_utc.strftime("%d %B %Y - Ore: %H:%M UTC")

    fig = plt.figure(figsize=(10, 8), facecolor='#ffffff')
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor('#f4f4f4')
    ax.set_extent([11.3, 14.1, 41.0, 42.8], crs=ccrs.PlateCarree())

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

    fig.text(0.50, 0.93, "Modello ICON-2I - Lazio | Temperatura (°C) [Nativo]", fontsize=12, fontweight='bold', color='black', ha='center')
    fig.text(0.50, 0.89, f"Valido il: {valid_time_str}", fontsize=10, fontweight='bold', color='#333333', ha='center')

    ax.text(0.97, 0.03, 'www.meteonerola.it', transform=ax.transAxes, fontsize=9, fontweight='bold', color='black', 
            ha='right', va='bottom', zorder=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#aaaaaa'))

    filename = f"temperature_2m_h{h:02d}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

print("Mappe native generate con successo in mappe_native/")
