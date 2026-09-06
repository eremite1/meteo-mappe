import os
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== GENERAZIONE MAPPA CONTINUA (GRIGLIA LAZIO) ===")

# Definizione della griglia di punti che coprono il Lazio
lats = np.linspace(41.3, 42.8, 15)  # da Sud a Nord
lons = np.linspace(11.8, 13.9, 15)  # da Ovest a Est

Lon, Lat = np.meshgrid(lons, lats)
Data_Grid = np.zeros_like(Lon)

# Download dei dati puntuali per popolare la griglia
for i in range(lats.shape[0]):
    for j in range(lons.shape[0]):
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lats[i]}&longitude={lons[j]}&current=temperature_2m&models=icon_seamless"
        response = requests.get(url)
        if response.status_code == 200:
            current = response.json().get("current", {})
            Data_Grid[i, j] = current.get("temperature_2m", 0)

os.makedirs("mappe_output", exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 8))

# Generazione delle sfumature continue stile modello meteo
contour = ax.contourf(Lon, Lat, Data_Grid, levels=20, cmap='Spectral_r', extend='both')

plt.title("Meteo Lazio - Temperatura a 2m (Modello ICON)", fontsize=13, fontweight='bold', color='white', pad=15)
fig.patch.set_facecolor('#1e1e1e')
ax.set_facecolor('#1e1e1e')

cbar = fig.colorbar(contour, orientation='horizontal', pad=0.05, shrink=0.8)
cbar.set_label('Temperatura (°C)', color='white')
cbar.ax.tick_params(labelsize=9, colors='white')

output_path = "mappe_output/mappa_continua_lazio.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print(f"Mappa continua generata con successo in {output_path}!")
