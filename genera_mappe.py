import os
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== INIZIO GENERAZIONE MAPPA LAZIO ===")

citta = {
    "Nerola": {"lat": 42.15, "lon": 12.75},
    "Roma": {"lat": 41.89, "lon": 12.51},
    "Viterbo": {"lat": 42.42, "lon": 12.10},
    "Rieti": {"lat": 42.40, "lon": 12.86},
    "Latina": {"lat": 41.46, "lon": 12.90},
    "Frosinone": {"lat": 41.64, "lon": 13.35}
}

dati_meteo = {}

for nome, coords in citta.items():
    url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&models=icon_seamless"
    response = requests.get(url)
    print(f"Richiesta {nome}: status {response.status_code}")
    if response.status_code == 200:
        current = response.json().get("current", {})
        dati_meteo[nome] = {
            "temp": current.get("temperature_2m", 0),
            "umid": current.get("relative_humidity_2m", 0),
            "pioggia": current.get("precipitation", 0),
            "vento": current.get("wind_speed_10m", 0)
        }

os.makedirs("mappe_output", exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')

plt.title("Meteo Lazio - Modello ICON (Open-Meteo)", fontsize=14, fontweight='bold', pad=20, color='#1f4e79')

ax.text(0.08, 0.85, "Località", fontsize=11, fontweight='bold', transform=ax.transAxes, color='#333333')
ax.text(0.35, 0.85, "Temperatura", fontsize=11, fontweight='bold', transform=ax.transAxes, color='#333333')
ax.text(0.60, 0.85, "Umidità", fontsize=11, fontweight='bold', transform=ax.transAxes, color='#333333')
ax.text(0.80, 0.85, "Vento", fontsize=11, fontweight='bold', transform=ax.transAxes, color='#333333')

y_pos = 0.73
for nome, vals in dati_meteo.items():
    ax.text(0.08, y_pos, f"• {nome}", fontsize=11, transform=ax.transAxes, fontweight='semibold')
    ax.text(0.35, y_pos, f"{vals['temp']} °C", fontsize=11, transform=ax.transAxes)
    ax.text(0.60, y_pos, f"{vals['umid']}%", fontsize=11, transform=ax.transAxes)
    ax.text(0.80, y_pos, f"{vals['vento']} km/h", fontsize=11, transform=ax.transAxes)
    y_pos -= 0.10

output_path = "mappe_output/mappa_lazio.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Mappa generata con successo in {output_path}!")
