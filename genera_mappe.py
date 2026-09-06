import os
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=== INIZIO GENERAZIONE ARCHIVIO MAPPE ORARIE LAZIO ===")

citta = {
    "Nerola": {"lat": 42.15, "lon": 12.75},
    "Roma": {"lat": 41.89, "lon": 12.51},
    "Viterbo": {"lat": 42.42, "lon": 12.10},
    "Rieti": {"lat": 42.40, "lon": 12.86},
    "Latina": {"lat": 41.46, "lon": 12.90},
    "Frosinone": {"lat": 41.64, "lon": 13.35}
}

# Parametri da generare
parametri = {
    "temp": {"titolo": "Temperatura (°C", "unita": "°C"},
    "umid": {"titolo": "Umidità Relativa", "unita": "%"},
    "pioggia": {"titolo": "Precipitazioni", "unita": "mm"},
    "nuvole": {"titolo": "Copertura Nuvolosa", "unita": "%"}
}

# Scarichiamo i dati orari per ogni città (orizzonte 24 ore, step ogni 3 ore)
step_ore = [0, 3, 6, 9, 12, 15, 18, 21, 24]
dati_orari = {}

# Inizializziamo la struttura dati per le ore
for ora in step_ore:
    dati_orari[ora] = {}
    for nome, coords in citta.items():
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&hourly=temperature_2m,relative_humidity_2m,precipitation,cloud_cover&models=icon_seamless"
        response = requests.get(url)
        if response.status_code == 200:
            hourly = response.json().get("hourly", {})
            # Prendiamo i valori corrispondenti all'ora iesima
            dati_orari[ora][nome] = {
                "temp": hourly.get("temperature_2m", [0]*25)[ora],
                "umid": hourly.get("relative_humidity_2m", [0]*25)[ora],
                "pioggia": hourly.get("precipitation", [0]*25)[ora],
                "nuvole": hourly.get("cloud_cover", [0]*25)[ora]
            }

os.makedirs("mappe_output", exist_ok=True)

# Generazione delle immagini per ogni parametro e per ogni ora
for param_key, param_info in parametri.items():
    for ora in step_ore:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')

        # Titolo dinamico con parametro e ora
        titolo_grafico = f"Meteo Lazio - {param_info['titolo']} (+{ora}h)"
        plt.title(titolo_grafico, fontsize=13, fontweight='bold', pad=20, color='#1f4e79')

        # Intestazioni tabella
        ax.text(0.10, 0.85, "Località", fontsize=11, fontweight='bold', transform=ax.transAxes, color='#333333')
        ax.text(0.55, 0.85, f"Valore ({param_info['unita']})", fontsize=11, fontweight='bold', transform=ax.transAxes, color='#333333')

        y_pos = 0.73
        for nome in citta.keys():
            valore = dati_orari[ora][nome][param_key]
            ax.text(0.10, y_pos, f"• {nome}", fontsize=11, transform=ax.transAxes, fontweight='semibold')
            ax.text(0.55, y_pos, f"{valore} {param_info['unita']}", fontsize=11, transform=ax.transAxes)
            y_pos -= 0.10

        # Salvataggio con nome strutturato (es. temp_03h.png, pioggia_12h.png)
        output_path = f"mappe_output/{param_key}_{ora:02d}h.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

print("Tutte le mappe multi-parametro e multi-orario sono state generate con successo!")
