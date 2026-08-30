import os
import requests
import matplotlib.pyplot as plt

print("Inizio generazione mappe meteo da Open-Meteo...")

# Coordinate di esempio centrate sull'Italia / zona centrale
url = "https://api.open-meteo.com/v1/forecast?latitude=42.15&longitude=12.75&current=temperature_2m,cloud_cover,rain"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    temp = data["current"]["temperature_2m"]
    clouds = data["current"]["cloud_cover"]
    
    print(f"Dati ricevuti con successo! Temp: {temp}°C, Nuvole: {clouds}%")
    
    # Creiamo una cartella locale per le immagini se non esiste
    os.makedirs("mappe_output", exist_ok=True)
    
    # Generiamo un'immagine di test semplice con matplotlib
    plt.figure(figsize=(6, 6))
    plt.title(f"Test Mappa Meteo - Temp: {temp} C")
    plt.text(0.5, 0.5, f"Temperatura: {temp} C\nNuvolosità: {clouds}%", 
             fontsize=14, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')
    
    # Salviamo l'immagine
    output_path = "mappe_output/mappa_test.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Immagine salvata correttamente in {output_path}")
else:
    print("Errore nel recupero dati da Open-Meteo")
