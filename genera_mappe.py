import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ftplib import FTP

print("Inizio elaborazione mappa ICON focalizzata sul Lazio...")

# 1. Richiesta dati a Open-Meteo (usando il modello ICON se disponibile o il globale ottimizzato)
# Coordinate centrate sul Lazio con bounding box per la mappa
url = "https://api.open-meteo.com/v1/forecast?latitude=42.0&longitude=12.5&current=temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m&models=icon_seamless"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    current = data.get("current", {})
    temp = current.get("temperature_2m", 0)
    precip = current.get("precipitation", 0)
    wind = current.get("wind_speed_10m", 0)
    
    print(f"Dati meteo scaricati con successo. Temp media stimata: {temp}°C")
    
    os.makedirs("mappe_output", exist_ok=True)
    
    # 2. Creazione della mappa geografica focalizzata sul Lazio con Cartopy
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Impostiamo i confini geografici mirati sul Lazio (Lon min, Lon max, Lat min, Lat max)
    ax.set_extent([11.5, 14.2, 41.0, 43.1], crs=ccrs.PlateCarree())
    
    # Aggiungiamo elementi geografici di riferimento
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
    ax.add_feature(cfeature.OCEAN, facecolor='#d0e0e3')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
    ax.add_feature(cfeature.STATES, linestyle='--', linewidth=0.5, edgecolor='gray')
    
    # Griglia di coordinate
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Titolo e riquadro informativo con i dati meteo
    plt.title("Meteo Lazio - Modello ICON (Open-Meteo)", fontsize=12, fontweight='bold', pad=15)
    
    info_text = f"Area: Lazio\nTemp. Riferimento: {temp}°C\nPrecipitazioni: {precip} mm\nVento: {wind} km/h"
    plt.figtext(0.15, 0.15, info_text, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
    
    output_path = "mappe_output/mappa_lazio.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Mappa geografica del Lazio salvata in {output_path}")

    # 3. Caricamento automatico su Aruba via FTP nella cartella modelli
    ftp_server = os.environ.get("FTP_SERVER")
    ftp_user = os.environ.get("FTP_USERNAME")
    ftp_pass = os.environ.get("FTP_PASSWORD")
    
    if ftp_server and ftp_user and ftp_pass:
        print("Connessione al server FTP di Aruba...")
        try:
            ftp = FTP(ftp_server)
            ftp.login(ftp_user, ftp_pass)
            print("Login FTP effettuato con successo!")
            
            with open(output_path, "rb") as file:
                ftp.storbinary("STOR /www.meteonerola.it/modelli/mappa_lazio.png", file)
            
            print("File caricato con successo in /www.meteonerola.it/modelli/mappa_lazio.png!")
            ftp.quit()
            
        except Exception as e:
            print(f"Errore durante l'upload FTP: {e}")
    else:
        print("Credenziali FTP non trovate nelle variabili d'ambiente.")

else:
    print(f"Errore nel recupero dati da Open-Meteo: {response.status_code}")
