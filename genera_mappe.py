import os
import requests
from ftplib import FTP

print("Inizio generazione mappa meteo da Open-Meteo...")

url = "https://api.open-meteo.com/v1/forecast?latitude=42.15&longitude=12.75&current=temperature_2m,cloud_cover,rain"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    temp = data["current"]["temperature_2m"]
    clouds = data["current"]["cloud_cover"]
    
    print(f"Dati ricevuti con successo! Temp: {temp}°C, Nuvole: {clouds}%")
    
    os.makedirs("mappe_output", exist_ok=True)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.title(f"Test Mappa Meteo - Temp: {temp} C")
    plt.text(0.5, 0.5, f"Temperatura: {temp}°C\nNuvolosità: {clouds}%",
             fontsize=14, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')
    
    output_path = "mappe_output/mappa_test.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Immagine salvata localmente in {output_path}")

    # Caricamento via FTP nativo con percorso corretto
    ftp_server = os.environ.get("FTP_SERVER")
    ftp_user = os.environ.get("FTP_USERNAME")
    ftp_pass = os.environ.get("FTP_PASSWORD")
    
    if ftp_server and ftp_user and ftp_pass:
        print("Connessione al server FTP di Aruba...")
        try:
            ftp = FTP(ftp_server)
            ftp.login(ftp_user, ftp_pass)
            print("Login FTP effettuato con successo!")
            
            # Navigazione passo-passo nelle cartelle esistenti su Aruba
            ftp.cwd("www.meteonerola.it")
            
            try:
                ftp.cwd("modelli")
            except:
                print("La cartella modelli non esiste, la creo...")
                ftp.mkd("modelli")
                ftp.cwd("modelli")
            
            with open(output_path, "rb") as file:
                ftp.storbinary("STOR mappa_test.png", file)
            
            print("File caricato con successo nella cartella modelli via FTP!")
            ftp.quit()
            
        except Exception as e:
            print(f"Errore durante l'upload FTP: {e}")
    else:
        print("Credenziali FTP non trovate nelle variabili d'ambiente.")

else:
    print("Errore nel recupero dati da Open-Meteo")
