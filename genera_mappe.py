import os
import requests
from ftplib import FTP

print("Inizio generazione mappa meteo da Open-Meteo...")

# 1. Coordinate di esempio centralizzate sull'Italia
url = "https://api.open-meteo.com/v1/forecast?latitude=42.15&longitude=12.75&current=temperature_2m,cloud_cover,rain"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    temp = data["current"]["temperature_2m"]
    clouds = data["current"]["cloud_cover"]
    
    print(f"Dati ricevuti con successo! Temp: {temp}°C, Nuvole: {clouds}%")
    
    # Creiamo una cartella locale temporanea
    os.makedirs("mappe_output", exist_ok=True)
    
    # Usiamo matplotlib per generare l'immagine della mappa
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

    # 2. Caricamento automatico via FTP nativo in Python
    # Leggiamo i dati direttamente dalle "Secrets" di GitHub
    ftp_server = os.environ.get("FTP_SERVER")
    ftp_user = os.environ.get("FTP_USERNAME")
    ftp_pass = os.environ.get("FTP_PASSWORD")
    
    if ftp_server and ftp_user and ftp_pass:
        print("Connessione al server FTP di Aruba...")
        try:
            ftp = FTP(ftp_server)
            ftp.login(ftp_user, ftp_pass)
            print("Login FTP effettuato con successo!")
            
            # Entriamo nella cartella di destinazione (creandola se non esiste)
            target_dir = "/www.meteonerola.it/modelli"
            
            # Navighiamo o creiamo i rami delle cartelle se necessario
            # Semplifichiamo spostandoci nella cartella principale del sito
            try:
                ftp.cwd(target_dir)
            except:
                print(f"La cartella {target_dir} non esiste o non è raggiungibile.")
            
            # Carichiamo il file
            with open(output_path, "rb") as file:
                ftp.storbinary("STOR mappa_test.png", file)
            
            print("File caricato con successo via FTP!")
            ftp.quit()
            
        except Exception as e:
            print(f"Errore durante l'upload FTP: {e}")
    else:
        print("Credenziali FTP non trovate nelle variabili d'ambiente.")

else:
    print("Errore nel recupero dati da Open-Meteo")
