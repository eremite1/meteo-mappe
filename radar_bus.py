import math
from google.transit import gtfs_realtime_pb2
import requests
import folium

# CONFIGURAZIONE (Puoi modificare coordinate e raggio in km)
CENTRO_LAT = 41.9028  # Latitudine del centro mappa
CENTRO_LON = 12.4964  # Longitudine del centro mappa
RAGGIO_KM = 3.0  # Raggio di ricerca bus

URL_ATAC = (
    "https://romamobilita.it/sites/default/files/rome_gtfs_rt_vehicle_positions.pb"
)


def calcola_distanza(lat1, lon1, lat2, lon2):
  R = 6371.0
  dlat = math.radians(lat2 - lat1)
  dlon = math.radians(lon2 - lon1)
  a = (
      math.sin(dlat / 2) ** 2
      + math.cos(math.radians(lat1))
      * math.cos(math.radians(lat2))
      * math.sin(dlon / 2) ** 2
  )
  c = 2 * math.asin(math.sqrt(a))
  return R * c


def main():
  mappa = folium.Map(location=[CENTRO_LAT, CENTRO_LON], zoom_start=13)

  folium.Circle(
      location=[CENTRO_LAT, CENTRO_LON],
      radius=RAGGIO_KM * 1000,
      color="blue",
      fill=True,
      fill_opacity=0.1,
      popup=f"Raggio {RAGGIO_KM} km",
  ).add_to(mappa)

  folium.Marker(
      [CENTRO_LAT, CENTRO_LON],
      popup="Punto di riferimento",
      icon=folium.Icon(color="red", icon="home"),
  ).add_to(mappa)

  try:
    response = requests.get(URL_ATAC, timeout=15)
    if response.status_code == 200:
      feed = gtfs_realtime_pb2.FeedMessage()
      feed.ParseFromString(response.content)

      count = 0
      for entity in feed.entity:
        if entity.HasField("vehicle"):
          veh = entity.vehicle
          lat = veh.position.latitude
          lon = veh.position.longitude
          route_id = veh.trip.route_id if veh.HasField("trip") else "N/D"
          veh_id = veh.vehicle.id if veh.HasField("vehicle") else "N/D"

          dist = calcola_distanza(CENTRO_LAT, CENTRO_LON, lat, lon)
          if dist <= RAGGIO_KM:
            count += 1
            popup_text = f"<b>Linea:</b> {route_id}<br><b>Mezzo:</b> {veh_id}<br><b>Distanza:</b> {dist:.2f} km"
            folium.Marker(
                [lat, lon],
                popup=popup_text,
                icon=folium.Icon(color="green", icon="bus", prefix="fa"),
            ).add_to(mappa)
      print(f"Trovati {count} autobus nel raggio.")
  except Exception as e:
      print(f"Errore nel recupero dati: {e}")

  # Salva la mappa dei bus in un file dedicato
  mappa.save("mappa_bus.html")
  print("Mappa bus aggiornata con successo!")


if __name__ == "__main__":
  main()
