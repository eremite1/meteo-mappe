import math
from google.transit import gtfs_realtime_pb2
import requests
import folium

# CONFIGURAZIONE (Modifica coordinate e raggio in km)
CENTRO_LAT = 41.9028  # Latitudine del centro (es. Roma)
CENTRO_LON = 12.4964  # Longitudine del centro
RAGGIO_KM = 5.0  # Raggio di ricerca bus in km

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
      fill_opacity=0.08,
      popup=f"Raggio di ricerca: {RAGGIO_KM} km",
  ).add_to(mappa)

  folium.Marker(
      [CENTRO_LAT, CENTRO_LON],
      popup="Punto di riferimento",
      icon=folium.Icon(color="red", icon="home", prefix="fa"),
  ).add_to(mappa)

  try:
    print("Scaricamento dati in tempo reale da ATAC...")
    response = requests.get(URL_ATAC, timeout=15)

    if response.status_code == 200:
      feed = gtfs_realtime_pb2.FeedMessage()
      feed.ParseFromString(response.content)

      bus_trovati = 0
      for entity in feed.entity:
        if entity.HasField("vehicle"):
          veh = entity.vehicle
          lat = veh.position.latitude
          lon = veh.position.longitude

          # Verifica che le coordinate siano valide (intorno a Roma)
          if 41.0 < lat < 42.5 and 11.5 < lon < 13.5:
            route_id = (
                veh.trip.route_id if veh.HasField("trip") else "Sconosciuta"
            )
            veh_id = (
                veh.vehicle.id if veh.HasField("vehicle") else "ID Sconosciuto"
            )

            distanza = calcola_distanza(CENTRO_LAT, CENTRO_LON, lat, lon)

            if distanza <= RAGGIO_KM:
              bus_trovati += 1
              popup_text = f"<b>Linea:</b> {route_id}<br><b>Mezzo ID:</b> {veh_id}<br><b>Distanza:</b> {distanza:.2f} km"

              folium.Marker(
                  [lat, lon],
                  popup=popup_text,
                  icon=folium.Icon(color="green", icon="bus", prefix="fa"),
              ).add_to(mappa)

      print(
          f"Trovati e mappati {bus_trovati} autobus nel raggio di"
          f" {RAGGIO_KM} km."
      )
    else:
      print(f"Errore download ATAC: {response.status_code}")
  except Exception as e:
    print(f"Errore: {e}")

  mappa.save("mappa_bus.html")
  print("Mappa generata con successo!")


if __name__ == "__main__":
  main()
