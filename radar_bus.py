import math
from google.transit import gtfs_realtime_pb2
import requests
import folium

CENTRO_LAT = 41.9028
CENTRO_LON = 12.4964
RAGGIO_KM = 5.0

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
  ).add_to(mappa)

  folium.Marker(
      [CENTRO_LAT, CENTRO_LON],
      popup="Centro",
      icon=folium.Icon(color="red", icon="home", prefix="fa"),
  ).add_to(mappa)

  print("Scaricamento dati in tempo reale da ATAC...")
  response = requests.get(URL_ATAC, timeout=15)
  print(f"Codice HTTP risposta: {response.status_code}")
  print(f"Dimensione dati scaricati: {len(response.content)} bytes")

  feed = gtfs_realtime_pb2.FeedMessage()
  feed.ParseFromString(response.content)

  print(f"Totale entità nel feed: {len(feed.entity)}")

  bus_trovati = 0
  for entity in feed.entity:
    if entity.HasField("vehicle"):
      veh = entity.vehicle
      lat = veh.position.latitude
      lon = veh.position.longitude

      # Stampiamo le prime coordinate che incontra per diagnostica
      if bus_trovati < 3:
        print(f"Trovato veicolo grezzo: lat={lat}, lon={lon}")

      if 40.0 < lat < 43.0 and 11.0 < lon < 14.0:
        route_id = veh.trip.route_id if veh.HasField("trip") else "N/D"
        veh_id = veh.vehicle.id if veh.HasField("vehicle") else "N/D"

        distanza = calcola_distanza(CENTRO_LAT, CENTRO_LON, lat, lon)

        if distanza <= RAGGIO_KM:
          bus_trovati += 1
          popup_text = f"<b>Linea:</b> {route_id}<br>Mezzo: {veh_id}<br>Dist: {distanza:.2f}km"
          folium.Marker(
              [lat, lon],
              popup=popup_text,
              icon=folium.Icon(color="green", icon="bus", prefix="fa"),
          ).add_to(mappa)

  print(f"Totale autobus inseriti nel raggio: {bus_trovati}")
  mappa.save("mappa_bus.html")
  print("Mappa salvata correttamente.")


if __name__ == "__main__":
  main()
