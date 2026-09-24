import math
from google.transit import gtfs_realtime_pb2
import requests
import folium

CENTRO_LAT = 41.9028
CENTRO_LON = 12.4964
RAGGIO_KM = 5.0

# URL Ufficiale estratto direttamente dal portale Open Data ATAC
URL_ATAC = "https://romamobilita.it/sites/default/files/rome_rtgtfs_vehicle_positions_feed.pb"


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

  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      )
  }

  bus_trovati = 0
  try:
    print("Scaricamento dal link ufficiale ATAC...")
    response = requests.get(URL_ATAC, headers=headers, timeout=15)
    print(f"HTTP Status: {response.status_code}, Bytes: {len(response.content)}")

    if response.status_code == 200 and len(response.content) > 100:
      feed = gtfs_realtime_pb2.FeedMessage()
      feed.ParseFromString(response.content)

      for entity in feed.entity:
        if entity.HasField("vehicle"):
          veh = entity.vehicle
          lat = veh.position.latitude
          lon = veh.position.longitude

          if 40.0 < lat < 43.0 and 11.0 < lon < 14.0:
            route_id = veh.trip.route_id if veh.HasField("trip") else "N/D"
            veh_id = veh.vehicle.id if veh.HasField("vehicle") else "N/D"
            distanza = calcola_distanza(CENTRO_LAT, CENTRO_LON, lat, lon)

            if distanza <= RAGGIO_KM:
              bus_trovati += 1
              folium.Marker(
                  [lat, lon],
                  popup=f"Linea: {route_id} (ID: {veh_id})",
                  icon=folium.Icon(color="green", icon="bus", prefix="fa"),
              ).add_to(mappa)

      print(f"Trovati {bus_trovati} autobus reali nel raggio.")
  except Exception as e:
    print(f"Errore: {e}")

  mappa.save("mappa_bus.html")
  print("Mappa salvata con successo.")


if __name__ == "__main__":
  main()
