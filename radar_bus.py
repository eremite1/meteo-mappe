import math
from google.transit import gtfs_realtime_pb2
import requests
import folium

CENTRO_LAT = 41.9028
CENTRO_LON = 12.4964
RAGGIO_KM = 35.0

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
  mappa = folium.Map(
      location=[CENTRO_LAT, CENTRO_LON], zoom_start=11, tiles="OpenStreetMap"
  )

  # Cerchio blu
  folium.Circle(
      location=[CENTRO_LAT, CENTRO_LON],
      radius=RAGGIO_KM * 1000,
      color="#3388ff",
      weight=2,
      fill=True,
      fill_color="#3388ff",
      fill_opacity=0.08,
      popup=f"Raggio: {RAGGIO_KM} km",
  ).add_to(mappa)

  # Centro Roma
  folium.Marker(
      [CENTRO_LAT, CENTRO_LON],
      popup="Centro Roma",
      icon=folium.Icon(color="red", icon="home", prefix="fa"),
  ).add_to(mappa)

  try:
    response = requests.get(URL_ATAC, timeout=15)
    print(f"Dimensione dati scaricati: {len(response.content)} bytes")

    if response.status_code == 200:
      feed = gtfs_realtime_pb2.FeedMessage()
      feed.ParseFromString(response.content)

      totale_veicoli = 0
      mappati = 0

      for entity in feed.entity:
        if entity.HasField("vehicle"):
          totale_veicoli += 1
          veh = entity.vehicle
          lat = veh.position.latitude
          lon = veh.position.longitude

          # Estrazione sicura degli ID
          route_id = (
              veh.trip.route_id
              if (veh.HasField("trip") and veh.trip.HasField("route_id"))
              else "Generico"
          )
          veh_id = (
              veh.vehicle.id
              if (veh.HasField("vehicle") and veh.vehicle.HasField("id"))
              else "N/D"
          )

          dist = calcola_distanza(CENTRO_LAT, CENTRO_LON, lat, lon)

          if dist <= RAGGIO_KM:
            mappati += 1
            # Usiamo un marker classico verde con icona bus per evitare problemi di visualizzazione
            folium.Marker(
                [lat, lon],
                popup=f"Linea: {route_id} - Mezzo ID: {veh_id} - Distanza: {dist:.2f} km",
                icon=folium.Icon(color="green", icon="bus", prefix="fa"),
            ).add_to(mappa)

      print(
          f"Trovati {totale_veicoli} veicoli totali nel feed, {mappati} nel"
          " raggio."
      )
  except Exception as e:
    print(f"Errore: {e}")

  mappa.save("mappa_bus.html")
  print("Mappa salvata.")


if __name__ == "__main__":
  main()
