import math
from google.transit import gtfs_realtime_pb2
import requests
import folium

# CONFIGURAZIONE: Riportato al centro di Roma con un raggio ampio di 35 km
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
  # Mappa pulita con OpenStreetMap
  mappa = folium.Map(
      location=[CENTRO_LAT, CENTRO_LON], zoom_start=12, tiles="OpenStreetMap"
  )

  # Cerchio blu del raggio
  folium.Circle(
      location=[CENTRO_LAT, CENTRO_LON],
      radius=RAGGIO_KM * 1000,
      color="blue",
      fill=True,
      fill_opacity=0.05,
      popup=f"Raggio di ricerca: {RAGGIO_KM} km",
  ).add_to(mappa)

  # Punto centrale Roma
  folium.Marker(
      [CENTRO_LAT, CENTRO_LON],
      popup="Centro di riferimento",
      icon=folium.Icon(color="red", icon="home", prefix="fa"),
  ).add_to(mappa)

  bus_trovati = 0
  try:
    response = requests.get(URL_ATAC, timeout=15)
    if response.status_code == 200:
      feed = gtfs_realtime_pb2.FeedMessage()
      feed.ParseFromString(response.content)

      for entity in feed.entity:
        if entity.HasField("vehicle"):
          veh = entity.vehicle
          lat = veh.position.latitude
          lon = veh.position.longitude

          route_id = veh.trip.route_id if veh.HasField("trip") else "N/D"
          headsign = (
              veh.trip.trip_headsign if veh.HasField("trip") else "Non disponibile"
          )
          veh_id = veh.vehicle.id if veh.HasField("vehicle") else "N/D"

          dist = calcola_distanza(CENTRO_LAT, CENTRO_LON, lat, lon)

          if dist <= RAGGIO_KM:
            bus_trovati += 1

            html_icon = f"""
                        <div style="
                            background: #00ffcc; 
                            color: #0b0f19; 
                            font-weight: bold; 
                            font-size: 11px; 
                            padding: 2px 5px; 
                            border-radius: 4px; 
                            border: 2px solid #000000;
                            box-shadow: 0 0 5px rgba(0,0,0,0.3);
                            text-align: center;
                            white-space: nowrap;">
                            🚌 {route_id}
                        </div>
                        """
            icona_custom = folium.DivIcon(
                html=html_icon, class_name="bus-tag", icon_size=(40, 20)
            )

            popup_text = f"""
                        <div style="font-family: Arial; font-size: 12px;">
                            <b>Linea:</b> {route_id}<br>
                            <b>Direzione:</b> {headsign}<br>
                            <b>ID Mezzo:</b> {veh_id}<br>
                            <b>Distanza:</b> {dist:.2f} km
                        </div>
                        """

            folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_text, max_width=250),
                icon=icona_custom,
            ).add_to(mappa)

      print(f"Trovati e mappati {bus_trovati} autobus.")
  except Exception as e:
    print(f"Errore: {e}")

  mappa.save("mappa_bus.html")
  print("Mappa salvata.")


if __name__ == "__main__":
  main()
