"""Sonda temporanea: legge solo i parametri e i riquadri che servono, usando motore/fonti.py."""
import json, time, traceback, os, sys
import numpy as np
sys.path.insert(0, ".")
from motore import fonti

OUT = "campioni"; os.makedirs(OUT, exist_ok=True)
LIV = ["1000hPa", "925hPa", "850hPa", "700hPa", "600hPa", "500hPa", "300hPa", "250hPa"]
VAR = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "showers",
       "snowfall_water_equivalent", "snow_depth", "snowfall_height", "freezing_level_height",
       "wind_speed_10m", "wind_direction_10m", "wind_u_component_10m", "wind_v_component_10m",
       "wind_gusts_10m", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
       "pressure_msl", "visibility", "shortwave_radiation", "cape", "convective_inhibition",
       "lightning_potential", "total_column_integrated_water_vapour", "sea_surface_temperature",
       "boundary_layer_height"]
for l in LIV:
    VAR += [f"{x}_{l}" for x in ("temperature", "geopotential_height", "relative_humidity",
                                 "wind_u_component", "wind_v_component", "wind_speed",
                                 "wind_direction", "vertical_velocity")]
ITALIA = [5.0, 20.0, 35.0, 48.4]
EUROPA = [-24.0, 44.0, 29.0, 72.0]
LAZIO = [10.9, 14.5, 40.6, 43.2]
DS = {"italia_meteo_arpae_icon_2i": ITALIA, "meteofrance_arome_france_hd": LAZIO,
      "ecmwf_ifs": ITALIA, "ecmwf_ifs025": EUROPA, "ukmo_global_deterministic_10km": ITALIA,
      "dwd_icon_eu": ITALIA}
rapporto = {}
for ds, rq in DS.items():
    info = {}
    try:
        j = fonti.ultimo(ds)
        info["corsa"] = j["reference_time"]; info["n_tempi"] = len(j["valid_times"])
        info["tempi"] = j["valid_times"][:12] + ["..."] + j["valid_times"][-3:]
        info["bbox"] = j["bbox"]; info["ridotta"] = j["ridotta"]
        info["variabili_latest"] = j.get("variables")
        vt = j["valid_times"]
        sel = vt[3:6] if ds != "ecmwf_ifs025" else vt[1:3]
        info["letture"] = []
        for t in sel:
            t0 = time.time()
            f = fonti.FileOM(ds, j, t)
            t_ap = time.time() - t0
            salva = {}
            t1 = time.time()
            for v in VAR:
                if f.ha(v):
                    la, lo, val = f.leggi(v, rq)
                    salva[v] = val.astype(np.float16)
            salva["_lat"] = la.astype(np.float32); salva["_lon"] = lo.astype(np.float32)
            np.savez_compressed(f"{OUT}/{ds}__{t.replace(':', '')}.npz", **salva)
            info["letture"].append(dict(t=t, apertura=round(t_ap, 1), lettura=round(time.time() - t1, 1),
                                        nvar=len(salva) - 2, forma=list(val.shape)))
            # statistiche per capire unità
            info["statistiche"] = {k: [float(np.nanmin(v)), float(np.nanmean(v)), float(np.nanmax(v))]
                                   for k, v in salva.items() if not k.startswith("_")}
            info["mancanti"] = [v for v in VAR if v not in salva]
            if not j["ridotta"]:
                T = salva.get("temperature_2m")
                if T is not None:
                    info["T_sud_nord"] = [float(np.nanmean(T[0].astype(float))), float(np.nanmean(T[-1].astype(float)))]
    except Exception as e:
        info["errore"] = f"{e} {traceback.format_exc()[-1200:]}"
    rapporto[ds] = info
    print(ds, json.dumps(info)[:300], flush=True)
json.dump(rapporto, open(f"{OUT}/rapporto.json", "w"), indent=1, default=str)
