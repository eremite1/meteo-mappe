"""Sonda temporanea: legge l'archivio Open-Meteo e salva piccoli campioni per lo sviluppo."""
import json, time, traceback, os
import numpy as np, requests, fsspec
import omfiles

B = "https://openmeteo.s3.amazonaws.com/data_spatial/"
OUT = "campioni"; os.makedirs(OUT, exist_ok=True)
rapporto = {}

def coords_regolari(bbox, shape):
    lat0, lon0, lat1, lon1 = bbox
    return np.linspace(lat0, lat1, shape[0]), np.linspace(lon0, lon1, shape[1])

def o1280():
    n = 1280
    x, _ = np.polynomial.legendre.leggauss(2 * n)
    lat_rows = np.degrees(np.arcsin(x))[::-1]           # da nord a sud
    lats, lons = [], []
    for i in range(2 * n):
        k = i + 1 if i < n else 2 * n - i
        npt = 4 * k + 16
        lats.append(np.full(npt, lat_rows[i])); lons.append(np.arange(npt) * 360.0 / npt)
    lat = np.concatenate(lats); lon = np.concatenate(lons); lon = np.where(lon >= 180, lon - 360, lon)
    return lat, lon

VOGLIO = None  # tutte

DATASET = {
  "italia_meteo_arpae_icon_2i": dict(crop=[5, 20, 35, 48]),
  "meteofrance_arome_france_hd": dict(crop=[11, 14.5, 40.5, 43.5]),
  "ecmwf_ifs025": dict(crop=[-25, 45, 30, 72]),
  "ecmwf_ifs": dict(crop=[5, 20, 35, 48]),
  "ukmo_global_deterministic_10km": dict(crop=[-25, 45, 30, 72]),
  "dwd_icon_eu": dict(crop=[-25, 45, 30, 72], coarse=2),
}

def leggi(child):
    sh = child.shape
    try:
        return np.asarray(child[...])
    except Exception:
        return np.asarray(child.read_array(tuple(slice(0, s) for s in sh)))

fs = fsspec.filesystem("s3", anon=True)
for m, cfg in DATASET.items():
    info = {}
    try:
        j = requests.get(B + m + "/latest.json", timeout=30).json()
        info["latest"] = {k: v for k, v in j.items() if k != "crs_wkt"}
        info["crs"] = j["crs_wkt"][-200:]
        rt = j["reference_time"]; vt = j["valid_times"]
        d = rt[:10].replace("-", "/"); hh = rt[11:13] + rt[14:16]
        scelti = [vt[3], vt[4]] if len(vt) > 5 else vt[:2]
        if m in ("ecmwf_ifs025",):
            scelti = [vt[2], vt[3]]
        info["file"] = []
        for t in scelti:
            tt = t.replace(":", "").replace("Z", "")
            path = f"openmeteo/data_spatial/{m}/{d}/{hh}Z/{tt}.om"
            t0 = time.time()
            rd = omfiles.OmFileReader.from_fsspec(fs, path)
            nomi = [rd.get_child_by_index(i).name for i in range(rd.num_children)]
            info["variabili"] = nomi
            # scalari
            sc = {}
            for nm in ("forecast_reference_time", "valid_time", "coordinates", "created_at"):
                try:
                    c = rd.get_child_by_name(nm)
                    sc[nm] = str(c.read_scalar())[:300]
                except Exception as e:
                    sc[nm] = f"ERR {e}"
            info["scalari"] = sc
            arr = {}
            t1 = time.time()
            for nm in nomi:
                c = rd.get_child_by_name(nm)
                if not c.is_array:
                    continue
                if m == "ukmo_global_deterministic_10km" and not any(k in nm for k in (
                    "2m", "10m", "msl", "precip", "rain", "snow", "cloud", "cape", "inhibition", "visib",
                    "850hPa", "700hPa", "500hPa", "300hPa", "250hPa", "925hPa", "freezing", "water_vapour")):
                    continue
                a = leggi(c).astype(np.float32)
                arr[nm] = a
            dt_read = time.time() - t1
            # ritaglio
            nome0 = next(iter(arr))
            shape = arr[nome0].shape
            lo0, lo1, la0, la1 = cfg["crop"]
            if m == "ecmwf_ifs":
                lat, lon = o1280()
                sel = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
                salva = {k: v.reshape(-1)[sel].astype(np.float16) for k, v in arr.items()}
                salva["_lat"] = lat[sel].astype(np.float32); salva["_lon"] = lon[sel].astype(np.float32)
            else:
                bb = [float(x) for x in j["crs_wkt"].split("BBOX[")[1].split("]")[0].split(",")]
                lat, lon = coords_regolari(bb, shape)
                iy = np.where((lat >= la0) & (lat <= la1))[0]; ix = np.where((lon >= lo0) & (lon <= lo1))[0]
                c = cfg.get("coarse", 1)
                iy = iy[::c]; ix = ix[::c]
                salva = {k: v[np.ix_(iy, ix)].astype(np.float16) for k, v in arr.items()}
                salva["_lat"] = lat[iy].astype(np.float32); salva["_lon"] = lon[ix].astype(np.float32)
                # verifica orientamento: temperatura a Nord (Alpi) vs Sud (Sahara)
                if "temperature_2m" in arr:
                    T = arr["temperature_2m"]
                    info["T_riga0_media"] = float(np.nanmean(T[0])); info["T_ultima_media"] = float(np.nanmean(T[-1]))
            np.savez_compressed(f"{OUT}/{m}__{tt}.npz", **salva)
            info["file"].append(dict(t=t, tempo_apertura=round(t1 - t0, 1), tempo_lettura=round(dt_read, 1), nvar=len(arr), shape=list(shape)))
    except Exception as e:
        info["errore"] = f"{e} {traceback.format_exc()[-800:]}"
    rapporto[m] = info
    print(m, json.dumps(info)[:500], flush=True)

json.dump(rapporto, open(f"{OUT}/rapporto.json", "w"), indent=1)
