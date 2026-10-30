import time, os, sys, json, requests, numpy as np
sys.path.insert(0, ".")
import omfiles, fsspec
from motore import fonti
out = []
def W(m): print(m, flush=True); out.append(m)
VAR = None
def prova(ds, rq, nvar=45):
    j = fonti.ultimo(ds); t = j["valid_times"][6]
    p = fonti.percorso(ds, j["reference_time"], t)
    url = "https://openmeteo.s3.amazonaws.com/" + p[len("openmeteo/"):]
    # A: download completo
    t0 = time.time()
    with requests.get(url, stream=True, timeout=600) as r:
        with open("/tmp/x.om", "wb") as f:
            for ch in r.iter_content(8 << 20): f.write(ch)
    td = time.time() - t0; sz = os.path.getsize("/tmp/x.om") / 1e6
    rd = omfiles.OmFileReader("/tmp/x.om")
    nomi = [rd.get_child_by_index(i).name for i in range(rd.num_children) if rd.get_child_by_index(i).is_array]
    sel = [n for n in nomi if any(k in n for k in ("2m", "10m", "msl", "precip", "cloud", "cape", "850hPa", "700hPa", "500hPa", "300hPa"))][:nvar]
    class Fake: pass
    t1 = time.time()
    fo = fonti.FileOM.__new__(fonti.FileOM); fo.rd = rd; fo.info = j; fo.dataset = ds
    for n in sel: fo.leggi(n, rq)
    tl = time.time() - t1
    W(f"{ds}: download {sz:.0f} MB in {td:.1f}s, lettura locale {len(sel)} var in {tl:.1f}s")
    # B: s3fs con blocchi grandi
    for bs in (16 << 20, 64 << 20):
        fs = fsspec.filesystem("s3", anon=True, default_block_size=bs, default_cache_type="blockcache", skip_instance_cache=True)
        t2 = time.time()
        rd2 = omfiles.OmFileReader.from_fsspec(fs, p)
        fo.rd = rd2
        for n in sel[:10]: fo.leggi(n, rq)
        W(f"{ds}: s3fs blocchi {bs>>20}MB, 10 var in {time.time()-t2:.1f}s")
    os.remove("/tmp/x.om")
prova("ukmo_global_deterministic_10km", [-24, 44, 29, 72])
prova("italia_meteo_arpae_icon_2i", [5, 20, 35, 48.4])
prova("ecmwf_ifs", [5, 20, 35, 48.4])
print("::warning::" + " || ".join(out))
