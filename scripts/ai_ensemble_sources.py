"""Open AI + physics global ensembles, cut to a Western US/NE Pacific domain (added 2026-10-10).

Sources, all checked live 2026-10-10:
  aifs_ens  ECMWF AIFS ENS (AI), 50 perturbed members, 6-hourly to 360 h, 0.25 deg. ECMWF open
            data (CC BY 4.0) on AWS (ecmwf-forecasts, eu-central-1), data.ecmwf.int as fallback.
            One GRIB2 file per step holding every member, plus a JSON-lines .index of byte offsets.
  ifs_ens   ECMWF IFS ENS (physics), same layout ("-ef" files, 50 members); the physics baseline.
  aigefs    NOAA AIGEFS (AI; GraphCast-based, GEFS-initialized), mem000-mem030, 6-hourly to 384 h,
            0.25 deg. NOMADS only (no AWS bucket, no grib filter), ~120 hits/min cap. Separate
            "pres" (13 levels) and "sfc" (10 m wind, 2 m T, PRMSL, 6-h APCP) files with .idx.
  gefs      NOAA GEFS (physics), gec00 + gep01-30, 0.5 deg pgrb2a on AWS (noaa-gefs-pds).
  weathernext3  Stub. Google DeepMind access pending; see WeatherNext3 below.

Every source publishes global fields, so this pulls just the messages it needs by HTTP byte range
(the nbm_noaa.py approach), decodes them with gribberish (ecCodes has no Windows wheel), and keeps
  - the domain grid, subsampled to the config resolution (0.5 deg), and
  - the nearest native grid point to each named point (gradient / 850 hPa wind sites).
NOMADS answers multi-range requests (multipart/byteranges), so one request per file there; S3
doesn't, so nearby ranges are merged and fetched one at a time with more threads.

Units out: z500 m, msl hPa, u850/v850 m/s, t850 deg C, tp mm accumulated since init.

Promoted 2026-10-10 from the incoming/2026-10-10_ai_ensembles sandbox; settings in
config/ai_ensemble_settings.json, paths via ai_ensemble_common.py, docs in docs/ai_ensembles/README.md.
"""
import datetime as dt
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

G = 9.80665
UA = {"User-Agent": "FireAnalystToolkit ai-ensembles"}
ACC_RE = re.compile(r"^(\d+)-(\d+) hour acc")


# ----------------------------------------------------------------------------- HTTP

class OverRate(RuntimeError):
    """The server says we're over its request rate (NOMADS "Over Rate" page)."""


class _Limiter:
    def __init__(self, per_min):
        self.per_min, self.stamps, self.lock = per_min, [], threading.Lock()
        self.pause_until = 0.0

    def cooldown(self, seconds):
        """Every thread waits until now + seconds before its next request."""
        with self.lock:
            self.pause_until = max(self.pause_until, time.time() + seconds)

    def wait(self):
        while True:
            with self.lock:
                now = time.time()
                if now < self.pause_until:
                    pause = self.pause_until - now
                    self.stamps = []
                else:
                    pause = None
            if pause is not None:
                time.sleep(pause)
                continue
            with self.lock:
                now = time.time()
                self.stamps = [t for t in self.stamps if now - t < 60]
                if len(self.stamps) < self.per_min:
                    self.stamps.append(now)
                    return
                pause = 60 - (now - self.stamps[0]) + 0.1
            time.sleep(pause)


class Http:
    def __init__(self, per_min, workers, multirange, merge_gap=256_000):
        self.limiter = _Limiter(per_min)
        self.workers, self.multirange, self.merge_gap = workers, multirange, merge_gap
        self.local = threading.local()
        self.nbytes, self.nreq, self.lock = 0, 0, threading.Lock()
        self.failed_files = 0

    def _session(self):
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
        return self.local.s

    def get(self, url, headers=None, tries=8, stream=False):
        for attempt in range(tries):
            self.limiter.wait()
            try:
                r = self._session().get(url, headers=dict(UA, **(headers or {})), timeout=180, stream=stream)
                with self.lock:
                    self.nreq += 1
                if r.status_code in (403, 404):
                    return None
                if r.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                    # Throttling / transient server error (S3 answers "503 Slow Down" when too many
                    # requests hit one prefix, 2026-10-10): back off exponentially, with jitter.
                    r.close()
                    time.sleep(min(90, 3 * 2 ** attempt) * (0.75 + 0.5 * random.random()))
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException:
                if attempt == tries - 1:
                    raise
                time.sleep(min(90, 3 * 2 ** attempt))

    def exists(self, url):
        r = self.get(url, headers={"Range": "bytes=0-0"})
        return r is not None

    def _one_range(self, url, a, b):
        r = self.get(url, headers={"Range": f"bytes={a}-{'' if b is None else b}"}, stream=True)
        if r is None:
            raise RuntimeError(f"{url}: range request 404")
        if r.status_code != 206:
            r.close()
            raise RuntimeError(f"{url}: server ignored the byte range (HTTP {r.status_code})")
        data = r.content
        with self.lock:
            self.nbytes += len(data)
        return data

    def ranges(self, url, spans):
        """Bytes for each inclusive (start, end) span, in input order; end None = to end of file."""
        if not spans:
            return []
        if self.multirange and len(spans) > 1:
            return self._multipart(url, spans)
        out = [None] * len(spans)
        for i, (a, b) in enumerate(spans):
            if b is None:
                out[i] = self._one_range(url, a, None)
        rest = [i for i, (a, b) in enumerate(spans) if b is not None]
        if not rest:
            return out
        order = sorted(rest, key=lambda i: spans[i][0])
        groups, cur = [], [order[0]]
        for i in order[1:]:
            if spans[i][0] - spans[cur[-1]][1] <= self.merge_gap:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        for g in groups:
            a = spans[g[0]][0]
            b = max(spans[i][1] for i in g)
            blob = self._one_range(url, a, b)
            for i in g:
                out[i] = blob[spans[i][0] - a: spans[i][1] - a + 1]
        return out

    def _multipart(self, url, spans):
        hdr = "bytes=" + ",".join(f"{a}-{'' if b is None else b}" for a, b in spans)
        r = self.get(url, headers={"Range": hdr}, stream=True)
        if r is None:
            raise RuntimeError(f"{url}: 404")
        if r.status_code != 206:
            r.close()
            raise RuntimeError(f"{url}: multi-range refused (HTTP {r.status_code})")
        body = r.content
        with self.lock:
            self.nbytes += len(body)
        ctype = r.headers.get("Content-Type", "")
        if "multipart/byteranges" not in ctype:      # a single merged range came back
            a0 = spans[0][0]
            return [body[a - a0: None if b is None else b - a0 + 1] for a, b in spans]
        boundary = re.search(r"boundary=([^;]+)", ctype).group(1).strip('"').encode()
        parts = {}
        for chunk in body.split(b"--" + boundary):
            if b"\r\n\r\n" not in chunk:
                continue
            head, data = chunk.split(b"\r\n\r\n", 1)
            m = re.search(rb"Content-Range:\s*bytes (\d+)-(\d+)/", head, re.I)
            if not m:
                continue
            a, b = int(m.group(1)), int(m.group(2))
            parts[a] = data[: b - a + 1]
        out = []
        for a, b in spans:
            if a not in parts or (b is not None and len(parts[a]) != b - a + 1):
                raise RuntimeError(f"{url}: multipart response missing range {a}-{b}")
            out.append(parts[a])
        return out


# ----------------------------------------------------------------------------- decoding

def _attr(md, name):
    v = getattr(md, name)
    return v() if callable(v) else v


def _scan_mode(msg):
    pos = 16
    while pos < len(msg) - 5:
        length, num = int.from_bytes(msg[pos:pos + 4], "big"), msg[pos + 4]
        if num == 3:
            template = int.from_bytes(msg[pos + 12:pos + 14], "big")
            if template != 0:
                raise RuntimeError(f"expected a lat/lon grid (template 3.0), got 3.{template}")
            return msg[pos + 72 - 1]
        pos += length
    raise RuntimeError("GRIB2 grid section not found")


class Sampler:
    """Domain grid (subsampled) + nearest native point for each named point."""

    def __init__(self, domain, points):
        res = domain["res"]
        self.lats = np.round(np.arange(domain["lat_max"], domain["lat_min"] - 1e-6, -res), 4)
        self.lons = np.round(np.arange(domain["lon_min"], domain["lon_max"] + 1e-6, res), 4)
        self.pnames = list(points)
        self.plat = np.array([points[p][0] for p in self.pnames])
        self.plon = np.array([points[p][1] for p in self.pnames])
        self.cache, self.lock = {}, threading.Lock()

    def _index(self, lat1d, lon1d):
        key = (len(lat1d), len(lon1d), float(lat1d[0]), float(lon1d[0]))
        with self.lock:
            if key not in self.cache:
                lon360 = np.mod(lon1d, 360)

                def nearest_lon(targets):
                    d = np.abs(((lon360[None, :] - np.mod(targets, 360)[:, None]) + 180) % 360 - 180)
                    return d.argmin(axis=1)

                def nearest_lat(targets):
                    return np.abs(lat1d[None, :] - targets[:, None]).argmin(axis=1)

                gj, gi = nearest_lat(self.lats), nearest_lon(self.lons)
                pj, pi = nearest_lat(self.plat), nearest_lon(self.plon)
                step = abs(float(lat1d[1] - lat1d[0]))
                if np.abs(lat1d[gj] - self.lats).max() > step or step > 1.0:
                    raise RuntimeError("source grid doesn't cover the domain")
                self.cache[key] = (gj, gi, pj, pi)
            return self.cache[key]

    def sample(self, blob):
        import gribberish
        msg = gribberish.parse_grib_message(blob, 0)
        md = msg.metadata
        shape = tuple(_attr(md, "grid_shape"))
        a = np.asarray(msg.data(), dtype=np.float32).reshape(shape)
        if _scan_mode(blob) & 0x10:
            a[1::2] = a[1::2, ::-1]
        lat, lon = (np.asarray(v, dtype=np.float64) for v in _attr(md, "latlng"))
        if lat.ndim == 2:
            lat, lon = lat[:, 0], lon[0, :]
        gj, gi, pj, pi = self._index(lat, lon)
        return a[np.ix_(gj, gi)], a[pj, pi]


# ----------------------------------------------------------------------------- storage

class Store:
    """In-memory arrays for one model run; written out by ens_fetch."""

    def __init__(self, fields, n_mem, steps, sampler):
        self.fields, self.steps = fields, steps
        ny, nx, npt = len(sampler.lats), len(sampler.lons), len(sampler.pnames)
        self.grid = {f: np.full((n_mem, len(steps), ny, nx), np.nan, np.float32) for f in fields}
        self.pts = {f: np.full((n_mem, len(steps), npt), np.nan, np.float32) for f in fields}
        self.buckets = {}          # NOAA precip: (member, a, b) -> (grid, pts)
        self.lock = threading.Lock()

    def put(self, field, m, step, grid, pts):
        k = self.steps.index(step)
        with self.lock:
            self.grid[field][m, k] = grid
            self.pts[field][m, k] = pts

    def put_bucket(self, m, a, b, grid, pts):
        with self.lock:
            self.buckets[(m, a, b)] = (grid, pts)

    def accumulate_buckets(self, n_mem):
        """6-h APCP buckets -> totals since init at the pattern steps (NaN after a gap)."""
        for m in range(n_mem):
            by_start = {a: (b, g, p) for (mm, a, b), (g, p) in self.buckets.items() if mm == m}
            tot_g = np.zeros_like(self.grid["tp"][0, 0])
            tot_p = np.zeros_like(self.pts["tp"][0, 0])
            t, ok = 0, True
            totals = {0: (tot_g.copy(), tot_p.copy())}
            while t in by_start:
                b, g, p = by_start[t]
                tot_g, tot_p, t = tot_g + g, tot_p + p, b
                totals[t] = (tot_g.copy(), tot_p.copy())
            for k, s in enumerate(self.steps):
                if s in totals:
                    self.grid["tp"][m, k], self.pts["tp"][m, k] = totals[s]


# ----------------------------------------------------------------------------- sources

# Cycles the run looks at; ai_ensemble_fetch.run sets it from the settings' cycle_hours (2026-10-29: 00Z + 12Z).
CYCLE_HOURS = (0, 6, 12, 18)


def _candidate_cycles(now_utc, back_h=48):
    base = now_utc.replace(minute=0, second=0, microsecond=0)
    base -= dt.timedelta(hours=base.hour % 6)
    return [c for c in (base - dt.timedelta(hours=h) for h in range(0, back_h + 1, 6)) if c.hour in CYCLE_HOURS]


# Humidity / instability fields added 2026-10-11 (700 hPa dryness, surface RH, CAPE). Specific
# humidity is stored in g/kg so the files' 0.1 rounding keeps it.
HUMIDITY_ECMWF = {"t700": ("t", "700", 1, -273.15), "q700": ("q", "700", 1000, 0),
                  "t2m": ("2t", None, 1, -273.15), "d2m": ("2d", None, 1, -273.15),
                  "u10": ("10u", None, 1, 0), "v10": ("10v", None, 1, 0)}


class EcmwfEns:
    AWS = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/"
    GCS = "https://storage.googleapis.com/ecmwf-open-data/"
    ECMWF = "https://data.ecmwf.int/forecasts/"
    # Each model starts on its own bucket: with both on AWS, the 2026-10-12 run (12 fields each after
    # the humidity / 10 m wind additions) drew constant S3 "503 Slow Down" and AIFS + IFS took ~3x as
    # long. The other mirrors are fallbacks for a cycle the first one doesn't have yet.
    MIRROR_ORDER = {"aifs_ens": [AWS, GCS, ECMWF], "ifs_ens": [GCS, AWS, ECMWF]}

    def __init__(self, name):
        self.name = name
        if name == "aifs_ens":
            self.stream, self.ftype = "aifs-ens", "pf"
            self.params = {"z500": ("z", "500", 1 / G, 0), "msl": ("msl", None, 0.01, 0),
                           "u850": ("u", "850", 1, 0), "v850": ("v", "850", 1, 0),
                           "t850": ("t", "850", 1, -273.15), "tp": ("tp", None, 1, 0)}
            self.params.update(HUMIDITY_ECMWF)
        else:
            self.stream, self.ftype = "ifs", "ef"
            self.params = {"z500": ("gh", "500", 1, 0), "msl": ("msl", None, 0.01, 0),
                           "u850": ("u", "850", 1, 0), "v850": ("v", "850", 1, 0),
                           "t850": ("t", "850", 1, -273.15), "tp": ("tp", None, 1000, 0)}
            self.params.update(HUMIDITY_ECMWF)
            self.params["cape"] = ("mucape", None, 1, 0)     # most-unstable CAPE, J/kg (IFS ENS only)
        # 8 threads each: AIFS and IFS run at the same time against the same bucket, and 2 x 12
        # drew S3 "503 Slow Down" (2026-10-10).
        self.http = Http(per_min=3000, workers=8, multirange=False)
        self.MIRRORS = self.MIRROR_ORDER[name]
        self.base = self.MIRRORS[0]

    def _path(self, cycle, step):
        return (f"{cycle:%Y%m%d}/{cycle:%H}z/{self.stream}/0p25/enfo/"
                f"{cycle:%Y%m%d%H}0000-{step}h-enfo-{self.ftype}")

    def pick_cycle(self, now_utc, max_lead, log):
        for base in self.MIRRORS:
            for c in _candidate_cycles(now_utc):
                if self.http.exists(base + self._path(c, max_lead) + ".index"):
                    self.base = base
                    return c
            log(f"  {self.name}: no complete cycle on {base}")
        raise RuntimeError(f"{self.name}: no cycle reaching {max_lead} h in the last 48 h")

    def members(self, cycle, log):
        import json
        r = self.http.get(self.base + self._path(cycle, 24) + ".index")
        nums = sorted({int(json.loads(l)["number"]) for l in r.text.splitlines()
                       if json.loads(l).get("type") == "pf"})
        return nums

    def fetch(self, cycle, steps, fields, sampler, log, mem_limit=None):
        import json
        mems = self.members(cycle, log)[:mem_limit]
        mpos = {n: i for i, n in enumerate(mems)}
        store = Store(fields, len(mems), steps, sampler)
        want = {self.params[f][:2]: f for f in fields}

        def one(step):
            r = self.http.get(self.base + self._path(cycle, step) + ".index")
            if r is None:
                raise RuntimeError(f"{self.name}: no index for step {step}")
            rows = [json.loads(l) for l in r.text.splitlines()]
            rows = [x for x in rows if x.get("type") == "pf"
                    and (x["param"], x.get("levelist")) in want and int(x["number"]) in mpos]
            spans = [(x["_offset"], x["_offset"] + x["_length"] - 1) for x in rows]
            blobs = self.http.ranges(self.base + self._path(cycle, step) + ".grib2", spans)
            for x, blob in zip(rows, blobs):
                f = want[(x["param"], x.get("levelist"))]
                g, p = sampler.sample(blob)
                _, _, scale, off = self.params[f]
                store.put(f, mpos[int(x["number"])], step, g * scale + off, p * scale + off)
            return step, len(rows)

        _run(self.http, [lambda s=s: one(s) for s in steps], self.name, log)
        if "tp" in fields:       # step 0 has no precip message: zero by definition
            k0 = steps.index(0) if 0 in steps else None
            if k0 is not None:
                store.grid["tp"][:, k0] = np.where(np.isnan(store.grid["tp"][:, k0]), 0, store.grid["tp"][:, k0])
                store.pts["tp"][:, k0] = np.where(np.isnan(store.pts["tp"][:, k0]), 0, store.pts["tp"][:, k0])
        return store, [f"pf{n:02d}" for n in mems]


class NoaaEns:
    def __init__(self, name):
        self.name = name
        if name == "aigefs":
            self.base = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigefs/prod/"
            self.mems = [f"mem{i:03d}" for i in range(31)]
            self.prod = {"z500": "pres", "u850": "pres", "v850": "pres", "t850": "pres",
                         "msl": "sfc", "tp": "sfc", "t700": "pres", "q700": "pres", "t2m": "sfc",
                         "u10": "sfc", "v10": "sfc"}
            # NOMADS blocks an address briefly when it goes over ~120 hits/min (an "Over Rate" page,
            # 2026-10-10 after a day of test runs); 90 leaves room for retries.
            self.http = Http(per_min=90, workers=6, multirange=True)
        else:
            self.base = "https://noaa-gefs-pds.s3.amazonaws.com/"
            self.mems = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
            self.prod = {f: "a" for f in ("z500", "u850", "v850", "t850", "msl", "tp",
                                          "t700", "rh700", "t2m", "rh2m", "cape", "u10", "v10")}
            self.http = Http(per_min=3000, workers=12, multirange=False)
        self.vars = {"z500": ("HGT", "500 mb", 1, 0), "u850": ("UGRD", "850 mb", 1, 0),
                     "v850": ("VGRD", "850 mb", 1, 0), "t850": ("TMP", "850 mb", 1, -273.15),
                     "msl": ("PRMSL", "mean sea level", 0.01, 0), "tp": ("APCP", "surface", 1, 0),
                     "t700": ("TMP", "700 mb", 1, -273.15), "q700": ("SPFH", "700 mb", 1000, 0),
                     "rh700": ("RH", "700 mb", 1, 0), "t2m": ("TMP", "2 m above ground", 1, -273.15),
                     "rh2m": ("RH", "2 m above ground", 1, 0), "cape": ("CAPE", "180-0 mb above ground", 1, 0),
                     "u10": ("UGRD", "10 m above ground", 1, 0), "v10": ("VGRD", "10 m above ground", 1, 0)}

    def _path(self, cycle, mem, prod, f):
        if self.name == "aigefs":
            return (f"aigefs.{cycle:%Y%m%d}/{cycle:%H}/{mem}/model/atmos/grib2/"
                    f"aigefs.t{cycle:%H}z.{prod}.f{f:03d}.grib2")
        return f"gefs.{cycle:%Y%m%d}/{cycle:%H}/atmos/pgrb2ap5/{mem}.t{cycle:%H}z.pgrb2a.0p50.f{f:03d}"

    def pick_cycle(self, now_utc, max_lead, log):
        prods = sorted(set(self.prod.values()))
        for c in _candidate_cycles(now_utc):
            if all(self.http.exists(self.base + self._path(c, self.mems[-1], p, max_lead) + ".idx") for p in prods):
                return c
        raise RuntimeError(f"{self.name}: no cycle reaching {max_lead} h in the last 48 h")

    def fetch(self, cycle, steps, fields, sampler, log, precip_step=6, mem_limit=None):
        self.mems = self.mems[:mem_limit]
        store = Store(fields, len(self.mems), steps, sampler)
        jobs = []                      # (member index, prod, fhour, [fields])
        files = {}
        for prod in sorted(set(self.prod[f] for f in fields)):
            pf = [f for f in fields if self.prod[f] == prod]
            for s in steps:
                files[(prod, s)] = [f for f in pf if not (f == "tp" and s == 0)]
            if "tp" in pf:
                for s in range(precip_step, max(steps) + 1, precip_step):
                    files.setdefault((prod, s), [])
                    if "tp" not in files[(prod, s)]:
                        files[(prod, s)].append("tp")
        for m in range(len(self.mems)):
            for (prod, s), fl in files.items():
                if fl:
                    jobs.append((m, prod, s, fl))

        def one(m, prod, s, fl):
            url = self.base + self._path(cycle, self.mems[m], prod, s)
            idx = self.http.get(url + ".idx")
            if idx is None:
                raise RuntimeError(f"{url}.idx missing")
            rows = [ln.split(":") for ln in idx.text.strip().splitlines()]
            if not rows or any(len(r) < 6 or not r[1].isdigit() for r in rows):
                # NOMADS answers over its rate limit with an HTML "Over Rate" page (2026-10-10)
                if "over rate" in idx.text.lower():
                    raise OverRate(f"{url}.idx: NOMADS over-rate page")
                raise RuntimeError(f"{url}.idx is not a GRIB index ({idx.text[:60]!r})")
            picks = []
            for i, r in enumerate(rows):
                for f in fl:
                    var, lev = self.vars[f][:2]
                    if r[3] == var and r[4] == lev:
                        if f == "tp":
                            mm = ACC_RE.match(r[5])
                            if not mm or int(mm.group(2)) != s or int(mm.group(2)) - int(mm.group(1)) != precip_step:
                                continue
                        end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else None
                        picks.append((f, i, int(r[1]), end))
            blobs = self.http.ranges(url, [(a, b) for _, _, a, b in picks])
            for (f, i, _, _), blob in zip(picks, blobs):
                g, p = sampler.sample(blob)
                _, _, scale, off = self.vars[f]
                g, p = g * scale + off, p * scale + off
                if f == "tp":
                    store.put_bucket(m, s - precip_step, s, g, p)
                if s in steps and f != "tp":
                    store.put(f, m, s, g, p)
            return s

        _run(self.http, [lambda j=j: one(*j) for j in jobs], self.name, log)
        if "tp" in fields:
            store.accumulate_buckets(len(self.mems))
        return store, list(self.mems)


class WeatherNext3:
    """Placeholder until Google DeepMind grants access (request form; 5-7 business days).

    Known from developers.google.com/weathernext/guides/gcs (2026-10-10):
      full ensemble   gs://weathernext3_spatial/weathernext_3_0_0/zarr/  (64 members, 13 pressure
                      levels, requester pays -- pass x-goog-user-project), real time under
                      2026_to_present/<YYYYMMDD_HHhr_XX_preds>/predictions.zarr
      statistics      gs://weathernext3_statistics_spatial/weathernext_3_0_0_statistics/zarr/
                      (surface only: _mean/_p10.._p90, free egress)
      dims            init_time, lead_time (6-hourly), lead_subtime, sample, lat_0p1/lon_0p1, lon 0-360
      open with       obstore GCSStore -> zarr.storage.ObjectStore -> xr.open_zarr(chunks={})
    Real-time data is under GDM "Experimental Data" terms -- sandbox comparison only.
    Variable names are unconfirmed: list ds.data_vars once access works, then map them here.
    """

    def __init__(self, name):
        self.name = name

    def pick_cycle(self, *a, **k):
        raise NotImplementedError("WeatherNext 3: access pending -- see the class docstring")


def make_source(name):
    if name in ("aifs_ens", "ifs_ens"):
        return EcmwfEns(name)
    if name in ("aigefs", "gefs"):
        return NoaaEns(name)
    if name == "weathernext3":
        return WeatherNext3(name)
    raise ValueError(name)


OVER_RATE_PAUSE_S = 180
OVER_RATE_MAX = 5


def _retrying(task, name, log, http, tries=4):
    """One file's download+decode, retried with backoff: a single bad response (throttling page,
    truncated range) shouldn't fail the whole model run. An over-rate answer pauses EVERY thread
    for OVER_RATE_PAUSE_S (the block outlasts a per-file backoff) and doesn't use up an attempt."""
    def run():
        k = over = 0
        while True:
            try:
                return task()
            except OverRate as e:
                over += 1
                if over > OVER_RATE_MAX:
                    raise
                if http.limiter.pause_until <= time.time():
                    log(f"    {name}: server says over rate -- all downloads paused {OVER_RATE_PAUSE_S} s ({over}/{OVER_RATE_MAX})")
                http.limiter.cooldown(OVER_RATE_PAUSE_S)
            except Exception as e:  # noqa: BLE001
                k += 1
                if k >= tries:
                    raise
                msg = str(e).replace("https://", "")
                log(f"    {name}: retry {k}/{tries - 1} after {type(e).__name__}: ...{msg[-160:]}")
                time.sleep(20 * k * (0.75 + 0.5 * random.random()))
    return run


MAX_FAILED_SHARE = 0.02


def _run(http, tasks, name, log):
    started, n = time.time(), len(tasks)
    ex = ThreadPoolExecutor(http.workers)
    futs = [ex.submit(_retrying(t, name, log, http)) for t in tasks]
    failed = []
    try:
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                # A few files that still fail after their retries are left missing (NaN for those
                # members/steps) instead of throwing away the whole model run; more than
                # MAX_FAILED_SHARE means something is really wrong, so the model fails.
                failed.append(e)
                if len(failed) > max(2, MAX_FAILED_SHARE * n):
                    raise RuntimeError(f"{len(failed)} of {n} files failed; last: {type(e).__name__}: {e}") from e
                log(f"    {name}: WARNING file left missing after retries ({type(e).__name__}: ...{str(e)[-120:]})")
            if k % max(1, n // 20) == 0 or k == n:
                el = time.time() - started
                log(f"    {name}: {k}/{n} files  {http.nbytes / 1e9:.2f} GB  {http.nreq} requests  "
                    f"{el / 60:.1f} min (eta {el / k * (n - k) / 60:.0f} min)")
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    http.failed_files = len(failed)
    if failed:
        log(f"    {name}: {len(failed)} of {n} files missing (members/steps left blank)")
