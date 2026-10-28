"""Observed daily weather predictors 2001-present for calibrating the ensemble products.

    python scripts/ai_ensemble_history_obs.py              # download what's missing, then rebuild daily_predictors.csv
    python scripts/ai_ensemble_history_obs.py --refresh    # re-download the current year

Three observation sets, each the "truth" an analyst would check the ensembles against:
  ASOS sea-level pressure (hourly METAR SLP) for every gradient endpoint in config/ai_ensemble_settings.json --
     Iowa Environmental Mesonet (IEM) ASOS archive, one request per station per year.
  Upper-air soundings (00Z/12Z), 850/700/500 hPa -- IEM RAOB archive: KOAK (Oakland), KVBG
     (Vandenberg), KNKX (San Diego), KREV (Reno), KMFR (Medford).
  NCEP R1 daily-mean 500 hPa height and 850 hPa temperature on the 2.5 deg grid (NOAA PSL
     THREDDS subset service), the same reanalysis the anomaly climatology comes from -- used for pattern regimes
     and region-mean T850 anomalies.

Everything is put on the satellite growth days' FIRE DAY (Pacific local time minus 6 h, so
06:00-05:59; satellite_growth_fire_day_start_hour):
  - gradients: the hourly A-minus-B difference; the fire day's minimum (most offshore) value
  - soundings: the 12Z sounding on date D (05 PDT, the fire day's morning) -> fire day D;
    the 00Z sounding on D+1 (17 PDT, its afternoon) -> fire day D
  - R1 daily means are UTC days -> the fire day with the same date (a 6-8 h offset; fine for
    500 hPa patterns)

Outputs in <data_dir>/history/obs/: asos/<STN>_<year>.csv, raob/<STN>_<year>.csv, r1_daily.nc,
daily_predictors.csv (one row per fire day).

Promoted 2026-10-10 from the incoming/2026-10-10_ai_ensembles sandbox; settings in
config/ai_ensemble_settings.json, paths via ai_ensemble_common.py, docs in docs/ai_ensembles/README.md.
"""
import argparse
import datetime as dt
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import ai_ensemble_common as common

OBS = common.Paths(common.load_settings()).obs
UA = {"User-Agent": "FireAnalystToolkit ai-ensembles (history)"}
ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
RAOB_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py"
R1 = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/"
RAOB_STATIONS = ["KOAK", "KVBG", "KNKX", "KREV", "KMFR"]
LEVELS = [850, 700, 500]
FIRST_YEAR = 2001
TZ = "America/Los_Angeles"
DAY_START_H = 6


def _get(url, params, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=300)
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 503):
                time.sleep(30 * (k + 1))
                continue
            r.raise_for_status()
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"{url} kept failing")


def asos_stations(cfg):
    names = {g[x] for g in cfg["gradients"] for x in ("a", "b")} | set(cfg.get("rh_stations", {}))
    names |= {cfg["sundowner"][x] for x in ("a", "b")} if "sundowner" in cfg else set()     # CEFA crosswalk
    return sorted(names)


ASOS_FIELDS = "mslp,relh,sknt,gust"     # RH + wind/gust added 2026-10-11 (surface dryness predictors)


def _has_all_fields(path):
    try:
        with open(path, encoding="utf-8") as f:
            return "relh" in f.readline()
    except OSError:
        return False


def fetch_asos(stations, years, refresh):
    out = OBS / "asos"
    out.mkdir(parents=True, exist_ok=True)
    this = dt.date.today().year
    for s in stations:
        for y in years:
            p = out / f"{s}_{y}.csv"
            if p.exists() and _has_all_fields(p) and not (refresh and y == this):
                continue
            end = min(dt.date(y + 1, 1, 1), dt.date.today() + dt.timedelta(days=1))
            txt = _get(ASOS_URL, {"station": s, "data": ASOS_FIELDS, "year1": y, "month1": 1, "day1": 1,
                                  "year2": end.year, "month2": end.month, "day2": end.day, "tz": "Etc/UTC",
                                  "format": "onlycomma", "latlon": "no", "missing": "M", "report_type": 3})
            p.write_text(txt)
            print(f"  ASOS {s} {y}: {txt.count(chr(10)) - 1} obs", flush=True)
            time.sleep(1)


def fetch_raob(years, refresh):
    out = OBS / "raob"
    out.mkdir(parents=True, exist_ok=True)
    this = dt.date.today().year
    for s in RAOB_STATIONS:
        for y in years:
            p = out / f"{s}_{y}.csv"
            if p.exists() and not (refresh and y == this):
                continue
            txt = _get(RAOB_URL, {"station": s, "sts": f"{y}-01-01T00:00Z", "ets": f"{y + 1}-01-01T00:00Z"})
            df = pd.read_csv(io.StringIO(txt), na_values=["M"])
            df = df[df.pressure_mb.isin(LEVELS)]
            df.to_csv(p, index=False)
            print(f"  RAOB {s} {y}: {df.validUTC.nunique()} soundings", flush=True)
            time.sleep(1)


def _ntime(p):
    import xarray as xr
    try:
        with xr.open_dataset(p) as d:
            return d.sizes.get("time", 0)
    except Exception:  # noqa: BLE001 -- unreadable cache file: re-pull it
        return 0


def fetch_r1(cfg, years, refresh):
    """One THREDDS NetCDF Subset Service request per variable-year (PSL rate-limits the many small
    requests OPeNDAP slicing makes, HTTP 429), cached as <data_dir>/history/obs/r1/<var>_<year>.nc, then
    stitched into r1_daily.nc. The current year is always re-pulled (it grows daily)."""
    import xarray as xr
    d = cfg["domain"]
    cache = OBS / "r1"
    cache.mkdir(parents=True, exist_ok=True)
    this = dt.date.today().year
    q = {"north": d["lat_max"] + 5, "south": d["lat_min"] - 5, "west": d["lon_min"] - 5, "east": d["lon_max"] + 5,
         "temporal": "all", "accept": "netcdf3"}     # without temporal=all NCSS returns ONE time step
    for y in years:
        for var, lev in (("hgt", 500), ("air", 850)):
            p = cache / f"{var}_{y}.nc"
            if p.exists() and y != this and not refresh and _ntime(p) >= 360:
                continue
            url = f"https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep.reanalysis.dailyavgs/pressure/{var}.{y}.nc"
            for k in range(15):
                try:
                    r = requests.get(url, params=dict(q, var=var, vertCoord=lev), headers=UA, timeout=300)
                    if r.status_code == 200 and r.content[:3] == b"CDF":
                        tmp = p.with_suffix(".part")
                        tmp.write_bytes(r.content)
                        tmp.replace(p)
                        break
                    print(f"    R1 {var} {y}: HTTP {r.status_code}, retrying", flush=True)
                except requests.RequestException as e:
                    print(f"    R1 {var} {y}: {type(e).__name__}, retrying", flush=True)
                time.sleep(min(300, 20 * (k + 1)))
            else:
                raise RuntimeError(f"R1 {var} {y} failed")
            time.sleep(5)
        print(f"  R1 {y}: ok", flush=True)
    parts = []
    for y in years:
        z = xr.open_dataset(cache / f"hgt_{y}.nc").load()
        t = xr.open_dataset(cache / f"air_{y}.nc").load()
        zz, tt = z["hgt"].squeeze(drop=True), t["air"].squeeze(drop=True)
        parts.append(xr.Dataset({"z500": zz, "t850": tt - 273.15}))
        z.close(); t.close()
    ds = xr.concat(parts, "time")
    ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    ds = ds.assign_coords(time=pd.to_datetime(ds.time.values).normalize())
    ds = ds.drop_vars([v for v in ds.coords if v not in ("time", "lat", "lon")])
    tmp = OBS / "r1_daily.tmp.nc"
    ds.to_netcdf(tmp)
    tmp.replace(OBS / "r1_daily.nc")
    if ds.sizes["time"] < 300 * len(years):
        raise RuntimeError(f"r1_daily.nc would hold only {ds.sizes['time']} days for {len(years)} years")
    print(f"  r1_daily.nc: {ds.sizes['time']} days, lat {float(ds.lat.min())}-{float(ds.lat.max())}, "
          f"lon {float(ds.lon.min())}-{float(ds.lon.max())}")


def fire_day(ts_utc):
    """UTC timestamps -> fire day (Pacific local - 6 h) as dates."""
    loc = pd.DatetimeIndex(ts_utc).tz_localize("UTC").tz_convert(TZ) - pd.Timedelta(hours=DAY_START_H)
    return loc.tz_localize(None).normalize()


def build_daily(cfg):
    # --- ASOS gradients
    slp, rh, wind = {}, {}, {}
    for p in sorted((OBS / "asos").glob("*_*.csv")):
        s = p.stem.split("_")[0]
        df = pd.read_csv(p, na_values=["M"])
        if df.empty:
            continue
        df["valid"] = pd.to_datetime(df.valid).dt.round("h")
        slp.setdefault(s, []).append(df.dropna(subset=["mslp"]).groupby("valid").mslp.mean())
        if "relh" in df:
            rh.setdefault(s, []).append(pd.to_numeric(df.relh, errors="coerce").groupby(df.valid).min())
        if "sknt" in df:
            w = pd.concat([pd.to_numeric(df.sknt, errors="coerce"), pd.to_numeric(df.get("gust"), errors="coerce")], axis=1).max(axis=1)
            wind.setdefault(s, []).append(w.groupby(df.valid).max())
    slp = pd.DataFrame({s: pd.concat(v).groupby(level=0).mean() for s, v in slp.items()})
    slp = slp[(slp > 940) & (slp < 1070)]
    out = {}
    for g in cfg["gradients"]:
        if g["a"] in slp and g["b"] in slp:
            d = (slp[g["a"]] - slp[g["b"]]).dropna()
            fd = fire_day(d.index)
            grp = d.groupby(fd)
            n = grp.size()
            out[f"{g['name']}_min"] = grp.min().where(n >= 12)      # need half a day of hourly pairs
            out[f"{g['name']}_n"] = n
            # Min over the synoptic hours only -- what a 6-hourly model sample can see; the hourly
            # minimum is more offshore than any model step, so model thresholds calibrate on this.
            syn = d[d.index.hour.isin([0, 6, 12, 18])]
            gs = syn.groupby(fire_day(syn.index))
            out[f"{g['name']}_min6"] = gs.min().where(gs.size() >= 3)
    # --- surface dryness / wind at the RH stations (settings rh_stations): the fire day's minimum RH
    # (all hours, and the synoptic hours 00/06/12/18Z only -- what a model step can see) and its
    # strongest wind or gust (kt)
    for s_ in cfg.get("rh_stations", {}):
        if s_ in rh:
            r = pd.concat(rh[s_]).groupby(level=0).min()
            r = r[(r >= 0) & (r <= 100)]
            grp = r.groupby(fire_day(r.index))
            out[f"{s_}_rhmin"] = grp.min().where(grp.size() >= 12)
            syn = r[r.index.hour.isin([0, 6, 12, 18])]
            gs = syn.groupby(fire_day(syn.index))
            out[f"{s_}_rhmin6"] = gs.min().where(gs.size() >= 3)
        if s_ in wind:
            w = pd.concat(wind[s_]).groupby(level=0).max()
            gw = w.groupby(fire_day(w.index))
            out[f"{s_}_windmax"] = gw.max().where(gw.size() >= 12)
    daily = pd.DataFrame(out)

    # --- soundings: 12Z on D -> D; 00Z on D+1 -> D
    rows = []
    for p in sorted((OBS / "raob").glob("*_*.csv")):
        df = pd.read_csv(p)
        if df.empty:
            continue
        df["validUTC"] = pd.to_datetime(df.validUTC)
        rows.append(df)
    if rows:
        r = pd.concat(rows)
        r = r[r.validUTC.dt.hour.isin([0, 12])]
        r["fday"] = np.where(r.validUTC.dt.hour == 12, r.validUTC.dt.normalize(),
                             r.validUTC.dt.normalize() - pd.Timedelta(days=1))
        r["tag"] = np.where(r.validUTC.dt.hour == 12, "12z", "00z")
        r = r.drop_duplicates(["station", "validUTC", "pressure_mb"])
        for lev in LEVELS:
            sub = r[r.pressure_mb == lev]
            for col, short in (("tmpc", "t"), ("dwpc", "td"), ("drct", "dir"), ("speed_kts", "spd"), ("height_m", "z")):
                if lev != 500 and short == "z":
                    continue
                if lev == 500 and short in ("td",):
                    continue
                wide = sub.pivot_table(index="fday", columns=["station", "tag"], values=col, aggfunc="first")
                wide.columns = [f"{st[1:]}{lev}_{short}_{tag}" for st, tag in wide.columns]
                daily = daily.join(wide, how="outer")
    daily.index.name = "fire_day"
    daily = daily.sort_index()
    daily = daily[daily.index >= pd.Timestamp(f"{FIRST_YEAR}-01-01")]
    daily.to_csv(OBS / "daily_predictors.csv", float_format="%.2f")
    print(f"daily_predictors.csv: {len(daily)} fire days, {daily.shape[1]} columns "
          f"({daily.index.min():%Y-%m-%d} to {daily.index.max():%Y-%m-%d})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    cfg = common.load_settings()
    years = list(range(FIRST_YEAR, dt.date.today().year + 1))
    if not a.skip_download:
        print("ASOS sea-level pressure:")
        fetch_asos(asos_stations(cfg), years, a.refresh)
        print("Upper-air soundings:")
        fetch_raob(years, a.refresh)
        print("NCEP R1 daily z500 / t850:")
        fetch_r1(cfg, years, a.refresh)
    build_daily(cfg)


if __name__ == "__main__":
    sys.exit(main())
