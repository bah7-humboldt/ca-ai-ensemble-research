"""ERA5 history for the AI ensemble calibration: a model-consistent, gap-free record of the same
predictors the ensembles forecast, plus the ones stations can't give (area-wide dryness, mixing
height, CAPE / dry lightning, Hot-Dry-Windy).

    python scripts/ai_ensemble_history_era5.py              # download missing years, then derive
    python scripts/ai_ensemble_history_era5.py --derive     # derive only (files already downloaded)
    python scripts/ai_ensemble_history_era5.py --years 2020 2021

Added 2026-10-11 (roadmap item 3; analyst: "go for all of it"). Source: ECMWF ERA5 hourly reanalysis
via the Copernicus Climate Data Store (free account; the API key lives in %USERPROFILE%\\.cdsapirc,
which only the analyst creates -- never this script). Licence: "Licence to use Copernicus Products"
(attribution required: "Contains modified Copernicus Climate Change Service information").

Requests (one per year per dataset, 00/06/12/18 UTC, California box 31-45 N, 127-112 W, 0.25 deg):
  single levels   mean_sea_level_pressure, 2m_temperature, 2m_dewpoint_temperature, 10m_u/v,
                  instantaneous_10m_wind_gust, boundary_layer_height,
                  convective_available_potential_energy, total_precipitation
  pressure levels temperature, specific_humidity at 700 hPa (all derive() uses; more levels or
                  variables put a year over CDS's request cost limit -- a "too large" refusal
                  halves the months and retries)
-> <data_dir>/history/era5/era5_{sfc,pl}_<year>.nc. Single levels arrive as a ZIP of netCDFs
(accumulated + instantaneous fields); they're merged into one file.

Derived (fire day = Pacific - 6 h, as the satellite growth days) -> history/era5/era5_daily.csv:
  <A-B>_era5           fire-day minimum MSLP gradient between the ASOS endpoints' nearest cells
                       (model-consistent twin of <A-B>_min6, no station gaps)
  <STN>_era5_rhmin     fire-day minimum 2 m RH at the rh_stations' nearest cells
  (vpd_wind, rh_min and the dry-thunderstorm CAPE test use 00/12 UTC only, the models' times)
  <FP>_vpd_wind        fire-day max of the footprint mean (2 m vapour-pressure deficit x 10 m wind),
                       a surface Hot-Dry-Windy analogue (Srock et al. 2018 use the lowest 500 m)
  <FP>_blh_max         fire-day max of the footprint-mean boundary-layer height (m)
  <FP>_rh_min          fire-day minimum of the footprint-mean 2 m RH (%)
  <FP>_dry_tstorm      share of footprint cells with CAPE >= 250 J/kg, 700 hPa dewpoint depression
                       >= 5 C and < 2.5 mm precipitation that fire day (dry-thunderstorm setup).
                       2026-10-12: was 500 J/kg / 10 C, which scored the June 2008 and August 2020
                       lightning sieges 0.00 / 0.09 -- high-based dry storms run on weak elevated CAPE
                       with the moisture near 700 hPa. 250 / 5 scores the 2008, 2015, 2017, 2020 and
                       2021 sieges 0.28-0.73 and reaches 0.2 on ~9 Jun-Sep days a year.
  <FP>_capep90         thunderstorm setup: 90th-percentile CAPE over the footprint's cells, the higher of
                       the fire day's 00 and 12 UTC values (J/kg). Fit to Vaisala NLDN strikes
                       2001-2025 (2026-10-13): AUC 0.91 for >= 100-strike footprint days and 0.93 for
                       >= 500 (May-Oct, every footprint 0.90-0.94), against 0.70-0.84 for dry_tstorm;
                       requiring a dry 700 hPa layer only made it worse (storms need the moisture).
  <FP>_rain            fire-day footprint-mean precipitation (mm; four 6-hourly samples x 6), to tell
                       wet from dry lightning
FP = each footprint in config/ai_ensemble_settings.json; footprint cells = cell centres inside its
PSAs. ai_ensemble_calibrate.py scores these like the station predictors.
"""
import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd

import ai_ensemble_common as common

AREA = [45, -127, 31, -112]          # N, W, S, E
TIMES = ["00:00", "06:00", "12:00", "18:00"]
SFC_VARS = ["mean_sea_level_pressure", "2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind",
            "10m_v_component_of_wind", "instantaneous_10m_wind_gust", "boundary_layer_height",
            "convective_available_potential_energy", "total_precipitation"]
# Only what derive() uses: 700 hPa temperature + humidity (dry-thunderstorm dewpoint depression).
# Station soundings and the models cover the rest; the pressure-level cost limit is tight.
PL_VARS = ["temperature", "specific_humidity"]
LEVELS = ["700"]
FIRST_YEAR = 2001
# Regions (2026-10-13: Hawaii added -- analyst: "California and Hawaii for focus"). Hawaii's fire
# weather is trade-wind strength and the trade inversion, so its pressure levels add 850 / 700 hPa
# wind with the temperature and humidity.
REGIONS = {
    "ca": {"area": AREA, "dir": "era5", "pl_vars": PL_VARS, "levels": LEVELS, "tz": "America/Los_Angeles"},
    "hi": {"area": [28, -165, 15, -150], "dir": "era5_hi", "tz": "Pacific/Honolulu", "levels": ["850", "700"],
           "pl_vars": ["temperature", "specific_humidity", "u_component_of_wind", "v_component_of_wind"]},
}
DTS_CAPE, DTS_DD700 = 250, 5       # dry-thunderstorm cell: CAPE (J/kg), 700 hPa dewpoint depression (C)
TZ = "America/Los_Angeles"


def era5_dir(cfg, region="ca"):
    d = common.Paths(cfg).history / REGIONS[region]["dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(cfg, years, log=print, region="ca"):
    import cdsapi
    c = cdsapi.Client(quiet=True)
    R = REGIONS[region]
    out = era5_dir(cfg, region)
    this = dt.date.today().year
    for y in years:
        months = [f"{m:02d}" for m in range(1, 13)] if y < this else [f"{m:02d}" for m in range(1, dt.date.today().month + 1)]
        days = [f"{d:02d}" for d in range(1, 32)]
        for kind, dataset, extra in (("sfc", "reanalysis-era5-single-levels", {"variable": SFC_VARS}),
                                     ("pl", "reanalysis-era5-pressure-levels", {"variable": R["pl_vars"], "pressure_level": R["levels"]})):
            f = out / f"era5_{kind}_{y}.nc"
            if f.exists() and y != this:
                continue
            # A year of pressure levels (5 variables x 3 levels x 4 times) is over CDS's per-request
            # cost limit ("Your request is too large", 2026-10-11); a year of single levels fits. So the
            # pressure levels go in half-years and are merged.
            # Whole year first; a "too large" refusal (CDS cost limit) splits the months in half and
            # retries, down to single months.
            parts, queue, n = [], [months], 0
            while queue:
                mon = queue.pop(0)
                n += 1
                part = f.with_name(f"era5_{kind}_{y}_{mon[0]}-{mon[-1]}.nc")
                if part.exists() and y != this:
                    parts.append(part)
                    continue
                req = {"product_type": ["reanalysis"], "year": [str(y)], "month": mon, "day": days, "time": TIMES,
                       "area": R["area"], "data_format": "netcdf", "download_format": "unarchived", **extra}
                log(f"  ERA5 {region} {kind} {y} {mon[0]}-{mon[-1]}: requesting (CDS queues requests; this can take a while)", flush=True)
                tmp = part.with_suffix(".part")
                try:
                    c.retrieve(dataset, req, str(tmp))
                except Exception as e:  # noqa: BLE001
                    if "too large" in str(e).lower() and len(mon) > 1:
                        h = len(mon) // 2
                        log(f"  ERA5 {kind} {y}: too large for one request -- splitting {mon[0]}-{mon[-1]}", flush=True)
                        queue[:0] = [mon[:h], mon[h:]]
                        continue
                    raise
                tmp.replace(part)
                _unzip_if_needed(part)
                parts.append(part)
            chunks = parts
            import xarray as xr
            parts.sort(key=lambda p_: p_.stem.split("_")[-1])
            if len(parts) == 1:
                parts[0].replace(f)
            else:
                ds = xr.concat([xr.open_dataset(p_).load() for p_ in parts], dim="valid_time").sortby("valid_time")
                ds.to_netcdf(f)
                for p_ in parts:
                    p_.unlink()
            log(f"  ERA5 {kind} {y}: {f.stat().st_size / 1e6:.0f} MB", flush=True)


def _unzip_if_needed(path):
    """CDS answers a multi-variable single-levels request with a ZIP of several netCDFs when the
    variables mix accumulated (precipitation) and instantaneous fields, whatever download_format says
    (2026-10-11). Merge the members into one netCDF in place."""
    import zipfile
    import xarray as xr
    if not zipfile.is_zipfile(path):
        return path
    tmpdir = path.with_suffix(".unzip")
    tmpdir.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as z:
        z.extractall(tmpdir)
    parts = [xr.open_dataset(f).load() for f in sorted(tmpdir.glob("*.nc"))]
    merged = xr.merge(parts, compat="override", join="outer")
    for p_ in parts:
        p_.close()
    out = path.with_suffix(".merged.nc")
    merged.to_netcdf(out)
    path.unlink()
    out.replace(path)
    for f in tmpdir.glob("*"):
        f.unlink()
    tmpdir.rmdir()
    return path


def _open_year(path):
    import xarray as xr
    _unzip_if_needed(path)
    ds = xr.open_dataset(path)
    ren = {k: v for k, v in (("valid_time", "time"), ("pressure_level", "level"), ("latitude", "lat"), ("longitude", "lon")) if k in ds.dims or k in ds.coords}
    return ds.rename(ren)


def fire_days(times):
    t = pd.DatetimeIndex(times).tz_localize("UTC").tz_convert(TZ) - pd.Timedelta(hours=6)
    return t.tz_localize(None).normalize()


def es(t_c):
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def derive(cfg, log=print):
    import geopandas as gpd
    import shapely
    import xarray as xr
    out = era5_dir(cfg)
    years = sorted({int(p.stem.split("_")[-1]) for p in out.glob("era5_sfc_*.nc")})
    if not years:
        log("  no ERA5 files to derive from")
        return None
    pts = dict(cfg["points"])
    pts.update(cfg.get("rh_stations", {}))
    psas = []
    for f in (common.ROOT / "config" / "oncc_psa" / "ONCC_PSA.shp", common.ROOT / "config" / "oscc_psa" / "OSCC_PSA.shp"):
        if f.exists():
            psas.append(gpd.read_file(f).to_crs(4326)[["PSANationa", "geometry"]])
    psas = pd.concat(psas) if psas else None
    rows = []
    masks = None
    for y in years:
        s = _open_year(out / f"era5_sfc_{y}.nc")
        pl_path = out / f"era5_pl_{y}.nc"
        p = _open_year(pl_path) if pl_path.exists() else None
        if p is not None and (p.sizes["time"] != s.sizes["time"] or (p.time.values != s.time.values).any()):
            # the current year's two datasets can end at different hours (preliminary ERA5T data)
            p = p.reindex(time=s.time.values)
        lat, lon = s.lat.values, s.lon.values
        if masks is None and psas is not None:
            yy, xx = np.meshgrid(lat, lon, indexing="ij")
            masks = {}
            for fid, f in cfg.get("footprints", {}).items():
                geom = psas[psas.PSANationa.isin(f["psas"])].geometry.union_all()
                masks[fid] = shapely.contains_xy(geom, xx, yy)
        fd = fire_days(s.time.values)
        # The ensembles all arrive 12-hourly (00/12 UTC), so the footprint fields use those two
        # times only -- with all four, ERA5's fire-day max/min would be systematically more extreme
        # than any model's and the calibrated cutoffs out of the models' reach (2026-10-12).
        syn = np.isin(pd.DatetimeIndex(s.time.values).hour, (0, 12))
        msl = s.msl.values / 100.0
        t2 = s.t2m.values - 273.15
        d2 = s.d2m.values - 273.15
        rh = np.clip(100 * es(d2) / es(t2), 0, 100)
        wind = np.hypot(s.u10.values, s.v10.values)
        vpd = np.clip(es(t2) - es(d2), 0, None)
        hdw = vpd * wind
        blh = s.blh.values
        cape = s.cape.values
        # ERA5 hourly tp is the total of the hour ending at that time; with four sampled hours a day,
        # each stands for its 6-hour block, so x6 estimates the fire day's precipitation.
        tp = s.tp.values * 1000.0 * 6
        dd7 = None
        if p is not None:
            t7 = p.t.sel(level=700).values - 273.15
            q7 = p.q.sel(level=700).values
            e7 = q7 * 700 / (0.622 + 0.378 * q7)
            ln = np.log(np.clip(e7, 1e-4, None) / 6.112)
            dd7 = t7 - 243.5 * ln / (17.67 - ln)
        J = {k: int(np.abs(lat - v[0]).argmin()) for k, v in pts.items()}
        I = {k: int(np.abs(lon - v[1]).argmin()) for k, v in pts.items()}
        frame = {}
        for g in cfg["gradients"]:
            a, b = g["a"], g["b"]
            if a in J and b in J:
                frame[f"{g['name']}_era5"] = msl[:, J[a], I[a]] - msl[:, J[b], I[b]]
        for st in cfg.get("rh_stations", {}):
            frame[f"{st}_era5_rhmin"] = rh[:, J[st], I[st]]
        df = pd.DataFrame(frame, index=fd)
        agg = {c: "min" for c in df.columns}
        daily = df.groupby(level=0).agg(agg)
        for fid, m in (masks or {}).items():
            if not m.any():
                continue
            # footprint means at each time step; the fire day keeps the worst of its four steps
            ext = pd.DataFrame({"hdw": hdw[:, m].mean(axis=1), "blh": blh[:, m].mean(axis=1),
                                "rh": rh[:, m].mean(axis=1)}, index=fd)
            g_ = ext.groupby(level=0)
            g12 = ext[syn].groupby(level=0)
            daily[f"{fid}_vpd_wind"] = g12.hdw.max()
            daily[f"{fid}_blh_max"] = g_.blh.max()
            daily[f"{fid}_rh_min"] = g12.rh.min()
            c90 = pd.Series(np.nanpercentile(cape[:, m], 90, axis=1), index=fd)
            daily[f"{fid}_capep90"] = c90[syn].groupby(level=0).max()
            daily[f"{fid}_rain"] = pd.Series(tp[:, m].mean(axis=1), index=fd).groupby(level=0).sum()
            if dd7 is not None:
                conv = (cape[:, m] >= DTS_CAPE) & (dd7[:, m] >= DTS_DD700) & syn[:, None]
                cdf = pd.DataFrame(conv.astype(float), index=fd).groupby(level=0).max()
                pdf = pd.DataFrame(tp[:, m], index=fd).groupby(level=0).sum()
                dry = (cdf.values > 0) & (pdf.values < 2.5)
                daily[f"{fid}_dry_tstorm"] = pd.Series(dry.mean(axis=1), index=cdf.index)
        rows.append(daily)
        log(f"  ERA5 derived {y}: {len(daily)} fire days")
    res = pd.concat(rows).sort_index()
    res = res[~res.index.duplicated(keep="last")]
    res.to_csv(out / "era5_daily.csv", float_format="%.3f")
    log(f"  era5_daily.csv: {len(res)} fire days, {res.shape[1]} columns")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derive", action="store_true", help="derive only")
    ap.add_argument("--years", type=int, nargs="+")
    ap.add_argument("--region", choices=sorted(REGIONS), default="ca")
    a = ap.parse_args()
    cfg = common.load_settings()
    years = a.years or list(range(FIRST_YEAR, dt.date.today().year + 1))
    if not a.derive:
        import pathlib
        if not (pathlib.Path.home() / ".cdsapirc").exists():
            raise SystemExit("No %USERPROFILE%\\.cdsapirc -- create it from your CDS profile page first (see this script's docstring).")
        download(cfg, years, region=a.region)
    if a.region == "hi":
        import ai_ensemble_history_hawaii
        ai_ensemble_history_hawaii.derive_era5(cfg)
    else:
        derive(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
