"""Hawaii satellite large-fire growth days, observations and ERA5 summary for the AI ensemble calibration.

    python scripts/ai_ensemble_history_hawaii.py              # footprints -> fire occurrence -> FIRMS/MTBS/WFIGS -> growth days, obs
    python scripts/ai_ensemble_history_hawaii.py --prep-only  # re-date only (sources already downloaded)
    python scripts/ai_ensemble_history_hawaii.py --obs-only   # ASOS + soundings only
    python scripts/ai_ensemble_history_era5.py --region hi    # ERA5 Hawaii box (download), then derive_era5() below

Added 2026-10-13 (analyst: "Let's go California and Hawaii for focus, get started on full build out for
Hawaii"; ONCC dispatches Hawaii). Hawaii has no PSAs -- NIFC's national PSA layer carries one "No PSA
Assigned" ONCC polygon for the islands -- so its footprints are the four island groups, written as a
PSA-style shapefile the toolkit's pipeline reads unchanged:
    HI01 Kauai (with Niihau)   HI02 Oahu   HI03 Maui Nui (Maui, Molokai, Lanai, Kahoolawe)   HI04 Hawaii Island
Everything goes under <data_dir>/history/hawaii/ (production data/ and the ONCC store untouched), plus a
copy of the shapefile in config/hawaii_psa/ for the calibration and the research viewer.

Fire days run 06 to 06 Hawaii time (Pacific/Honolulu, no daylight time), the same convention as the
mainland's Pacific fire days.

Observations (daily, Hawaii fire day):
  ASOS (IEM): PHLI Lihue, PHNL Honolulu, PHOG Kahului, PHNY Lanai, PHMK Molokai, PHKO Kona, PHTO Hilo,
              PHJR Kalaeloa; sea-level pressure + RH at the synoptic hours (the models' sampling)
    <A>-<B>_min6 / _max6  gradients (settings hawaii.gradients), fire-day min / max over 00/06/12/18 UTC
    <STN>_rhmin6          fire-day minimum RH at the synoptic hours
    <STN>_windmax         strongest wind or gust (kt)
  Soundings (IEM RAOB): PHLI Lihue, PHTO Hilo (IEM id PITO), 00 and 12 UTC
    <STN>850_spd_<t>      850 hPa wind speed (kt): trade-wind strength
    <STN>850_dd_<t>, 700_dd_<t>   dewpoint depression (C): dryness under / above the trade inversion
    <STN>_inv_<t>         trade-inversion base (hPa; the lowest level above 950 hPa where temperature
                          stops falling, 0 if none below 600 hPa)
ERA5 (derive_era5): footprint means of the same measures as the mainland (Hot-Dry-Windy analogue, min
RH, 850 hPa wind) plus the north-south pressure gradient across the islands (25 N minus 19 N along
157 W: the subtropical high's push; the August 2023 Maui downslope wind event was the high to the
north with Hurricane Dora to the south): HI25N-HI19N_max6 and HI30N-HI19N_max6 along 157 W, named as
the models' point gradients so the forecast side reads them the same way.
"""
import argparse
import datetime as dt
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ai_ensemble_common as common

TOOLKIT = common.ROOT
PSA_URL = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
           "DMP_Predictive_Service_Area__PSA_Boundaries_Public/FeatureServer/0/query")
TZ = "Pacific/Honolulu"
MAIN_ISLANDS = (-161.0, 18.5, -154.5, 22.5)
GROUPS = [("HI01", "Kauai", -161.0, -159.0), ("HI02", "Oahu", -158.5, -157.5),
          ("HI03", "Maui Nui", -157.5, -155.9), ("HI04", "Hawaii Island", -156.2, -154.5)]
ASOS = ["PHLI", "PHNL", "PHOG", "PHNY", "PHMK", "PHKO", "PHTO", "PHJR"]
RAOB = {"PHLI": "PHLI", "PHTO": "PITO"}      # our name -> IEM RAOB id (IEM files Hilo as PITO)


def out_dir(cfg=None):
    d = common.Paths(cfg or common.load_settings()).history / "hawaii"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rel(p):
    return str(p.relative_to(TOOLKIT)).replace("\\", "/")


def fetch_footprints(cfg):
    """NIFC's Hawaii polygon -> island groups HI01-HI04 (PSA-style shapefile)."""
    import geopandas as gpd
    import requests
    OUT = out_dir(cfg)
    shp = OUT / "HAWAII_PSA.shp"
    if shp.exists():
        return shp
    r = requests.get(PSA_URL, params={"where": "GACCUnitID='USCAONCC' AND PSANAME='No PSA Assigned'",
                                      "outFields": "*", "outSR": 4269, "f": "geojson"}, timeout=120)
    r.raise_for_status()
    g = gpd.read_file(r.text).set_crs(4269, allow_override=True).explode(index_parts=False)
    x0, y0, x1, y1 = MAIN_ISLANDS
    c = g.geometry.representative_point()
    g = g[(c.x >= x0) & (c.x <= x1) & (c.y >= y0) & (c.y <= y1)].copy()
    c = g.geometry.representative_point()
    code = pd.Series(None, index=g.index, dtype=object)
    for k, name, lo, hi in GROUPS:
        sel = (c.x >= lo) & (c.x < hi) & code.isna()
        if k == "HI03":
            sel &= c.y > 20.3            # Maui Nui; Hawaii Island's north tip reaches -155.9
        code[sel] = k
    g["PSANationa"] = code
    g = g[g.PSANationa.notna()].dissolve(by="PSANationa").reset_index()
    g["PSANAME"] = g.PSANationa.map({k: n for k, n, *_ in GROUPS})
    g["GACCName"] = "Northern California Geographic Area Coordination Center"
    g["GACCUnitID"] = "USCAONCC"
    g[["PSANAME", "GACCName", "GACCUnitID", "PSANationa", "geometry"]].to_file(shp)
    cdir = TOOLKIT / "config" / "hawaii_psa"
    cdir.mkdir(exist_ok=True)
    for f in OUT.glob("HAWAII_PSA.*"):
        shutil.copy2(f, cdir / f.name)
    print(f"Hawaii footprints: {', '.join(f'{a} {b}' for a, b in zip(g.PSANationa, g.PSANAME))}")
    return shp


def settings_for_hawaii(shp, cfg):
    import pipeline_settings
    OUT = out_dir(cfg)
    s = dict(pipeline_settings.load_settings())
    s.update({
        "fire_occurrence_gacc": "HI",
        "fire_occurrence_psa_shapefile": rel(shp),
        "fire_occurrence_dir": rel(OUT / "fire_occurrence"),
        "fire_occurrence_timezone": TZ,
        "satellite_growth_dir": rel(OUT / "satellite_growth"),
        "nfdrs_ovf_gacc": "HI",
        "imsr_growth_dir": rel(OUT / "imsr_growth"),     # doesn't exist -> no IMSR cause join
    })
    return s


def growth(cfg, prep_only=False):
    shp = fetch_footprints(cfg)
    s = settings_for_hawaii(shp, cfg)
    import fire_occurrence
    import satellite_growth_data
    import satellite_growth_prep
    OUT = out_dir(cfg)
    if not prep_only:
        src = OUT / "fire_occurrence" / "source"
        src.mkdir(parents=True, exist_ok=True)
        for z in (TOOLKIT / "data" / "fire_occurrence" / "source").glob("RDS-2013-0009.*_Data_Format2_GDB.zip"):
            if not (src / z.name).exists():
                print(f"copying {z.name} from the toolkit's FOD cache")
                shutil.copy2(z, src / z.name)
        print("== fire occurrence (FPA FOD + InFORM FODR) for the Hawaii island groups")
        fire_occurrence.run(s)
        years = list(range(s["satellite_growth_first_year"], dt.date.today().year + 1))
        print(f"== FIRMS {years[0]}-{years[-1]}, MTBS, WFIGS for Hawaii's envelope")
        for attempt in range(6):
            try:
                satellite_growth_data.fetch_firms(s, years)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 5:
                    raise
                print(f"  FIRMS failed ({type(e).__name__}); retrying in {60 * (attempt + 1)} s", flush=True)
                time.sleep(60 * (attempt + 1))
        satellite_growth_data.fetch_mtbs(s)
        satellite_growth_data.fetch_wfigs(s)
    print("== growth days")
    satellite_growth_prep.run(s)


# ----------------------------------------------------------------------------- observations

def fire_day(ts_utc):
    loc = pd.DatetimeIndex(ts_utc).tz_localize("UTC").tz_convert(TZ) - pd.Timedelta(hours=6)
    return loc.tz_localize(None).normalize()


def fetch_obs(cfg, refresh=False):
    import ai_ensemble_history_obs as H
    OUT = out_dir(cfg) / "obs"
    (OUT / "asos").mkdir(parents=True, exist_ok=True)
    (OUT / "raob").mkdir(parents=True, exist_ok=True)
    this = dt.date.today().year
    for s in ASOS:
        for y in range(2001, this + 1):
            p = OUT / "asos" / f"{s}_{y}.csv"
            if p.exists() and not (refresh and y == this) and y != this:
                continue
            end = min(dt.date(y + 1, 1, 1), dt.date.today() + dt.timedelta(days=1))
            txt = H._get(H.ASOS_URL, {"station": s, "data": ["mslp", "relh", "sknt", "gust"], "year1": y, "month1": 1, "day1": 1,
                                      "year2": end.year, "month2": end.month, "day2": end.day, "tz": "Etc/UTC",
                                      "format": "onlycomma", "latlon": "no", "missing": "M", "report_type": 3})
            p.write_text(txt)
            print(f"  ASOS {s} {y}: {txt.count(chr(10)) - 1} obs", flush=True)
            time.sleep(1)
    for s, iem in RAOB.items():
        for y in range(2001, this + 1):
            p = OUT / "raob" / f"{s}_{y}.csv"
            if p.exists() and y != this and p.stat().st_size > 1000:
                continue
            txt = H._get(H.RAOB_URL, {"station": iem, "sts": f"{y}-01-01T00:00:00Z", "ets": f"{y + 1}-01-01T00:00:00Z"})
            p.write_text(txt)
            print(f"  RAOB {s} {y}: {txt.count(chr(10)) - 1} rows", flush=True)
            time.sleep(1)


def build_daily(cfg):
    OUT = out_dir(cfg) / "obs"
    slp, rh, wind = {}, {}, {}
    for p in sorted((OUT / "asos").glob("*_*.csv")):
        s = p.stem.split("_")[0]
        d = pd.read_csv(p, na_values=["M"])
        if d.empty:
            continue
        d["valid"] = pd.to_datetime(d.valid).dt.round("h")
        slp.setdefault(s, []).append(d.dropna(subset=["mslp"]).groupby("valid").mslp.mean())
        rh.setdefault(s, []).append(pd.to_numeric(d.relh, errors="coerce").groupby(d.valid).min())
        w = pd.concat([pd.to_numeric(d.sknt, errors="coerce"), pd.to_numeric(d.get("gust"), errors="coerce")], axis=1).max(axis=1)
        wind.setdefault(s, []).append(w.groupby(d.valid).max())
    slp = pd.DataFrame({s: pd.concat(v).groupby(level=0).mean() for s, v in slp.items()})
    slp = slp[(slp > 980) & (slp < 1050)]
    out = {}
    for g in cfg.get("hawaii", {}).get("gradients", []):
        a, b = g["a"], g["b"]
        if a in slp and b in slp:
            d = (slp[a] - slp[b]).dropna()
            syn = d[d.index.hour.isin([0, 6, 12, 18])]
            gs = syn.groupby(fire_day(syn.index))
            out[f"{g['name']}_min6"] = gs.min().where(gs.size() >= 3)
            out[f"{g['name']}_max6"] = gs.max().where(gs.size() >= 3)
    for s in ASOS:
        if s in rh:
            r = pd.concat(rh[s]).groupby(level=0).min()
            r = r[(r >= 0) & (r <= 100)]
            syn = r[r.index.hour.isin([0, 6, 12, 18])]
            gs = syn.groupby(fire_day(syn.index))
            out[f"{s}_rhmin6"] = gs.min().where(gs.size() >= 3)
        if s in wind:
            w = pd.concat(wind[s]).groupby(level=0).max()
            gw = w.groupby(fire_day(w.index))
            out[f"{s}_windmax"] = gw.max().where(gw.size() >= 12)
    daily = pd.DataFrame(out)
    rows = []
    for p in sorted((OUT / "raob").glob("*_*.csv")):
        d = pd.read_csv(p)
        if d.empty or "pressure_mb" not in d:
            continue
        if "speed_kts" in d and "sknt" not in d:
            d = d.rename(columns={"speed_kts": "sknt"})
        for c_ in ("pressure_mb", "tmpc", "dwpc", "sknt"):
            if c_ in d:
                d[c_] = pd.to_numeric(d[c_], errors="coerce")
        d["t"] = pd.to_datetime(d.validUTC)
        d = d[d.t.dt.hour.isin([0, 12])]
        for t, snd in d.groupby("t"):
            snd = snd.sort_values("pressure_mb", ascending=False)
            rec = {"t": t, "stn": p.stem.split("_")[0]}
            for lev in (850, 700):
                lv = snd[snd.pressure_mb == lev]
                if len(lv):
                    rec[f"dd{lev}"] = float(lv.tmpc.iloc[0] - lv.dwpc.iloc[0])
                    if lev == 850 and "sknt" in lv:
                        rec["spd850"] = float(lv.sknt.iloc[0])
            prof = snd[(snd.pressure_mb <= 950) & (snd.pressure_mb >= 600)].dropna(subset=["tmpc"])
            inv = 0.0
            if len(prof) > 2:
                dT = np.diff(prof.tmpc.values)
                k = np.where(dT >= 0)[0]
                inv = float(prof.pressure_mb.values[k[0]]) if len(k) else 0.0
            rec["inv"] = inv
            rows.append(rec)
    if rows:
        r = pd.DataFrame(rows)
        # 12 UTC on D -> D (02 HST, before the fire day starts at 06: the morning profile); 00 UTC on
        # D+1 -> D (14 HST: the afternoon profile)
        r["tag"] = np.where(r.t.dt.hour == 12, "12z", "00z")
        r["day"] = np.where(r.tag == "12z", r.t.dt.normalize(), r.t.dt.normalize() - pd.Timedelta(days=1))
        for (stn, tag), grp in r.groupby(["stn", "tag"]):
            gi = grp.set_index("day")
            for col, name in (("spd850", f"{stn}850_spd_{tag}"), ("dd850", f"{stn}850_dd_{tag}"),
                              ("dd700", f"{stn}700_dd_{tag}"), ("inv", f"{stn}_inv_{tag}")):
                if col in gi:
                    daily = daily.join(gi[col].groupby(level=0).first().rename(name), how="outer")
    daily.index.name = "fire_day"
    daily.sort_index().to_csv(OUT / "hawaii_daily_predictors.csv", float_format="%.2f")
    print(f"hawaii_daily_predictors.csv: {len(daily)} days, {daily.shape[1]} columns")
    return daily


# ----------------------------------------------------------------------------- ERA5

def derive_era5(cfg, log=print):
    """<data_dir>/history/era5_hi/era5_{sfc,pl}_<y>.nc -> history/hawaii/era5_hi_daily.csv: per island
    group the mainland footprint measures (00/12 UTC as the models) plus Hawaii's own trade-wind ones."""
    import geopandas as gpd
    import shapely
    import ai_ensemble_history_era5 as E
    src = E.era5_dir(cfg, "hi")
    years = sorted({int(p.stem.split("_")[-1]) for p in src.glob("era5_sfc_*.nc")})
    if not years:
        log("  no ERA5 Hawaii files yet")
        return None
    fps = gpd.read_file(fetch_footprints(cfg)).to_crs(4326)
    rows, masks = [], None
    for y in years:
        s = E._open_year(src / f"era5_sfc_{y}.nc")
        pp = src / f"era5_pl_{y}.nc"
        p = E._open_year(pp) if pp.exists() else None
        if p is not None and (p.sizes["time"] != s.sizes["time"] or (p.time.values != s.time.values).any()):
            p = p.reindex(time=s.time.values)
        lat, lon = s.lat.values, s.lon.values
        if masks is None:
            yy, xx = np.meshgrid(lat, lon, indexing="ij")
            # island cells: centres inside the island, or within 0.15 deg (small islands hold few 0.25 deg cells)
            masks = {r.PSANationa: shapely.contains_xy(r.geometry.buffer(0.15), xx, yy) for r in fps.itertuples()}
        times = pd.DatetimeIndex(s.time.values)
        syn = np.isin(times.hour, (0, 12))
        fd = fire_day(s.time.values)
        t2, d2 = s.t2m.values - 273.15, s.d2m.values - 273.15
        rh = np.clip(100 * E.es(d2) / E.es(t2), 0, 100)
        wind = np.hypot(s.u10.values, s.v10.values)
        hdw = np.clip(E.es(t2) - E.es(d2), 0, None) * wind
        msl = s.msl.values / 100.0
        tp = s.tp.values * 1000.0 * 6
        J = lambda la: int(np.abs(lat - la).argmin())
        I = lambda lo: int(np.abs(lon - lo).argmin())
        # named as the models' point gradients (settings hawaii.model_gradients), fire-day max at 00/12 UTC
        frame = pd.DataFrame({"HI25N-HI19N_max6": msl[:, J(25), I(-157)] - msl[:, J(19), I(-157)],
                              "HI30N-HI19N_max6": msl[:, J(30), I(-157)] - msl[:, J(19), I(-157)]}, index=fd)
        daily = frame[syn].groupby(level=0).max()
        if p is not None:
            u8, v8 = p.u.sel(level=850).values, p.v.sel(level=850).values
            t7, q7 = p.t.sel(level=700).values - 273.15, p.q.sel(level=700).values
            e7 = q7 * 700 / (0.622 + 0.378 * q7)
            ln = np.log(np.clip(e7, 1e-4, None) / 6.112)
            dd7 = t7 - 243.5 * ln / (17.67 - ln)
            # trade-wind box upstream of the islands (NE of the chain): mean 850 hPa wind speed and the
            # easterly component, the trades' strength
            box = (lat[:, None] >= 20) & (lat[:, None] <= 24) & (lon[None, :] >= -158) & (lon[None, :] <= -153)
            sp = np.hypot(u8, v8)[:, box].mean(axis=1)
            ue = (-u8)[:, box].mean(axis=1)
            tw = pd.DataFrame({"HI_850spd": sp, "HI_850east": ue, "HI_700dd": dd7[:, box].mean(axis=1)}, index=fd)
            g12 = tw[syn].groupby(level=0)
            daily["HI_850spd_max"] = g12.HI_850spd.max()
            daily["HI_850east_max"] = g12.HI_850east.max()
            daily["HI_700dd_max"] = g12.HI_700dd.max()
        for fid, m in masks.items():
            if not m.any():
                continue
            ext = pd.DataFrame({"hdw": hdw[:, m].mean(axis=1), "rh": rh[:, m].mean(axis=1), "wind": wind[:, m].mean(axis=1),
                                "rain": tp[:, m].mean(axis=1)}, index=fd)
            g12 = ext[syn].groupby(level=0)
            daily[f"{fid}_vpd_wind"] = g12.hdw.max()
            daily[f"{fid}_rh_min"] = g12.rh.min()
            daily[f"{fid}_wind_max"] = g12.wind.max()
            daily[f"{fid}_rain"] = ext.groupby(level=0).rain.sum()
        rows.append(daily)
        log(f"  ERA5 Hawaii derived {y}: {len(daily)} fire days")
    res = pd.concat(rows).sort_index()
    res = res[~res.index.duplicated(keep="last")]
    res.index.name = "fire_day"
    res.to_csv(out_dir(cfg) / "era5_hi_daily.csv", float_format="%.3f")
    log(f"  era5_hi_daily.csv: {len(res)} fire days, {res.shape[1]} columns")
    return res


# ----------------------------------------------------------------------------- NFDRS

def build_nfdrs(cfg):
    """Hawaii RAWS (the 46 full-record stations backfill_hawaii_raws_stations.py downloaded into
    data/historic_fires_research/Hawaii/_Backfill/) -> island group by location ->
      config/hawaii_raws_stations.csv     roster (GACC HI, FDRA = island group) for the FEMS 7-day forecast
      <climatology_dir>/HI/Historical/<HI0x>/<HI0x>_AvgDailyExtremes.csv   daily mean ERC/BI over the
                                          island's stations (what ai_ensemble_nfdrs.load_history reads)"""
    import geopandas as gpd
    import openpyxl
    import pipeline_settings
    xl = TOOLKIT / "config" / "fems_stations.xlsx"
    wb = openpyxl.load_workbook(xl, read_only=True, data_only=True)
    ws = wb["RAWS POR"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h).strip().lower() if h else "" for h in rows[0]]
    ix = {k: hdr.index(k) for k in ("station_id", "station_name", "latitude", "longitude") if k in hdr}
    st = []
    for r in rows[1:]:
        try:
            lat, lon = float(r[ix["latitude"]]), float(r[ix["longitude"]])
        except (TypeError, ValueError):
            continue
        x0, y0, x1, y1 = MAIN_ISLANDS
        if x0 <= lon <= x1 and y0 <= lat <= y1:
            st.append({"StationId": str(r[ix["station_id"]]).strip(), "StationName": r[ix["station_name"]], "lat": lat, "lon": lon})
    st = pd.DataFrame(st)
    fps = gpd.read_file(fetch_footprints(cfg)).to_crs(4326)
    pts = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs=4326)
    j = gpd.sjoin_nearest(pts, fps[["PSANationa", "PSANAME", "geometry"]], how="left", max_distance=0.2)
    j = j.drop_duplicates("StationId")
    back = TOOLKIT / "data" / "historic_fires_research" / "Hawaii" / "_Backfill"
    j = j[[(back / f"{sid}.csv").exists() for sid in j.StationId]]
    roster = pd.DataFrame({"GACC": "HI", "FDRA": j.PSANationa, "FDRAName": j.PSANAME, "StationId": j.StationId,
                           "StationName": j.StationName, "FuelModel": "Y", "LargeFireThreshold": 300, "OverlayYear": 2018})
    roster.to_csv(TOOLKIT / "config" / "hawaii_raws_stations.csv", index=False)
    base = TOOLKIT / pipeline_settings.load_settings()["climatology_dir"] / "HI" / "Historical"
    for psa, grp in roster.groupby("FDRA"):
        frames = []
        for sid in grp.StationId:
            d = pd.read_csv(back / f"{sid}.csv", usecols=["ObservationTime", "ERC", "BI"], parse_dates=["ObservationTime"])
            frames.append(d.groupby(d.ObservationTime.dt.normalize())[["ERC", "BI"]].mean())
        m = pd.concat(frames, keys=range(len(frames))).groupby(level=1).mean()
        m.index.name = "ObservationTime"
        (base / psa).mkdir(parents=True, exist_ok=True)
        m.reset_index().to_csv(base / psa / f"{psa}_AvgDailyExtremes.csv", index=False, float_format="%.2f")
        print(f"  NFDRS {psa}: {len(grp)} stations, {len(m):,} days {m.index.min():%Y-%m-%d}..{m.index.max():%Y-%m-%d}")
    print(f"hawaii_raws_stations.csv: {len(roster)} stations")
    return roster


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep-only", action="store_true")
    ap.add_argument("--obs-only", action="store_true")
    ap.add_argument("--era5", action="store_true", help="derive the ERA5 summary only")
    ap.add_argument("--nfdrs", action="store_true", help="island ERC/BI history + forecast roster only")
    a = ap.parse_args()
    cfg = common.load_settings()
    if a.era5:
        derive_era5(cfg)
        return 0
    if a.nfdrs:
        build_nfdrs(cfg)
        return 0
    if not a.obs_only:
        growth(cfg, prep_only=a.prep_only)
    if not a.prep_only:
        fetch_obs(cfg)
    build_daily(cfg)
    build_nfdrs(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
