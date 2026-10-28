"""Source data for satellite large-fire growth days (scripts/satellite_growth_prep.py): NASA FIRMS
active-fire detections, MTBS burned-area perimeters and NIFC WFIGS incident perimeters, all cut to
the GACC's PSA envelope. Added 2026-10-07, ported from the incoming/VIIRS experiment (2026-10-06).

    python scripts/satellite_growth_data.py                 # everything missing, satellite_growth_first_year..today
    python scripts/satellite_growth_data.py --current-year  # nightly: re-pull this year's FIRMS + WFIGS only
    python scripts/satellite_growth_data.py --years 2020 2021 --refresh

FIRMS (https://firms.modaps.eosdis.nasa.gov/): standard-processing active fire detections from
MODIS (2000-11 on), VIIRS S-NPP (2012-01 on) and VIIRS NOAA-20 (2018-04 on). NOAA-21 (2024 on) is
left out on purpose, so every year is built from the same satellites. Two ways in:
  - Yearly country archive, data/country/<sensor>/<year>/<sensor>_<year>_United_States.csv. No
    key. Covers whole past years once NASA has reprocessed them (2001-2024 when written). The
    full US file is streamed and only the envelope is kept.
  - Area API (api/area/csv/<MAP_KEY>/<source>/<bbox>/<1-5 days>/<date>) for years the archive
    doesn't have yet, and always for the current year. Standard processing (_SP) through the
    API's data_availability date, near-real-time (_NRT) after. NRT is reprocessed later, so the
    current year is re-pulled every --current-year run. Needs a free MAP_KEY
    (https://firms.modaps.eosdis.nasa.gov/api/map_key/) in config/firms_config.json; without
    one, those years are skipped with a warning and the archive years still build.
Output: <dir>/source/firms/<sensor>_<year>.csv, same columns either way (latitude, longitude,
acq_date, acq_time UTC HHMM, confidence, frp, type...).

MTBS (Monitoring Trends in Burn Severity, USGS/USFS): the national burned-area boundaries
composite, fires >= 1,000 ac in the West, 1984 through MTBS's latest year. ~390 MB zip, kept
only as the extracted shapefile cut to the envelope (<dir>/source/mtbs/mtbs_perims.gpkg).
Refreshed when the zip on USGS is newer than our copy (checked once per --refresh run).

WFIGS Interagency Fire Perimeters (NIFC Open Data, 2020-present): one latest perimeter per fire
with its IRWIN UniqueFireIdentifier. Re-pulled in full every run (~1,700 fires in ONCC's envelope,
under a minute) -> <dir>/source/wfigs/wfigs_perimeters.gpkg.
"""
import argparse
import csv
import datetime as dt
import io
import json
import pathlib
import time
import zipfile

import requests

import pipeline_settings

ROOT = pipeline_settings.ROOT
FIRMS_CONFIG = ROOT / "config" / "firms_config.json"
FIRMS_ARCHIVE = "https://firms.modaps.eosdis.nasa.gov/data/country/{s}/{y}/{s}_{y}_United_States.csv"
FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api"
# our file name -> (archive folder name, API source prefix, first year with data)
SENSORS = {"modis": ("modis", "MODIS", 2000), "viirs-snpp": ("viirs-snpp", "VIIRS_SNPP", 2012),
           "viirs-noaa20": ("viirs-jpss1", "VIIRS_NOAA20", 2018)}
API_DAYS = 5          # the area API's maximum day range per request
ENVELOPE_BUFFER_DEG = 0.25
WFIGS_FIELDS = ("OBJECTID,attr_UniqueFireIdentifier,attr_IrwinID,attr_IncidentName,attr_FireDiscoveryDateTime,"
                "attr_ContainmentDateTime,attr_FireOutDateTime,poly_GISAcres,attr_IncidentSize,"
                "attr_IncidentTypeCategory,attr_POOPredictiveServiceAreaID,poly_FeatureCategory,"
                "poly_DateCurrent,attr_CpxName")
HTTP_HEADERS = {"User-Agent": "Fire-Analyst-Toolkit satellite_growth_data.py"}


def source_dir(settings):
    return ROOT / settings["satellite_growth_dir"] / "source"


def envelope(settings):
    import geopandas as gpd
    b = gpd.read_file(ROOT / settings["fire_occurrence_psa_shapefile"]).to_crs(4326).total_bounds
    return (b[0] - ENVELOPE_BUFFER_DEG, b[1] - ENVELOPE_BUFFER_DEG, b[2] + ENVELOPE_BUFFER_DEG, b[3] + ENVELOPE_BUFFER_DEG)


def map_key():
    if not FIRMS_CONFIG.exists():
        return None
    try:
        return (json.loads(FIRMS_CONFIG.read_text()).get("map_key") or "").strip() or None
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: can't read {FIRMS_CONFIG.relative_to(ROOT)}: {e}")
        return None


def _get(url, tries=4, **kw):
    for i in range(tries):
        try:
            r = requests.get(url, headers=HTTP_HEADERS, timeout=kw.pop("timeout", 120), **kw)
            if r.status_code in (200, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"request failed after {tries} tries: {url.split('?')[0].split('/api/')[0]}")


# ---- FIRMS -----------------------------------------------------------------------------------

def _write_rows(out, header, rows):
    tmp = out.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    tmp.replace(out)


def firms_archive_year(sensor, year, env, out):
    """Stream the yearly US file, keep the envelope. -> detections kept, or None if not published."""
    folder = SENSORS[sensor][0]
    x0, y0, x1, y1 = env
    with requests.get(FIRMS_ARCHIVE.format(s=folder, y=year), headers=HTTP_HEADERS, stream=True, timeout=300) as r:
        if r.status_code == 404:
            return None
        r.raise_for_status()
        rd = csv.reader(ln.decode("utf-8") for ln in r.iter_lines() if ln)
        header = next(rd)
        ilat, ilon = header.index("latitude"), header.index("longitude")
        rows = [row for row in rd if x0 <= float(row[ilon]) <= x1 and y0 <= float(row[ilat]) <= y1]
    _write_rows(out, header, rows)
    return len(rows)


def firms_api_availability(key):
    r = _get(f"{FIRMS_API}/data_availability/csv/{key}/all", timeout=60)
    return {row["data_id"]: dt.date.fromisoformat(row["max_date"]) for row in csv.DictReader(io.StringIO(r.text))}


def firms_api_year(key, avail, sensor, year, env, out):
    """Pull a year (to today) through the area API in 5-day requests, SP where available, NRT after."""
    prefix = SENSORS[sensor][1]
    bbox = ",".join(f"{v:.3f}" for v in env)
    day, end = dt.date(year, 1, 1), min(dt.date(year, 12, 31), dt.date.today())
    header, rows = None, []
    while day <= end:
        n = min(API_DAYS, (end - day).days + 1)
        sp_end = avail.get(f"{prefix}_SP")
        src = f"{prefix}_SP" if sp_end and day + dt.timedelta(days=n - 1) <= sp_end else f"{prefix}_NRT"
        text = _get(f"{FIRMS_API}/area/csv/{key}/{src}/{bbox}/{n}/{day}").text
        if text.startswith("Invalid"):
            raise RuntimeError(f"FIRMS API refused {src} {day}: {text[:120]}")
        rd = csv.reader(io.StringIO(text))
        h = next(rd, None)
        if h:
            header = header or h
            idx = [h.index(c) if c in h else None for c in header]
            rows += [[row[i] if i is not None else "" for i in idx] for row in rd if row]
        day += dt.timedelta(days=n)
    if header is None:
        header = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "type"]
    _write_rows(out, header, rows)
    return len(rows)


def fetch_firms(settings, years, refresh=False):
    out_dir = source_dir(settings) / "firms"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = envelope(settings)
    this_year = dt.date.today().year
    key, avail = map_key(), None
    for year in years:
        for sensor, (_, _, first) in SENSORS.items():
            if year < first:
                continue
            out = out_dir / f"{sensor}_{year}.csv"
            # the current year is never final -> always re-pulled; past years only when missing/--refresh
            if out.exists() and not refresh and year < this_year:
                continue
            n = firms_archive_year(sensor, year, env, out) if year < this_year else None
            how = "archive"
            if n is None:
                if not key:
                    print(f"  {sensor} {year}: not in the FIRMS archive and no MAP_KEY in "
                          f"{FIRMS_CONFIG.relative_to(ROOT)} -- skipped")
                    continue
                avail = avail or firms_api_availability(key)
                n, how = firms_api_year(key, avail, sensor, year, env, out), "API"
            print(f"  {sensor} {year}: {n:,} detections ({how})", flush=True)


# ---- MTBS + WFIGS ----------------------------------------------------------------------------

def fetch_mtbs(settings, refresh=False):
    import geopandas as gpd
    out_dir = source_dir(settings) / "mtbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mtbs_perims.gpkg"
    url = settings["satellite_growth_mtbs_url"]
    meta_path = out_dir / "mtbs_source.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if out.exists() and not refresh:
        return
    head = requests.head(url, headers=HTTP_HEADERS, timeout=60, allow_redirects=True)
    modified = head.headers.get("Last-Modified", "")
    if out.exists() and modified and modified == meta.get("last_modified"):
        print(f"  MTBS: unchanged since {modified}")
        return
    print(f"  MTBS: downloading the national perimeter composite (~390 MB)...", flush=True)
    zpath = out_dir / "mtbs_perimeter_data.zip"
    with requests.get(url, headers=HTTP_HEADERS, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        shp = next(n for n in z.namelist() if n.endswith(".shp"))
        stem = shp[:-4]
        for n in z.namelist():
            if n.startswith(stem):
                z.extract(n, out_dir / "extract")
    g = gpd.read_file(out_dir / "extract" / shp, bbox=envelope(settings))
    g["geometry"] = g.geometry.make_valid()
    g.to_file(out, driver="GPKG")
    meta_path.write_text(json.dumps({"url": url, "last_modified": modified, "features": len(g),
                                     "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, indent=2))
    for p in sorted((out_dir / "extract").rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    (out_dir / "extract").rmdir()
    zpath.unlink()
    print(f"  MTBS: {len(g):,} perimeters in the envelope ({modified or 'no Last-Modified'})")


def fetch_wfigs(settings):
    import geopandas as gpd
    import pandas as pd
    out_dir = source_dir(settings) / "wfigs"
    out_dir.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = envelope(settings)
    url = settings["satellite_growth_wfigs_url"].rstrip("/") + "/query"
    frames, offset = [], 0
    while True:
        r = _get(url, params={"where": "1=1", "geometry": f"{x0},{y0},{x1},{y1}", "geometryType": "esriGeometryEnvelope",
                              "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": WFIGS_FIELDS,
                              "returnGeometry": "true", "outSR": 4326, "resultOffset": offset,
                              "resultRecordCount": 500, "f": "geojson"}, timeout=300)
        feats = r.json().get("features", [])
        if not feats:
            break
        frames.append(gpd.GeoDataFrame.from_features(feats, crs=4326))
        offset += len(feats)
    if not frames:
        raise RuntimeError("WFIGS returned no perimeters")
    g = pd.concat(frames, ignore_index=True).drop_duplicates("OBJECTID")
    for c in ("attr_FireDiscoveryDateTime", "attr_ContainmentDateTime", "attr_FireOutDateTime", "poly_DateCurrent"):
        g[c] = (pd.to_datetime(g[c], unit="ms", utc=True, errors="coerce")
                .dt.tz_convert(settings["fire_occurrence_timezone"]).dt.tz_localize(None))
    g = gpd.GeoDataFrame(g, crs=4326)
    g["geometry"] = g.geometry.make_valid()
    tmp = out_dir / "wfigs_perimeters.part.gpkg"
    g.to_file(tmp, driver="GPKG")
    tmp.replace(out_dir / "wfigs_perimeters.gpkg")
    print(f"  WFIGS: {len(g):,} perimeters")


def main():
    settings = pipeline_settings.load_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="+", help="FIRMS years (default satellite_growth_first_year..this year)")
    ap.add_argument("--current-year", action="store_true", help="nightly: this year's FIRMS + WFIGS only")
    ap.add_argument("--refresh", action="store_true", help="re-download FIRMS years that already exist; re-check MTBS")
    args = ap.parse_args()
    this_year = dt.date.today().year
    if args.current_year:
        years = [this_year]
    else:
        years = args.years or list(range(settings["satellite_growth_first_year"], this_year + 1))
    print(f"FIRMS detections {years[0]}-{years[-1]} (envelope of {settings['fire_occurrence_psa_shapefile']}):")
    fetch_firms(settings, years, refresh=args.refresh)
    if not args.current_year:
        fetch_mtbs(settings, refresh=args.refresh)
    fetch_wfigs(settings)


if __name__ == "__main__":
    main()
