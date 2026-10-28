"""Historical + current-year wildfire occurrence for a GACC's PSAs/FDRAs -- our own replacement
for the mpanunto/FDRA_FODR_FOD ArcGIS Pro toolbox (added 2026-09-24, analyst request).

WHY OUR OWN TOOL: FDRA_FODR_FOD's script is embedded encrypted in its .tbx. Its rules were
reverse-engineered from one run's inputs + output (data/historic_fires_research/HXFires/; an
independent re-implementation reproduced all 123,933 FDRA-assigned records and every FF+ cause
code -- see docs/data-sources.md "FDRA_FODR_FOD reverse-engineered"). This module applies the
same rules, minus two problems found along the way:
  - the toolbox silently drops the LAST DAY of every date range (end date treated as 00:00);
  - it collapses the NWCG cause into FireFamily Plus's 9-code statistical cause. Both FF+ codes
    2 and 9 are many-to-one, so the detail was lost. Here the full NWCG Cause Classification /
    General Cause / Specific Cause are kept AND the FF+ code is still produced for FF+ analysis.

SOURCES (both public; citations in docs/data-sources.md and on the fdra-charts viewer page):
  - FPA FOD (Short, Karen C.; USFS Research Data Archive RDS-2013-0009.x). The NEWEST edition is
    used and auto-detected: every run probes the RDS catalog for edition N+1 and, when one is
    published, downloads its FileGDB and rebuilds the FOD block. FOD covers 1992 through its
    last FIRE_YEAR (7th Edition, FPA_FOD_20260615: 1992-2024).
  - InFORM FODR (NIFC, live ArcGIS feature service): everything AFTER FOD's last year, re-pulled
    in full every run (a few thousand records in ONCC's PSA envelope; 4 requests of 2,000).
    A full re-pull, rather than "modified since", is what makes the nightly run correct as
    well as appending: new fires are added, edited records update in place, and records NIFC
    removes/quarantines are deactivated.

RULES (verified against the toolbox's own output, 2026-09-24):
  - FOD discovery date/time are taken as recorded, no timezone conversion (HHMM; blank ->
    00:00). FODR FireDiscoveryDateTime is UTC -> converted to fire_occurrence_timezone
    (US/Pacific for ONCC, DST-aware).
  - FODR query: IncidentTypeCategory='WF' AND IncidentSize IS NOT NULL AND FireCauseID IS NOT NULL.
  - FDRA/PSA = point-in-polygon against fire_occurrence_psa_shapefile (fires outside every
    polygon are dropped).
  - FF+ statistical cause from the NWCG general cause per config/nwcg_general_cause.csv's
    FFP_StatisticalCause column; general cause blank -> Natural class = 1, anything else = 9.

STORE: data/fire_occurrence/<gacc>_fire_occurrence.sqlite. `fires` table, one row per
(source, source_id), with a PERMANENT fire_number assigned the first time a fire is seen and
never reused -- FireFamily Plus requires a unique Fire Number and silently drops duplicates
(NWCG "Defining Custom Agencies and Importing Custom Fire Occurrence"). FOD_ID is stable across
FOD editions, so a FOD fire keeps its number through an edition upgrade. A FODR fire whose
year a new FOD edition later covers is deactivated in favor of the FOD record (which gets its
own number); that is the only way a number can go away.

OUTPUTS (data/fire_occurrence/, active fires only; config/oncc_fire_events.csv is NOT touched):
  <gacc>_fire_occurrence.csv      -- detailed. A superset of config/oncc_fire_events.csv's
                                     columns (GACC, FDRA_Name, FireDiscoveryDate, FireNumber,
                                     TotalAcres, FireCauseFireFamilyPlusID, FireName), so
                                     FDRACharts.py / the web viewer could point at it later
                                     with no code change, plus the NWCG cause fields etc.
  <gacc>_fire_occurrence_ffp.csv  -- FireFamily Plus custom fire import, NWCG's documented
                                     column order: Discovery Date (MM/DD/YYYY), Total Acres,
                                     Statistical Cause, Fire Name, Fire Number, Latitude,
                                     Longitude, FDRA (= FF+ "Region ID").
  <gacc>_fire_occurrence_sources.json -- provenance (FOD edition + its verbatim RDS citation,
                                     FODR service + last pull), read by fdra_charts_web_export.py.
"""
import argparse
import csv
import datetime as dt
import html
import json
import pathlib
import re
import sqlite3
import time
import zoneinfo
import zipfile

import requests

import pipeline_settings

ROOT = pipeline_settings.ROOT
NWCG_GENERAL_CAUSE_CSV = ROOT / "config" / "nwcg_general_cause.csv"
FIRE_SIZE_CLASS_CSV = ROOT / "config" / "fire_size_class.csv"

RDS_CATALOG = "https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.{n}"
RDS_GDB_ZIP = "https://www.fs.usda.gov/rds/archive/products/RDS-2013-0009.{n}/RDS-2013-0009.{n}_Data_Format2_GDB.zip"
FODR_QUERY = ("IncidentTypeCategory = 'WF' AND IncidentSize IS NOT NULL AND FireCauseID IS NOT NULL")
FODR_FIELDS = ["UniqueFireIdentifier", "IncidentName", "FireDiscoveryDateTime", "IncidentSize",
               "FireCauseID", "FireCauseGeneralID", "FireCauseSpecificID", "FireCauseSpecificDetailID", "POOState",
               "POOProtectingUnit", "LocalIncidentIdentifier"]
FODR_PAGE = 2000  # the service's maxRecordCount
HTTP_HEADERS = {"User-Agent": "Fire-Analyst-Toolkit fire_occurrence.py (ONCC)"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS fires (
    fire_number      INTEGER PRIMARY KEY,
    source           TEXT NOT NULL,          -- 'FOD' | 'FODR'
    source_id        TEXT NOT NULL,          -- FOD_ID | UniqueFireIdentifier
    fire_name        TEXT,
    discovery_local  TEXT,                   -- 'YYYY-MM-DD HH:MM' local (see module docstring)
    acres            REAL,
    cause_class      TEXT,                   -- NWCG Cause Classification
    general_cause    TEXT,                   -- NWCG General Cause (PMS 910 label)
    general_cause_id INTEGER,                -- InFORM FireCauseGeneralDomain id
    specific_cause   TEXT,                   -- NWCG Specific Cause (FODR only)
    specific_cause_detail TEXT,              -- NWCG Specific Cause Detail (FODR only)
    ffp_cause        INTEGER,                -- FireFamily Plus statistical cause 1-9
    fdra             TEXT,
    latitude         REAL,
    longitude        REAL,
    poo_state        TEXT,
    reporting_unit   TEXT,
    local_incident_id TEXT,
    source_edition   TEXT,                   -- e.g. FPA_FOD_20260615 | InFORM FODR (live)
    active           INTEGER NOT NULL DEFAULT 1,
    inactive_reason  TEXT,
    first_seen_utc   TEXT,
    last_seen_utc    TEXT,
    UNIQUE (source, source_id)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS runs (
    run_utc TEXT, step TEXT, fetched INTEGER, added INTEGER, updated INTEGER,
    deactivated INTEGER, note TEXT
);
"""

DATA_COLUMNS = ["fire_name", "discovery_local", "acres", "cause_class", "general_cause", "general_cause_id",
                "specific_cause", "specific_cause_detail", "ffp_cause", "fdra", "latitude", "longitude", "poo_state", "reporting_unit",
                "local_incident_id", "source_edition"]


def _now_utc():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _http_get(url, params=None, tries=4, **kw):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=kw.get("timeout", 120),
                             stream=kw.get("stream", False))
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return r
            msg = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            msg = str(e)
        if attempt < tries - 1:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {tries} tries: {msg}")


# ---- cause handling ---------------------------------------------------------------------------
def _txt(value):
    """Text field from pandas/pyogrio -> stripped str, or None for None/NaN/blank."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    value = str(value).strip()
    return value or None


def _norm_label(text):
    """'Power generation / transmission / distribution' and 'Power generation/transmission/
    distribution' (FODR vs FOD wording) -> one comparable key."""
    return re.sub(r"\s*/\s*", "/", (_txt(text) or "").lower())


def load_nwcg_general_causes(path=NWCG_GENERAL_CAUSE_CSV):
    """config/nwcg_general_cause.csv -> (by_id, by_fod_label), each value a row dict."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_id = {int(r["FireCauseGeneralID"]): r for r in rows}
    by_label = {}
    for r in rows:
        by_label[_norm_label(r["FPA_FOD_Label"])] = r
        by_label[_norm_label(r["GeneralCause"])] = r
    return by_id, by_label


def ffp_cause_for(general_row, cause_class):
    """FF+ statistical cause: the general cause's crosswalk; no general cause -> Natural 1 else 9
    (the toolbox's observed rule for FODR records with a blank FireCauseGeneralID)."""
    if general_row is not None:
        return int(general_row["FFP_StatisticalCause"])
    return 1 if cause_class == "Natural" else 9


# ---- geometry --------------------------------------------------------------------------------
def load_psas(shapefile, fdra_field):
    import geopandas as gpd
    psa = gpd.read_file(shapefile)
    if fdra_field not in psa.columns:
        raise SystemExit(f"{shapefile} has no '{fdra_field}' field (fields: {list(psa.columns)})")
    return psa[[fdra_field, "geometry"]].rename(columns={fdra_field: "fdra"}).to_crs(4269)


def assign_fdra(points_gdf, psa):
    """Point-in-polygon; points outside every PSA are dropped (the toolbox's HXFires run kept
    them with a blank FDRA only because its boundary layer had an extra unnamed polygon)."""
    import geopandas as gpd
    joined = gpd.sjoin(points_gdf.to_crs(4269), psa, how="inner", predicate="intersects")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined.drop(columns="index_right")


# ---- FPA FOD ---------------------------------------------------------------------------------
def fod_citation(n):
    """The edition's own "How to Cite" text from its RDS catalog page (verbatim), or None if the
    page can't be read. Stored in the store's meta once per edition."""
    try:
        r = _http_get(RDS_CATALOG.format(n=n), tries=2, timeout=60)
    except RuntimeError:
        return None
    if r.status_code != 200:
        return None
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", r.text, flags=re.S)
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text)))
    m = re.search(r"please use the following citation:\s*(.+?https://doi\.org/\S+)", text)
    return m.group(1).strip() if m else None


def latest_fod_edition(known_n):
    """Probe the RDS catalog upward from edition `known_n` (e.g. 7 = RDS-2013-0009.7) and return
    the highest one whose page lists a FileGDB product. Network failure -> known_n (soft)."""
    n = known_n
    while True:
        try:
            r = _http_get(RDS_CATALOG.format(n=n + 1), tries=2, timeout=60)
        except RuntimeError as e:
            print(f"  FOD edition check failed ({e}); staying on RDS-2013-0009.{n}")
            return n
        if r.status_code != 200 or f"RDS-2013-0009.{n + 1}_Data_Format2_GDB.zip" not in r.text:
            return n
        n += 1


def ensure_fod_gdb(source_dir, n):
    """Download + extract RDS-2013-0009.<n>'s FileGDB into source_dir if not already there.
    Returns the .gdb path."""
    existing = sorted(source_dir.glob(f"RDS-2013-0009.{n}_extracted/**/*.gdb"))
    if existing:
        return existing[0]
    zip_path = source_dir / f"RDS-2013-0009.{n}_Data_Format2_GDB.zip"
    if not zip_path.exists():
        url = RDS_GDB_ZIP.format(n=n)
        print(f"  Downloading FPA FOD edition RDS-2013-0009.{n} ({url}) ...")
        r = _http_get(url, stream=True, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f"FOD download failed: HTTP {r.status_code}")
        tmp = zip_path.with_suffix(".zip.part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(zip_path)
    out = source_dir / f"RDS-2013-0009.{n}_extracted"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    found = sorted(out.glob("**/*.gdb"))
    if not found:
        raise RuntimeError(f"No .gdb inside {zip_path}")
    return found[0]


def read_fod(gdb, psa, start_date, by_label):
    """FOD 'Fires' layer -> list of record dicts for fires inside the PSAs from start_date on.
    Returns (records, edition_tag, last_year)."""
    import pandas as pd
    import pyogrio
    layers = [name for name, _ in pyogrio.list_layers(gdb)]
    layer = "Fires" if "Fires" in layers else layers[0]
    cols = ["FOD_ID", "FIRE_NAME", "FIRE_YEAR", "DISCOVERY_DATE", "DISCOVERY_TIME", "NWCG_CAUSE_CLASSIFICATION",
            "NWCG_GENERAL_CAUSE", "FIRE_SIZE", "STATE", "NWCG_REPORTING_UNIT_ID", "LOCAL_INCIDENT_ID"]
    gdf = pyogrio.read_dataframe(gdb, layer=layer, columns=cols, bbox=tuple(psa.total_bounds))
    # Edition coverage end = the national max FIRE_YEAR (not just this GACC's, which could lag).
    last_year = int(pyogrio.read_dataframe(gdb, layer=layer, columns=["FIRE_YEAR"],
                                           read_geometry=False).FIRE_YEAR.max())
    gdf = assign_fdra(gdf, psa)
    t = gdf.DISCOVERY_TIME.fillna("").astype(str).str.strip()
    t = t.where(t.str.fullmatch(r"\d{4}"), "0000")
    day = pd.to_datetime(gdf.DISCOVERY_DATE).dt.strftime("%Y-%m-%d")
    gdf["discovery_local"] = day + " " + t.str[:2] + ":" + t.str[2:]
    gdf = gdf[gdf.discovery_local >= start_date]
    edition = pathlib.Path(gdb).stem
    recs = []
    for row in gdf.itertuples(index=False):
        cls = _txt(row.NWCG_CAUSE_CLASSIFICATION)
        cls = "Undetermined" if (cls or "").lower().startswith("missing") else cls
        gen = by_label.get(_norm_label(row.NWCG_GENERAL_CAUSE))
        pt = row.geometry
        recs.append({
            "source": "FOD", "source_id": str(int(row.FOD_ID)), "fire_name": _txt(row.FIRE_NAME),
            "discovery_local": row.discovery_local, "acres": float(row.FIRE_SIZE) if pd.notna(row.FIRE_SIZE) else None,
            "cause_class": cls, "general_cause": gen["GeneralCause"] if gen else None,
            "general_cause_id": int(gen["FireCauseGeneralID"]) if gen else None, "specific_cause": None,
            "specific_cause_detail": None,
            "ffp_cause": ffp_cause_for(gen, cls), "fdra": row.fdra, "latitude": round(pt.y, 6),
            "longitude": round(pt.x, 6), "poo_state": _txt(row.STATE),
            "reporting_unit": _txt(row.NWCG_REPORTING_UNIT_ID), "local_incident_id": _txt(row.LOCAL_INCIDENT_ID), "source_edition": edition})
    return recs, edition, last_year


# ---- InFORM FODR -----------------------------------------------------------------------------
def fodr_domains(url):
    """Coded-value domains of the live layer ({field: {code: label}}) -- read from the service
    each run rather than hardcoded, so NIFC domain edits flow through."""
    meta = _http_get(url, params={"f": "json"}).json()
    return {f["name"]: {cv["code"]: cv["name"] for cv in (f.get("domain") or {}).get("codedValues", [])}
            for f in meta.get("fields", []) if (f.get("domain") or {}).get("codedValues")}


def fetch_fodr(url, psa, start_utc):
    """Every FODR record matching the toolbox's query, discovered at/after start_utc, inside the
    PSA envelope (the point-in-polygon test happens afterwards). Paged by resultOffset."""
    b = psa.total_bounds
    where = f"{FODR_QUERY} AND FireDiscoveryDateTime >= TIMESTAMP '{start_utc:%Y-%m-%d %H:%M:%S}'"
    feats, offset = [], 0
    while True:
        params = {"where": where, "geometry": f"{b[0]},{b[1]},{b[2]},{b[3]}", "geometryType": "esriGeometryEnvelope",
                  "inSR": 4269, "spatialRel": "esriSpatialRelIntersects", "outFields": ",".join(FODR_FIELDS),
                  "outSR": 4269, "orderByFields": "OBJECTID", "resultOffset": offset,
                  "resultRecordCount": FODR_PAGE, "f": "json"}
        page = _http_get(f"{url}/query", params=params).json()
        if "error" in page:
            raise RuntimeError(f"FODR query error: {page['error']}")
        batch = page.get("features", [])
        feats.extend(batch)
        if not page.get("exceededTransferLimit") and len(batch) < FODR_PAGE:
            break
        offset += len(batch)
        if not batch:
            break
    return feats


def fodr_records(feats, psa, domains, by_id, tz):
    import geopandas as gpd
    import pandas as pd
    rows = [dict(f["attributes"], x=f["geometry"]["x"], y=f["geometry"]["y"]) for f in feats if f.get("geometry")]
    if not rows:
        return []
    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=4269)
    gdf = assign_fdra(gdf, psa)
    gdf["discovery_local"] = (pd.to_datetime(gdf.FireDiscoveryDateTime, unit="ms", utc=True)
                              .dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M"))
    cls_dom, spec_dom = domains.get("FireCauseID", {}), domains.get("FireCauseSpecificID", {})
    detail_dom = domains.get("FireCauseSpecificDetailID", {})
    recs = []
    for row in gdf.itertuples(index=False):
        cls = cls_dom.get(row.FireCauseID)
        gen = by_id.get(int(row.FireCauseGeneralID)) if pd.notna(row.FireCauseGeneralID) else None
        recs.append({
            "source": "FODR", "source_id": row.UniqueFireIdentifier, "fire_name": _txt(row.IncidentName),
            "discovery_local": row.discovery_local, "acres": float(row.IncidentSize), "cause_class": cls,
            "general_cause": gen["GeneralCause"] if gen else None,
            "general_cause_id": int(gen["FireCauseGeneralID"]) if gen else None,
            "specific_cause": spec_dom.get(row.FireCauseSpecificID) if pd.notna(row.FireCauseSpecificID) else None,
            "specific_cause_detail": (detail_dom.get(row.FireCauseSpecificDetailID)
                                      if pd.notna(row.FireCauseSpecificDetailID) else None),
            "ffp_cause": ffp_cause_for(gen, cls), "fdra": row.fdra, "latitude": round(row.y, 6),
            "longitude": round(row.x, 6), "poo_state": _txt(row.POOState),
            "reporting_unit": _txt(row.POOProtectingUnit), "local_incident_id": _txt(row.LocalIncidentIdentifier), "source_edition": "InFORM FODR (live)"})
    return recs


# ---- store -----------------------------------------------------------------------------------
def open_store(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    # Columns added after a store was first built (CREATE TABLE IF NOT EXISTS won't add them).
    have = {r[1] for r in con.execute("PRAGMA table_info(fires)")}
    if "specific_cause_detail" not in have:  # added 2026-09-24
        con.execute("ALTER TABLE fires ADD COLUMN specific_cause_detail TEXT")
    return con


def meta_get(con, key, default=None):
    row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def meta_set(con, key, value):
    con.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)))


def upsert(con, records, now):
    """Insert new (source, source_id) rows with the next fire_number; update changed ones.
    Returns (added, updated, seen_keys)."""
    existing = {(s, i): (n, a, *rest) for s, i, n, a, *rest in
                con.execute(f"SELECT source, source_id, fire_number, active, {', '.join(DATA_COLUMNS)} FROM fires")}
    next_no = (con.execute("SELECT MAX(fire_number) FROM fires").fetchone()[0] or 0) + 1
    added = updated = 0
    seen = set()
    records = sorted(records, key=lambda r: (r["discovery_local"] or "", r["source"], r["source_id"]))
    for r in records:
        key = (r["source"], r["source_id"])
        if key in seen:
            continue
        seen.add(key)
        values = [r[c] for c in DATA_COLUMNS]
        if key not in existing:
            con.execute(f"INSERT INTO fires (fire_number, source, source_id, {', '.join(DATA_COLUMNS)}, active, "
                        f"first_seen_utc, last_seen_utc) VALUES ({', '.join('?' * (len(DATA_COLUMNS) + 6))})",
                        [next_no, *key, *values, 1, now, now])
            next_no += 1
            added += 1
        else:
            number, active, *old = existing[key]
            changed = list(old) != values or not active
            con.execute(f"UPDATE fires SET {', '.join(c + ' = ?' for c in DATA_COLUMNS)}, active = 1, "
                        "inactive_reason = NULL, last_seen_utc = ? WHERE fire_number = ?", [*values, now, number])
            updated += int(changed)
    return added, updated, seen


def deactivate_missing(con, source, seen, reason, extra_where="", params=()):
    """Active rows of `source` (optionally narrowed) not in `seen` -> inactive. Returns count."""
    rows = con.execute(f"SELECT fire_number, source_id FROM fires WHERE source = ? AND active = 1 {extra_where}",
                       (source, *params)).fetchall()
    gone = [n for n, sid in rows if (source, sid) not in seen]
    con.executemany("UPDATE fires SET active = 0, inactive_reason = ? WHERE fire_number = ?",
                    [(reason, n) for n in gone])
    return len(gone)


# ---- exports ---------------------------------------------------------------------------------
def _size_class(acres, classes):
    if acres is None:
        return ""
    for c in classes:
        if float(c["min_acres"]) <= acres < float(c["max_acres"]):
            return c["fire_size_class"]
    return ""


def export_sources(con, out_dir, gacc, settings):
    """<gacc>_fire_occurrence_sources.json -- provenance for whoever displays these fires
    (scripts/fdra_charts_web_export.py cites from it): FOD edition + its verbatim citation, the
    FODR service and last pull time. Written every run so it always matches the CSVs."""
    n = int(meta_get(con, "fod_edition_n"))
    if not meta_get(con, "fod_citation"):
        cite = fod_citation(n)
        if cite:
            meta_set(con, "fod_citation", cite)
            con.commit()
    payload = {
        "tool": "scripts/fire_occurrence.py",
        "fod_edition": meta_get(con, "fod_edition"), "fod_edition_n": n,
        "fod_last_year": int(meta_get(con, "fod_last_year")),
        "fod_citation": meta_get(con, "fod_citation"),
        "fod_doi": f"https://doi.org/10.2737/RDS-2013-0009.{n}",
        "fodr_url": settings["fire_occurrence_fodr_url"],
        "fodr_last_pull_utc": meta_get(con, "fodr_last_pull_utc"),
    }
    path = out_dir / f"{gacc.lower()}_fire_occurrence_sources.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_csvs(con, out_dir, gacc):
    with open(FIRE_SIZE_CLASS_CSV, newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        size_classes = list(csv.DictReader(f, dialect=csv.Sniffer().sniff(sample, delimiters=",\t;")))
    rows = con.execute(f"SELECT fire_number, source, source_id, {', '.join(DATA_COLUMNS)} FROM fires "
                       "WHERE active = 1 AND fdra IS NOT NULL ORDER BY discovery_local, fire_number").fetchall()
    detail_path = out_dir / f"{gacc.lower()}_fire_occurrence.csv"
    ffp_path = out_dir / f"{gacc.lower()}_fire_occurrence_ffp.csv"
    detail_cols = ["GACC", "FDRA_Name", "FireDiscoveryDate", "FireNumber", "TotalAcres", "FireCauseFireFamilyPlusID",
                   "FireName", "FireDiscoveryDateTime", "FireSizeClass", "FireCauseClassification", "FireCauseGeneral",
                   "FireCauseGeneralID", "FireCauseSpecific", "FireCauseSpecificDetail", "Latitude", "Longitude", "POOState", "ReportingUnit",
                   "LocalIncidentID", "Source", "SourceID", "SourceEdition"]
    ffp_cols = ["Discovery Date", "Total Acres", "Statistical Cause", "Fire Name", "Fire Number", "Latitude",
                "Longitude", "FDRA"]
    tmp_d, tmp_f = detail_path.with_suffix(".csv.tmp"), ffp_path.with_suffix(".csv.tmp")
    with open(tmp_d, "w", newline="", encoding="utf-8") as fd, open(tmp_f, "w", newline="", encoding="utf-8") as ff:
        wd, wf = csv.writer(fd), csv.writer(ff)
        wd.writerow(detail_cols)
        wf.writerow(ffp_cols)
        for (number, source, sid, name, disc, acres, cls, gen, gen_id, spec, spec_detail, ffp, fdra, lat, lon,
             state, unit, local_id, edition) in rows:
            date_mdy = dt.datetime.strptime(disc[:10], "%Y-%m-%d").strftime("%m/%d/%Y")
            wd.writerow([gacc, fdra, date_mdy, number, acres, ffp, name, disc, _size_class(acres, size_classes),
                         cls, gen, gen_id, spec, spec_detail, lat, lon, state, unit, local_id, source, sid, edition])
            # FF+ rejects records with missing required data; a blank name gets a placeholder.
            wf.writerow([date_mdy, acres, ffp, name or "UNNAMED", number, lat, lon, fdra])
    tmp_d.replace(detail_path)
    tmp_f.replace(ffp_path)
    return detail_path, ffp_path, len(rows)


# ---- orchestration ---------------------------------------------------------------------------
def run(settings, rebuild_fod=False, skip_fodr=False):
    gacc = settings["fire_occurrence_gacc"]
    out_dir = ROOT / settings["fire_occurrence_dir"]
    source_dir = out_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    con = open_store(out_dir / f"{gacc.lower()}_fire_occurrence.sqlite")
    now = _now_utc()
    tz = settings["fire_occurrence_timezone"]
    psa = load_psas(ROOT / settings["fire_occurrence_psa_shapefile"], settings["fire_occurrence_fdra_field"])
    by_id, by_label = load_nwcg_general_causes()

    # 1. FOD: newest edition; rebuild the FOD block only when the edition changes (or forced).
    current_n = int(meta_get(con, "fod_edition_n", settings["fire_occurrence_fod_min_edition"]))
    newest_n = latest_fod_edition(current_n) if settings["fire_occurrence_fod_auto_update"] else current_n
    have_fod = meta_get(con, "fod_edition_n") is not None
    if rebuild_fod or not have_fod or newest_n != current_n:
        print(f"Building FOD block from RDS-2013-0009.{newest_n} ...")
        gdb = ensure_fod_gdb(source_dir, newest_n)
        recs, edition, last_year = read_fod(gdb, psa, settings["fire_occurrence_start_date"], by_label)
        added, updated, seen = upsert(con, recs, now)
        gone = deactivate_missing(con, "FOD", seen, f"not in {edition}")
        meta_set(con, "fod_edition_n", newest_n)
        meta_set(con, "fod_edition", edition)
        meta_set(con, "fod_last_year", last_year)
        meta_set(con, "fod_citation", fod_citation(newest_n) or "")
        con.execute("INSERT INTO runs VALUES (?, 'FOD', ?, ?, ?, ?, ?)", (now, len(recs), added, updated, gone, edition))
        con.commit()
        print(f"  FOD {edition} (through {last_year}): {len(recs)} fires in PSAs; +{added} new, "
              f"{updated} changed, {gone} deactivated")
    last_year = int(meta_get(con, "fod_last_year"))

    # A FODR fire in a year the FOD edition now covers is superseded by FOD.
    cutoff_local = f"{last_year + 1}-01-01 00:00"
    superseded = con.execute("UPDATE fires SET active = 0, inactive_reason = 'superseded by FPA FOD' "
                             "WHERE source = 'FODR' AND active = 1 AND discovery_local < ?", (cutoff_local,)).rowcount

    # 2. FODR: full re-pull of everything after FOD's last year.
    if not skip_fodr:
        # Local midnight Jan 1 in the FDRA timezone, expressed in UTC for the service query.
        start_utc = dt.datetime(last_year + 1, 1, 1, tzinfo=zoneinfo.ZoneInfo(tz)).astimezone(dt.timezone.utc)
        url = settings["fire_occurrence_fodr_url"]
        print(f"Pulling InFORM FODR from {start_utc:%Y-%m-%d %H:%M} UTC ({url}) ...")
        feats = fetch_fodr(url, psa, start_utc)
        recs = fodr_records(feats, psa, fodr_domains(url), by_id, tz)
        recs = [r for r in recs if r["discovery_local"] >= cutoff_local]
        before = con.execute("SELECT COUNT(*) FROM fires WHERE source = 'FODR' AND active = 1").fetchone()[0]
        # Guard: a failed/empty pull must never wipe the FODR block.
        if before > 50 and len(recs) < 0.5 * before:
            raise SystemExit(f"FODR pull returned {len(recs)} fires vs {before} active -- refusing to apply "
                             "(service problem?). Nothing changed.")
        added, updated, seen = upsert(con, recs, now)
        gone = deactivate_missing(con, "FODR", seen, "no longer returned by InFORM FODR",
                                  "AND discovery_local >= ?", (cutoff_local,))
        con.execute("INSERT INTO runs VALUES (?, 'FODR', ?, ?, ?, ?, ?)",
                    (now, len(feats), added, updated, gone, f"superseded={superseded}"))
        meta_set(con, "fodr_last_pull_utc", now)
        con.commit()
        print(f"  FODR: {len(feats)} fetched, {len(recs)} in PSAs; +{added} new, {updated} changed, "
              f"{gone} deactivated, {superseded} superseded by FOD")

    detail, ffp, n = export_csvs(con, out_dir, gacc)
    export_sources(con, out_dir, gacc, settings)
    con.close()
    print(f"Wrote {n} active fires -> {detail.name}, {ffp.name} ({out_dir})")
    return detail, ffp


def main():
    settings = pipeline_settings.load_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild-fod", action="store_true",
                    help="re-read the FOD edition even if unchanged (e.g. after a PSA boundary edit)")
    ap.add_argument("--skip-fodr", action="store_true", help="don't contact the live InFORM FODR service")
    args = ap.parse_args()
    run(settings, rebuild_fod=args.rebuild_fod, skip_fodr=args.skip_fodr)


if __name__ == "__main__":
    main()
