"""Daily large-fire growth per fire per PSA from satellite active-fire detections -- the satellite
alternative to scripts/imsr_growth_prep.py's IMSR growth days for the interactive charts' growth
overlay (fdra_charts_web_large_fire_growth_source). Added 2026-10-07, ported from the
incoming/VIIRS experiment (2026-10-06, analyst request: "match the VIIRS/MODIS hits that fall
inside the perimeters AND inside the date range ... another set of growth days by fire that more
closely matches the PSA boundaries").

    python scripts/satellite_growth_prep.py               # all years with FIRMS data
    python scripts/satellite_growth_prep.py --years 2026  # re-date just these years, merged into the outputs
    python scripts/satellite_growth_prep.py --nightly     # scheduled task: refresh + re-date this year
                                                          #   (no-op unless the source setting is Satellite)

IDEA: after Sonoma Technology's satellite growth dataset ("Consistent, high-accuracy mapping of
daily and sub-daily wildfire growth with satellite observations", Int. J. Wildland Fire 2023,
https://publish.csiro.au/WF/WF22048; https://sonomatech.com/projects/4741), simplified. They delineated fires from detections; we already have each fire's
final perimeter, so detections only have to DATE the burned area inside it. Inputs come from
scripts/satellite_growth_data.py (FIRMS, MTBS, WFIGS) and scripts/fire_occurrence.py's store.

1. FIRE -> PERIMETER ("events"), from the fire occurrence store (FPA FOD + InFORM FODR):
   a. FPA FOD's MTBS_ID == MTBS event_id (1984-latest MTBS year, >= 1,000 ac West). FOD records a
      complex as its individual starts, several sharing one MTBS_ID -> one event: discovery =
      earliest member's, CONT_DATE = latest member's, PSA of origin / cause = the largest member's.
   b. WFIGS Interagency Fire Perimeters (2020-present) for store fires in no MTBS event: FOD by
      FPA_ID minus its source prefix (INF-2021-CABTU-009205 -> 2021-CABTU-009205), FODR by
      UniqueFireIdentifier; perimeter >= satellite_growth_wfigs_min_acres; skipped when it mostly
      overlaps a SAME-YEAR MTBS event (a member start's ID pointing at the same burn -- reburns of
      older scars are new fires). Where both exist they agree closely (2020-24, 66 fires: median
      WFIGS/MTBS area 0.985, median IoU 0.96). This is how the current season and MTBS-less fires
      (SITES 2024) get perimeters. WFIGS acre attributes can be stale (ELEPHANT 2026 polygon
      16,177 ac vs poly_GISAcres 13,930); the polygon is used.
2. WINDOW: start = discovery - 1 day; MTBS ig_date - 1 when FOD's discovery is more than
   WINDOW_DISAGREE_DAYS off it (flagged; RANCH 2018 carried a June member). End of growth = the run
   of in-perimeter detection days from the start: everything through CONT_DATE + 1, then until a
   gap > satellite_growth_max_gap_days. CONT_DATE is a floor, not a cap -- some FOD CONT_DATEs
   are the discovery date (HONEYDEW 2003).
3. DETECTIONS: VIIRS confidence nominal/high, MODIS >= 30, FIRMS type 0 (vegetation fire) or blank
   (NRT has no type); inside the perimeter + half a pixel's spread (VIIRS 375 m, MODIS 1 km).
4. FIRE DAY = acquisition time in fire_occurrence_timezone minus satellite_growth_fire_day_start_hour
   (6), so the ~01:30 night overpass counts toward the previous afternoon's burning period.
5. DAY OF BURN: the perimeter rasterized at satellite_growth_cell_m (100 m, EPSG:3310). Each cell
   gets the EARLIEST fire day among detections within max(nearest distance, one pixel: 375 m VIIRS,
   1 km MODIS) -- nearest-detection interpolation (Parks 2014) with a pixel-sized tie radius, so
   later re-detections of smoldering interior don't re-date ground that already burned. VIIRS
   wherever a VIIRS hit is within 1.5 km, MODIS otherwise (all of 2001-2011).
6. PSA = the PSA polygon containing the cell center. Daily acres per (fire, PSA) sum to the
   perimeter's gridded area -- so a fire that burns across a PSA line puts each day's acres in
   the PSA they burned in. (IMSR growth can't: it puts a whole fire in one PSA.)

VALIDATION (2026-10-06, 2001-2026, see docs/data-sources.md): 462 of 565 events dated (the rest
pre-2001 or no detections; all rejects <= 6,242 ac); gridded acres / perimeter acres median 1.001;
59 fires burned in more than one PSA, 12.2% of all acres outside the fire's PSA of origin. Against
IMSR daily growth (147 fires): median Spearman 0.60 same day, 0.75 with satellite +1 day -- the
satellite fire day lands a day before IMSR's burn date (report - 1) on most fires. Big fast runs
agree same day (CAMP 11/8/2018); NORTH COMPLEX's Bear run (night of 9/8/2020) is 9/8 here, 9/9
in IMSR. Plausibly 209 acreage is often last night's IR. Unresolved; the overlay shows each
source's own dates. Known limits: cloudy/smoky days with no detections push their growth to the
next detected day (gap_days per fire); MODIS years are coarse (1 km blobs, Voronoi wedges far from
any hit); the earliest-day rule pulls edge cells up to one pixel early.

OUTPUTS (satellite_growth_dir, default data/satellite_growth/):
  <gacc>_satellite_growth_days.csv  one row per fire per PSA per fire day. Carries the columns
                                    fdra_charts_web_export.py's growth overlay reads from the IMSR
                                    file (fire_name, fdra, unit, burn_date, growth_acres, cause_*),
                                    so either file can feed it; acres outside every PSA have
                                    psa=OUTSIDE and blank fdra/unit (never counted). fire_label is
                                    what the overlay shows (fire_label()): MTBS maps some complexes
                                    as one perimeter per main fire (LNU 2020 = HENNESSEY + WALLBRIDGE
                                    + MEYERS), so "HENNESSEY (LNU LIGHTNING COMPLEX)". fire_name is
                                    the perimeter's own name, complex_name FOD/WFIGS's. cause_method
                                    largest_member:<name> = a complex's cause, from its largest start.
  <gacc>_satellite_fires.csv        one row per event: perimeter source + join, dates, detections,
                                    PSAs burned, flags (review file).
  events.gpkg                       event perimeters (EPSG:3310) for scripts/satellite_growth_charts.py
  burnday/<event_id>.tif            day-of-burn grid per event (value = day index + 1; tag day0)
"""
import argparse
import datetime as dt
import pathlib
import re
import sqlite3
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from scipy.spatial import cKDTree

import pipeline_settings
import satellite_growth_data

ROOT = pipeline_settings.ROOT
CRS = 3310                      # California Albers
R_VIIRS, R_MODIS = 375.0, 1000.0  # "same pixel" radius for the earliest-day rule, and perimeter buffer
VIIRS_REACH_M = 1500.0           # a cell uses VIIRS if a VIIRS hit is this close
WINDOW_DISAGREE_DAYS = 3
MAX_SEARCH_DAYS = 180            # how far past the start detections are looked for
K_NEIGHBORS = 24                 # neighbors searched for the earliest-day rule
WFIGS_DUP_SHARE = 0.5
ACRES_PER_M2 = 1 / 4046.8564224
OUTSIDE = "OUTSIDE"
# Analyst cause corrections kept for the IMSR growth days (scripts/imsr_growth_prep.py); also
# applied here, so a fire's cause reads the same on either growth source.
CAUSE_OVERRIDES_CSV = ROOT / "config" / "imsr_fire_fdra_overrides.csv"
CAUSE_FIELDS = ("cause_class", "cause_general", "cause_specific", "cause_detail")


class Cfg:
    def __init__(self, s):
        self.s = s
        self.gacc = s["fire_occurrence_gacc"].lower()
        self.dir = ROOT / s["satellite_growth_dir"]
        self.src = self.dir / "source"
        self.tz = s["fire_occurrence_timezone"]
        self.cell = float(s["satellite_growth_cell_m"])
        self.day_start = int(s["satellite_growth_fire_day_start_hour"])
        self.max_gap = int(s["satellite_growth_max_gap_days"])
        self.wfigs_min = float(s["satellite_growth_wfigs_min_acres"])
        self.store = ROOT / s["fire_occurrence_dir"] / f"{self.gacc}_fire_occurrence.sqlite"
        self.psa_field = s["fire_occurrence_fdra_field"]
        self.psa = gpd.read_file(ROOT / s["fire_occurrence_psa_shapefile"]).to_crs(CRS)[[self.psa_field, "geometry"]]
        self.days_csv = self.dir / f"{self.gacc}_satellite_growth_days.csv"
        self.fires_csv = self.dir / f"{self.gacc}_satellite_fires.csv"
        self.events_gpkg = self.dir / "events.gpkg"
        self.burn_dir = self.dir / "burnday"


def _clean_name(text):
    """Upper-case, odd characters (FOD's stray non-breaking spaces, U+FFFD) -> space, spaces collapsed."""
    t = "".join(ch if ch.isprintable() and ord(ch) < 0x2000 and ch != "\ufffd" else " " for ch in str(text or ""))
    return re.sub(r"\s+", " ", t).strip().upper()


def _name_key(text):
    return re.sub(r"[^A-Z0-9]", "", text)


def fire_label(fire_name, complex_name):
    """What the charts call a fire: the individual fire, with its final complex in parentheses
    (analyst rule 2026-10-08: "if the individual fires made the big run prior to the merge, list
    as such but with the final complex name in parentheses").
      - MTBS often names a complex's perimeters "<COMPLEX> (<FIRE>)" (SODA COMPLEX (BACK)): the
        fire is the part in parentheses, the outer name the complex unless FOD/WFIGS give one.
      - The final complex name is FOD COMPLEX_NAME / WFIGS CpxName, minus FOD's year tags
        ("LIME COMPLEX (2008)", "SOUTH COMPLEX (2015 CA)").
      - A complex's namesake fire is just the complex (BLUE COMPLEX (BLUE) -> BLUE COMPLEX,
        ROUTE COMPLEX, LAKE COMPLEX); punctuation doesn't count (CHINA BACK / CHINA-BACK COMPLEX).
    -> "BACK (SODA COMPLEX)", "HENNESSEY (LNU LIGHTNING COMPLEX)", "DIXIE"."""
    name = _clean_name(fire_name)
    cpx = _clean_name(complex_name) if isinstance(complex_name, str) else ""
    cpx = re.sub(r"\s*\((?:[A-Z]{2}\s*)?\d{4}(?:\s*[A-Z]{2})?\)$", "", cpx).strip()
    m = re.match(r"^(.*\b(?:COMPLEX|COMPLEXES|FIRE)\b)\s*\((.+)\)$", name)
    if m:
        name = m.group(2).strip()
        cpx = cpx or m.group(1).strip()
    if not cpx:
        return name
    n, c = _name_key(name), _name_key(cpx)
    if n == c or c in (n + "COMPLEX", n + "COMPLEXES") or c in n:
        return cpx if len(c) >= len(n) else name
    return f"{name} ({cpx})"


def norm_unit(unit):
    """NWCG unit ID "USCABTU" -> IMSR-style "CA-BTU" (what the overlay's GACC rule reads)."""
    u = (unit or "").upper().strip()
    return f"{u[2:4]}-{u[4:]}" if re.fullmatch(r"US[A-Z]{2}[A-Z0-9]+", u) else u


# ---- 1. events ----------------------------------------------------------------------------------

def load_store(cfg):
    if not cfg.store.exists():
        sys.exit(f"{cfg.store.relative_to(ROOT)} not found -- run scripts/fire_occurrence.py first "
                 f"(reference data step 7)")
    con = sqlite3.connect(cfg.store)
    st = pd.read_sql("SELECT fire_number, source, source_id, fire_name, discovery_local, acres, fdra, cause_class, "
                     "general_cause, specific_cause, specific_cause_detail, reporting_unit FROM fires "
                     "WHERE active = 1", con)
    con.close()
    return st


def fod_extras(cfg, fod):
    gdbs = sorted((ROOT / cfg.s["fire_occurrence_dir"] / "source").glob("RDS-2013-0009.*_extracted/**/*.gdb"))
    if not gdbs:
        sys.exit("FPA FOD FileGDB not found under fire_occurrence_dir/source -- run scripts/fire_occurrence.py")
    bbox = tuple(cfg.psa.to_crs(4326).total_bounds)
    # bbox read: string WHERE clauses on this FileGDB silently return nothing
    g = pyogrio.read_dataframe(gdbs[-1], layer="Fires", bbox=bbox, read_geometry=False,
                               columns=["FOD_ID", "FPA_ID", "MTBS_ID", "COMPLEX_NAME", "CONT_DATE"])
    fod = fod.assign(FOD_ID=fod.source_id.astype("int64")).merge(g, on="FOD_ID", how="left")
    fod["uid"] = fod.FPA_ID.fillna("").str.replace(r"^[A-Z]+-", "", regex=True).str.upper()
    return fod


def _event_row(eid, name, year, members, geom, acres, perim_source, join, cont, complex_name="", irwin=""):
    big = members.sort_values("acres", ascending=False).iloc[0]
    return {"event_id": eid, "fire_name": name, "year": year, "fdra_origin": big.fdra,
            "unit": norm_unit(big.reporting_unit), "cause_class": big.cause_class,
            "cause_general": big.general_cause, "cause_specific": big.specific_cause,
            "cause_detail": big.specific_cause_detail,
            # a complex's cause is its largest start's; the viewer names that start
            "cause_method": "located_fire" if len(members) == 1 else f"largest_member:{big.fire_name}",
            "discovery": min(members.discovery_local)[:10], "cont_date": cont,
            "located_acres": members.acres.sum(), "perimeter_acres": acres, "perim_source": perim_source,
            "join_method": join, "n_members": len(members),
            "fire_numbers": ";".join(str(n) for n in sorted(members.fire_number)),
            "complex_name": complex_name, "irwinid": irwin, "geometry": geom}


def build_events(cfg):
    st = load_store(cfg)
    fod = fod_extras(cfg, st[st.source == "FOD"])
    fodr = st[st.source == "FODR"].assign(uid=lambda d: d.source_id.str.upper(), CONT_DATE=None)
    mtbs = gpd.read_file(cfg.src / "mtbs" / "mtbs_perims.gpkg").to_crs(CRS)
    rows = []
    has = fod[fod.MTBS_ID.fillna("").str.strip() != ""]
    for eid, grp in has.groupby("MTBS_ID"):
        per = mtbs[mtbs.event_id == eid]
        if per.empty:
            continue
        cont = pd.to_datetime(grp.CONT_DATE, errors="coerce", utc=True).dt.tz_localize(None).dt.date.dropna()
        rows.append(_event_row(eid, per.incid_name.iloc[0], int(eid[-8:-4]), grp, per.geometry.union_all(),
                               float(per.burnbndac.iloc[0]), "MTBS", "FOD MTBS_ID",
                               str(max(cont)) if len(cont) else "",
                               next((c for c in grp.COMPLEX_NAME if isinstance(c, str) and c.strip()), ""),
                               per.irwinid.iloc[0] if isinstance(per.irwinid.iloc[0], str) else ""))
    n_mtbs = len(rows)
    ig = mtbs.set_index("event_id").ig_date

    # WFIGS for fires in no MTBS event
    wf_path = cfg.src / "wfigs" / "wfigs_perimeters.gpkg"
    if wf_path.exists():
        used = {n for r in rows for n in r["fire_numbers"].split(";")}
        taken = gpd.GeoDataFrame({"year": [r["year"] for r in rows]}, geometry=[r["geometry"] for r in rows], crs=CRS)
        w = gpd.read_file(wf_path).to_crs(CRS)
        w["uid"] = w.attr_UniqueFireIdentifier.str.upper().str.strip()
        w = w[(w.poly_GISAcres >= cfg.wfigs_min) & (w.attr_IncidentTypeCategory == "WF")]
        cand = pd.concat([fod.assign(src="FOD"), fodr.assign(src="FODR")], ignore_index=True)
        cand = cand[~cand.fire_number.astype(str).isin(used) & (cand.uid != "")]
        dup = 0
        for r in cand.merge(w[["uid", "attr_IncidentName", "attr_ContainmentDateTime", "poly_FeatureCategory",
                               "attr_CpxName", "attr_IrwinID", "geometry"]], on="uid").itertuples():
            geom = r.geometry if r.geometry.is_valid else r.geometry.buffer(0)
            year = int(r.uid[:4])
            near = taken.iloc[taken.sindex.query(geom)]
            near = near[near.year == year]
            if len(near) and near.intersection(geom).area.sum() > WFIGS_DUP_SHARE * geom.area:
                dup += 1
                continue
            cont = r.attr_ContainmentDateTime
            if pd.isna(cont) and isinstance(r.CONT_DATE, str):
                cont = pd.to_datetime(r.CONT_DATE, errors="coerce")
            members = cand[cand.fire_number == r.fire_number].iloc[:1]
            rows.append(_event_row(f"WFIGS-{r.uid}", str(r.attr_IncidentName).upper(), year, members, geom,
                                   geom.area * ACRES_PER_M2, f"WFIGS {r.poly_FeatureCategory or ''}".strip(),
                                   f"WFIGS ({r.src} id)", str(pd.Timestamp(cont).date()) if pd.notna(cont) else "",
                                   r.attr_CpxName if isinstance(r.attr_CpxName, str) else "", r.attr_IrwinID or ""))
        print(f"Perimeters: {n_mtbs} MTBS + {len(rows) - n_mtbs} WFIGS events "
              f"({dup} WFIGS skipped as same-year duplicates of an MTBS event)")
    else:
        print(f"Perimeters: {n_mtbs} MTBS events (no WFIGS file -- run satellite_growth_data.py)")
    ev = gpd.GeoDataFrame(rows, crs=CRS).sort_values(["year", "discovery"]).reset_index(drop=True)
    ev["mtbs_ig_date"] = ev.event_id.map(ig).fillna("")
    ev["fire_label"] = [fire_label(n, c) for n, c in zip(ev.fire_name, ev.complex_name)]
    return ev


def apply_cause_overrides(cfg, ev):
    """config/imsr_fire_fdra_overrides.csv's cause_class / cause_general corrections (DIXIE 2021 ->
    Human, Power Generation/Transmission/Distribution; PARK 2024 -> Arson), keyed by IMSR fire_id.
    An override reaches every satellite event holding one of that IMSR fire's located fire records:
    the IMSR fire's matched_fire_number (<gacc>_imsr_fires.csv, which already honours the
    override's own match_fire_numbers) plus match_fire_numbers itself. Applied the way
    imsr_growth_prep.py does: every cause field blanked, then class/general set; cause_method
    "override". Only these shared-record links are used -- the QA charts' looser spatial/name
    pairing would carry 2017 LAPORTE's cause onto CASCADE (checked 2026-10-08). The table's fdra /
    continues / exclude columns are IMSR-listing fixes and don't apply here."""
    if not CAUSE_OVERRIDES_CSV.exists():
        return ev
    for f in CAUSE_FIELDS + ("cause_method", "cause_override"):
        ev[f] = ev[f].astype(object).where(ev[f].notna(), "")  # all-blank columns read as float
    ov = pd.read_csv(CAUSE_OVERRIDES_CSV, dtype=str, encoding="utf-8-sig").fillna("")
    ov = ov[(ov.cause_class.str.strip() != "") | (ov.cause_general.str.strip() != "")]
    imsr_csv = ROOT / cfg.s["imsr_growth_dir"] / f"{cfg.s['nfdrs_ovf_gacc'].lower()}_imsr_fires.csv"
    matched = (pd.read_csv(imsr_csv, dtype=str).fillna("").set_index("fire_id").matched_fire_number
               if imsr_csv.exists() else pd.Series(dtype=str))
    members = [set(str(n).split(";")) for n in ev.fire_numbers]
    applied, conflicts, unmatched = {}, [], []
    for r in ov.itertuples():
        nums = {n for n in (str(matched.get(r.fire_id, "")) + ";" + r.match_fire_numbers).split(";") if n.strip()}
        hits = [i for i, m in enumerate(members) if nums & m]
        if not hits:
            unmatched.append(r.fire_id)
        cause = (r.cause_class.strip(), r.cause_general.strip())
        for i in hits:
            if i in applied and applied[i][1] != cause:
                conflicts.append(f"{ev.at[i, 'fire_name']}: {applied[i][0]} vs {r.fire_id}")
                continue
            applied[i] = (r.fire_id, cause)
    for i, (fire_id, (cls, gen)) in applied.items():
        for f in CAUSE_FIELDS:
            ev.at[i, f] = ""
        ev.at[i, "cause_class"], ev.at[i, "cause_general"] = cls, gen
        ev.at[i, "cause_method"] = "override"
        ev.at[i, "cause_override"] = fire_id
    print(f"Cause overrides ({CAUSE_OVERRIDES_CSV.name}): {len(applied)} fires corrected"
          + (f"; {len(unmatched)} overrides match no satellite fire" if unmatched else "")
          + (f"; CONFLICTS kept first: {conflicts}" if conflicts else ""))
    return ev


# ---- 2-6. dating ---------------------------------------------------------------------------------

def load_detections(cfg, year):
    frames = []
    for f in sorted((cfg.src / "firms").glob(f"*_{year}.csv")):
        d = pd.read_csv(f, dtype={"acq_time": str, "confidence": str, "type": str})
        if d.empty:
            continue
        sensor = f.stem.rsplit("_", 1)[0]
        if sensor == "modis":
            ok = pd.to_numeric(d.confidence, errors="coerce") >= 30
        else:
            ok = d.confidence.str.lower().isin(["n", "h", "nominal", "high"])
        if "type" in d:
            ok &= d["type"].isna() | (d["type"].astype(str).str.strip().isin(["0", ""]))
        frames.append(d[ok].assign(sensor=sensor))
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    utc = pd.to_datetime(d.acq_date + " " + d.acq_time.str.zfill(4), format="%Y-%m-%d %H%M", utc=True)
    local = utc.dt.tz_convert(cfg.tz).dt.tz_localize(None)
    d["fire_day"] = (local - pd.Timedelta(hours=cfg.day_start)).dt.normalize()
    d["is_viirs"] = d.sensor != "modis"
    d["frp"] = pd.to_numeric(d.frp, errors="coerce").fillna(0)
    g = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.longitude, d.latitude), crs=4326).to_crs(CRS)
    g["x"], g["y"] = g.geometry.x, g.geometry.y
    return g


def window(ev):
    disc = pd.Timestamp(ev.discovery)
    ig = pd.Timestamp(ev.mtbs_ig_date) if ev.mtbs_ig_date else disc
    flags = []
    start = min(disc, ig)
    if abs((disc - ig).days) > WINDOW_DISAGREE_DAYS:
        start = ig
        flags.append(f"FOD discovery {disc.date()} vs MTBS ig_date {ig.date()}; used MTBS")
    start -= pd.Timedelta(days=1)
    cont = pd.Timestamp(ev.cont_date) if ev.cont_date else None
    if cont is None:
        flags.append("no CONT_DATE")
    return start, start + pd.Timedelta(days=MAX_SEARCH_DAYS), cont, flags


def growth_end(days, start, cont, max_gap):
    """Follow the run of detection days from the start, allowing gaps up to max_gap; everything
    through CONT_DATE + 1 is always kept."""
    floor = (cont + pd.Timedelta(days=1)) if cont is not None else start
    last = start
    for d in sorted(days):
        if d <= floor or (d - last).days <= max_gap:
            last = max(last, d)
        else:
            break
    return last


def date_cells(cx, cy, det, r):
    """Earliest fire day (day index) among detections within max(nearest, r) of each cell."""
    pts = np.c_[det.x.values, det.y.values]
    k = min(K_NEIGHBORS, len(pts))
    dist, idx = cKDTree(pts).query(np.c_[cx, cy], k=k)
    dist, idx = dist.reshape(len(cx), k), idx.reshape(len(cx), k)
    d1 = dist[:, 0]
    within = dist <= (np.maximum(d1, r) + 1e-6)[:, None]
    return np.where(within, det.day_i.values[idx], np.iinfo(np.int64).max).min(axis=1), d1


def date_event(cfg, ev, det):
    """-> (growth rows DataFrame or None, summary dict)."""
    start, end, cont, flags = window(ev)
    geom = ev.geometry
    w = det[(det.fire_day >= start) & (det.fire_day <= end)]
    w = w[w.geometry.within(geom.buffer(R_MODIS))]
    near = pd.concat([w[~w.is_viirs], w[w.is_viirs & w.geometry.within(geom.buffer(R_VIIRS))]])
    if near.empty:
        return None, {"flags": "; ".join(flags + ["no detections in perimeter"])}
    stop = growth_end(near.fire_day.unique(), start, cont, cfg.max_gap)
    first = near.fire_day.min()
    near = near[near.fire_day <= stop].copy()
    if near.empty:
        return None, {"flags": "; ".join(flags + [f"first detection in perimeter {first.date()} more than "
                                                  f"{cfg.max_gap} d after start {start.date()}"])}
    if cont is not None and stop > cont + pd.Timedelta(days=WINDOW_DISAGREE_DAYS + 1):
        flags.append(f"growth ran past CONT_DATE {cont.date()} to {stop.date()}")
    t0 = near.fire_day.min()
    near["day_i"] = (near.fire_day - t0).dt.days

    x0, y0, x1, y1 = geom.bounds
    x0, y1 = np.floor(x0 / cfg.cell) * cfg.cell, np.ceil(y1 / cfg.cell) * cfg.cell
    shape = (int(np.ceil((y1 - y0) / cfg.cell)), int(np.ceil((x1 - x0) / cfg.cell)))
    transform = from_origin(x0, y1, cfg.cell, cfg.cell)
    mask = features.rasterize([(geom, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8")
    psa_ids = features.rasterize([(g, i + 1) for i, g in enumerate(cfg.psa.geometry)], out_shape=shape,
                                 transform=transform, fill=0, dtype="int32")
    rr, cc = np.nonzero(mask)
    cx, cy = transform.c + (cc + 0.5) * cfg.cell, transform.f - (rr + 0.5) * cfg.cell

    day = np.full(len(rr), -1, dtype=np.int64)
    viirs, modis = near[near.is_viirs], near[~near.is_viirs]
    use_v = np.zeros(len(rr), dtype=bool)
    if len(viirs):
        dv, d1v = date_cells(cx, cy, viirs, R_VIIRS)
        use_v = d1v <= VIIRS_REACH_M
        day[use_v] = dv[use_v]
    rest = ~use_v
    if rest.any():  # MODIS where no VIIRS is close; nearest VIIRS anyway if there's no MODIS
        day[rest] = date_cells(cx[rest], cy[rest], modis if len(modis) else viirs, R_MODIS if len(modis) else R_VIIRS)[0]

    ras = np.zeros(shape, dtype="int16")
    ras[rr, cc] = day + 1
    with rasterio.open(cfg.burn_dir / f"{ev.event_id}.tif", "w", driver="GTiff", height=shape[0], width=shape[1],
                       count=1, dtype="int16", crs=f"EPSG:{CRS}", transform=transform, nodata=0,
                       compress="deflate") as dst:
        dst.write(ras, 1)
        dst.update_tags(day0=str(t0.date()))

    names = np.array([OUTSIDE] + list(cfg.psa[cfg.psa_field]))
    cells = pd.DataFrame({"day_i": day, "psa": names[psa_ids[rr, cc]], "viirs": use_v})
    g = cells.groupby(["day_i", "psa"]).agg(cells=("viirs", "size"), viirs_cells=("viirs", "sum")).reset_index()
    col = np.clip(((near.x - transform.c) // cfg.cell).astype(int), 0, shape[1] - 1)
    row = np.clip(((transform.f - near.y) // cfg.cell).astype(int), 0, shape[0] - 1)
    near["psa"] = names[psa_ids[row, col]]
    dd = near.groupby(["day_i", "psa"]).agg(detections=("x", "size"), frp_sum=("frp", "sum")).reset_index()
    g = g.merge(dd, on=["day_i", "psa"], how="outer").fillna({"cells": 0, "viirs_cells": 0, "detections": 0, "frp_sum": 0})
    cell_ac = cfg.cell * cfg.cell * ACRES_PER_M2
    g["growth_acres"] = (g.cells * cell_ac).round(1)
    g["viirs_share"] = np.where(g.cells > 0, g.viirs_cells / g.cells.where(g.cells > 0, 1), np.nan).round(3)
    g["burn_date"] = (t0 + pd.to_timedelta(g.day_i, unit="D")).dt.strftime("%Y-%m-%d")
    inside = g.psa != OUTSIDE
    g["fdra"] = np.where(inside, g.psa, "")
    g["unit"] = np.where(inside, ev.unit or "", "")
    for c in ("event_id", "fire_name", "fire_label", "complex_name", "year", "cause_class", "cause_general", "cause_specific", "cause_detail",
              "cause_method", "perim_source"):
        g[c] = getattr(ev, c)
    active = sorted(near.fire_day.dt.date.unique())
    span = (active[-1] - active[0]).days + 1
    summ = {"window_start": start.date(), "growth_end": stop.date(), "first_det_day": active[0],
            "last_det_day": active[-1], "det_days": len(active), "gap_days": span - len(active),
            "detections": len(near), "viirs_detections": int(near.is_viirs.sum()),
            "grid_acres": round(len(rr) * cell_ac, 1),
            "psas_burned": ";".join(sorted(set(cells.psa) - {OUTSIDE})),
            "max_day_acres": round(g.groupby("burn_date").growth_acres.sum().max(), 1), "flags": "; ".join(flags)}
    return g, summ


DAY_COLUMNS = ["event_id", "fire_name", "fire_label", "complex_name", "year", "fdra", "unit", "psa", "burn_date", "growth_acres", "detections",
               "frp_sum", "viirs_share", "cause_class", "cause_general", "cause_specific", "cause_detail",
               "cause_method", "perim_source"]
FIRE_COLUMNS = ["event_id", "fire_name", "fire_label", "year", "fdra_origin", "psas_burned", "unit", "discovery", "cont_date",
                "window_start", "growth_end", "first_det_day", "last_det_day", "det_days", "gap_days", "detections",
                "viirs_detections", "perimeter_acres", "located_acres", "grid_acres", "max_day_acres",
                "perim_source", "join_method", "n_members", "fire_numbers", "complex_name", "irwinid",
                "mtbs_ig_date", "cause_class", "cause_general", "cause_specific", "cause_detail", "cause_method",
                "cause_override", "flags"]


def run(settings, years=None):
    cfg = Cfg(settings)
    cfg.dir.mkdir(parents=True, exist_ok=True)
    cfg.burn_dir.mkdir(exist_ok=True)
    if not (cfg.src / "mtbs" / "mtbs_perims.gpkg").exists():
        sys.exit("MTBS perimeters not found -- run scripts/satellite_growth_data.py first")
    ev = build_events(cfg)
    ev["cause_override"] = ""
    ev = apply_cause_overrides(cfg, ev)
    ev.to_file(cfg.events_gpkg, layer="events", driver="GPKG")
    have = sorted({int(p.stem.rsplit("_", 1)[1]) for p in (cfg.src / "firms").glob("*_*.csv")})
    todo = [y for y in (years or have) if y in have]
    days, summ = [], []
    for y in todo:
        det = load_detections(cfg, y)
        sub = ev[ev.year == y]
        print(f"  {y}: {len(sub)} fires, {0 if det is None else len(det):,} QA'd detections", flush=True)
        for e in sub.itertuples():
            g, s = date_event(cfg, e, det) if det is not None else (None, {"flags": "no FIRMS detections that year"})
            summ.append({**{c: getattr(e, c) for c in FIRE_COLUMNS if hasattr(e, c)}, **s})
            if g is not None:
                days.append(g)
    days = pd.concat(days)[DAY_COLUMNS] if days else pd.DataFrame(columns=DAY_COLUMNS)
    fires = pd.DataFrame(summ).reindex(columns=FIRE_COLUMNS)
    if years and cfg.days_csv.exists():  # merge: keep the other years' rows
        old_d, old_f = pd.read_csv(cfg.days_csv, dtype=str), pd.read_csv(cfg.fires_csv, dtype=str)
        days = pd.concat([old_d[~old_d.year.astype(int).isin(todo)], days.astype(str)])
        fires = pd.concat([old_f[~old_f.year.astype(int).isin(todo)], fires.astype(str)])
    days.sort_values(["year", "event_id", "burn_date", "psa"]).to_csv(cfg.days_csv, index=False)
    fires.sort_values(["year", "discovery", "event_id"]).to_csv(cfg.fires_csv, index=False)
    dated = pd.to_numeric(fires.det_days, errors="coerce").notna()
    print(f"Wrote {cfg.days_csv.relative_to(ROOT)} ({len(days):,} fire-PSA-days) and "
          f"{cfg.fires_csv.relative_to(ROOT)} ({int(dated.sum())} of {len(fires)} fires dated)")


def main():
    settings = pipeline_settings.load_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="+", help="re-date only these years, merged into the existing outputs")
    ap.add_argument("--nightly", action="store_true",
                    help="refresh this year's FIRMS + WFIGS and re-date it; does nothing unless "
                         "fdra_charts_web_large_fire_growth_source is Satellite")
    args = ap.parse_args()
    if args.nightly:
        if str(settings["fdra_charts_web_large_fire_growth_source"]).strip().lower() != "satellite":
            print("fdra_charts_web_large_fire_growth_source is not Satellite -- nightly satellite refresh skipped")
            return
        this_year = dt.date.today().year
        satellite_growth_data.fetch_firms(settings, [this_year])
        satellite_growth_data.fetch_wfigs(settings)
        run(settings, [this_year])
        return
    run(settings, args.years)


if __name__ == "__main__":
    main()
