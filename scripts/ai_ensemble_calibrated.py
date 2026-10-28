"""Calibrated forecast products: the ensembles scored against cutoffs and regimes learned from
history (scripts/ai_ensemble_calibrate.py -> <data_dir>/history/calibration/). Called by
ai_ensemble_products.py when that folder exists.

Every forecast predictor is built to match its historical twin:
  <A-B>_min6              fire-day minimum of the model's MSLP gradient over its steps (the history
                          uses ASOS at 00/06/12/18Z; a 12-hourly run sees 2 of those 4 hours, so its
                          minimum is a little less offshore -- slightly conservative)
  <STN>850_offshore_kt_*  850 hPa wind at the sounding site (nearest 0.5 deg cell), interpolated to
  <STN>850_t_* , 500_z_*  12Z on the fire day (its morning) / 00Z the next date (its afternoon);
                          offshore = from 0-110 deg, else 0 kt (same rule as the soundings)
  <GACC>_t850_anom/_z500  UTC-day mean over the region (1.25 deg buffer, like the 2.5 deg history)
Regimes: each member's daily-mean 500 hPa anomaly, sampled on the R1 2.5 deg box, projected onto the
history's EOFs and assigned to the nearest regime centroid.

Promoted 2026-10-10 from the incoming/2026-10-10_ai_ensembles sandbox; settings in
config/ai_ensemble_settings.json, paths via ai_ensemble_common.py, docs in docs/ai_ensembles/README.md.
"""
import datetime as dt
import json
import re
import zoneinfo
from pathlib import Path

import numpy as np
import pandas as pd

import ai_ensemble_common as common

CAL = common.Paths(common.load_settings()).cal
PACIFIC = zoneinfo.ZoneInfo("America/Los_Angeles")
SITES = {"OAK": (37.72, -122.22), "VBG": (34.75, -120.57), "NKX": (32.85, -117.12),
         "REV": (39.57, -119.80), "MFR": (42.37, -122.87)}
OFFSHORE = (0, 110)
THRESH = "5000"
# Published operational thresholds for measures this project also calibrates, shown as slider presets on
# their outlook rows. "lift" = x usual odds of >= 1,000-ac growth in the footprint at that level, 2001-2025
# (ai_ensemble_cefa_patterns.nws_objective). Source: NWS Riverside, "Critical Fire Weather Patterns of the
# United States" (NWCC compilation, 2008): LAX to TPH > 10 mb potential, > 12 mb very strong.
_RIV = "NWS Riverside, Critical Fire Weather Patterns of the United States (NWCC, 2008)"
REF_CUTOFFS = {"LAX-TPH_min6": [{"label": "NWS: potential", "cutoff": -10.0, "lift": 4.85, "source": _RIV},
                                {"label": "NWS: very strong", "cutoff": -12.0, "lift": 7.81, "source": _RIV}]}
INK, MUTED = "#2b2b2b", "#6b6b6b"


def base_rate(cal, entity, sug, thresh=THRESH):
    """Season base rate of the predictor's own outcome: its footprint's PSAs, else the GACC
    (for a lightning row: the season's rate of >= 100-strike days in the footprint)."""
    if "base_ltg" in sug:
        return sug["base_ltg"]
    if "base" in sug:                     # Hawaii rows carry their own (choose_hawaii)
        return sug["base"]
    fp = sug.get("footprint")
    return cal["base_rates"].get(f"{entity}:{fp}_{thresh}" if fp else f"{entity}_{thresh}", np.nan)


def footprint_label(cal, sug):
    fp = sug.get("footprint")
    return cal.get("footprints", {}).get(fp, {}).get("label") if fp else None


def available():
    return (CAL / "calibration.json").exists()


def regimes_available():
    return (CAL / "regimes.nc").exists() and (CAL / "regime_stats.csv").exists()


def load():
    import xarray as xr
    cal = json.loads((CAL / "calibration.json").read_text())
    reg = xr.open_dataset(CAL / "regimes.nc").load() if regimes_available() else None
    rstats = pd.read_csv(CAL / "regime_stats.csv") if regimes_available() else None
    skill = pd.read_csv(CAL / "predictor_skill.csv")
    return cal, reg, rstats, skill


def fire_day_of(valid_utc):
    return [((v.replace(tzinfo=dt.timezone.utc).astimezone(PACIFIC)) - dt.timedelta(hours=6)).date()
            for v in valid_utc]


def _interp_time(series, valid, target):
    """series (mem, lead) at valid times -> (mem,) at target (linear; NaN outside the run)."""
    t = np.array([(v - valid[0]).total_seconds() for v in valid])
    x = (target - valid[0]).total_seconds()
    if x < t[0] or x > t[-1]:
        return np.full(series.shape[0], np.nan)
    j = int(np.searchsorted(t, x))
    if t[j] == x:
        return series[:, j]
    w = (x - t[j - 1]) / (t[j] - t[j - 1])
    return series[:, j - 1] * (1 - w) + series[:, j] * w


_DOY_CLIMO = None


def _site_normal(name, days):
    """Day-of-year normal of a sounding or ERA5 footprint predictor (history/calibration/predictor_doy_climo.csv)."""
    global _DOY_CLIMO
    if _DOY_CLIMO is None:
        p = CAL / "predictor_doy_climo.csv"
        _DOY_CLIMO = pd.read_csv(p, index_col=0) if p.exists() else pd.DataFrame()
    if name not in _DOY_CLIMO:
        return None
    out = []
    for d in days:
        n = d.timetuple().tm_yday
        if d.year % 4 == 0 and n >= 60:
            n = max(59, n - 1)
        out.append(_DOY_CLIMO.at[n, name])
    return np.array(out)


def _es(t_c):
    """Saturation vapour pressure (hPa), Magnus over water."""
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def _td_from_e(e):
    e = np.clip(e, 1e-4, None)
    ln = np.log(e / 6.112)
    return 243.5 * ln / (17.67 - ln)


def humidity_fields(ds, level):
    """(T, Td) arrays (member, lead, lat, lon) at 700 hPa ('700') or 2 m ('2m'), from whichever
    humidity field the model has (specific humidity g/kg, dewpoint, or RH); None if it has none."""
    if level == "700":
        if "t700" not in ds:
            return None
        t = ds.t700.values
        if "q700" in ds:
            q = ds.q700.values / 1000.0
            e = q * 700.0 / (0.622 + 0.378 * q)
            return t, _td_from_e(e)
        if "rh700" in ds:
            return t, _td_from_e(np.clip(ds.rh700.values, 1, 100) / 100 * _es(t))
        return None
    if "t2m" not in ds:
        return None
    t = ds.t2m.values
    if "d2m" in ds:
        return t, ds.d2m.values
    if "rh2m" in ds:
        return t, _td_from_e(np.clip(ds.rh2m.values, 1, 100) / 100 * _es(t))
    return None


SURFACE_SITES = {}      # filled from settings rh_stations by region_masks()
_FP_MASK = {}


def _footprint_mask(fid, lats, lons):
    key = (fid, len(lats), len(lons))
    if key not in _FP_MASK:
        import geopandas as gpd
        import shapely
        f = FOOTPRINTS.get(fid)
        shp = common.ROOT / "config" / ({"ONCC": "oncc_psa/ONCC_PSA.shp", "HI": "hawaii_psa/HAWAII_PSA.shp"}.get(
            (f or {}).get("gacc"), "oscc_psa/OSCC_PSA.shp"))
        if not f or not shp.exists():
            _FP_MASK[key] = None
        else:
            g = gpd.read_file(shp).to_crs(4326)
            geom = g[g.PSANationa.isin(f["psas"])].geometry.union_all()
            yy, xx = np.meshgrid(lats, lons, indexing="ij")
            # cell centres inside the PSAs, as the ERA5 history (a buffer pulled desert / ocean cells
            # into the model mean); buffered only if a footprint is too small to hold a 0.5 deg cell
            m = shapely.contains_xy(geom, xx, yy) if f["gacc"] != "HI" else np.zeros_like(xx, dtype=bool)
            # islands hold few grid cells: take the cells near the coast too (ERA5 history: 0.15 deg)
            for buf in ((0.15, 0.3) if f["gacc"] == "HI" else (0.25,)):
                if m.any():
                    break
                m = shapely.contains_xy(geom.buffer(buf), xx, yy)
            _FP_MASK[key] = m if m.any() else None
    return _FP_MASK[key]


def footprint_series(ds, mask, kind):
    """(member, step) footprint mean of 2 m VPD x 10 m wind ('vpd_wind') or 2 m RH ('rh_min'), or
    None when the run lacks the fields. Also archived per run (ai_ensemble_fetch.archive_run) so
    the models' bias against ERA5 can be measured once ERA5 covers the same days."""
    if kind == "capep90":       # thunderstorm setup: 90th-percentile CAPE over the footprint's cells
        return None if "cape" not in ds else np.nanpercentile(ds.cape.values[:, :, mask], 90, axis=2)
    hf = humidity_fields(ds, "2m")
    if hf is None or (kind == "vpd_wind" and "u10" not in ds):
        return None
    t, td = hf[0][:, :, mask], hf[1][:, :, mask]
    if kind == "rh_min":
        return np.nanmean(np.clip(100 * _es(td) / _es(t), 0, 100), axis=2)
    w = np.hypot(ds.u10.values[:, :, mask], ds.v10.values[:, :, mask])
    return np.nanmean(np.clip(_es(t) - _es(td), 0, None) * w, axis=2)


def footprint_field(run, fid, kind, days, lats, lons):
    """Model version of an ERA5 footprint predictor, (member, day). vpd_wind: fire-day max of the
    footprint-mean (2 m VPD x 10 m wind); rh_min: fire-day min of the footprint-mean 2 m RH;
    dry_tstorm: share of footprint cells with CAPE >= 250, 700 hPa dewpoint depression >= 5 C and
    < 2.5 mm that fire day (IFS ENS / GEFS only -- the others have no CAPE). blh_max: no model has it."""
    ds = run["ds"]
    if kind == "ltgformula":
        # lightning formula (ai_ensemble_lightning_lopez.py, 2026-10-22): the footprint's expected strikes per IFS ENS
        # member and fire day, from that run's own lightning_lopez_<cycle>.npz; IFS ENS only, days 1-14 (max_lead_h 360 since 2026-10-23)
        import ai_ensemble_lightning_lopez as LL
        if not str(run.get("file", "")).startswith("ifs_ens"):
            return None
        cfg = common.load_settings()
        f = common.Paths(cfg).runs / f"lightning_lopez_{run['cycle']:%Y%m%d%H}.npz"
        if not f.exists():                 # the pattern run's cycle differs: the newest formula run within a day
            g = LL.latest_npz(cfg)
            if g is not None and abs((dt.datetime.strptime(g.stem.rsplit("_", 1)[1], "%Y%m%d%H")
                                      - run["cycle"]).total_seconds()) <= 86400:
                f = g
        idx = LL.footprint_cells(cfg).get(fid)
        return LL.footprint_forecast(f, idx, days) if f.exists() and idx else None
    mask = _footprint_mask(fid, lats, lons)
    if mask is None or kind == "blh_max":
        return None
    fd = fire_day_of(run["valid"])
    out = np.full((ds.sizes["member"], len(days)), np.nan)
    if kind == "capep90":
        series = footprint_series(ds, mask, kind)
        if series is None:
            return None
        for k, d in enumerate(days):
            sel = [x for x, f in enumerate(fd) if f == d]
            if sel:
                out[:, k] = np.nanmax(series[:, sel], axis=1)
        return out
    if kind in ("vpd_wind", "rh_min"):
        series = footprint_series(ds, mask, kind)
        if series is None:
            return None
        agg = np.nanmin if kind == "rh_min" else np.nanmax
        for k, d in enumerate(days):
            sel = [x for x, f in enumerate(fd) if f == d]
            if sel:
                out[:, k] = agg(series[:, sel], axis=1)
        return out
    if kind == "dry_tstorm":
        if "cape" not in ds:
            return None
        hf = humidity_fields(ds, "700")
        if hf is None:
            return None
        dd = hf[0][:, :, mask] - hf[1][:, :, mask]
        conv = (ds.cape.values[:, :, mask] >= DTS_CAPE) & (dd >= DTS_DD700)   # same rule as the ERA5 history
        tp = ds.tp.values[:, :, mask]
        for k, d in enumerate(days):
            sel = [x for x, f in enumerate(fd) if f == d]
            if not sel:
                continue
            prev = [x for x in range(len(fd)) if fd[x] < d]
            rain = tp[:, sel[-1]] - (tp[:, prev[-1]] if prev else 0)
            out[:, k] = np.mean(conv[:, sel].any(axis=1) & (rain < 2.5), axis=1)
        return out
    return None


def model_predictor(run, name, days, climo, masks, lats, lons):
    """(member, day) array of predictor `name` on fire days `days`, or None if not computable."""
    if name.startswith(("PH", "HI")):
        return hawaii_predictor(run, name, days, lats, lons)
    name = _alias(name)
    m = re.match(r"^([a-z_]+)_(vpd_wind|blh_max|rh_min|dry_tstorm|capep90|ltgformula)(_anom)?$", name)
    if m:                                  # footprint-wide fields (ERA5-calibrated)
        raw = footprint_field(run, m.group(1), m.group(2), days, lats, lons)
        if raw is None or not m.group(3):
            return raw
        normal = _site_normal(name[:-5], days)     # same ERA5 date normal the calibration used
        return None if normal is None else raw - normal[None, :]
    m = re.match(r"^([A-Z]{3})_rhmin6$", name)
    if m:                                  # fire-day minimum 2 m RH at the station's nearest cell
        ll = SURFACE_SITES.get(m.group(1))
        hf = humidity_fields(run["ds"], "2m")
        if ll is None or hf is None:
            return None
        j, i = int(np.abs(lats - ll[0]).argmin()), int(np.abs(lons - ll[1]).argmin())
        t, td = hf[0][:, :, j, i], hf[1][:, :, j, i]
        rh = np.clip(100 * _es(td) / _es(t), 0, 100)
        out = np.full((rh.shape[0], len(days)), np.nan)
        fd = fire_day_of(run["valid"])
        for k, d in enumerate(days):
            sel = [x for x, f in enumerate(fd) if f == d]
            if sel:
                out[:, k] = np.nanmin(rh[:, sel], axis=1)
        return out
    if name.endswith("_anom") and re.match(r"^[A-Z]{3}(850|700|500)_", name):
        raw = model_predictor(run, name[:-5], days, climo, masks, lats, lons)
        normal = _site_normal(name[:-5], days)
        return None if raw is None or normal is None else raw - normal[None, :]
    ds, valid = run["ds"], run["valid"]
    n_mem = ds.sizes["member"]
    out = np.full((n_mem, len(days)), np.nan)
    m = re.match(r"^(.+)_min6?$", name)
    if m and "-" in m.group(1):
        a, b = m.group(1).split("-")
        if a not in ds.point.values or b not in ds.point.values:
            return None
        g = ds.msl_pt.sel(point=a).values - ds.msl_pt.sel(point=b).values
        fd = fire_day_of(valid)
        for k, d in enumerate(days):
            sel = [i for i, x in enumerate(fd) if x == d]
            if sel:
                out[:, k] = np.nanmin(g[:, sel], axis=1)
        return out
    m = re.match(r"^([A-Z]{3})(850|700|500)_(offshore_kt|t|z|dd)_(12z|00z)$", name)
    if m:
        st, lev, kind, tag = m.groups()
        if st not in SITES:
            return None
        j = int(np.abs(lats - SITES[st][0]).argmin())
        i = int(np.abs(lons - SITES[st][1]).argmin())
        if kind == "dd" and lev == "700":
            hf = humidity_fields(ds, "700")
            if hf is None:
                return None
            dd = hf[0][:, :, j, i] - hf[1][:, :, j, i]
            for k, d in enumerate(days):
                target = dt.datetime.combine(d, dt.time(12)) if tag == "12z" else dt.datetime.combine(d + dt.timedelta(days=1), dt.time(0))
                out[:, k] = _interp_time(dd, valid, target)
            return out
        if kind == "offshore_kt":
            u, v = ds.u850.values[:, :, j, i], ds.v850.values[:, :, j, i]
        elif kind == "t" and lev == "850":
            t = ds.t850.values[:, :, j, i]
        elif kind == "z" and lev == "500":
            z = ds.z500.values[:, :, j, i]
        else:
            return None
        for k, d in enumerate(days):
            target = dt.datetime.combine(d, dt.time(12)) if tag == "12z" else dt.datetime.combine(d + dt.timedelta(days=1), dt.time(0))
            if kind == "offshore_kt":
                uu, vv = _interp_time(u, valid, target), _interp_time(v, valid, target)
                spd = np.hypot(uu, vv) * 1.943844
                dfrom = (np.degrees(np.arctan2(-uu, -vv)) + 360) % 360
                out[:, k] = np.where((dfrom >= OFFSHORE[0]) & (dfrom <= OFFSHORE[1]), spd, 0.0)
                out[np.isnan(uu), k] = np.nan
            elif kind == "t":
                out[:, k] = _interp_time(t, valid, target)
            else:
                out[:, k] = _interp_time(z, valid, target)
        return out
    m = re.match(r"^(ONCC|OSCC|CA)_(t850|z500)_anom$", name)
    if m:
        region, f = m.groups()
        if region not in masks or not masks[region].any():
            return None
        arr = ds[f].values[:, :, masks[region]]
        for k, d in enumerate(days):
            start = dt.datetime.combine(d, dt.time(0))
            sel = [i for i, v in enumerate(valid) if start <= v < start + dt.timedelta(days=1)]
            if not sel:
                continue
            doy = d.timetuple().tm_yday
            if d.year % 4 == 0 and doy >= 60:
                doy = max(59, doy - 1)
            cl = climo[f][doy - 1][masks[region]]
            out[:, k] = np.nanmean(np.nanmean(arr[:, sel], axis=1) - cl[None], axis=1)
        return out
    return None


def region_masks(cfg, lats, lons):
    SURFACE_SITES.update(cfg.get("rh_stations", {}))
    FOOTPRINTS.update(cfg.get("footprints", {}))
    hi = cfg.get("hawaii", {})
    SURFACE_SITES.update(hi.get("stations", {}))
    FOOTPRINTS.update({k: dict(v, gacc="HI") for k, v in hi.get("footprints", {}).items()})
    import geopandas as gpd
    import shapely
    yy, xx = np.meshgrid(lats, lons, indexing="ij")
    masks = {}
    for g, rp in cfg["regions"].items():
        p = common.resolve(rp)
        if p.exists():
            geom = gpd.read_file(p).to_crs(4326).geometry.union_all()
            masks[g] = shapely.contains_xy(geom.buffer(1.25), xx, yy)
    if "ONCC" in masks and "OSCC" in masks:
        masks["CA"] = masks["ONCC"] | masks["OSCC"]
    return masks


# Which measures belong to which region (analyst 2026-10-10: "the mfr to rdd is matched to Southern
# CA fires, that's not making sense"). A Northern California gradient can score well against
# Southern California growth because the same Great Basin high drives both regions' offshore events;
# that's a real association but not a local driver, so the outlook only pairs a region with its own
# stations, soundings and region means. The calibration report still scores everything.
LOCAL = {
    "ONCC": {"gradients": {"SFO-WMC", "SFO-SAC", "RDD-MFR"}, "soundings": {"OAK", "MFR", "REV"},
             "surface": {"SAC", "RDD", "STS"}},
    "OSCC": {"gradients": {"LAX-DAG", "LAX-TPH"}, "soundings": {"VBG", "NKX"}, "surface": {"BUR", "ONT", "PRB"}},
}
# Each region's signature offshore gradients: shown whenever they have a "high" (x4) cutoff even if
# their overall AUC is modest. SFO-WMC is the case in point -- many of Northern California's big fall
# growth days are long-running fires under hot ridges, which pulls its AUC down, but past -10 hPa
# big days are ~5x as likely as usual, the strongest signal there.
SIGNATURE = {"ONCC": ["SFO-WMC_min6"], "OSCC": ["LAX-TPH_min6", "LAX-DAG_min6"]}


from ai_ensemble_history_era5 import DTS_CAPE, DTS_DD700
FOOTPRINTS = {}      # filled from settings by region_masks(): id -> {gacc, psas, members}
ERA5_FP = "vpd_wind|blh_max|rh_min|dry_tstorm|capep90|ltgformula"


def _alias(key):
    """ERA5 twins use the same forecast as the station versions: A-B_era5 -> A-B_min6, STN_era5_rhmin
    -> STN_rhmin6 (the calibration differs, the model quantity is the same)."""
    return re.sub(r"_era5(_rhmin)?$", lambda m: "_rhmin6" if m.group(1) else "_min6", key)


def is_local(region, key):
    m = re.match(r"^([a-z_]+)_(vpd_wind|blh_max|rh_min|dry_tstorm|capep90|ltgformula)(_anom)?$", key)
    if m:
        return FOOTPRINTS.get(m.group(1), {}).get("gacc") == region
    key = _alias(key)
    m = re.match(r"^(.+)_min6?$", key)
    if m and "-" in m.group(1):
        return m.group(1) in LOCAL.get(region, {}).get("gradients", set())
    m = re.match(r"^([A-Z]{3})(850|700|500)_", key)
    if m:
        return m.group(1) in LOCAL.get(region, {}).get("soundings", set())
    m = re.match(r"^([A-Z]{3})_(rhmin6?|windmax)$", key)
    if m:
        return m.group(1) in LOCAL.get(region, {}).get("surface", set())
    m = re.match(r"^(ONCC|OSCC|CA)_", key)
    return bool(m) and m.group(1) == region


def family(key):
    """Station / gradient / region a predictor comes from -- one per family per entity, so a
    single balloon site can't fill the list with its 00Z, 12Z and 500 hPa versions (and a station
    measure and its ERA5 twin count as one)."""
    m = re.match(r"^([a-z_]+)_(vpd_wind|blh_max|rh_min|dry_tstorm|capep90|ltgformula)(_anom)?$", key)
    if m:
        return key
    key = _alias(key)
    m = re.match(r"^(.+)_min6?$", key)
    if m and "-" in m.group(1):
        return m.group(1)
    m = (re.match(r"^([A-Z]{3})(850|700|500)_", key) or re.match(r"^([A-Z]{3})_(rhmin6?|windmax)$", key)
         or re.match(r"^(ONCC|OSCC|CA)_", key))
    return m.group(1) if m else key


def choose_predictors(cal, skill, months, per_entity=3, min_auc=0.65, min_auc_fire_season=0.60):
    """For each calibration entity (GACC fire season, GACC offshore season) whose months cover most of
    the forecast days: the region's signature offshore gradients that have a "high" cutoff, then its
    top local predictors by AUC (>= 5,000 ac; AUC >= min_auc in the offshore seasons,
    min_auc_fire_season otherwise), one per station/region family -- only predictors with an
    'elevated' cutoff that the fetched fields can reproduce, and only the region's own measures."""
    picks = []
    ents = [e for e in ("ONCC-offshore", "OSCC-offshore", "ONCC", "OSCC")
            if e in cal["seasons"] and np.mean([m in cal["seasons"][e] for m in months]) >= 0.5]
    # A region in its offshore season shows only that set (its general fire-season rows would mostly
    # repeat the same stations with weaker scores).
    ents = [e for e in ents if e.endswith("offshore") or f"{e}-offshore" not in ents]
    for g in ents:
        region = g.split("-")[0]
        sug = cal["suggested"].get(g, {}).get(THRESH, {})
        s = skill[(skill.gacc == g) & (skill.threshold_acres == int(THRESH))].sort_values("auc", ascending=False)
        auc = dict(zip(s.predictor, s.auc))

        floor = min_auc if g.endswith("offshore") else min_auc_fire_season   # summer weather alone is weaker

        def usable(p):
            return (is_local(region, p) and not p.endswith("_min") and p in sug and "elevated" in sug[p]
                    and "850_dd_" not in p and not re.search(r"_(t|z|dd)_(00z|12z)$", p)
                    and not p.endswith("_rhmin") and not p.endswith("_windmax") and not p.endswith("_blh_max")
                    # Hot-Dry-Windy: history only until the models' bias against ERA5 is measured
                    # (GEFS ran past the Santa Ana cutoff on every member for 8 days, 2026-10-12;
                    # the archive now keeps each run's footprint series for that check)
                    and not re.search(r"_vpd_wind(_anom)?$", p)
                    and not any(family(p) == family(q) and g == e for e, q, _, _ in picks))
        if g.endswith("offshore"):
            for p in SIGNATURE.get(region, []):
                if usable(p) and "high" in sug[p]:
                    picks.append((g, p, sug[p], auc.get(p, np.nan)))
        n = sum(1 for e, *_ in picks if e == g)
        for row in s.itertuples():
            if n >= per_entity + len(SIGNATURE.get(region, [])) or row.auc < floor:
                break         # sorted: everything after is weaker (summer heat alone is ~0.6)
            if usable(row.predictor):
                picks.append((g, row.predictor, sug[row.predictor], row.auc))
                n += 1
    # Thunderstorm / lightning-starts rows (calibration "lightning", from NLDN strikes): one per
    # footprint whose lightning months cover most of the forecast days, Northern then Southern
    # California, each by strike-day AUC.
    for fid, L in sorted(cal.get("lightning", {}).items(), key=lambda kv: (kv[1]["gacc"], -kv[1]["auc_100"])):
        if np.mean([m in L["months"] for m in months]) < 0.5:
            continue
        sug = dict(L, footprint=fid)
        picks.append((f"{L['gacc']}-lightning", L["predictor"], sug, L["auc_100"]))
    picks += choose_hawaii(cal, months)
    return picks


# ----------------------------------------------------------------------------- Hawaii (2026-10-13)

HI_SOUNDINGS = {"PHLI": (21.98, -159.35), "PHTO": (19.72, -155.05)}
HI_BOX = (20.0, 24.0, -158.0, -153.0)          # trade-wind box upstream (NE) of the chain, as the ERA5 history
# History-only for now: the inversion and 850 hPa dryness (no model field), station wind, Hot-Dry-Windy
# (as the mainland), and airport minimum RH -- the models' nearest cells (0.25-0.5 deg, mostly ocean)
# ran 5-20 % moister than the leeward airports on 2026-10-10..12 (Kahului 63-75 % vs 52-55 % observed),
# so station cutoffs would almost never fire. The run archive keeps each model's airport values, for
# the bias correction once a few weeks have built up.
HI_HISTORY_ONLY = re.compile(r"(_inv_|850_dd_|_windmax$|_vpd_wind(_anom)?$|^PH[A-Z]{2}_rhmin6$)")


def _hi_fire_day_agg(series, fd, days, agg):
    out = np.full((series.shape[0], len(days)), np.nan)
    for k, d in enumerate(days):
        sel = [x for x, f in enumerate(fd) if f == d]
        if sel:
            out[:, k] = agg(series[:, sel], axis=1)
    return out


def hawaii_predictor(run, name, days, lats, lons):
    """Model version of a Hawaii calibration measure (ai_ensemble_calibrate_hawaii.py), (member, day);
    None when the run can't make it. Fire days: the models' 00/12 UTC steps fall on the same Hawaii
    (HST - 6 h) fire day as the Pacific one, so fire_day_of() serves both."""
    ds, valid = run["ds"], run["valid"]
    fd = fire_day_of(valid)
    m = re.match(r"^(.+)-(.+)_(min|max)6$", name)
    if m:
        a, b, kind = m.groups()
        if a not in ds.point.values or b not in ds.point.values:
            return None
        g = ds.msl_pt.sel(point=a).values - ds.msl_pt.sel(point=b).values
        return _hi_fire_day_agg(g, fd, days, np.nanmin if kind == "min" else np.nanmax)
    m = re.match(r"^(PH[A-Z]{2})_rhmin6$", name)
    if m:
        ll = SURFACE_SITES.get(m.group(1))
        hf = humidity_fields(ds, "2m")
        if ll is None or hf is None:
            return None
        j, i = int(np.abs(lats - ll[0]).argmin()), int(np.abs(lons - ll[1]).argmin())
        rh = np.clip(100 * _es(hf[1][:, :, j, i]) / _es(hf[0][:, :, j, i]), 0, 100)
        return _hi_fire_day_agg(rh, fd, days, np.nanmin)
    m = re.match(r"^(PH[A-Z]{2})(850|700)_(spd|dd)_(12z|00z)$", name)
    if m:
        st, lev, kind, tag = m.groups()
        if st not in HI_SOUNDINGS or (kind, lev) not in (("spd", "850"), ("dd", "700")):
            return None
        j = int(np.abs(lats - HI_SOUNDINGS[st][0]).argmin())
        i = int(np.abs(lons - HI_SOUNDINGS[st][1]).argmin())
        if kind == "spd":
            f = np.hypot(ds.u850.values[:, :, j, i], ds.v850.values[:, :, j, i]) * 1.943844
        else:
            hf = humidity_fields(ds, "700")
            if hf is None:
                return None
            f = hf[0][:, :, j, i] - hf[1][:, :, j, i]
        out = np.full((ds.sizes["member"], len(days)), np.nan)
        for k, d in enumerate(days):
            # 12 UTC on D (02 HST) -> D; 00 UTC on D+1 (14 HST) -> D, as the sounding history
            target = dt.datetime.combine(d, dt.time(12)) if tag == "12z" else dt.datetime.combine(d + dt.timedelta(days=1), dt.time(0))
            out[:, k] = _interp_time(f, valid, target)
        return out
    m = re.match(r"^HI_(850spd|850east|700dd)_max$", name)
    if m:
        la0, la1, lo0, lo1 = HI_BOX
        box = (lats[:, None] >= la0) & (lats[:, None] <= la1) & (lons[None, :] >= lo0) & (lons[None, :] <= lo1)
        kind = m.group(1)
        if kind == "700dd":
            hf = humidity_fields(ds, "700")
            if hf is None:
                return None
            f = (hf[0] - hf[1])[:, :, box].mean(axis=2)
        else:
            u, v = ds.u850.values[:, :, box], ds.v850.values[:, :, box]
            f = (np.hypot(u, v) if kind == "850spd" else -u).mean(axis=2)
        steps = [x for x, t in enumerate(valid) if t.hour in (0, 12)]
        f = f[:, steps]
        return _hi_fire_day_agg(f, [fd[x] for x in steps], days, np.nanmax)
    m = re.match(r"^(HI0\d)_(vpd_wind|rh_min|wind_max)(_anom)?$", name)
    if m:
        fid, kind, anom = m.groups()
        mask = _footprint_mask(fid, lats, lons)
        if mask is None:
            return None
        if kind == "wind_max":
            if "u10" not in ds:
                return None
            series = np.hypot(ds.u10.values[:, :, mask], ds.v10.values[:, :, mask]).mean(axis=2)
        else:
            series = footprint_series(ds, mask, kind)
            if series is None:
                return None
        raw = _hi_fire_day_agg(series, fd, days, np.nanmin if kind == "rh_min" else np.nanmax)
        if not anom:
            return raw
        normal = _site_normal(name[:-5], days)
        return None if normal is None else raw - normal[None, :]
    return None


HI_SIGNATURE = ("HI25N-HI19N_max6",)


def hawaii_family(key):
    m = re.match(r"^(PH[A-Z]{2})", key)
    if m and "-" not in key:
        return m.group(1)
    m = re.match(r"^(HI0\d)_(\w+?)(_anom)?$", key)
    if m:
        return key
    return key.split("_")[0]


def choose_hawaii(cal, months, per_entity=4, floor=0.62):
    """Hawaii rows: the season covers most forecast days -> its top measures at >= main_threshold acres
    (AUC >= floor), one per station / gradient / island field, skipping history-only measures (the
    trade inversion and 850 hPa dryness, which the models don't carry, and Hot-Dry-Windy until the bias
    check)."""
    H = cal.get("hawaii")
    if not H or np.mean([m in H["seasons"]["HI"] for m in months]) < 0.5:
        return []
    t = str(H.get("main_threshold", 1000))
    sug = H["suggested"].get(t, {})
    p = CAL / "hawaii_skill.csv"
    if not p.exists():
        return []
    sk = pd.read_csv(p)
    sk = sk[sk.threshold_acres == int(t)].sort_values("auc", ascending=False)
    picks, fams = [], set()
    for r in sk.itertuples():
        if len(picks) >= per_entity or r.auc < floor:
            break
        k = r.predictor
        if k not in sug or "elevated" not in sug[k] or HI_HISTORY_ONLY.search(k) or hawaii_family(k) in fams:
            continue
        s_ = dict(sug[k])
        fid = s_.get("footprint")
        s_["base"] = H["base_rates"].get(f"HI:{fid}_{t}" if fid else f"HI_{t}")
        s_["footprint_hi"] = fid
        picks.append(("HI", k, s_, r.auc))
        fams.add(hawaii_family(k))
    # signature row (2026-10-15), like the mainland's offshore gradients: the ERA5 trade-wind gradient is
    # only AUC ~0.60 year-round, but ~x2 seasonally adjusted and at its 98.8th percentile on Aug 8, 2023 (Maui)
    for k in HI_SIGNATURE:
        if k in sug and "elevated" in sug[k] and hawaii_family(k) not in fams:
            s_ = dict(sug[k])
            s_["base"] = H["base_rates"].get(f"HI_{t}")
            s_["footprint_hi"] = None
            picks.append(("HI", k, s_, float(sk.loc[sk.predictor == k, "auc"].max())))
            fams.add(hawaii_family(k))
    return picks


# ----------------------------------------------------------------------------- gridded lightning potential (2026-10-14)

_LTG_GRID = None


def lightning_grid():
    global _LTG_GRID
    if _LTG_GRID is None:
        f = CAL / "lightning_grid.json"
        _LTG_GRID = json.loads(f.read_text()) if f.exists() else {}
    return _LTG_GRID


def lightning_potential(run, days):
    """{"p1": (day, cell), "p10": (day, cell)} ensemble-mean probability of >= 1 / >= 10 strikes in each 0.5 deg
    California cell on each fire day (ai_ensemble_lightning_grid.py's logistic equations on this run's members:
    CAPE, 700 hPa temperature / dewpoint, 2 m dewpoint, 2 m minus 700 hPa temperature, cell climatology), or None
    when the run lacks CAPE (the AI models) or the fit hasn't been built."""
    J = lightning_grid()
    ds = run["ds"]
    if not J or "cape" not in ds or "t700" not in ds:
        return None
    h7, h2 = humidity_fields(ds, "700"), humidity_fields(ds, "2m")
    if h7 is None or h2 is None:
        return None
    lats, lons = ds.lat.values, ds.lon.values
    cells = J["cells"]
    jj, ii, keep = [], [], []
    for n, (j, i) in enumerate(cells):
        la, lo = J["lat0"] - j * J["grid_res"], J["lon0"] + i * J["grid_res"]
        a, b = np.abs(lats - la), np.abs(lons - lo)
        if a.min() < 0.01 and b.min() < 0.01:
            jj.append(int(a.argmin())); ii.append(int(b.argmin())); keep.append(n)
    if not keep:
        return None
    jj, ii = np.array(jj), np.array(ii)
    valid = run["valid"]
    fd = fire_day_of(valid)
    hrs = np.array([t.hour for t in valid])
    cape = ds.cape.values[:, :, jj, ii]                   # (member, step, cell)
    t7, td7 = h7[0][:, :, jj, ii], h7[1][:, :, jj, ii]
    t2, td2 = h2[0][:, :, jj, ii], h2[1][:, :, jj, ii]
    out = {}
    for name, T in J["targets"].items():
        coef, mu, sd = np.array(T["coef"]), np.array(T["mu"]), np.array(T["sd"])
        P_ = np.full((len(days), len(cells)), np.nan)
        for k, d in enumerate(days):
            syn = [x for x in range(len(valid)) if fd[x] == d and hrs[x] in (0, 12)]
            z00 = [x for x in syn if hrs[x] == 0]
            if not syn or not z00:
                continue
            feats = {
                "log_cape": np.log1p(np.clip(np.nanmax(cape[:, syn], axis=1), 0, None)),
                "t700": np.nanmean(t7[:, syn], axis=1),
                "td700": np.nanmax(td7[:, syn], axis=1),
                "lapse": np.nanmax(t2[:, z00] - t7[:, z00], axis=1),
                "td2m": np.nanmax(td2[:, z00], axis=1),
            }
            mo = d.month
            cl = np.array([T["clim"].get(f"{cells[n][0]},{cells[n][1]},{mo}", T["clim_month"].get(str(mo), 0.01)) for n in keep])
            feats["clim_logit"] = np.broadcast_to(np.log(cl / (1 - cl)), feats["t700"].shape)
            X = np.stack([feats[f] for f in J["features"]], axis=-1)          # (member, cell, feature)
            z = coef[0] + ((X - mu) / sd) @ coef[1:]
            P_[k, keep] = np.nanmean(1 / (1 + np.exp(-z)), axis=0)
        out[name] = P_
    return out


def product_calibrated(runs, cfg, dates, climo, out, lats, lons):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from ai_ensemble_products import MODEL_STYLE
    cal, reg, rstats, skill = load()
    masks = region_masks(cfg, lats, lons)
    picks = choose_predictors(cal, skill, [d.month for d in dates])
    if not picks:
        return [], {}
    days = list(dates)
    fig, axes = plt.subplots(len(picks), 1, figsize=(13, 2.15 * len(picks) + 1.0), squeeze=False, sharex=True)
    summary = {}
    for ax, (g, p, sug, a) in zip(axes[:, 0], picks):
        tail = sug["tail"]
        el, hi = sug["elevated"], sug.get("high")
        base = base_rate(cal, g, sug)
        fpl = footprint_label(cal, sug)
        for name, r in runs.items():
            x = model_predictor(r, p, days, climo, masks, lats, lons)
            if x is None:
                continue
            beyond = (x <= el["cutoff"]) if tail == "low" else (x >= el["cutoff"])
            pr = np.where(np.isnan(x).all(axis=0), np.nan, np.nanmean(np.where(np.isnan(x), np.nan, beyond), axis=0))
            st = MODEL_STYLE.get(name, {"color": "#888", "ls": "-"})
            ax.plot(days, 100 * pr, color=st["color"], ls=st["ls"], lw=2, marker="o", ms=3.5, label=r["label"])
            entry = {"p_elevated": [None if np.isnan(v) else round(float(v), 3) for v in pr]}
            if hi:
                bh = (x <= hi["cutoff"]) if tail == "low" else (x >= hi["cutoff"])
                ph = np.nanmean(np.where(np.isnan(x), np.nan, bh), axis=0)
                entry["p_high"] = [None if np.isnan(v) else round(float(v), 3) for v in ph]
            summary.setdefault(f"{g}:{p}", {})[name] = entry
        op = "<=" if tail == "low" else ">="
        hi_txt = f"; high {op} {hi['cutoff']:.1f} = x{hi['lift_adj']:.1f}" if hi else ""
        ax.set_title(f"{g} | {p} {op} {el['cutoff']:.1f}: historically x{el['lift_adj']:.1f} the usual >=5k-ac "
                     f"growth-day odds for the date{' in ' + fpl.split(' (')[0] if fpl else ''}, catches "
                     f"{el['share_of_large']:.0%} of them{hi_txt}  [AUC {a:.2f}, season base {base:.1%}]",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_ylim(-3, 103)
        ax.set_ylabel("% members", fontsize=8)
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.tick_params(labelsize=7, colors=MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1, 0].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%a\n%m/%d"))
    axes[0, 0].legend(fontsize=7.5, ncol=4, frameon=False, loc="upper right")
    fig.suptitle("Calibrated outlook: % of members beyond the 'elevated' cutoff learned from 2001-present "
                 "satellite growth days (fire days, Pacific; predictors with AUC >= 0.65)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    p = out / "calibrated_predictors.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return [p.name], summary


def assign_regimes(run, reg, climo, lats, lons, dates):
    from ai_ensemble_products import daily_mean, doy
    z = daily_mean(run["ds"].z500.values, run["valid"], dates)        # (mem, day, y, x)
    blat, blon = reg.box_lat.values, reg.box_lon.values
    jj = [int(np.abs(lats - v).argmin()) for v in blat]
    ii = [int(np.abs(lons - v).argmin()) for v in blon]
    w = reg.weights.values
    eof, mean, cent = reg.eof.values, reg.feature_mean.values, reg.centroid_pc.values
    lab = np.full(z.shape[:2], -1)
    for k, d in enumerate(dates):
        if np.isnan(z[:, k]).all():
            continue
        an = z[:, k] - climo["z500"][doy(d) - 1][None]
        box = an[:, jj][:, :, ii]
        feat = (box * w[None]).reshape(len(box), -1)
        pcs = (feat - mean[None]) @ eof.T
        dist = ((pcs[:, None, :] - cent[None]) ** 2).sum(axis=2)
        lab[:, k] = dist.argmin(axis=1)
    return lab


def product_regimes(runs, cfg, dates, climo, bm_unused, out, lats, lons):
    import matplotlib.pyplot as plt
    from ai_ensemble_products import Basemap, anom_cmap, ANOM_LEVELS
    cal, reg, rstats, skill = load()
    k = reg.sizes["regime"]
    letters = list(reg.regime.values)
    days = [d for d in dates]
    probs = {}
    for name, r in runs.items():
        lab = assign_regimes(r, reg, climo, lats, lons, days)
        p = np.full((k, len(days)), np.nan)
        for j in range(len(days)):
            col = lab[:, j]
            if (col >= 0).any():
                p[:, j] = [np.mean(col[col >= 0] == c) for c in range(k)]
        probs[name] = p
    pooled = np.nanmean(np.stack(list(probs.values())), axis=0)
    col = "lift_adj" if "lift_adj" in rstats else "lift"
    lift = {g: rstats[(rstats.gacc == g) & (rstats.threshold_acres == int(THRESH))].set_index("regime").reindex(letters)[col].values
            for g in ("ONCC", "OSCC") if g in set(rstats.gacc)}
    names = list(runs)
    nrow = 2 + len(names) + 1
    hr = [1.25, 1.25] + [0.85] * len(names) + [1.3]
    fig = plt.figure(figsize=(16, sum(hr) * 1.55 + 0.8))
    gs = fig.add_gridspec(nrow, k, height_ratios=hr, hspace=0.55)
    rlat, rlon = reg.lat.values, reg.lon.values
    bm = Basemap(cfg, rlat, rlon)
    cmap = anom_cmap()
    for c in range(k):
        ax = fig.add_subplot(gs[0, c])
        ax.contourf(rlon, rlat, reg.composite_anom.values[c], levels=ANOM_LEVELS, cmap=cmap, extend="both")
        bm.draw(ax)
        lt = ", ".join(f"{g} x{lift[g][c]:.1f}" for g in lift)
        ax.set_title(f"{letters[c]}  ({lt})", fontsize=8, color=INK, pad=2)

    def heat(ax, p, label, annotate):
        im = ax.imshow(100 * p, aspect="auto", cmap="Blues", vmin=0, vmax=80, interpolation="nearest")
        ax.set_yticks(range(k))
        ax.set_yticklabels(letters, fontsize=7)
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels([f"{d:%m/%d}" for d in days] if annotate else [], fontsize=7)
        ax.set_ylabel(label, fontsize=7.5, rotation=0, ha="right", va="center", color=INK)
        ax.tick_params(length=0, colors=MUTED)
        if annotate:
            for (i, j), v in np.ndenumerate(100 * p):
                if not np.isnan(v) and v >= 5:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if v >= 45 else INK)
        return im

    ax = fig.add_subplot(gs[1, :])
    im = heat(ax, pooled, "All models\n(equal weight)", True)
    ax.set_title("Regime probability by day (% of members; Day 1 = today)", fontsize=9, color=INK, loc="left")
    for i, n in enumerate(names):
        heat(fig.add_subplot(gs[2 + i, :]), probs[n], runs[n]["label"].replace(" (", "\n("), i == len(names) - 1)
    ax = fig.add_subplot(gs[-1, :])
    styles = {"ONCC": ("-", INK), "OSCC": ("--", MUTED)}
    summary = {"pooled": {letters[c]: [None if np.isnan(v) else round(float(v), 3) for v in pooled[c]] for c in range(k)}}
    for g, lv in lift.items():
        e = np.nansum(pooled * lv[:, None], axis=0)
        e[np.isnan(pooled).all(axis=0)] = np.nan
        ls, col = styles[g]
        ax.plot(range(len(days)), e, ls=ls, color=col, lw=2, marker="o", ms=3.5)
        last = np.where(~np.isnan(e))[0]
        if len(last):
            ax.text(last[-1] + 0.15, e[last[-1]], g, fontsize=8, color=col, va="center")
        summary[f"expected_lift_{g}"] = [None if np.isnan(v) else round(float(v), 2) for v in e]
    ax.axhline(1, color="#bdbdbd", lw=0.8)
    ax.set_xlim(-0.5, len(days) - 0.5)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([f"{d:%a\n%m/%d}" for d in days], fontsize=7)
    ax.set_ylabel("x base rate", fontsize=8)
    ax.set_title("Pattern-implied odds of a >= 5,000-acre growth day vs the usual odds for the date (sum of regime "
                 "probability x that regime's date-adjusted historical lift)", fontsize=9, color=INK, loc="left")
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, color="#e6e6e6", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.suptitle(f"Historical 500 hPa regimes (R1 2001-2025 fire seasons, k={k}) in the current ensembles",
                 fontsize=11, color=INK)
    p = out / "regime_outlook.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return [p.name], summary
