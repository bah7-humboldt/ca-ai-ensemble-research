"""Interactive web viewer for the AI ensemble pattern products -> <output_dir>/web/ (PROTOTYPE).

    python scripts/ai_ensemble_web_export.py            # build web/index.html + web/aie_data.js
    python scripts/ai_ensemble_web_export.py --open     # and open it in the browser
    python scripts/ai_ensemble_web_export.py --publish  # and push it to GitHub Pages

Added 2026-10-10 to see how the products would play out as an interactive GitHub Pages site
(analyst: "lets look at how an interactive viewer would play out on a new github.io site"; then "make sure
this is sanitized as purely experimental for research using publicly available resources with a
focus on the state of california"). ai_ensemble_daily.py builds it on every run (web_export) and,
when web_publish is true, pushes it with scripts/github_pages_publish.py (clone key
ai_ensemble_clone_path, history replaced each push -- the same as the Rx window's site).

Same pattern as the other toolkit viewers: one self-contained page (scripts/templates/
ai_ensemble_viewer.html, no external libraries) plus a data file loaded with a <script> tag, so it
works double-clicked from disk (file://) and on Pages. The data is built from the newest fetched
runs exactly as ai_ensemble_products.py / ai_ensemble_calibrated.py build the PNGs, re-using their
functions:
  grid      1.0 deg (every other 0.5 deg cell): per model + all-model mean, per day 1-15, Int8
            base64 -- 500 hPa anomaly (m / 4), 850 hPa temperature anomaly (C x 5), % of members
            with wetting rain since init, and the pooled % of members with above-normal heights
  contours  ensemble-mean 500 hPa height every 6 dam, precomputed polylines
  series    every member's gradient / 850 hPa offshore wind at each step, so the viewer can
            recompute probabilities for any cutoff the user drags to
  weeks     days 3-7 and 8-14: % of members above normal and the k-means scenarios (composites,
            contours, overall and per-model shares) -- the original scenarios_d*.png, interactive
  outlook   calibrated predictors (% members past the elevated cutoff by fire day) and the
            regime probabilities + pattern-implied odds
~1-2 MB a day. WeatherNext 3 must never go on a public page (experimental terms); the exporter
skips any model whose kind isn't a public source.
"""
import argparse
import base64
import datetime as dt
import json
import sys
import webbrowser

import numpy as np
import pandas as pd

import ai_ensemble_common as common

PUBLIC_MODELS = ("aifs_ens", "aigefs", "ifs_ens", "gefs")   # open licences (CC BY 4.0 / public domain)
TEMPLATE = common.ROOT / "scripts" / "templates" / "ai_ensemble_viewer.html"
REPO_README = """# California & Hawaii AI Ensemble Research — 1–15 Day Patterns

**Experimental research, not a forecast.** This site compares openly available AI and physics-based
weather ensembles for large-scale California and Hawaii fire-weather patterns, 1–15 days ahead. It isn't an
official forecast, outlook, watch or warning, and it hasn't been reviewed or endorsed by any agency.
Don't use it for operational decisions. For official information see the
[National Weather Service fire weather pages](https://www.weather.gov/fire/) and the
[NIFC Predictive Services outlooks](https://www.nifc.gov/nicc/predictive-services/outlooks).

The page (`index.html`, data in `aie_data.js`) is regenerated once a day. Each update replaces this
repository's history, so only the current day is kept here.

## Data and credits

Everything comes from publicly available sources:

- ECMWF AIFS ENS and IFS ENS open data © European Centre for Medium-Range Weather Forecasts (ECMWF),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Modified: cut to the region, averaged
  and compared against thresholds. ECMWF doesn't accept any liability for errors or omissions here.
- NOAA NCEP AIGEFS and GEFS; NCEP/NCAR Reanalysis 1 data provided by NOAA PSL, Boulder, Colorado,
  USA. U.S. Government works, public domain.
- Active-fire detections: NASA FIRMS (LANCE / ESDIS). Fire perimeters: MTBS (USGS/USFS) and NIFC
  WFIGS. Fire records: FPA FOD (Short, USDA Forest Service Research Data Archive) and NIFC InFORM.
  Region boundaries: NIFC.
- Surface and upper-air observations: Iowa Environmental Mesonet (Iowa State University). Map lines:
  Natural Earth.
- Lightning formula layer: after the ECMWF IFS lightning parameterization, Lopez, P., 2016: A lightning
  parameterization for the ECMWF Integrated Forecasting System. Mon. Wea. Rev., 144, 3057-3075,
  [doi:10.1175/MWR-D-16-0026.1](https://doi.org/10.1175/MWR-D-16-0026.1). Rebuilt on IFS ENS open data and
  calibrated with ERA5 (contains modified Copernicus Climate Change Service information) to Vaisala NLDN
  strikes (licensed; only derived probabilities are shown). Not ECMWF's lightning product.

Independent, non-commercial research. Data are provided as-is with no warranty.
"""


def b64_int8(a):
    a = np.where(np.isfinite(a), np.clip(np.round(a), -127, 127), -128).astype(np.int8)
    return base64.b64encode(a.tobytes()).decode()


def contours(lons, lats, field, levels, tol=0.08):
    """Polylines [[level, [lon, lat, lon, lat, ...]], ...], simplified (tol deg) and rounded to
    0.1 deg -- invisible at the page's map scale, and a third the size."""
    import contourpy
    import shapely
    gen = contourpy.contour_generator(lons, lats, np.where(np.isfinite(field), field, np.nan),
                                      line_type=contourpy.LineType.Separate)
    out = []
    for lv in levels:
        for seg in gen.lines(lv):
            if len(seg) >= 3:
                c = np.asarray(shapely.LineString(seg).simplify(tol).coords)
                if len(c) >= 2:
                    out.append([int(lv), np.round(c.ravel(), 1).tolist()])
    return out


def b64_int16(a, scale):
    """Members x steps -> base64 little-endian Int16 of value*scale (-32768 = missing)."""
    a = np.asarray(a, dtype=np.float64) * scale
    a = np.where(np.isfinite(a), np.clip(np.round(a), -32767, 32767), -32768).astype("<i2")
    return base64.b64encode(a.tobytes()).decode()


def _lines_of(geom):
    """Coordinate arrays of every LineString in a (Multi)LineString / GeometryCollection."""
    out = []
    for g in getattr(geom, "geoms", [geom]):
        if g.geom_type == "LineString":
            out.append(np.asarray(g.coords)[:, :2])
        elif hasattr(g, "geoms"):
            out += _lines_of(g)
    return out


def simplify_lines(segs, tol=0.04):
    import shapely
    out = []
    for s in segs:
        ls = shapely.LineString(s).simplify(tol)
        c = np.asarray(ls.coords)
        if len(c) >= 2:
            out.append(np.round(c.ravel(), 2).tolist())
    return out


def build_map(runs, names, dates, glat, glon, wet_mm, P):
    """Pattern-map data for one set of runs on the page's 1.0 deg grid (glat, glon): daily means and the
    12-hourly steps, per model and all-model mean -- z500 / t850 anomalies, wetting rain, height contours and
    the share of members above normal. A run on an older, smaller domain (lat_min 20 N before 2026-10-14) is
    placed on the same grid with no data where it doesn't reach."""
    climo_cache = {}

    def grid_of(ds):
        lats, lons = ds.lat.values, ds.lon.values
        key = (len(lats), len(lons), float(lats[0]), float(lons[0]))
        if key not in climo_cache:
            jj = np.array([int(np.abs(lats - v).argmin()) if np.abs(lats - v).min() < 0.01 else -1 for v in glat])
            ii = np.array([int(np.abs(lons - v).argmin()) if np.abs(lons - v).min() < 0.01 else -1 for v in glon])
            climo_cache[key] = (lats, lons, jj, ii, P.load_climo(lats, lons))
        return climo_cache[key]

    def sub(arr, jj, ii):
        """(..., y, x) on the run grid -> (..., glat, glon), NaN where the run has no row / column."""
        out = arr[..., np.clip(jj, 0, None), :][..., np.clip(ii, 0, None)].astype(np.float64)
        out[..., jj < 0, :] = np.nan
        out[..., :, ii < 0] = np.nan
        return out

    grids = {"z500a": {}, "t850a": {}, "wet": {}}
    cont = {}
    above = np.zeros((len(dates), len(glat), len(glon)))
    n_above = np.zeros(len(dates))
    zmean_all, mean_parts = [], {"z500a": [], "t850a": [], "wet": []}
    steps = [dt.datetime.combine(d, dt.time(h)) + dt.timedelta(days=1 if h == 0 else 0) for d in dates for h in (12, 0)]
    hgrids = {"z500a": {}, "t850a": {}, "wet": {}}
    hcont = {}
    habove = np.zeros((len(steps), len(glat), len(glon)))
    hn = np.zeros(len(steps))
    hz_all, hmean = [], {"z500a": [], "t850a": [], "wet": []}
    for name in names:
        r = runs[name]
        ds = r["ds"]
        lats, lons, jj, ii, climo = grid_of(ds)
        # daily means
        z = P.daily_mean(ds.z500.values, r["valid"], dates)          # (mem, day, y, x)
        t = P.daily_mean(ds.t850.values, r["valid"], dates)
        cl_z = np.stack([climo["z500"][P.doy(d) - 1] for d in dates])
        cl_t = np.stack([climo["t850"][P.doy(d) - 1] for d in dates])
        za = np.nanmean(z, axis=0) - cl_z
        ta = np.nanmean(t, axis=0) - cl_t
        tp = ds.tp.values          # wetting rain since init: the last step inside each day
        wet = np.full((len(dates), len(lats), len(lons)), np.nan)
        for k, d in enumerate(dates):
            end_ = dt.datetime.combine(d + dt.timedelta(days=1), dt.time(0))
            sel = [i for i, v in enumerate(r["valid"]) if v <= end_]
            if sel and r["valid"][sel[-1]] > end_ - dt.timedelta(hours=13):
                wet[k] = 100 * np.nanmean(tp[:, sel[-1]] >= wet_mm, axis=0)
        grids["z500a"][name] = b64_int8(sub(za, jj, ii) / 4)
        grids["t850a"][name] = b64_int8(sub(ta, jj, ii) * 5)
        grids["wet"][name] = b64_int8(sub(wet, jj, ii))
        zm = np.nanmean(z, axis=0)
        zmean_all.append(sub(zm, jj, ii))
        cont[name] = [contours(lons, lats, zm[k] / 10, range(480, 612, 6)) if not np.isnan(zm[k]).all() else []
                      for k in range(len(dates))]
        valid_days = ~np.isnan(z).all(axis=(0, 2, 3))
        ab = sub(np.nanmean(z > cl_z[None], axis=0), jj, ii)
        above[valid_days] += np.nan_to_num(ab[valid_days])
        n_above[valid_days] += 1
        for key, arr in (("z500a", za), ("t850a", ta), ("wet", wet)):
            mean_parts[key].append(sub(arr, jj, ii))
        # 12-hourly steps
        vt = list(r["valid"])
        pick = [vt.index(t_) if t_ in vt else None for t_ in steps]
        shp = (len(steps), len(lats), len(lons))
        hzm, hza, hta, hwet = (np.full(shp, np.nan) for _ in range(4))
        hab = np.full(shp, np.nan)
        for k, (t_, i) in enumerate(zip(steps, pick)):
            if i is None:
                continue
            local = (t_ - dt.timedelta(hours=7)).date()        # the step's California date, for the normal
            cz, ct = climo["z500"][P.doy(local) - 1], climo["t850"][P.doy(local) - 1]
            zz = ds.z500.values[:, i]
            hzm[k] = np.nanmean(zz, axis=0)
            hza[k] = hzm[k] - cz
            hta[k] = np.nanmean(ds.t850.values[:, i], axis=0) - ct
            hwet[k] = 100 * np.nanmean(ds.tp.values[:, i] >= wet_mm, axis=0)
            hab[k] = np.nanmean(zz > cz[None], axis=0)
        hgrids["z500a"][name] = b64_int8(sub(hza, jj, ii) / 4)
        hgrids["t850a"][name] = b64_int8(sub(hta, jj, ii) * 5)
        hgrids["wet"][name] = b64_int8(sub(hwet, jj, ii))
        hcont[name] = [contours(lons, lats, hzm[k] / 10, range(480, 612, 6)) if not np.isnan(hzm[k]).all() else []
                       for k in range(len(steps))]
        hab = sub(hab, jj, ii)
        ok = ~np.isnan(hab).all(axis=(1, 2))
        habove[ok] += np.nan_to_num(hab[ok])
        hn[ok] += 1
        hz_all.append(sub(hzm, jj, ii))
        for key, arr in (("z500a", hza), ("t850a", hta), ("wet", hwet)):
            hmean[key].append(sub(arr, jj, ii))
    scale = {"z500a": 1 / 4, "t850a": 5, "wet": 1}
    for key in ("z500a", "t850a", "wet"):
        grids[key]["mean"] = b64_int8(np.nanmean(np.stack(mean_parts[key]), axis=0) * scale[key])
        hgrids[key]["mean"] = b64_int8(np.nanmean(np.stack(hmean[key]), axis=0) * scale[key])
    zall = np.nanmean(np.stack(zmean_all), axis=0)
    hz = np.nanmean(np.stack(hz_all), axis=0)
    cont["mean"] = [contours(glon, glat, zall[k] / 10, range(480, 612, 6)) if not np.isnan(zall[k]).all() else []
                    for k in range(len(dates))]
    hcont["mean"] = [contours(glon, glat, hz[k] / 10, range(480, 612, 6)) if not np.isnan(hz[k]).all() else []
                     for k in range(len(steps))]
    agree = np.where(n_above[:, None, None] > 0, 100 * above / np.maximum(n_above, 1)[:, None, None], np.nan)
    hagree = np.where(hn[:, None, None] > 0, 100 * habove / np.maximum(hn, 1)[:, None, None], np.nan)
    return {"days": [d.isoformat() for d in dates], "steps": [t_.strftime("%Y-%m-%dT%H:00Z") for t_ in steps],
            "cycles": {n: runs[n]["cycle"].strftime("%Y-%m-%dT%HZ") for n in names},
            "fields": grids, "agree": b64_int8(agree), "contours": cont,
            "hfields": hgrids, "hagree": b64_int8(hagree), "hcontours": hcont}


def load_cycle_runs(cfg, P, date):
    """Each public model's 00 UTC run of `date` (or its latest run that date) from <data_dir>/runs/."""
    import xarray as xr
    out = {}
    for name, m in cfg["models"].items():
        if not m["enabled"] or name not in PUBLIC_MODELS:
            continue
        files = sorted(P.DATA.glob(f"{name}_{date:%Y%m%d}??.nc"))
        if not files:
            continue
        f = next((x for x in files if x.stem.endswith("00")), files[-1])
        ds = xr.open_dataset(f)
        cyc = dt.datetime.strptime(ds.attrs["cycle"], "%Y-%m-%dT%H:00Z")
        out[name] = {"ds": ds, "cycle": cyc, "valid": np.array([cyc + dt.timedelta(hours=int(h)) for h in ds.lead_h.values]),
                     "label": m["label"], "kind": m["kind"], "file": f.name}
    return out


def build_lightning(runs, names, dates, C, lopez_npz=None):
    """Gridded lightning potential for the pattern map: P(>= 1) and P(>= 10) strikes per 0.5 deg California cell
    and fire day, per CAPE-carrying model and their mean, as int8 % on the lightning grid's own small box.
    2026-10-22: plus the lightning formula layer (ai_ensemble_lightning_lopez.py, IFS ENS) as p1/p10["lopez"], kept
    out of the mean, with its verification under "lopez"."""
    J = C.lightning_grid()
    if not J:
        return None
    lopez = None
    if lopez_npz is not None:
        import ai_ensemble_lightning_lopez as LL
        cal = LL.calibration()
        if cal and [tuple(c) for c in np.load(lopez_npz)["cells"]] == [tuple(c) for c in J["cells"]]:
            lopez = LL.forecast(lopez_npz, dates)
            lopez_meta = {"cycle": str(np.load(lopez_npz)["cycle"]), "credit": cal["credit"],
                          "verify": {t: {k: {kk: cal["verify"][t][k][kk] for kk in ("auc", "bss")} for k in ("lopez", "grid")}
                                     for t in ("p1", "p10")}, "test_years": cal["test_years"], "months": cal["months"]}
    cells = J["cells"]
    j0, j1 = min(c[0] for c in cells), max(c[0] for c in cells)
    i0, i1 = min(c[1] for c in cells), max(c[1] for c in cells)
    ny, nx = j1 - j0 + 1, i1 - i0 + 1
    res = {"lat0": J["lat0"] - j0 * J["grid_res"], "lon0": J["lon0"] + i0 * J["grid_res"], "res": J["grid_res"],
           "ny": ny, "nx": nx, "models": [], "p1": {}, "p10": {},
           "verify": {k: {kk: v["verify"][kk] for kk in ("auc", "auc_climatology", "bss", "base_rate")} for k, v in J["targets"].items()}}
    parts = {"p1": [], "p10": []}
    for name in names:
        lp = C.lightning_potential(runs[name], dates)
        if lp is None:
            continue
        res["models"].append(name)
        for key in ("p1", "p10"):
            g = np.full((len(dates), ny, nx), np.nan)
            for n, (j, i) in enumerate(cells):
                g[:, j - j0, i - i0] = lp[key][:, n]
            res[key][name] = b64_int8(100 * g)
            parts[key].append(g)
    if not res["models"] and lopez is None:
        return None
    for key in ("p1", "p10"):
        if parts[key]:
            res[key]["mean"] = b64_int8(100 * np.nanmean(np.stack(parts[key]), axis=0))
    if lopez is not None:
        for key in ("p1", "p10"):
            g = np.full((len(dates), ny, nx), np.nan)
            for n, (j, i) in enumerate(cells):
                g[:, j - j0, i - i0] = lopez[key][:, n]
            res[key]["lopez"] = b64_int8(100 * g)
        res["lopez"] = lopez_meta
    return res


HISTORY_DAYS = 3        # earlier daily runs offered by the pattern map's run selector


def run(cfg):
    import ai_ensemble_products as P
    import ai_ensemble_calibrated as C
    paths = common.Paths(cfg)
    P.DATA, P.OUTROOT, P.CLIMO = paths.runs, paths.output, paths.climo
    runs = {k: v for k, v in P.load_runs(cfg).items() if k in PUBLIC_MODELS}
    if not runs:
        raise SystemExit("no model runs -- run scripts/ai_ensemble_fetch.py first")
    any_ds = next(iter(runs.values()))["ds"]
    lats, lons = any_ds.lat.values, any_ds.lon.values
    newest = max(r["cycle"] for r in runs.values())
    dates = common.forecast_dates(newest)          # Day 1 = today
    climo = P.load_climo(lats, lons)
    bm = P.Basemap(cfg, lats, lons)
    # 1.0 deg grid for the page
    jj = np.arange(0, len(lats), 2)
    ii = np.arange(0, len(lons), 2)
    glat, glon = lats[jj], lons[ii]
    names = list(runs)

    wet_mm = cfg["wetting_rain_mm"]
    M = build_map(runs, names, dates, glat, glon, wet_mm, P)
    import ai_ensemble_lightning_lopez as LL
    lopez_on = cfg.get("lightning_lopez", {}).get("enabled", False)
    M["ltg"] = build_lightning(runs, names, dates, C, LL.latest_npz(cfg) if lopez_on else None)
    grids, cont, steps = M["fields"], M["contours"], [dt.datetime.strptime(x, "%Y-%m-%dT%H:00Z") for x in M["steps"]]
    hgrids, hcont = M["hfields"], M["hcontours"]
    agree_b64, hagree_b64 = M["agree"], M["hagree"]
    # Earlier runs for the pattern map's run selector (2026-10-14, analyst: "add a selector to look at the
    # last 4 days of runs"): the 00 UTC runs of the previous HISTORY_DAYS days still in <data_dir>/runs/
    # (keep_cycles >= 4), each on its own dates (Day 1 = its run date), in web/runs/<YYYYMMDD>/.
    prev_runs = []
    for back in range(1, HISTORY_DAYS + 1):
        d0 = newest.date() - dt.timedelta(days=back)
        rr = load_cycle_runs(cfg, P, d0)
        if len(rr) < 2:
            continue
        nm = [n for n in names if n in rr]
        Mb = build_map(rr, nm, [d0 + dt.timedelta(days=i) for i in range(len(dates))], glat, glon, wet_mm, P)
        Mb["ltg"] = build_lightning(rr, nm, [d0 + dt.timedelta(days=i) for i in range(len(dates))], C,
                                    LL.latest_npz(cfg, d0) if lopez_on else None)
        prev_runs.append((d0, nm, Mb))
        for r_ in rr.values():
            r_["ds"].close()
    # --- member series: gradients + 850 hPa offshore wind
    series = []
    for g in cfg["gradients"]:
        item = {"key": g["name"], "label": g["label"], "units": "hPa", "tail": "low",
                "ref": g["thresholds"], "models": {}}
        for name in names:
            r = runs[name]
            ds = r["ds"]
            v = ds.msl_pt.sel(point=g["a"]).values - ds.msl_pt.sel(point=g["b"]).values
            item["models"][name] = {"t": [x.strftime("%Y-%m-%dT%H:00Z") for x in r["valid"]],
                                    "n": int(v.shape[0]), "m16": b64_int16(v, 10)}
        series.append(item)
    for w in cfg["wind850"]:
        lo, hi = w["dir_from"]
        item = {"key": w["point"], "label": w["label"], "units": "kt", "tail": "high",
                "ref": [w["min_kt"]], "models": {}}
        for name in names:
            r = runs[name]
            ds = r["ds"]
            u, v = ds.u850_pt.sel(point=w["point"]).values, ds.v850_pt.sel(point=w["point"]).values
            spd = np.hypot(u, v) * 1.943844
            dfrom = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
            ins = (dfrom >= lo) & (dfrom <= hi) if lo <= hi else (dfrom >= lo) | (dfrom <= hi)
            item["models"][name] = {"t": [x.strftime("%Y-%m-%dT%H:00Z") for x in r["valid"]],
                                    "n": int(spd.shape[0]), "m16": b64_int16(np.where(ins, spd, 0.0), 10)}
        series.append(item)

    # --- week 1 / week 2: above-normal probability + k-means scenarios (ai_ensemble_products.compute_scenarios,
    # the same calculation as scenarios_d*.png)
    weeks = []
    letters = "ABCDEFGH"
    for W in P.compute_scenarios(runs, cfg, dates, climo, lats, lons):
        sc_list = []
        for col, c in enumerate(W["order"]):
            sel = W["lab"] == c
            comp_a = W["A"][sel].mean(axis=0)
            comp_f = W["F"][sel].mean(axis=0)
            sc_list.append({"letter": letters[col], "share": round(float(W["overall"][c]), 3), "members": int(sel.sum()),
                            "by_model": {n: round(float(W["share"][c, mi]), 3) for mi, n in enumerate(W["mods"])},
                            "anom": b64_int8(comp_a[jj][:, ii] / 4),
                            "contours": contours(lons, lats, comp_f / 10, range(480, 612, 6))})
        weeks.append({"d0": W["d0"], "d1": W["d1"], "start": dates[W["d0"] - 1].isoformat(),
                      "end": dates[W["d1"] - 1].isoformat(), "n": int(len(W["lab"])), "models": W["mods"],
                      "above": b64_int8(100 * W["above"][jj][:, ii]), "scenarios": sc_list})

    # --- calibrated outlook + regimes
    outlook, regimes = [], None
    import ai_ensemble_nfdrs
    fuels = ai_ensemble_nfdrs.load_forecast(cfg)
    if fuels is not None and fuels.attrs.get("age_hours", 0) > 48:
        print(f"  NFDRS forecast is {fuels.attrs['age_hours']:.0f} h old -- not used")
        fuels = None
    if C.available():
        cal, reg, rstats, skill = C.load()
        masks = C.region_masks(cfg, lats, lons)
        for g, p, sug, a in C.choose_predictors(cal, skill, [d.month for d in dates]):
            el, hi = sug["elevated"], sug.get("high")
            scale = 1.0
            row = {"entity": g, "key": p, "auc": round(float(a), 3), "tail": sug["tail"],
                   "elevated": el, "high": hi, "base": C.base_rate(cal, g, sug), "days": {}, "members": {}}
            fp = sug.get("footprint")
            if fp and fp in cal.get("footprints", {}):
                f = cal["footprints"][fp]
                row["footprint"] = {"id": fp, "label": f["label"], "psas": f["psas"]}
            if g == "HI":          # Hawaii rows (choose_hawaii): island group or all islands, own threshold
                fh = sug.get("footprint_hi")
                hf = cal.get("hawaii", {}).get("footprints", {})
                row["footprint_hi"] = fh
                row["thresh"] = int(sug.get("thresh", 300))
                row["footprint"] = ({"id": fh, "label": hf[fh]["label"], "psas": [fh]} if fh in hf else
                                    {"id": "HI", "label": "Hawaii (all islands)", "psas": sorted(hf)})
                # wind in mph, pressure in mb (2026-10-29): cutoffs and members scaled from the stored kt / m/s
                import ai_ensemble_web_pages
                row["units"], scale = ai_ensemble_web_pages.hawaii_units(p)
                for k in ("elevated", "high"):
                    if row[k]:
                        row[k] = {**row[k], "cutoff": round(row[k]["cutoff"] * scale, 2)}
            row["joint"] = sug.get("joint")
            row["starts"] = sug.get("starts")
            refs = C.REF_CUTOFFS.get(p)
            if refs:          # published operational thresholds for the same measure (slider presets + About page)
                row["refs"] = [dict(r) for r in refs]
            if g.endswith("-lightning"):
                # thunderstorm / lightning-starts row (calibration "lightning", NLDN strikes): strike-day
                # skill for the About page and the table; only derived numbers, never strike data
                row["kind"] = "lightning"
                row["ltg"] = {k: sug.get(k) for k in ("lightning_day", "base_ltg", "p_ltg_past", "p_big_past", "auc_500",
                                                       "share_ltg_caught", "years", "start_rate_dry", "start_rate_wet")}
            # forecast fuels for the row's area: FEMS forecast ERC percentile (PSA mean), days 1-7,
            # the last forecast day carried forward after that (flagged)
            area = row["footprint"]["psas"] if row.get("footprint") else [
                k for k in (fuels.psa.unique() if fuels is not None else []) if k.startswith("NC" if g.startswith("ONCC") else "SC")]
            erc_by_day, persisted_from = [], None
            if fuels is not None and len(area):
                fa = fuels[fuels.psa.isin(area)].groupby("date").ERC_pct.mean()
                last = None
                for d in dates:
                    v = fa.get(pd.Timestamp(d))
                    if v is not None and np.isfinite(v):
                        erc_by_day.append(round(float(v), 1))
                        last = float(v)
                    elif last is not None:
                        erc_by_day.append(round(last, 1))
                        persisted_from = persisted_from or d.isoformat()
                    else:
                        erc_by_day.append(None)
            row["fuels"] = {"erc_pct": erc_by_day, "persisted_from": persisted_from,
                            "dry_pct": (sug.get("joint") or {}).get("erc_dry_pct", 90)}
            for name in names:
                x = C.model_predictor(runs[name], p, dates, climo, masks, lats, lons)
                if x is None:
                    continue
                row["members"][name] = [[None if not np.isfinite(v) else round(float(v) * scale, 1) for v in col]
                                        for col in x.T]       # per day, per member
            if not row["members"]:
                print(f"  outlook: {g} {p} -- no model in this run carries it yet, row left out")
                continue
            outlook.append(row)
        if reg is not None:
            k = reg.sizes["regime"]
            letters = [str(v) for v in reg.regime.values]
            col = "lift_adj" if "lift_adj" in rstats else "lift"
            lift = {g: [float(v) for v in rstats[(rstats.gacc == g) & (rstats.threshold_acres == 5000)]
                        .set_index("regime").reindex(letters)[col].values] for g in ("ONCC", "OSCC")}
            probs = {}
            for name in names:
                lab = C.assign_regimes(runs[name], reg, climo, lats, lons, dates)
                probs[name] = [[None if (lab[:, j] < 0).all() else round(float(np.mean(lab[:, j][lab[:, j] >= 0] == c)), 3)
                                for j in range(len(dates))] for c in range(k)]
            rl, rn = reg.lat.values, reg.lon.values
            regimes = {"letters": letters, "lift": lift, "probs": probs,
                       "grid": {"lat0": float(rl[0]), "lon0": float(rn[0]), "dlat": float(rl[1] - rl[0]),
                                "dlon": float(rn[1] - rn[0]), "ny": len(rl), "nx": len(rn)},
                       "composites": [b64_int8(reg.composite_anom.values[c] / 4) for c in range(k)],
                       "freq": [float(rstats[(rstats.regime == L)].freq.iloc[0]) for L in letters]}

    data = {
        "meta": {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                 "days": [d.isoformat() for d in dates],
                 "steps": [t.strftime("%Y-%m-%dT%H:00Z") for t in steps],
                 "runs": [{"id": newest.strftime("%Y%m%d"), "date": newest.date().isoformat(), "current": True,
                           "cycles": M["cycles"]}] + [
                     {"id": d0.strftime("%Y%m%d"), "date": d0.isoformat(), "current": False, "cycles": Mb["cycles"],
                      "models": nm} for d0, nm, Mb in prev_runs],
                 "models": [{"key": n, "label": runs[n]["label"], "kind": runs[n]["kind"],
                             "cycle": runs[n]["cycle"].strftime("%Y-%m-%dT%HZ"),
                             "members": int(runs[n]["ds"].sizes["member"]),
                             "color": P.MODEL_STYLE[n]["color"], "dash": P.MODEL_STYLE[n]["ls"] != "-"}
                            for n in names],
                 "wetting_mm": wet_mm},
        "grid": {"lat0": float(glat[0]), "lon0": float(glon[0]), "dlat": float(glat[1] - glat[0]),
                 "dlon": float(glon[1] - glon[0]), "ny": len(glat), "nx": len(glon),
                 "fields": grids, "agree": agree_b64, "hfields": hgrids, "hagree": hagree_b64},
        "contours": cont, "hcontours": hcont, "ltg": M["ltg"],
        "lines": {"country": simplify_lines(bm.segs.get("country", [])),
                  "state": simplify_lines(bm.segs.get("state", [])),
                  "region": simplify_lines(bm.region_segs, 0.01)},
        "series": series, "outlook": outlook, "regimes": regimes, "weeks": weeks,
    }
    web = paths.output / "web"
    web.mkdir(parents=True, exist_ok=True)
    # Sub-pages (About each calibrated row / regime, How-to-read guides) -- ai_ensemble_web_pages.py
    try:
        import ai_ensemble_web_pages
        region_lines = {g: simplify_lines(_lines_of(geom.boundary), 0.01) for g, geom in bm.regions.items()}
        links = ai_ensemble_web_pages.generate(web, data, cfg, outlook, region_lines)
        for i, href in links["predictors"].items():
            outlook[i]["page"] = href
        if data["regimes"]:
            data["regimes"]["pages"] = links["regimes"]
        data["meta"]["guides"] = True
    except Exception as e:  # noqa: BLE001 -- the main page works without its sub-pages
        import traceback
        traceback.print_exc()
        print(f"WARNING: sub-pages not built ({type(e).__name__}: {e})")
    # 12-hourly map steps go in their own file, loaded only when the viewer switches to them (~3 MB)
    half = {"hfields": data["grid"].pop("hfields"), "hagree": data["grid"].pop("hagree"), "hcontours": data.pop("hcontours")}
    (web / "aie_half.js").write_text("window.__AIE_HALF__ = " + json.dumps(half, separators=(",", ":")) + ";\n", encoding="utf-8")
    # earlier runs: map.js (daily) and half.js (12-hourly) per run, each loaded on demand
    import shutil
    rdir = web / "runs"
    if rdir.is_dir():
        shutil.rmtree(rdir)
    for d0, nm, Mb in prev_runs:
        od = rdir / d0.strftime("%Y%m%d")
        od.mkdir(parents=True)
        key = d0.strftime("%Y%m%d")
        daily = {k: Mb[k] for k in ("days", "steps", "cycles", "fields", "agree", "contours", "ltg")}
        hh = {k: Mb[k] for k in ("hfields", "hagree", "hcontours")}
        (od / "map.js").write_text(f"(window.__AIE_RUNS__ = window.__AIE_RUNS__ || {{}})['{key}'] = "
                                   + json.dumps(daily, separators=(",", ":")) + ";\n", encoding="utf-8")
        (od / "half.js").write_text(f"(window.__AIE_RUNS__ = window.__AIE_RUNS__ || {{}})['{key}h'] = "
                                    + json.dumps(hh, separators=(",", ":")) + ";\n", encoding="utf-8")
    js = "window.__AIE__ = " + json.dumps(data, separators=(",", ":")) + ";\n"
    (web / "aie_data.js").write_text(js, encoding="utf-8")
    (web / "index.html").write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    (web / ".nojekyll").write_text("", encoding="utf-8")       # serve files as-is (no Jekyll build)
    (web / "README.md").write_text(REPO_README, encoding="utf-8")
    print(f"wrote {web / 'index.html'} + aie_data.js ({len(js) / 1e6:.2f} MB)")
    return web


def publish(web, cfg=None):
    """Pushes web/ to the Pages clone; True on success or nothing to do. Never raises."""
    import github_pages_publish
    when = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return github_pages_publish.publish_dir(web, commit_message=f"California & Hawaii AI ensemble research data {when} PT",
                                            clone_path_key="ai_ensemble_clone_path", replace_history=True,
                                            # regenerated in full each run: rows come and go with the season
                                            clean_subdirs=("predictors", "regimes", "runs", "guide"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--publish", action="store_true", help="push to GitHub Pages (ai_ensemble_clone_path)")
    a = ap.parse_args()
    cfg = common.load_settings()
    web = run(cfg)
    if a.publish:
        print("published" if publish(web, cfg) else "PUBLISH FAILED (see messages above)")
    if a.open:
        webbrowser.open((web / "index.html").as_uri())


if __name__ == "__main__":
    sys.exit(main())
