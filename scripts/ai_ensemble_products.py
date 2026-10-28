"""Pattern products from the newest fetched run of each model -> <output_dir>/<YYYYMMDD_HHMM>/.

    python scripts/ai_ensemble_products.py            # newest <data_dir>/runs/<model>_*.nc per enabled model
    python scripts/ai_ensemble_products.py --open     # also open index.html in the browser

Products (PNG + index.html + summary.json; nothing is published):
  z500_<model>.png       daily-mean ensemble-mean 500 hPa height (contours, dam) and anomaly vs
                         R1 1991-2020 (shading), days 1-15
  z500_compare.png       every model side by side on selected days, plus a pooled row: % of members
                         (each model weighted equally) with above-normal 500 hPa height
  offshore_gradients.png MSLP gradient plumes (median + 10-90%) and P(gradient <= threshold) by model
  offshore_850wind.png   850 hPa offshore-wind probability at the configured points
  regions.png            per region (ONCC, OSCC): T850 anomaly, P(T850 anom >= threshold), and the
                         ensemble-mean % of the region with wetting rain since init
  scenarios_dA-B.png     k-means "scenarios" of the window-mean 500 hPa anomaly over all members of
                         all models pooled (EOF-truncated), composites + each model's share

"Day d" is the calendar day d - 1 days after today (Day 1 = today; ai_ensemble_common.forecast_dates).
A model's daily mean uses whichever of its 12-hourly steps fall inside that UTC day (00/12Z or
06/18Z depending on the cycle).
The plume charts' reference thresholds in config/ai_ensemble_settings.json are starting points; the
calibrated outlook (ai_ensemble_calibrated.py) uses cutoffs learned from history instead.

Promoted 2026-10-10 from the incoming/2026-10-10_ai_ensembles sandbox; settings in
config/ai_ensemble_settings.json, paths via ai_ensemble_common.py, docs in docs/ai_ensembles/README.md.
"""
import argparse
import datetime as dt
import json
import sys
import webbrowser
import zoneinfo
from pathlib import Path

import numpy as np

import ai_ensemble_common as common

DATA = OUTROOT = CLIMO = None    # set from settings in run()
PACIFIC = zoneinfo.ZoneInfo("America/Los_Angeles")

# Categorical slots 1-4 of the dataviz reference palette, fixed per model (never by rank).
MODEL_STYLE = {
    "aifs_ens": {"color": "#2a78d6", "ls": "-"},
    "aigefs": {"color": "#eb6834", "ls": "-"},
    "ifs_ens": {"color": "#1baf7a", "ls": "--"},
    "gefs": {"color": "#eda100", "ls": "--"},
    "weathernext3": {"color": "#e87ba4", "ls": "-"},
}
ANOM_LEVELS = [-240, -180, -120, -90, -60, -30, 30, 60, 90, 120, 180, 240]
INK, MUTED = "#2b2b2b", "#6b6b6b"


# ----------------------------------------------------------------------------- loading

def load_runs(cfg):
    import xarray as xr
    runs = {}
    for name, m in cfg["models"].items():
        if not m["enabled"]:
            continue
        files = sorted(DATA.glob(f"{name}_??????????.nc"))
        if not files:
            print(f"  {name}: no data file -- skipped")
            continue
        ds = xr.open_dataset(files[-1])
        cyc = dt.datetime.strptime(ds.attrs["cycle"], "%Y-%m-%dT%H:00Z")
        valid = np.array([cyc + dt.timedelta(hours=int(h)) for h in ds.lead_h.values])
        runs[name] = {"ds": ds, "cycle": cyc, "valid": valid, "label": m["label"], "kind": m["kind"],
                      "file": files[-1].name}
    return runs


def load_climo(sampler_lats, sampler_lons):
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator
    path = CLIMO
    if not path.exists():
        raise SystemExit(f"{path} missing -- run scripts/ai_ensemble_climo.py first")
    c = xr.open_dataset(path)
    lat = c.lat.values
    order = np.argsort(lat)
    yy, xx = np.meshgrid(sampler_lats, sampler_lons, indexing="ij")
    out = {}
    for f in ("z500", "msl", "t850"):
        a = c[f].values[:, order, :]
        res = np.empty((365, len(sampler_lats), len(sampler_lons)), np.float32)
        for d in range(365):
            ip = RegularGridInterpolator((lat[order], c.lon.values), a[d])
            res[d] = ip(np.c_[yy.ravel(), xx.ravel()]).reshape(yy.shape)
        out[f] = res
    return out


def doy(d):
    n = d.timetuple().tm_yday
    if d.year % 4 == 0 and n >= 60:   # fold Feb 29 / leap years onto the 365-day climatology
        n = max(59, n - 1)
    return n


def daily_mean(arr, valid, dates):
    """arr (member, lead, ...) -> (member, day, ...) mean of steps valid inside each UTC date."""
    out = np.full((arr.shape[0], len(dates)) + arr.shape[2:], np.nan, np.float32)
    for k, d in enumerate(dates):
        start = dt.datetime.combine(d, dt.time(0))
        sel = [i for i, v in enumerate(valid) if start <= v < start + dt.timedelta(days=1)]
        if sel:
            out[:, k] = np.nanmean(arr[:, sel], axis=1)
    return out


# ----------------------------------------------------------------------------- map helpers

class Basemap:
    def __init__(self, cfg, lats, lons):
        import geopandas as gpd
        from shapely.geometry import box
        self.extent = (lons.min(), lons.max(), lats.min(), lats.max())
        bb = box(lons.min() - 1, lats.min() - 1, lons.max() + 1, lats.max() + 1)
        ne = common.resolve(cfg["natural_earth_dir"])
        self.segs = {}
        for key, fname in (("country", "ne_50m_admin_0_countries.geojson"),
                           ("state", "ne_50m_admin_1_states_provinces_lakes.geojson")):
            p = ne / fname
            if not p.exists():
                print(f"  basemap: {p} missing -- no {key} lines")
                self.segs[key] = []
                continue
            g = gpd.read_file(p, bbox=tuple(bb.bounds))
            if key == "state":
                g = g[g["adm0_a3"].isin(["USA", "CAN", "MEX"])] if "adm0_a3" in g else g
            self.segs[key] = _lines(g.geometry.boundary.intersection(bb))
        self.regions = {}
        for rname, rpath in cfg["regions"].items():
            p = common.resolve(rpath)
            if p.exists():
                g = gpd.read_file(p).to_crs(4326)
                self.regions[rname] = g.geometry.union_all()
        self.region_segs = _lines([r.boundary for r in self.regions.values()])

    def draw(self, ax):
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(self.segs.get("state", []), colors="#8a8a8a", linewidths=0.35, zorder=3))
        ax.add_collection(LineCollection(self.segs.get("country", []), colors="#404040", linewidths=0.6, zorder=3))
        ax.add_collection(LineCollection(self.region_segs, colors="#111111", linewidths=0.9, zorder=4))
        x0, x1, y0, y1 = self.extent
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#bdbdbd")


def _lines(geoms):
    out = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        parts = getattr(g, "geoms", [g])
        for p in parts:
            if p.geom_type == "LineString":
                out.append(np.asarray(p.coords)[:, :2])
            elif hasattr(p, "geoms"):
                out.extend(np.asarray(q.coords)[:, :2] for q in p.geoms if q.geom_type == "LineString")
    return out


def anomaly_panel(ax, bm, lons, lats, full, anom, title, cmap):
    cf = ax.contourf(lons, lats, anom, levels=ANOM_LEVELS, cmap=cmap, extend="both", zorder=1)
    cs = ax.contour(lons, lats, full / 10, levels=np.arange(480, 612, 6), colors=INK, linewidths=0.6, zorder=2)
    ax.clabel(cs, fmt="%d", fontsize=5.5, inline_spacing=1)
    bm.draw(ax)
    ax.set_title(title, fontsize=8, color=INK, pad=2)
    return cf


def anom_cmap():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    base = plt.get_cmap("RdBu_r")
    n = len(ANOM_LEVELS) + 1
    cols = [base(x) for x in np.linspace(0.04, 0.96, n)]
    cols[n // 2] = (0.97, 0.97, 0.97, 1.0)              # neutral gray midpoint, not a hue
    cm = mcolors.ListedColormap(cols[1:-1])
    cm.set_under(cols[0])
    cm.set_over(cols[-1])
    return cm


# ----------------------------------------------------------------------------- products

def product_model_maps(runs, cfg, dates, climo, bm, out, lats, lons):
    import matplotlib.pyplot as plt
    cmap = anom_cmap()
    days = [d for d in cfg["map_days"] if d <= len(dates)]
    files = []
    for name, r in runs.items():
        z = daily_mean(r["ds"].z500.values, r["valid"], dates)            # (mem, day, y, x)
        zm = np.nanmean(z, axis=0)
        ncol = 5
        nrow = int(np.ceil(len(days) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.3, nrow * 2.75 + 0.9), squeeze=False)
        cf = None
        for k, ax in enumerate(axes.ravel()):
            if k >= len(days):
                ax.axis("off")
                continue
            d = days[k]
            dd = dates[d - 1]
            if np.isnan(zm[d - 1]).all():
                ax.axis("off")
                continue
            an = zm[d - 1] - climo["z500"][doy(dd) - 1]
            cf = anomaly_panel(ax, bm, lons, lats, zm[d - 1], an, f"Day {d}  {dd:%a %b %d}", cmap)
        fig.suptitle(f"{r['label']}  {r['cycle']:%Y-%m-%d %HZ}  ({len(r['ds'].member)} members)\n"
                     "Daily-mean ensemble-mean 500 hPa height (dam) and anomaly vs 1991-2020 (m)",
                     fontsize=11, color=INK)
        if cf is not None:
            cb = fig.colorbar(cf, ax=axes, orientation="horizontal", fraction=0.03, pad=0.02, aspect=60)
            cb.set_label("500 hPa height anomaly (m)", fontsize=8)
            cb.ax.tick_params(labelsize=7)
        p = out / f"z500_{name}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        files.append(p.name)
    return files


def product_compare(runs, cfg, dates, climo, bm, out, lats, lons):
    import matplotlib.pyplot as plt
    cmap = anom_cmap()
    days = [d for d in cfg["compare_days"] if d <= len(dates)]
    names = list(runs)
    fig, axes = plt.subplots(len(names) + 1, len(days), figsize=(len(days) * 2.6, (len(names) + 1) * 2.3 + 1.0),
                             squeeze=False)
    ridge = np.zeros((len(days), len(lats), len(lons)))
    have = np.zeros(len(days))
    cf = None
    for i, name in enumerate(names):
        r = runs[name]
        z = daily_mean(r["ds"].z500.values, r["valid"], dates)
        for j, d in enumerate(days):
            ax = axes[i, j]
            if np.isnan(z[:, d - 1]).all():
                ax.axis("off")
                continue
            cl = climo["z500"][doy(dates[d - 1]) - 1]
            zm = np.nanmean(z[:, d - 1], axis=0)
            cf = anomaly_panel(ax, bm, lons, lats, zm, zm - cl, f"Day {d}  {dates[d - 1]:%b %d}" if i == 0 else "", cmap)
            ridge[j] += np.nanmean(z[:, d - 1] > cl, axis=0)
            have[j] += 1
            if j == 0:
                ax.set_ylabel(f"{r['label']}\n{r['cycle']:%m/%d %HZ}", fontsize=7.5, color=INK)
    import matplotlib.colors as mcolors
    plev = [0, 10, 20, 33, 67, 80, 90, 100]
    pc = plt.get_cmap("RdBu_r")
    pcols = [pc(x) for x in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)]
    pcols[3] = (0.97, 0.97, 0.97, 1)
    pcm = mcolors.ListedColormap(pcols)
    pf = None
    for j, d in enumerate(days):
        ax = axes[-1, j]
        if not have[j]:
            ax.axis("off")
            continue
        pf = ax.contourf(lons, lats, 100 * ridge[j] / have[j], levels=plev, cmap=pcm, zorder=1)
        bm.draw(ax)
        if j == 0:
            ax.set_ylabel("All models\n% members above normal", fontsize=7.5, color=INK)
    fig.suptitle("500 hPa pattern, model by model (anomaly shaded, height contoured in dam), "
                 "with pooled agreement on above-normal heights", fontsize=11, color=INK)
    if cf is not None:
        cb = fig.colorbar(cf, ax=axes[:-1].ravel().tolist(), orientation="vertical", fraction=0.015, pad=0.01)
        cb.set_label("anomaly (m)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    if pf is not None:
        cb2 = fig.colorbar(pf, ax=axes[-1].ravel().tolist(), orientation="vertical", fraction=0.015, pad=0.01)
        cb2.set_label("% members", fontsize=8)
        cb2.ax.tick_params(labelsize=7)
    p = out / "z500_compare.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return [p.name]


def _time_axis(ax):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2, tz=PACIFIC))
    ax.xaxis.set_minor_locator(mdates.DayLocator(tz=PACIFIC))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%m/%d", tz=PACIFIC))
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _plume(ax, r, name, series, label):
    st = MODEL_STYLE.get(name, {"color": "#888", "ls": "-"})
    t = [v.replace(tzinfo=dt.timezone.utc) for v in r["valid"]]
    lo, med, hi = np.nanpercentile(series, [10, 50, 90], axis=0)
    ax.fill_between(t, lo, hi, color=st["color"], alpha=0.14, linewidth=0)
    ax.plot(t, med, color=st["color"], ls=st["ls"], lw=2, label=label)


def _prob(ax, r, name, p, label):
    st = MODEL_STYLE.get(name, {"color": "#888", "ls": "-"})
    t = [v.replace(tzinfo=dt.timezone.utc) for v in r["valid"]]
    ax.plot(t, 100 * p, color=st["color"], ls=st["ls"], lw=2, marker="o", ms=3, label=label)


def product_offshore(runs, cfg, out):
    import matplotlib.pyplot as plt
    files, summary = [], {}
    grads = cfg["gradients"]
    fig, axes = plt.subplots(len(grads), 2, figsize=(14, 2.6 * len(grads) + 0.8), squeeze=False,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    for i, g in enumerate(grads):
        axv, axp = axes[i]
        t1, t2 = g["thresholds"]
        for name, r in runs.items():
            ds = r["ds"]
            pa = ds.msl_pt.sel(point=g["a"]).values
            pb = ds.msl_pt.sel(point=g["b"]).values
            d = pa - pb                                       # (member, lead)
            _plume(axv, r, name, d, r["label"])
            p1 = np.nanmean(d <= t1, axis=0)
            _prob(axp, r, name, p1, r["label"])
            summary.setdefault(g["name"], {})[name] = {
                "max_p_le_t1": float(np.nanmax(p1)),
                "max_p_le_t2": float(np.nanmax(np.nanmean(d <= t2, axis=0))),
                "first_valid_p_ge_30pct": next((r["valid"][k].strftime("%Y-%m-%dT%HZ") for k in range(len(p1))
                                                if p1[k] >= 0.3), None)}
        for t, ls in ((t1, ":"), (t2, "--")):
            axv.axhline(t, color=MUTED, lw=1, ls=ls)
            axv.text(1.0, t, f" {t:+g}", transform=axv.get_yaxis_transform(), fontsize=7, color=MUTED, va="center")
        axv.axhline(0, color="#bdbdbd", lw=0.8)
        axv.set_ylabel(f"{g['name']} (hPa)", fontsize=8, color=INK)
        axv.set_title(f"{g['label']}: median and 10-90% of members", fontsize=9, color=INK, loc="left")
        axp.set_ylim(-3, 103)
        axp.set_ylabel(f"% members <= {t1:+g} hPa", fontsize=8, color=INK)
        axp.set_title(f"P({g['name']} <= {t1:+g} hPa)", fontsize=9, color=INK, loc="left")
        _time_axis(axv)
        _time_axis(axp)
    axes[0, 0].legend(fontsize=7.5, loc="upper left", ncol=2, frameon=False)
    fig.suptitle("Offshore pressure gradients (MSLP difference; negative = offshore). Times Pacific. "
                 "Thresholds are uncalibrated starting points.", fontsize=11, color=INK)
    fig.tight_layout()
    p = out / "offshore_gradients.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    files.append(p.name)

    winds = cfg["wind850"]
    fig, axes = plt.subplots(len(winds), 2, figsize=(14, 2.6 * len(winds) + 0.8), squeeze=False,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    for i, w in enumerate(winds):
        axv, axp = axes[i]
        lo, hi = w["dir_from"]
        for name, r in runs.items():
            ds = r["ds"]
            u = ds.u850_pt.sel(point=w["point"]).values
            v = ds.v850_pt.sel(point=w["point"]).values
            spd = np.hypot(u, v) * 1.943844
            dfrom = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
            insector = (dfrom >= lo) & (dfrom <= hi) if lo <= hi else (dfrom >= lo) | (dfrom <= hi)
            offshore_kt = np.where(insector, spd, 0.0)
            _plume(axv, r, name, offshore_kt, r["label"])
            p1 = np.nanmean(insector & (spd >= w["min_kt"]), axis=0)
            _prob(axp, r, name, p1, r["label"])
            summary.setdefault(w["point"] + "_850", {})[name] = {"max_p": float(np.nanmax(p1))}
        axv.axhline(w["min_kt"], color=MUTED, lw=1, ls="--")
        axv.set_ylabel("kt (0 if not offshore)", fontsize=8, color=INK)
        axv.set_title(f"{w['label']} ({lo}-{hi} deg): offshore-component speed, median and 10-90%",
                      fontsize=9, color=INK, loc="left")
        axp.set_ylim(-3, 103)
        axp.set_ylabel(f"% members >= {w['min_kt']} kt", fontsize=8, color=INK)
        axp.set_title(f"P(850 hPa from {lo}-{hi} deg at >= {w['min_kt']} kt)", fontsize=9, color=INK, loc="left")
        _time_axis(axv)
        _time_axis(axp)
    axes[0, 0].legend(fontsize=7.5, loc="upper left", ncol=2, frameon=False)
    fig.suptitle("850 hPa offshore wind at sample points (nearest native grid point). Times Pacific. "
                 "Uncalibrated thresholds.", fontsize=11, color=INK)
    fig.tight_layout()
    p = out / "offshore_850wind.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    files.append(p.name)
    return files, summary


def product_regions(runs, cfg, climo, bm, out, lats, lons):
    import matplotlib.pyplot as plt
    import shapely
    if not bm.regions:
        return [], {}
    yy, xx = np.meshgrid(lats, lons, indexing="ij")
    masks = {k: shapely.contains_xy(g, xx, yy) for k, g in bm.regions.items()}
    thr, wet = cfg["t850_anom_threshold_c"], cfg["wetting_rain_mm"]
    fig, axes = plt.subplots(len(masks), 3, figsize=(16, 3.0 * len(masks) + 0.8), squeeze=False)
    summary = {}
    for i, (rname, m) in enumerate(masks.items()):
        ax1, ax2, ax3 = axes[i]
        for name, r in runs.items():
            ds = r["ds"]
            t = ds.t850.values[:, :, m]                                   # (mem, lead, cells)
            cl = np.stack([climo["t850"][doy(v.date()) - 1][m] for v in r["valid"]])   # (lead, cells)
            an = np.nanmean(t - cl[None], axis=2)                          # (mem, lead)
            # Trailing 24-h mean: the 00/12Z vs 06/18Z steps otherwise zigzag with the diurnal cycle
            # (850 hPa sits near the ground over much of ONCC's terrain).
            per_day = max(1, 24 // cfg["step_h"])
            an = np.stack([np.nanmean(an[:, max(0, k - per_day + 1):k + 1], axis=1) for k in range(an.shape[1])], axis=1)
            _plume(ax1, r, name, an, r["label"])
            p = np.nanmean(an >= thr, axis=0)
            _prob(ax2, r, name, p, r["label"])
            tp = ds.tp.values[:, :, m]
            frac = np.nanmean(tp >= wet, axis=2)                          # (mem, lead)
            _prob(ax3, r, name, np.nanmean(frac, axis=0), r["label"])
            summary.setdefault(rname, {})[name] = {
                "max_p_t850_anom": float(np.nanmax(p)),
                "pct_area_wetting_by_end": float(100 * np.nanmean(frac[:, -1]))}
        ax1.axhline(0, color="#bdbdbd", lw=0.8)
        ax1.axhline(thr, color=MUTED, lw=1, ls="--")
        ax1.set_title(f"{rname}: 850 hPa temp anomaly (region mean, trailing 24 h), median and 10-90%",
                      fontsize=9, color=INK, loc="left")
        ax1.set_ylabel("deg C", fontsize=8)
        ax2.set_ylim(-3, 103)
        ax2.set_title(f"{rname}: P(24-h region-mean T850 anomaly >= +{thr:g} C)", fontsize=9, color=INK, loc="left")
        ax2.set_ylabel("% members", fontsize=8)
        ax3.set_ylim(-3, 103)
        ax3.set_title(f"{rname}: % of region with >= {wet:g} mm since init (ens mean)", fontsize=9, color=INK,
                      loc="left")
        ax3.set_ylabel("% of region", fontsize=8)
        for ax in (ax1, ax2, ax3):
            _time_axis(ax)
    axes[0, 0].legend(fontsize=7.5, loc="upper left", ncol=2, frameon=False)
    fig.suptitle("GACC regions: warmth aloft and wetting rain. Anomaly vs R1 1991-2020; precipitation "
                 "accumulates from each model's own init. Times Pacific.", fontsize=11, color=INK)
    fig.tight_layout()
    p = out / "regions.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return [p.name], summary


def _kmeans(X, k, restarts=25, seed=7):
    from scipy.cluster.vq import kmeans2
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        cent, lab = kmeans2(X, k, minit="++", seed=int(rng.integers(1 << 31)))
        if len(set(lab)) < k:
            continue
        inertia = sum(((X[lab == c] - cent[c]) ** 2).sum() for c in range(k))
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def compute_scenarios(runs, cfg, dates, climo, lats, lons):
    """The week-window 500 hPa scenarios (settings scenarios.windows_days: days 3-7 and 8-14), shared
    by the PNG product and the research viewer. Per window: every member's window-mean 500 hPa
    anomaly (A) and height (F) pooled across models, k-means labels on the EOF-truncated anomalies,
    each model's share of each scenario, the equal-weight overall share, the scenarios ordered by
    it, and the probability map: % of members above normal, each model weighted equally."""
    sc = cfg["scenarios"]
    b = sc["box"]
    jy = (lats >= b["lat_min"]) & (lats <= b["lat_max"])
    ix = (lons >= b["lon_min"]) & (lons <= b["lon_max"])
    w = np.sqrt(np.cos(np.radians(lats[jy])))[:, None] * np.ones(ix.sum())[None, :]
    names = list(runs)
    out = []
    for d0, d1 in sc["windows_days"]:
        if d1 > len(dates):
            continue
        anoms, full, owner = [], [], []
        for name in names:
            r = runs[name]
            z = daily_mean(r["ds"].z500.values, r["valid"], dates)[:, d0 - 1:d1]   # (mem, nd, y, x)
            if np.isnan(z).all(axis=(0, 2, 3)).any():
                print(f"  scenarios days {d0}-{d1}: {name} doesn't cover the window -- left out")
                continue
            cl = np.stack([climo["z500"][doy(dates[d - 1]) - 1] for d in range(d0, d1 + 1)])
            anoms.append(np.nanmean(z - cl[None], axis=1))
            full.append(np.nanmean(z, axis=1))
            owner += [name] * z.shape[0]
        if not anoms:
            continue
        A = np.concatenate(anoms)
        F = np.concatenate(full)
        owner = np.array(owner)
        X = (A[:, jy][:, :, ix] * w).reshape(len(A), -1)
        X = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        n_eof = min(sc["n_eofs"], len(S))
        pcs = U[:, :n_eof] * S[:n_eof]
        k = sc["k"]
        lab = _kmeans(pcs, k)
        mods = [n for n in names if n in set(owner)]
        share = np.array([[np.mean(lab[owner == n] == c) for n in mods] for c in range(k)])   # (k, model)
        overall = share.mean(axis=1)                                   # each model weighted equally
        order = np.argsort(-overall)
        above = np.mean([np.mean(A[owner == n] > 0, axis=0) for n in mods], axis=0)   # equal-weight models
        out.append({"d0": d0, "d1": d1, "A": A, "F": F, "owner": owner, "lab": lab, "mods": mods,
                    "share": share, "overall": overall, "order": order, "n_eof": n_eof, "k": k,
                    "above": above})
    return out


def product_scenarios(runs, cfg, dates, climo, bm, out, lats, lons):
    import matplotlib.pyplot as plt
    cmap = anom_cmap()
    files, summary = [], {}
    for W in compute_scenarios(runs, cfg, dates, climo, lats, lons):
        d0, d1, A, F, owner, lab = W["d0"], W["d1"], W["A"], W["F"], W["owner"], W["lab"]
        mods, share, overall, order, n_eof, k = W["mods"], W["share"], W["overall"], W["order"], W["n_eof"], W["k"]
        letters = "ABCDEFGH"
        fig = plt.figure(figsize=(4.2 * k, 7.2))
        gs = fig.add_gridspec(3, k, height_ratios=[3.0, 1.25, 0.12], hspace=0.45)
        cf = None
        for col, c in enumerate(order):
            ax = fig.add_subplot(gs[0, col])
            sel = lab == c
            cf = anomaly_panel(ax, bm, lons, lats, F[sel].mean(axis=0), A[sel].mean(axis=0),
                               f"Scenario {letters[col]}: {100 * overall[c]:.0f}% (models equal-weighted)\n"
                               f"{sel.sum()} of {len(lab)} members", cmap)
            axb = fig.add_subplot(gs[1, col])
            for mi, n in enumerate(mods):
                st = MODEL_STYLE.get(n, {"color": "#888"})
                axb.barh(mi, 100 * share[c, mi], color=st["color"], height=0.7,
                         hatch="//" if runs[n]["kind"] == "physics" else None, edgecolor="white", linewidth=0.5)
                axb.text(100 * share[c, mi] + 1.5, mi, f"{100 * share[c, mi]:.0f}%", va="center", fontsize=7,
                         color=INK)
            axb.set_yticks(range(len(mods)))
            axb.set_yticklabels([runs[n]["label"].replace(" (", "\n(") for n in mods] if col == 0 else [],
                                fontsize=6.5)
            axb.set_xlim(0, 105)
            axb.invert_yaxis()
            axb.tick_params(labelsize=6.5, colors=MUTED)
            axb.set_xlabel("% of that model's members", fontsize=7, color=MUTED)
            for s in ("top", "right"):
                axb.spines[s].set_visible(False)
            summary.setdefault(f"days_{d0}-{d1}", []).append(
                {"scenario": letters[col], "share": round(float(overall[c]), 3),
                 "by_model": {n: round(float(share[c, mi]), 3) for mi, n in enumerate(mods)}})
        fig.suptitle(f"Days {d0}-{d1} ({dates[d0 - 1]:%b %d}-{dates[d1 - 1]:%b %d}) 500 hPa scenarios: k-means on "
                     f"{len(lab)} members from {len(mods)} models (window-mean anomaly, {n_eof} EOFs)",
                     fontsize=11, color=INK)
        if cf is not None:
            cb = fig.colorbar(cf, cax=fig.add_subplot(gs[2, 1:k - 1] if k > 2 else gs[2, :]), orientation="horizontal")
            cb.set_label("composite 500 hPa anomaly (m); contours = composite height (dam)", fontsize=8)
            cb.ax.tick_params(labelsize=7)
        p = out / f"scenarios_d{d0}-{d1}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        files.append(p.name)
    return files, summary


def write_index(out, runs, files, summary):
    rows = "".join(
        f"<tr><td>{r['label']}</td><td>{r['cycle']:%Y-%m-%d %HZ}</td><td>{len(r['ds'].member)}</td>"
        f"<td>{r['ds'].attrs.get('download_gb', '')} GB / {r['ds'].attrs.get('minutes', '')} min</td>"
        f"<td>{r['file']}</td></tr>" for r in runs.values())
    imgs = "".join(f'<h2>{f}</h2><a href="{f}"><img src="{f}" alt="{f}"></a>' for f in files)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>AI Ensemble Patterns</title>
<style>
:root {{ --bg:#ffffff; --ink:#1f1f1f; --muted:#666; --rule:#ddd; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --ink:#f2f2f2; --muted:#bbb; --rule:#444; }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --ink:#f2f2f2; --muted:#bbb; --rule:#444; }}
body {{ background:var(--bg); color:var(--ink); font:14px/1.45 system-ui,sans-serif; margin:0 auto; max-width:1500px; padding:16px; }}
table {{ border-collapse:collapse; margin:8px 0 16px; }} td,th {{ border-bottom:1px solid var(--rule); padding:4px 10px; text-align:left; }}
img {{ max-width:100%; background:#fff; border:1px solid var(--rule); }} h2 {{ font-size:15px; margin-top:28px; }}
.note {{ color:var(--muted); }}
</style></head><body>
<h1>AI / physics ensemble patterns</h1>
<p class="note">Local products, not for distribution. Calibrated cutoffs come from 2001-present satellite growth days;
the plume charts' reference lines are uncalibrated starting points. Generated
{dt.datetime.now():%Y-%m-%d %H:%M} local.</p>
<table><tr><th>Model</th><th>Cycle</th><th>Members</th><th>Download</th><th>File</th></tr>{rows}</table>
{imgs}
<details><summary>summary.json</summary><pre>{json.dumps(summary, indent=1)}</pre></details>
</body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    out = run(common.load_settings())
    if a.open:
        webbrowser.open((out / "index.html").as_uri())


def run(cfg):
    """Builds every product from the newest fetched runs. Returns the output folder."""
    global DATA, OUTROOT, CLIMO
    paths = common.Paths(cfg)
    DATA, OUTROOT, CLIMO = paths.runs, paths.output, paths.climo
    import matplotlib
    matplotlib.use("Agg")
    runs = load_runs(cfg)
    if not runs:
        raise SystemExit(f"no model runs in {DATA} -- run scripts/ai_ensemble_fetch.py first")
    any_ds = next(iter(runs.values()))["ds"]
    lats, lons = any_ds.lat.values, any_ds.lon.values
    newest = max(r["cycle"] for r in runs.values())
    dates = common.forecast_dates(newest)          # Day 1 = today
    print("models: " + ", ".join(f"{n} {r['cycle']:%m/%d %HZ}" for n, r in runs.items()))
    climo = load_climo(lats, lons)
    bm = Basemap(cfg, lats, lons)
    out = OUTROOT / dt.datetime.now().strftime("%Y%m%d_%H%M")
    out.mkdir(parents=True, exist_ok=True)
    files, summary = [], {"cycles": {n: f"{r['cycle']:%Y-%m-%dT%HZ}" for n, r in runs.items()}}
    print("  z500 compare"); files += product_compare(runs, cfg, dates, climo, bm, out, lats, lons)
    print("  scenarios"); f, s = product_scenarios(runs, cfg, dates, climo, bm, out, lats, lons)
    files += f; summary["scenarios"] = s
    print("  offshore"); f, s = product_offshore(runs, cfg, out); files += f; summary["offshore"] = s
    print("  regions"); f, s = product_regions(runs, cfg, climo, bm, out, lats, lons)
    files += f; summary["regions"] = s
    import ai_ensemble_calibrated as ens_calibrated
    if ens_calibrated.available():
        print("  calibrated predictors"); f, s = ens_calibrated.product_calibrated(runs, cfg, dates, climo, out, lats, lons)
        files = f + files; summary["calibrated"] = s
        if ens_calibrated.regimes_available():
            print("  regime outlook"); f, s = ens_calibrated.product_regimes(runs, cfg, dates, climo, bm, out, lats, lons)
            files = f + files; summary["regimes"] = s
    else:
        print("  (no calibration yet -- run scripts/ai_ensemble_calibrate.py for the calibrated products)")
    print("  per-model maps"); files += product_model_maps(runs, cfg, dates, climo, bm, out, lats, lons)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_index(out, runs, files, summary)
    print(f"wrote {len(files)} images + index.html to {out}")
    return out


if __name__ == "__main__":
    sys.exit(main())
