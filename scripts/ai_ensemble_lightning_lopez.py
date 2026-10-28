"""Lightning formula layer: the ECMWF IFS lightning parameterization's shape (Lopez 2016) run on IFS ENS open-data
fields, calibrated to Vaisala NLDN strikes. P(>= 1) and P(>= 10) cloud-to-ground strikes per 0.5 deg California cell
and fire day, on the same cells as ai_ensemble_lightning_grid.py's logistic model, for side-by-side use.

    python scripts/ai_ensemble_lightning_lopez.py            # fetch the newest IFS ENS 00z run -> runs/lightning_lopez_<cycle>.npz
    python scripts/ai_ensemble_lightning_lopez.py --cycle 2026102100

Added 2026-10-22 (analyst: "incorporate this new lightning formula into the daily run with the proper credits given";
prototyped 2026-10-21 in incoming/2026-10-21_lightning_proxy/). Why a proxy: ECMWF computes total lightning flash
density in the IFS (Lopez 2016), but that field (litota*) isn't in the free open-data feed, the opencharts API only
serves it as a PNG, and ERA5 on the Copernicus CDS has no lightning variables. The formula is

    f = alpha * Q_R * sqrt(CAPE) * min(z_base, 1.8)**2                (Lopez 2016, flashes km-2 day-1)

with Q_R the charging term (graupel x (cloud liquid + snow) through the 0 to -25 C layer) and z_base the convective
cloud base (km). The open feed has no hydrometeor profiles or cloud base, so the stand-ins are
    graupel flux       -> convective precipitation rate (cp, mm/h)
    (beta q_l + q_s)   -> column cloud ice + 0.7 x column cloud liquid (tciw + 0.7 tclw, kg m-2; 0.7 = Lopez's land weight)
    CAPE               -> most-unstable CAPE (IFS mucape; ERA5's CAPE is also the most-unstable parcel's)
    z_base             -> the surface parcel's LCL, 0.125 km per K of 2 m dewpoint depression
and the exponents, scale and cp gate are fit to NLDN (ai_ensemble_lightning_lopez_calibrate.py, ERA5 2001-2020 Jun-Sep)
rather than taken as 1, 1, 1/2, 2. Its output is therefore NLDN cloud-to-ground strikes, not ECMWF's total flashes.
A cell's fire-day expected strikes (lambda) -> P(>= 1), P(>= 10) by logistic curves in log(lambda) fit on the same
history; the forecast probability is the members' mean.

Forecast side: IFS ENS 00z, the 50 perturbed members (the -enfo-ef files carry no control), every 6 h to
lightning_lopez.max_lead_h (default 180 h = seven fire days; 360 h = 14 since 2026-10-23, ~7.7 GB), cut to the California box at 0.25 deg. cp, tciw and tclw
entered the open feed on 2026-10-20 12z; older cycles can't be run. ~3.9 GB of byte ranges a day at 180 h.
Output <data_dir>/runs/lightning_lopez_<YYYYMMDDHH>.npz: lam (member, day, cell) expected strikes, days (fire days),
cells (the lightning grid's), cycle. ai_ensemble_web_export.build_lightning reads it.

Credits: ECMWF IFS ENS open data (CC BY 4.0); Lopez, P., 2016: A lightning parameterization for the ECMWF Integrated
Forecasting System. Mon. Wea. Rev., 144, 3057-3075, doi:10.1175/MWR-D-16-0026.1; ERA5 (Copernicus Climate Change
Service); Vaisala NLDN (licensed, derived products only).
"""
import argparse
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import ai_ensemble_common as common

BETA = 0.7            # Lopez's land weight on cloud liquid (0.45 over sea); the cells are all land
ZB_CAP = 1.8          # km, Lopez's cap on cloud-base height
RES = 0.25            # source grid (IFS open data and ERA5 are both 0.25 deg)
FIELDS = {"cp": "cp", "mucape": "cape", "tciw": "tciw", "tclw": "tclw", "2t": "t2", "2d": "d2"}   # IFS param -> name here
MIRRORS = ["https://storage.googleapis.com/ecmwf-open-data/", "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/"]
FIRST_CYCLE = dt.datetime(2026, 10, 20, 12)     # first open-data cycle with cp / tciw / tclw
_CACHE = {}
CREDIT = ("Lightning formula: after the ECMWF IFS lightning parameterization (Lopez 2016, Mon. Wea. Rev. 144, 3057-3075), "
          "rebuilt on ECMWF IFS ENS open data (CC BY 4.0) and calibrated to Vaisala NLDN strikes (licensed; derived only) "
          "with ERA5 (Copernicus Climate Change Service).")


# ----------------------------------------------------------------------------- formula

def lcl_km(t2, d2):
    return np.clip(0.125 * (t2 - d2), 0.0, None)


def flash_density(cp_rate, cape, tciw, tclw, t2, d2, p):
    """Strike density (NLDN strikes km-2 day-1 once p is calibrated). cp_rate mm/h; t2 / d2 in K or C (same units)."""
    q = np.clip(tciw, 0, None) + BETA * np.clip(tclw, 0, None)
    zb = np.minimum(lcl_km(t2, d2), ZB_CAP)
    cp = np.clip(cp_rate, 0, None)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        f = (np.exp(p["log_alpha"]) * cp ** p["a_cp"] * q ** p["b_q"] * np.clip(cape, 0, None) ** p["c_cape"]
             * zb ** p["d_zb"])
    return np.where((cp > p["cp_min"]) & (cape > 0), np.nan_to_num(f), 0.0)


def prob(lam, curve):
    """P(>= k strikes) from expected strikes lambda: logistic in log(lambda + eps)."""
    z = curve["a"] + curve["b"] * np.log(np.clip(lam, 0, None) + curve["eps"])
    return 1 / (1 + np.exp(-z))


# ----------------------------------------------------------------------------- cells

def grid_cells():
    """The lightning grid's cells [(j, i)], their centres, and the grid json (None if not built)."""
    cfg = common.load_settings()
    f = common.Paths(cfg).cal / "lightning_grid.json"
    if not f.exists():
        return None, None, None
    J = json.loads(f.read_text())
    cells = [tuple(c) for c in J["cells"]]
    cen = np.array([(J["lat0"] - j * J["grid_res"], J["lon0"] + i * J["grid_res"]) for j, i in cells])
    return cells, cen, J


def cell_weights(lats, lons, cen, half=0.25):
    """(n_cell, n_lat * n_lon) km2 weights: the 0.25 deg points inside each 0.5 deg cell, edge points halved and
    corners quartered, each times its own 0.25 deg area -> sum(w * density) = expected strikes in the cell."""
    W = np.zeros((len(cen), len(lats) * len(lons)))
    area = (RES * 111.2) ** 2 * np.cos(np.deg2rad(lats))
    for n, (la, lo) in enumerate(cen):
        for j, x in enumerate(lats):
            dy = abs(x - la)
            if dy > half + 1e-6:
                continue
            wy = 0.5 if abs(dy - half) < 1e-6 else 1.0
            for i, y in enumerate(lons):
                dx = abs(y - lo)
                if dx > half + 1e-6:
                    continue
                wx = 0.5 if abs(dx - half) < 1e-6 else 1.0
                W[n, j * len(lons) + i] = wy * wx * area[j]
    return W


def domain(cen, pad=0.5):
    return {"lat_min": float(np.floor(cen[:, 0].min() - pad)), "lat_max": float(np.ceil(cen[:, 0].max() + pad)),
            "lon_min": float(np.floor(cen[:, 1].min() - pad)), "lon_max": float(np.ceil(cen[:, 1].max() + pad)), "res": RES}


# ----------------------------------------------------------------------------- calibration

def calibration():
    cfg = common.load_settings()
    f = common.Paths(cfg).cal / "lightning_lopez.json"
    return json.loads(f.read_text()) if f.exists() else None


# ----------------------------------------------------------------------------- forecast fetch

def _path(cyc, step):
    return f"{cyc:%Y%m%d}/{cyc:%H}z/ifs/0p25/enfo/{cyc:%Y%m%d%H}0000-{step}h-enfo-ef"


def fetch(cfg, cycle=None, log=print):
    """Newest IFS ENS 00z cycle -> expected strikes per member, fire day and cell. Returns the npz path (cached)."""
    from ai_ensemble_sources import Http, Sampler
    L = cfg.get("lightning_lopez", {})
    cal = calibration()
    cells, cen, _ = grid_cells()
    if cal is None or cells is None:
        raise RuntimeError("lightning_lopez.json / lightning_grid.json not built -- run the calibrations first")
    max_lead = int(L.get("max_lead_h", 180))
    steps = list(range(0, max_lead + 1, 6))
    http = Http(per_min=3000, workers=8, multirange=False)
    paths = common.Paths(cfg)
    if cycle is None:
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)
        cands = [now.replace(hour=0) - dt.timedelta(days=k) for k in range(3)]
    else:
        cands = [dt.datetime.strptime(cycle, "%Y%m%d%H")]
    base = cyc = None
    for c in cands:
        if c < FIRST_CYCLE:
            continue
        base = next((b for b in MIRRORS if http.exists(b + _path(c, steps[-1]) + ".index")), None)
        if base:
            cyc = c
            break
    if cyc is None:
        raise RuntimeError("no IFS ENS 00z cycle with the lightning fields reaching "
                           f"{steps[-1]} h in the last 3 days")
    out = paths.runs / f"lightning_lopez_{cyc:%Y%m%d%H}.npz"
    if out.exists():
        log(f"  lightning formula: {cyc:%Y%m%d%H}z cached")
        return out
    dom = domain(cen)
    smp = Sampler(dom, {"c": (float(cen[0, 0]), float(cen[0, 1]))})
    ny, nx = len(smp.lats), len(smp.lons)
    arr = {n: np.full((51, len(steps), ny, nx), np.nan, np.float32) for n in FIELDS.values()}

    def one(step):
        r = http.get(base + _path(cyc, step) + ".index")
        if r is None:
            raise RuntimeError(f"lightning formula: no index for {cyc:%Y%m%d%H} step {step}")
        rows = [json.loads(x) for x in r.text.splitlines() if x.strip()]
        rows = [x for x in rows if x["param"] in FIELDS and x.get("levtype") == "sfc" and x.get("type") == "pf"]
        blobs = http.ranges(base + _path(cyc, step) + ".grib2",
                            [(x["_offset"], x["_offset"] + x["_length"] - 1) for x in rows])
        return step, [(FIELDS[x["param"]], int(x["number"]), smp.sample(b)[0]) for x, b in zip(rows, blobs)]

    t0 = dt.datetime.now()
    with ThreadPoolExecutor(8) as ex:
        for step, got in ex.map(one, steps):
            k = steps.index(step)
            for name, m, g in got:
                arr[name][m, k] = g
    mem = [m for m in range(51) if np.isfinite(arr["cp"][m, 1:]).all()]
    if len(mem) < 10:
        raise RuntimeError(f"lightning formula: only {len(mem)} complete members in {cyc:%Y%m%d%H}")
    for n in arr:
        arr[n] = arr[n][mem]
    lam, days = expected_strikes(arr, steps, cyc, smp.lats, smp.lons, cen, cal["params"])
    paths.runs.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.npz")
    np.savez_compressed(tmp, lam=lam.astype(np.float32), days=np.array([f"{d:%Y-%m-%d}" for d in days]),
                        cells=np.array(cells), cycle=f"{cyc:%Y%m%d%H}", members=np.array(mem))
    tmp.replace(out)
    keep = int(cfg.get("keep_cycles", 8))
    for old in sorted(paths.runs.glob("lightning_lopez_??????????.npz"))[:-keep]:
        old.unlink(missing_ok=True)
    log(f"  lightning formula: {cyc:%Y%m%d%H}z, {len(mem)} members, {len(days)} fire days, "
        f"{http.nbytes / 1e9:.1f} GB in {(dt.datetime.now() - t0).seconds / 60:.0f} min")
    return out


def expected_strikes(arr, steps, cyc, lats, lons, cen, p):
    """6-h windows -> fire-day expected strikes per member and cell.

    cp is accumulated, so a window's cp rate is its mean; the instantaneous fields are averaged over the window's two
    ends. A window belongs to the fire day of its midpoint; only fire days with all four windows are kept."""
    from ai_ensemble_calibrated import fire_day_of
    steps = np.array(steps)
    hrs = np.diff(steps)[None, :, None, None]
    cp_rate = np.diff(arr["cp"], axis=1) * 1000 / hrs
    f = 0.0
    for k in (slice(None, -1), slice(1, None)):
        f = f + flash_density(cp_rate, arr["cape"][:, k], arr["tciw"][:, k], arr["tclw"][:, k], arr["t2"][:, k],
                              arr["d2"][:, k], p)
    f = f / 2                                                    # (member, window, y, x) strikes km-2 day-1
    mids = [cyc + dt.timedelta(hours=int(s) - 3) for s in steps[1:]]
    fd = np.array(fire_day_of(mids))
    days = [d for d in sorted(set(fd)) if (fd == d).sum() == 24 // int(hrs.flat[0])]
    W = cell_weights(lats, lons, cen)
    nm = f.shape[0]
    lam = np.zeros((nm, len(days), len(cen)))
    for k, d in enumerate(days):
        dens = f[:, fd == d].mean(axis=1).reshape(nm, -1)         # strikes km-2 over the day
        lam[:, k] = dens @ W.T
    return lam, days


def forecast(npz, dates):
    """{"p1": (day, cell), "p10": (day, cell), "lam": (day, cell)} ensemble means on the given fire dates
    (NaN where the run doesn't cover a date), or None."""
    cal = calibration()
    if cal is None or npz is None:
        return None
    d = np.load(npz)
    days = list(d["days"])
    lam = d["lam"]
    out = {k: np.full((len(dates), lam.shape[2]), np.nan) for k in ("p1", "p10", "lam")}
    for n, x in enumerate(dates):
        s = f"{x:%Y-%m-%d}"
        if s in days:
            k = days.index(s)
            out["p1"][n] = prob(lam[:, k], cal["curves"]["p1"]).mean(0)
            out["p10"][n] = prob(lam[:, k], cal["curves"]["p10"]).mean(0)
            out["lam"][n] = lam[:, k].mean(0)
    return out


def footprint_cells(cfg):
    """{footprint id: [index into the grid cells]}: the calibrated outlook's own footprint cells
    (ai_ensemble_calibrated._footprint_mask on the 0.5 deg model grid, which the lightning grid shares), so the
    history (<fid>_ltgformula) and the forecast sum the same cells. California footprints only."""
    if "fp" in _CACHE:
        return _CACHE["fp"]
    import ai_ensemble_calibrated as C
    cells, _, J = grid_cells()
    if J is None:
        return {}
    d = cfg["domain"]
    lats = np.arange(d["lat_max"], d["lat_min"] - 0.01, -d["res"])
    lons = np.arange(d["lon_min"], d["lon_max"] + 0.01, d["res"])
    if abs(lats[0] - J["lat0"]) > 1e-6 or abs(lons[0] - J["lon0"]) > 1e-6 or abs(d["res"] - J["grid_res"]) > 1e-6:
        raise RuntimeError("lightning grid and model domain grids differ -- rebuild lightning_grid.json")
    C.region_masks(cfg, lats, lons)
    out = {}
    for fid, f in cfg.get("footprints", {}).items():
        m = C._footprint_mask(fid, lats, lons)
        idx = [n for n, (j, i) in enumerate(cells) if m is not None and m[j, i]]
        if idx:
            out[fid] = idx
    _CACHE["fp"] = out
    return out


def footprint_forecast(npz, idx, dates):
    """(member, day) expected strikes in a footprint (sum over its cells) on the given fire dates, NaN where the run
    doesn't cover a date -- the forecast side of the outlook's <fid>_ltgformula rows."""
    d = np.load(npz)
    days = list(d["days"])
    lam = d["lam"]
    out = np.full((lam.shape[0], len(dates)), np.nan)
    for n, x in enumerate(dates):
        s = f"{x:%Y-%m-%d}"
        if s in days:
            out[:, n] = lam[:, days.index(s)][:, idx].sum(axis=1)
    return out


def latest_npz(cfg, on_date=None):
    """The newest lightning_lopez run (or the one for cycle date on_date)."""
    runs = sorted(common.Paths(cfg).runs.glob("lightning_lopez_??????????.npz"))
    if on_date is not None:
        runs = [r for r in runs if r.stem.endswith(f"{on_date:%Y%m%d}00")]
    return runs[-1] if runs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycle", help="YYYYMMDDHH (default: newest 00z)")
    a = ap.parse_args()
    print(fetch(common.load_settings(), a.cycle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
