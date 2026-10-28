"""Fetch the newest complete run of each enabled ensemble into <data_dir>/runs/<model>_<YYYYMMDDHH>.nc.

    python scripts/ai_ensemble_fetch.py                     # every enabled model, newest complete cycle
    python scripts/ai_ensemble_fetch.py --models aifs_ens gefs
    python scripts/ai_ensemble_fetch.py --refresh           # re-download even if the cycle's file exists
    python scripts/ai_ensemble_fetch.py --estimate          # print request/download counts, fetch nothing

Models run in parallel (different servers); AIGEFS is the slow one (NOMADS ~100 requests/min,
~30-60 min a run). See ai_ensemble_sources.py for what each source is.

Promoted 2026-10-10 from the incoming/2026-10-10_ai_ensembles sandbox; settings in
config/ai_ensemble_settings.json, paths via ai_ensemble_common.py, docs in docs/ai_ensembles/README.md.
"""
import argparse
import datetime as dt
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

import ai_ensemble_common as common
import ai_ensemble_sources as ens_sources

DATA = None   # set from settings in main() / run()
_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True)


def load_config():
    return common.load_settings()


def write_run(path, name, cfg, cycle, members, store, sampler, src, elapsed):
    import xarray as xr
    pts = sampler.pnames
    ds = xr.Dataset(coords={
        "member": members, "lead_h": np.array(store.steps, dtype=np.int32),
        "lat": sampler.lats, "lon": sampler.lons, "point": pts,
        "point_lat": ("point", sampler.plat), "point_lon": ("point", sampler.plon)})
    units = {"z500": "m", "msl": "hPa", "u850": "m s-1", "v850": "m s-1", "t850": "degC",
             "tp": "mm", "t700": "degC", "q700": "g kg-1", "rh700": "%", "t2m": "degC", "d2m": "degC",
             "rh2m": "%", "cape": "J kg-1", "u10": "m s-1", "v10": "m s-1"}
    enc = {}
    for f in store.fields:
        ds[f] = (("member", "lead_h", "lat", "lon"), store.grid[f], {"units": units[f], **({"description": "accumulated since init"} if f == "tp" else {})})
        ds[f + "_pt"] = (("member", "lead_h", "point"), store.pts[f], {"units": units[f]})
        enc[f] = {"zlib": True, "complevel": 4, "least_significant_digit": 1}
    ds.attrs.update(model=name, label=cfg["models"][name]["label"], kind=cfg["models"][name]["kind"],
                    cycle=cycle.strftime("%Y-%m-%dT%H:00Z"), source=getattr(src, "base", ""),
                    fetched=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                    download_gb=round(src.http.nbytes / 1e9, 2), requests=src.http.nreq,
                    missing_files=int(getattr(src.http, "failed_files", 0)),
                    minutes=round(elapsed / 60, 1))
    tmp = path.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp, encoding=enc)
    tmp.replace(path)
    archive_run(ds, path)


def archive_run(ds, path):
    """Keep every run's member values at the named points (all fields, every step) plus the 500 hPa
    height on a coarse 2.5 deg grid (the regime box) in <data_dir>/archive/, forever -- about 1 MB a
    run. The full runs are pruned to keep_cycles; this is the verification record that lets the
    models be scored against what happened (added 2026-10-11, roadmap item "start archiving our own
    daily runs now")."""
    try:
        arch = path.parent.parent / "archive"
        arch.mkdir(exist_ok=True)
        keep = [v for v in ds.data_vars if v.endswith("_pt")]
        sub = ds[keep]
        z = ds["z500"].isel(lat=slice(None, None, 5), lon=slice(None, None, 5))   # 0.5 -> 2.5 deg
        sub = sub.assign(z500_2p5=z.rename({"lat": "lat_2p5", "lon": "lon_2p5"}))
        # footprint means of 2 m VPD x 10 m wind and 2 m RH (member, step), for measuring each
        # model's bias against the ERA5 footprint history (added 2026-10-12)
        try:
            import ai_ensemble_calibrated as C
            cfg = common.load_settings()
            C.FOOTPRINTS.update(cfg.get("footprints", {}))
            for fid in cfg.get("footprints", {}):
                mask = C._footprint_mask(fid, ds.lat.values, ds.lon.values)
                for kind in ("vpd_wind", "rh_min", "capep90"):
                    v = None if mask is None else C.footprint_series(ds, mask, kind)
                    if v is not None:
                        sub[f"fp_{fid}_{kind}"] = (("member", "lead_h"), v.astype("float32"))
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: footprint series not archived ({type(e).__name__}: {e})")
        enc = {v: {"zlib": True, "complevel": 5, "least_significant_digit": 1} for v in sub.data_vars}
        out = arch / (path.stem + "_archive.nc")
        tmp = out.with_suffix(".tmp.nc")
        sub.to_netcdf(tmp, encoding=enc)
        tmp.replace(out)
    except Exception as e:  # noqa: BLE001 -- the archive must never fail a fetch
        log(f"WARNING: archive of {path.name} failed ({type(e).__name__}: {e})")


def fields_for(name, cfg, src):
    """The common fields plus this model's extra_fields (settings), keeping only ones the source has."""
    have = getattr(src, "params", None) or getattr(src, "vars", {})
    extra = [f for f in cfg.get("extra_fields", {}).get(name, []) if f in have and f not in cfg["fields"]]
    return list(cfg["fields"]) + extra


def prune(name, keep):
    files = sorted(DATA.glob(f"{name}_??????????.nc"))
    for old in files[:-keep]:
        old.unlink(missing_ok=True)


def fetch_model(name, cfg, refresh, estimate, results, mem_limit=None):
    try:
        src = ens_sources.make_source(name)
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        cycle = src.pick_cycle(now, cfg["max_lead_h"], log)
        path = DATA / f"{name}_{cycle:%Y%m%d%H}.nc"
        steps = list(range(0, cfg["max_lead_h"] + 1, cfg["step_h"]))
        if path.exists() and not refresh and not estimate:
            log(f"{name}: {cycle:%Y-%m-%d %HZ} already in {path.name}")
            results[name] = ("cached", path)
            return
        if estimate:
            nf = len(cfg["fields"])
            mem = 50 if name in ("aifs_ens", "ifs_ens") else 31
            log(f"{name}: cycle {cycle:%Y-%m-%d %HZ}; ~{mem} members x {len(steps)} steps x {nf} fields "
                f"= {mem * len(steps) * nf} global messages (~0.4-1 MB each)")
            results[name] = ("estimate", None)
            return
        log(f"{name}: fetching {cycle:%Y-%m-%d %HZ}, {len(steps)} steps to {steps[-1]} h")
        sampler = ens_sources.Sampler(cfg["domain"], cfg["points"])
        t0 = time.time()
        if isinstance(src, ens_sources.NoaaEns):
            store, members = src.fetch(cycle, steps, fields_for(name, cfg, src), sampler, log, cfg["precip_step_h"], mem_limit)
        else:
            store, members = src.fetch(cycle, steps, fields_for(name, cfg, src), sampler, log, mem_limit)
        DATA.mkdir(parents=True, exist_ok=True)
        write_run(path, name, cfg, cycle, members, store, sampler, src, time.time() - t0)
        prune(name, cfg["keep_cycles"])
        miss = {f: float(np.isnan(store.grid[f]).mean()) for f in store.fields}
        log(f"{name}: wrote {path.name} ({path.stat().st_size / 1e6:.0f} MB; "
            f"{src.http.nbytes / 1e9:.2f} GB downloaded in {(time.time() - t0) / 60:.1f} min); "
            f"missing fraction {', '.join(f'{k} {v:.0%}' for k, v in miss.items() if v)}" if any(miss.values())
            else f"{name}: wrote {path.name} ({path.stat().st_size / 1e6:.0f} MB; "
                 f"{src.http.nbytes / 1e9:.2f} GB downloaded in {(time.time() - t0) / 60:.1f} min)")
        results[name] = ("ok", path)
    except NotImplementedError as e:
        log(f"{name}: skipped -- {e}")
        results[name] = ("skipped", str(e))
    except Exception as e:  # noqa: BLE001 -- one model failing shouldn't stop the others
        log(f"{name}: FAILED {type(e).__name__}: {e}")
        results[name] = ("failed", f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--max-lead", type=int, help="testing: stop at this lead (h)")
    ap.add_argument("--mem-limit", type=int, help="testing: first N members only")
    a = ap.parse_args()
    cfg = load_config()
    return run(cfg, a.models, a.refresh, a.estimate, a.max_lead, a.mem_limit)


def run(cfg, models=None, refresh=False, estimate=False, max_lead=None, mem_limit=None):
    """Fetches every enabled (or listed) model in parallel. Returns {model: (status, detail)}."""
    global DATA
    DATA = common.Paths(cfg).runs
    a = argparse.Namespace(models=models, refresh=refresh, estimate=estimate, max_lead=max_lead,
                           mem_limit=mem_limit)
    if a.max_lead:
        cfg["max_lead_h"] = a.max_lead
    if a.max_lead or a.mem_limit:
        DATA = common.Paths(cfg).data / "runs_test"
    names = a.models or [m for m, v in cfg["models"].items() if v["enabled"]]
    results = {}
    threads = [threading.Thread(target=fetch_model, args=(n, cfg, a.refresh, a.estimate, results, a.mem_limit)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("summary: " + "; ".join(f"{n} {results.get(n, ('?',))[0]}" for n in names))
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if all(v[0] in ("ok", "cached", "estimate", "skipped") for v in res.values()) else 1)
