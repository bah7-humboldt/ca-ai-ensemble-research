"""Daily AI ensemble pattern products: fetch the newest runs, build the products, refresh latest/.

    python scripts/ai_ensemble_daily.py           # what the scheduled task runs
    python scripts/ai_ensemble_daily.py --force   # run even if config/ai_ensemble_settings.json has enabled: false

Added 2026-10-10 (analyst: "set up a daily task that will have the latest products prepared by
0600 PDT"). Scheduled by scripts/setup_ai_ensemble_task.ps1 at task_hour_local:task_minute_local
(02:30 by default). All four 00Z runs are posted by ~01:00 PDT and AIGEFS takes ~1 h from NOMADS,
so products are ready ~03:45.

1. Fetch (ai_ensemble_fetch.run): every enabled model's newest complete cycle into
   <data_dir>/runs/. A model that fails keeps its previous run; the products use whatever is newest
   and the summary flags anything older than STALE_HOURS.
   In parallel: the FEMS 7-day NFDRS forecast per PSA (ai_ensemble_nfdrs.fetch_forecast) for the
   weather + fuels odds; if it fails the products go without them.
   Then the lightning formula layer's IFS ENS fields (ai_ensemble_lightning_lopez.fetch, ~5 min) when
   lightning_lopez.enabled; a failure is flagged on the summary line and the map keeps the previous run.
2. Products (ai_ensemble_products.run) -> <output_dir>/<YYYYMMDD_HHMM>/.
3. latest/: replaced by a copy of that folder only after the products finished, so it always
   holds one complete set. Dated folders beyond keep_output_runs are deleted.
4. Web viewer (web_export, ai_ensemble_web_export.py) -> <output_dir>/web/, pushed to GitHub Pages
   when web_publish is true. A failure here is logged and flagged but doesn't fail the run.
5. Logs: <output_dir>/logs/ai_ensembles_<yyyymmdd>.log (full output) and ai_ensembles_history.log
   (one line per run). Exit code 1 if the products failed or no model fetched.

The calibration history (ai_ensemble_history_*.py, ai_ensemble_calibrate.py) is not part of the
daily run; it changes only when it's rebuilt (see docs/ai_ensembles/README.md). Nothing is
published (local products only).
"""
import argparse
import datetime as dt
import shutil
import sys
import time

import ai_ensemble_common as common
import run_log

STALE_HOURS = 30


def refresh_latest(paths, out):
    tmp = paths.output / "latest.part"
    old = paths.output / "latest.old"
    for p in (tmp, old):
        if p.exists():
            shutil.rmtree(p)
    shutil.copytree(out, tmp)
    if paths.latest.exists():
        paths.latest.rename(old)
    tmp.rename(paths.latest)
    if old.exists():
        shutil.rmtree(old)


def prune_outputs(paths, keep):
    runs = sorted(p for p in paths.output.iterdir()
                  if p.is_dir() and len(p.name) == 13 and p.name[8] == "_" and p.name.replace("_", "").isdigit())
    for p in runs[:-keep] if keep > 0 else []:
        shutil.rmtree(p, ignore_errors=True)
    return max(0, len(runs) - keep)


def stale_models(cfg, paths):
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    out = []
    for name, m in cfg["models"].items():
        if not m["enabled"]:
            continue
        files = sorted(paths.runs.glob(f"{name}_??????????.nc"))
        if not files:
            out.append(f"{name} none")
            continue
        cyc = dt.datetime.strptime(files[-1].stem.rsplit("_", 1)[1], "%Y%m%d%H")
        age = (now - cyc).total_seconds() / 3600
        if age > STALE_HOURS:
            out.append(f"{name} {cyc:%m/%d %HZ} ({age:.0f} h old)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = common.load_settings()
    paths = common.Paths(cfg)
    log_dir = str(paths.logs.relative_to(common.ROOT)) if paths.logs.is_relative_to(common.ROOT) else str(paths.logs)
    run_log.start(log_dir, "ai_ensembles")
    if not cfg.get("enabled", True) and not a.force:
        print("ai_ensemble_settings.json has enabled: false -- nothing to do (--force to run anyway)")
        run_log.summary(log_dir, "ai_ensembles", "disabled -- skipped")
        return 0
    t0 = time.time()
    import threading
    import ai_ensemble_fetch
    import ai_ensemble_nfdrs
    import ai_ensemble_products
    # FEMS 7-day NFDRS forecast for the weather + fuels odds, alongside the model downloads (~5 min)
    fuels_note = {"v": "fuels: not run"}

    def fuels():
        try:
            t = ai_ensemble_nfdrs.fetch_forecast(cfg)
            fuels_note["v"] = f"fuels: {t.psa.nunique() if len(t) else 0} PSAs"
        except Exception as e:  # noqa: BLE001 -- the products just go without the fuels odds
            print(f"NFDRS FORECAST FAILED: {type(e).__name__}: {e}")
            fuels_note["v"] = f"FUELS FAILED ({type(e).__name__})"
    ft = threading.Thread(target=fuels)
    ft.start()
    try:
        results = ai_ensemble_fetch.run(cfg)
    except Exception as e:  # noqa: BLE001 -- products can still run on the previous runs
        print(f"FETCH FAILED: {type(e).__name__}: {e}")
        results = {}
    ft.join()
    # Lightning formula layer (2026-10-22): its own IFS ENS fields (cp, cloud ice/liquid, CAPE, 2 m T/Td) every 6 h.
    # A failure keeps the previous run's file; the map shows whatever is newest.
    ltg_note = ""
    if cfg.get("lightning_lopez", {}).get("enabled", False):
        try:
            import ai_ensemble_lightning_lopez
            f = ai_ensemble_lightning_lopez.fetch(cfg)
            ltg_note = f"; lightning formula {f.stem.rsplit('_', 1)[1]}z"
        except Exception as e:  # noqa: BLE001 -- the other products don't need it
            print(f"LIGHTNING FORMULA FAILED: {type(e).__name__}: {e}")
            ltg_note = f"; LIGHTNING FORMULA FAILED ({type(e).__name__})"
    fetched = [n for n, (st, _) in results.items() if st in ("ok", "cached")]
    failed = [n for n, (st, _) in results.items() if st == "failed"]
    try:
        out = ai_ensemble_products.run(cfg)
    except BaseException as e:  # noqa: BLE001 -- SystemExit included: log it, keep latest/ as it was
        print(f"PRODUCTS FAILED: {type(e).__name__}: {e}")
        run_log.summary(log_dir, "ai_ensembles", f"FAILED products ({type(e).__name__}: {e}); fetch ok "
                        f"{','.join(fetched) or 'none'}; failed {','.join(failed) or 'none'}")
        return 1
    refresh_latest(paths, out)
    pruned = prune_outputs(paths, int(cfg.get("keep_output_runs", 14)))
    web_note = ""
    if cfg.get("web_export", True):
        try:
            import ai_ensemble_web_export
            web = ai_ensemble_web_export.run(cfg)
            web_note = "; web built"
            if cfg.get("web_publish", False):
                web_note = "; web published" if ai_ensemble_web_export.publish(web, cfg) else "; WEB PUBLISH FAILED"
        except Exception as e:  # noqa: BLE001 -- the local products are already done
            print(f"WEB EXPORT FAILED: {type(e).__name__}: {e}")
            web_note = f"; WEB EXPORT FAILED ({type(e).__name__})"
    stale = stale_models(cfg, paths)
    line = (f"ok {out.name} in {(time.time() - t0) / 60:.0f} min; fetched {','.join(fetched) or 'none'}"
            + (f"; FAILED {','.join(failed)}" if failed else "")
            + (f"; STALE {', '.join(stale)}" if stale else "")
            + (f"; pruned {pruned}" if pruned else "") + web_note + ltg_note + "; " + fuels_note["v"])
    print(line)
    run_log.summary(log_dir, "ai_ensembles", line)
    return 0 if fetched else 1


if __name__ == "__main__":
    sys.exit(main())
