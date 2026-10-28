"""Refresh the CurrentYear NFDRS observations the AI ensemble products rank against -- added
2026-10-28 for the GitHub Actions run (analyst: move the AI ensemble daily run off the local PC).

    python scripts/ai_ensemble_refresh_current.py            # ONCC, OSCC and Hawaii
    python scripts/ai_ensemble_refresh_current.py --gacc HI

The products read <climatology_dir>/<GACC>/{Historical,CurrentYear}/<PSA>/<PSA>_AvgDailyExtremes.csv
(ai_ensemble_nfdrs.load_history, ai_ensemble_web_pages). Locally, only ONCC's CurrentYear is kept
current (scripts/climo_refresh.py, nightly from fdra_charts_web_export.py); OSCC's and Hawaii's
tree in climatology_dir was never refreshed. This does all three, against the same rosters the
AI ensemble's FEMS forecast uses (ai_ensemble_nfdrs.ROSTERS), so the Historical and CurrentYear
station sets match.

CurrentYear only: the forecast side is ai_ensemble_nfdrs.fetch_forecast's own job, written under
<data_dir>/nfdrs/. Same request and files as climo_refresh.refresh_current (one FEMS request per
station, Jan 1 .. today); a failed station is tried once more, then keeps its previous CSV (on the
runner, where station CSVs aren't kept, it's left out of that day's PSA average). Exit 1 when any
station failed, so the workflow can flag it; the run goes on with whatever is on disk.
"""
import argparse
import sys

import ai_ensemble_nfdrs
import pipeline_settings
from climo_common import climo_settings, load_roster, log, period_dir, toggle
from climo_daily import calculate_daily_extremes
from climo_download import download_period

PERIOD = "CurrentYear"


def refresh(gacc, roster_csv):
    """One GACC's CurrentYear station CSVs + daily-extremes files. Returns failed station ids."""
    s = climo_settings()
    rosters = load_roster(roster_csv, [gacc])
    if gacc not in rosters:
        log(f"  {gacc}: no rows in {roster_csv.name} -- skipped")
        return []
    df_stations = rosters[gacc]
    args = (df_stations, gacc, s["outdir"], PERIOD, s["hist_startdate"], s["hist_enddate"],
            s["season_enabled"], s["season_startdate"], s["season_enddate"])
    failed = download_period(*args)
    if failed:  # FEMS drops the odd connection; one more try for just those stations
        failed = download_period(*args, only_stations=set(failed))
    calculate_daily_extremes(df_stations, gacc, period_dir(s["outdir"], gacc, PERIOD),
                             s["index_list"], toggle(s["season_enabled"]))
    return sorted(set(failed))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gacc", action="append", choices=sorted(ai_ensemble_nfdrs.ROSTERS),
                    help="only this GACC (repeatable); default: all of them")
    a = ap.parse_args()
    pipeline_settings.load_settings()
    failed = []
    for gacc in a.gacc or sorted(ai_ensemble_nfdrs.ROSTERS):
        log(f"Refreshing {gacc} {PERIOD}...")
        f = refresh(gacc, ai_ensemble_nfdrs.ROSTERS[gacc])
        log(f"  {gacc}: " + ("ok" if not f else f"{len(f)} station(s) failed, old CSVs kept: {', '.join(f)}"))
        failed += f
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
