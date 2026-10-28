"""Reference data, step 1: download each roster station's NFDRS daily history from FEMS.
Step 2 (scripts/climo_daily.py) reads what this writes.

Moved 2026-10-03 from scripts/FDRACharts.py's fems_data_download() -- same request, same
per-station CSV (columns, order, "_Season" copy), same folder layout (scripts/climo_common.py):

    <climatology_dir>/<GACC>/<Period>/<FDRA>/<StationID>.csv
    <climatology_dir>/<GACC>/<Period>/<FDRA>/<StationID>_Season.csv   (Historical + season on)

Periods: Historical (climatology_hist_startdate..enddate -- the 20-year record the
percentiles come from), CurrentYear (Jan 1..Dec 31 this year) and Forecast (today..+7 days).
CurrentYear/Forecast only feed the static Pocket Card charts; the interactive charts and
danger maps pull their own current data.

FEMS's response changed after the existing ONCC history was pulled (June 2026): it now pads
every station back to startDate with blank rows and adds a duplicate "StationId" column. Both
are stripped (_match_original_format), so a new download has the same columns and dates as
the existing files -- checked on station 40102 2026-10-03. FEMS has also QC-revised ~1% of past
values since then, so a fresh download is close to, not identical with, the June pull.

Other differences from FDRACharts.py, none of which change a downloaded file:
  * Historical resumes: stations whose CSV already exists are skipped and only missing ones
    are downloaded (FDRACharts.py skipped the WHOLE download if any CSV existed). --force
    re-downloads every station.
  * A station that fails is listed at the end, and the exit code is 1 if any failed.
  * Requests time out after 300 s instead of waiting forever.
  * The Staging forecast URL had endDate misspelled "enddate", so FEMS ignored it there.

A full ONCC Historical pull is ~122 stations x 20 years -- expect tens of minutes. Use
--station to try one or two first.

    python scripts/climo_download.py                         # Historical, missing stations only
    python scripts/climo_download.py --station 40102         # just one station
    python scripts/climo_download.py --period Forecast --staging
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

import fems_api
from climo_common import PERIODS, climo_settings, load_roster, log, period_dir

REQUEST_TIMEOUT_S = 300


def period_dates(period, hist_startdate, hist_enddate):
    if period == "Historical":
        return hist_startdate, hist_enddate
    if period == "CurrentYear":
        current_year = datetime.now().year
        return f"{current_year}-01-01", f"{current_year}-12-31"
    datetime_now = datetime.now()
    return datetime_now.strftime("%Y-%m-%d"), (datetime_now + timedelta(days=7)).strftime("%Y-%m-%d")


NFDR_VALUE_COLS = ["1HrFM", "10HrFM", "100HrFM", "1000HrFM", "KBDI", "GSI", "WoodyFM", "HerbFM",
                   "IC", "ERC", "SC", "BI"]


def _match_original_format(station_df):
    """Puts a FEMS response back in the shape the existing history (pulled June 2026) has, which
    every later step was built on. Since then FEMS (checked 2026-10-03):
      * returns a row for every day from startDate, even years before the station existed, with
        every index blank -- np.percentile() in step 3 turns any blank into a NaN breakpoint, so
        those rows are dropped (a row with ANY index value is kept);
      * adds its own "StationId" column next to the "stationId" this script adds -- dropped."""
    station_df = station_df.drop(columns=["StationId"], errors="ignore")
    value_cols = [c for c in NFDR_VALUE_COLS if c in station_df.columns]
    if value_cols:
        station_df = station_df.dropna(subset=value_cols, how="all").reset_index(drop=True)
    return station_df


def download_period(df_stations, gacc, outdir, period, hist_startdate, hist_enddate,
                    season_enabled, season_startdate, season_enddate,
                    force=False, staging=False, only_stations=None):
    """Downloads one GACC's stations for one period. Returns the list of station ids that
    failed (empty = all good)."""
    log("​")
    log(f"DOWNLOADING '{period}' STATION DATA")
    startdate, enddate = period_dates(period, hist_startdate, hist_enddate)
    data_dir = period_dir(outdir, gacc, period)
    os.makedirs(data_dir, exist_ok=True)
    failed = []
    skipped = 0
    FDRA_list = list(df_stations['FDRA'].unique())
    FDRAname_list = list(df_stations['FDRAName'].unique())
    for i in range(0, len(FDRA_list)):
        FDRA = FDRA_list[i]
        FDRA_name = FDRAname_list[i]
        FDRA_dir = f"{data_dir}/{FDRA}"
        os.makedirs(FDRA_dir, exist_ok=True)
        log(f"....{FDRA}")
        station_list = list(df_stations[df_stations['FDRA'] == FDRA]["StationID"])
        for station in station_list:
            if only_stations and station not in only_stations:
                continue
            csv_outpath = f"{FDRA_dir}/{station}.csv"
            if period == "Historical" and not force and os.path.exists(csv_outpath):
                skipped += 1
                continue
            fuelmodel = df_stations[df_stations["StationID"] == station]["FuelModel"].iloc[0]
            log(f"......{station}")
            try:
                station_df = fems_api.download_daily_summary(
                    station, fuelmodel, startdate, enddate,
                    dataset="forecast" if period == "Forecast" else "observation",
                    staging=staging, timeout=REQUEST_TIMEOUT_S)
                station_df = _match_original_format(station_df)
                #Add stationId to the dataframe
                station_df["stationId"] = station
                #Add FDRA and FDRAName to the dataframe
                station_df["GACC"] = gacc
                station_df["FDRA"] = FDRA
                station_df["FDRAName"] = FDRA_name
                #Adjust field order
                index_cols = [col for col in station_df.columns if col not in ["GACC", "FDRA", "FDRAName", "stationId", "StationName", "ObservationTime"]]
                station_df = station_df[["GACC", "FDRA", "FDRAName", "stationId", "StationName", "ObservationTime"] + index_cols]
            except Exception as e:
                log(f"........FAILED TO DOWNLOAD ({e})")
                failed.append(station)
                continue
            station_df.to_csv(csv_outpath, index=False)
            #Season copy: only the rows inside the season_startdate..season_enddate MM-DD
            #window. Historical only -- CurrentYear/Forecast are short ranges already.
            if season_enabled and period == "Historical":
                if(len(station_df) > 1):
                    station_df['ObservationTime'] = pd.to_datetime(station_df['ObservationTime'])
                    station_df['ObservationTime_mmdd'] = station_df['ObservationTime'].dt.strftime('%m-%d')
                    start_mmdd = pd.to_datetime(season_startdate, format='%m-%d').strftime('%m-%d')
                    end_mmdd = pd.to_datetime(season_enddate, format='%m-%d').strftime('%m-%d')
                    season_test = (station_df['ObservationTime_mmdd'] >= start_mmdd) & (station_df['ObservationTime_mmdd'] <= end_mmdd)
                    station_df = station_df[season_test]
                    station_df['ObservationTime'] = station_df['ObservationTime'].dt.strftime("%Y-%m-%d")
                    station_df.drop(columns='ObservationTime_mmdd', inplace=True)
                station_df.to_csv(f"{FDRA_dir}/{station}_Season.csv", index=False)
    if skipped:
        log(f"..{skipped} station(s) already downloaded, skipped (--force to re-download)")
    return failed


def main():
    s = climo_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", choices=PERIODS, default="Historical")
    ap.add_argument("--gacc", action="append", help="only this GACC (repeatable); default: every GACC in the roster")
    ap.add_argument("--station", action="append", help="only this station id (repeatable)")
    ap.add_argument("--force", action="store_true", help="re-download stations that already have a CSV")
    ap.add_argument("--staging", action="store_true", help="use FEMS staging instead of production")
    args = ap.parse_args()
    failed = []
    for gacc, df_stations in load_roster(s["station_csv"], args.gacc).items():
        failed += download_period(df_stations, gacc, s["outdir"], args.period,
                                  s["hist_startdate"], s["hist_enddate"], s["season_enabled"],
                                  s["season_startdate"], s["season_enddate"],
                                  force=args.force, staging=args.staging, only_stations=args.station)
    if failed:
        log(f"\n{len(failed)} station(s) failed to download: {', '.join(failed)} -- re-run to retry them")
        sys.exit(1)


if __name__ == "__main__":
    main()
