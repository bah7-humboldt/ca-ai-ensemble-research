"""Shared pieces for the reference-data (climatology) build steps -- added 2026-10-03 when
scripts/FDRACharts.py's one big script was broken out into separate steps a new analyst can
run in order (see docs/reference_data_setup.md and scripts/build_reference_data.py):

    climo_download.py     step 1  FEMS daily history per roster station
    climo_daily.py        step 2  daily extremes + day-of-year climatology (station/PSA/GACC)
    climo_percentiles.py  step 3  percentile breakpoints + percentile tables (station/PSA/GACC)

This module holds what all three share: the climatology settings, the station roster, and
the folder layout. The layout is exactly what FDRACharts.py wrote, because everything
downstream (export_*.py, fdra_charts_web_export.py, the PSA danger maps' percentile curves)
reads those paths:

    <climatology_dir>/<GACC>/<Period>/<FDRA>/<StationID>.csv           raw FEMS download
    <climatology_dir>/<GACC>/<Period>/<FDRA>/<StationID>_DOY.csv       ...and its products
    <climatology_dir>/<GACC>/<Period>/<FDRA>/<FDRA>_AvgDailyExtremes.csv
    <climatology_dir>/<GACC>/<Period>/<GACC>_PercentileTable.csv
    (Period = Historical | CurrentYear | Forecast; "_Season" variants alongside)
"""
import pathlib

import pandas as pd

import pipeline_settings

ROOT = pathlib.Path(__file__).resolve().parent.parent
PERIODS = ("Historical", "CurrentYear", "Forecast")


def log(message):
    """print() that survives a console code page that can't encode a station name."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(str(message).encode("ascii", "ignore").decode("ascii"), flush=True)


def climo_settings():
    """The climatology_* settings from config/default_settings.json, with paths resolved
    against the repo root. Season dates are MM-DD; the toggle is a real bool here and is
    turned into FDRACharts.py's "true"/"false" strings only where the moved code needs it."""
    s = pipeline_settings.load_settings()
    return {
        "station_csv": ROOT / s["climatology_station_csv"],
        "index_list": list(s["climatology_index_list"]),
        "outdir": ROOT / s["climatology_dir"],
        "hist_startdate": s["climatology_hist_startdate"],
        "hist_enddate": s["climatology_hist_enddate"],
        "season_enabled": bool(s["climatology_season_enabled"]),
        "season_startdate": s["climatology_season_startdate"],
        "season_enddate": s["climatology_season_enddate"],
    }


def load_roster(station_csv, gaccs=None):
    """{GACC: df_stations} from the station roster CSV, normalized exactly the way
    FDRACharts.py's main loop did it (StationId -> StationID, sorted by GACC/FDRA/StationID,
    core + overlay columns, duplicates dropped). Row order matters: the steps walk FDRAs in
    df_stations order and pair FDRA with FDRAName by position, same as before."""
    station_df_all = pd.read_csv(station_csv, dtype=str).fillna("")
    if "StationId" in station_df_all.columns and "StationID" not in station_df_all.columns:
        station_df_all.rename(columns={"StationId": "StationID"}, inplace=True)
    rosters = {}
    for gacc in sorted(set(station_df_all["GACC"])):
        if gaccs and gacc not in gaccs:
            continue
        df_stations = station_df_all[station_df_all["GACC"] == gacc].copy()
        df_stations.sort_values(by=["GACC", "FDRA", "StationID"], inplace=True)
        core_cols = ["GACC", "FDRA", "FDRAName", "StationID", "StationName", "FuelModel"]
        overlay_cols = [col for col in ["OverlayYear", "overlay_year", "Overlay_Year"]
                        if col in df_stations.columns]
        rosters[gacc] = df_stations[core_cols + overlay_cols].drop_duplicates().reset_index(drop=True)
    return rosters


def period_dir(outdir, gacc, period):
    """Folder for one GACC's period, as the forward-slash string the moved code builds its
    paths from (FDRACharts.py used f"{outdir}/{gacc}/{period}")."""
    return f"{pathlib.Path(outdir).as_posix()}/{gacc}/{period}"


def roster_gacc():
    """The one GACC in the climatology roster. The export_*.py steps write flat files in
    config/ (one GACC per install), so a multi-GACC roster has to be split first."""
    gaccs = sorted(load_roster(climo_settings()["station_csv"]))
    if len(gaccs) != 1:
        raise SystemExit(f"climatology_station_csv lists {len(gaccs)} GACCs ({', '.join(gaccs)}); "
                         f"the config/ exports need exactly one")
    return gaccs[0]


def historical_dir():
    """<climatology_dir>/<GACC>/Historical -- what the export_*.py steps read."""
    return climo_settings()["outdir"] / roster_gacc() / "Historical"


def toggle(value):
    """bool -> FDRACharts.py's "true"/"false" string convention."""
    return "true" if value else "false"
