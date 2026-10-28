"""FEMS NFDRS daily-summary download -- one place for the request every reference-data step
makes (added 2026-10-03 with the FDRACharts.py breakout; see scripts/climo_common.py).

    GET https://fems.fs2c.usda.gov/api/climatology/download-nfdr-daily-summary/
        ?dataset=observation|forecast&startDate=YYYY-MM-DD&endDate=YYYY-MM-DD
        &dataFormat=csv&stationIds=<id>&fuelModels=<model>

One station per request: FEMS dropped the station id from the returned rows, so callers add
it back themselves. The browser-like User-Agent works around HTTP 403s FEMS returned to
plain requests (FDRACharts.py's own fix, kept as is).
"""
from io import StringIO

import pandas as pd
import requests

PRODUCTION = "https://fems.fs2c.usda.gov"
STAGING = "https://fems-stg.fs2c.usda.gov"
DAILY_SUMMARY_PATH = "/api/climatology/download-nfdr-daily-summary/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/csv",
}


def daily_summary_url(station, fuelmodel, startdate, enddate, dataset="observation", staging=False):
    host = STAGING if staging else PRODUCTION
    return (f"{host}{DAILY_SUMMARY_PATH}?dataset={dataset}&startDate={startdate}&endDate={enddate}"
            f"&dataFormat=csv&stationIds={station}&fuelModels={fuelmodel}")


def download_daily_summary(station, fuelmodel, startdate, enddate, dataset="observation",
                           staging=False, timeout=None):
    """One station's NFDRS daily summary as a DataFrame (FEMS's own columns). Raises on an
    HTTP error or unparseable body; callers decide whether that's fatal."""
    response = requests.get(daily_summary_url(station, fuelmodel, startdate, enddate, dataset, staging),
                            headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))
