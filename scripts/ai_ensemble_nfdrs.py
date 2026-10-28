"""NFDRS for the AI ensemble pattern products: PSA daily ERC/BI history and percentiles (both GACCs)
and the FEMS 7-day forecast per PSA, for the weather + fuels scoring.

    python scripts/ai_ensemble_nfdrs.py            # pull today's FEMS forecast, write the PSA table

Added 2026-10-11 (analyst: "go for all of it ... it needs to be as smart as it can with what's
available"). The history's lesson: weather and fuels multiply (Northern California 850 hPa warmth
with ERC >= 90th percentile ~8x the usual odds of a >= 5,000-acre day, without it ~0.9x), so the
calibrated outlook scores weather AND forecast fuels.

History: the toolkit's reference data, <climatology_dir>/<GACC>/{Historical,CurrentYear}/<PSA>/
<PSA>_AvgDailyExtremes.csv (ONCC from the production build, OSCC from scripts/oscc_reference_build.py).
Percentiles: the empirical rank of each value among that PSA's 2005-2025 days, the same definition
as the toolkit's charts (fdra_charts_web_export.SfdiModel._pct).

Forecast: climo_download.download_period(period="Forecast") -- the toolkit's own FEMS request -- for
every station in config/oncc_raws_stations.csv and config/oscc_raws_stations.csv, written under
<data_dir>/nfdrs/<GACC>/Forecast/ (never the production Forecast folder the pocket charts read).
PSA value = mean over the PSA's stations, as the reference build's FDRA averages. Output:
<data_dir>/nfdrs/forecast_psa.csv (date, gacc, psa, ERC, BI, ERC_pct, BI_pct, stations).
"""
import datetime as dt
import sys

import numpy as np
import pandas as pd

import ai_ensemble_common as common
import pipeline_settings

ROOT = common.ROOT
ROSTERS = {"ONCC": ROOT / "config" / "oncc_raws_stations.csv", "OSCC": ROOT / "config" / "oscc_raws_stations.csv",
           "HI": ROOT / "config" / "hawaii_raws_stations.csv"}     # Hawaii: island groups (ai_ensemble_history_hawaii.py)
INDICES = ("ERC", "BI")


def climatology_dir():
    return ROOT / pipeline_settings.load_settings()["climatology_dir"]


def load_history(gaccs=("ONCC", "OSCC", "HI")):
    """{PSA: DataFrame indexed by date with ERC, BI, ERC_pct, BI_pct} -- Historical + CurrentYear."""
    base, parts = climatology_dir(), {}
    for gacc in gaccs:
        for period in ("Historical", "CurrentYear"):
            d = base / gacc / period
            if d.is_dir():
                for f in d.glob("*/*_AvgDailyExtremes.csv"):
                    parts.setdefault(f.parent.name, []).append(
                        pd.read_csv(f, usecols=["ObservationTime", *INDICES], parse_dates=["ObservationTime"]))
    out = {}
    for psa, ps in parts.items():
        d = pd.concat(ps).drop_duplicates("ObservationTime", keep="last").set_index("ObservationTime").sort_index()
        hist = d[d.index.year <= 2025]
        for c in INDICES:
            d[c + "_pct"] = percentile(hist[c], d[c])
        out[psa] = d
    return out


def percentile(history, values):
    srt = np.sort(pd.Series(history).dropna().values)
    v = np.asarray(values, dtype=float)
    p = np.searchsorted(srt, v, side="right") / max(1, len(srt)) * 100
    return np.where(np.isfinite(v), p, np.nan)


def area_erc_pct(hist, psas, dates=None):
    """Mean ERC percentile over a set of PSAs, by date (NaN where none report)."""
    cols = [hist[p]["ERC_pct"] for p in psas if p in hist]
    if not cols:
        return pd.Series(dtype=float)
    s = pd.concat(cols, axis=1).mean(axis=1)
    return s if dates is None else s.reindex(dates)


def fetch_forecast(cfg, log=print):
    """FEMS 7-day forecast for every roster station -> per-PSA ERC/BI and percentiles."""
    import climo_download
    from climo_common import climo_settings, load_roster
    out_root = common.Paths(cfg).data / "nfdrs"
    c = climo_settings()
    hist = load_history()
    rows = []
    for gacc, roster in ROSTERS.items():
        if not roster.exists():
            log(f"  NFDRS forecast: no {roster.name} -- {gacc} skipped")
            continue
        df_st = load_roster(roster).get(gacc)
        if df_st is None:
            continue
        failed = climo_download.download_period(df_st, gacc, out_root, "Forecast", c["hist_startdate"], c["hist_enddate"],
                                                False, c["season_startdate"], c["season_enddate"])
        fdir = out_root / gacc / "Forecast"
        for psa in df_st["FDRA"].unique():
            frames = []
            for f in (fdir / psa).glob("*.csv"):
                try:
                    d = pd.read_csv(f, usecols=["ObservationTime", *INDICES])
                except (ValueError, pd.errors.EmptyDataError):
                    continue
                d["date"] = pd.to_datetime(d.ObservationTime).dt.normalize()
                frames.append(d.groupby("date")[list(INDICES)].mean())
            if not frames:
                continue
            allst = pd.concat(frames, keys=range(len(frames)))
            m = allst.groupby(level="date").mean()
            n = allst.groupby(level="date").size()
            for d, r in m.iterrows():
                h = hist.get(psa)
                rows.append({"date": d.strftime("%Y-%m-%d"), "gacc": gacc, "psa": psa,
                             "ERC": round(float(r.ERC), 1), "BI": round(float(r.BI), 1),
                             "ERC_pct": None if h is None else round(float(percentile(h.ERC[h.index.year <= 2025], [r.ERC])[0]), 1),
                             "BI_pct": None if h is None else round(float(percentile(h.BI[h.index.year <= 2025], [r.BI])[0]), 1),
                             "stations": int(n[d])})
        log(f"  NFDRS forecast {gacc}: {df_st['FDRA'].nunique()} PSAs, {len(df_st) - len(failed)}/{len(df_st)} stations")
    out = pd.DataFrame(rows)
    if len(out):
        out_root.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_root / "forecast_psa.csv", index=False)
    return out


def load_forecast(cfg):
    p = common.Paths(cfg).data / "nfdrs" / "forecast_psa.csv"
    if not p.exists():
        return None
    f = pd.read_csv(p, parse_dates=["date"])
    age = (pd.Timestamp.now() - pd.Timestamp(p.stat().st_mtime, unit="s")).total_seconds() / 3600
    f.attrs["age_hours"] = age
    return f


def main():
    t = fetch_forecast(common.load_settings())
    print(f"forecast_psa.csv: {len(t)} PSA-days" + (f", {t.date.min()} .. {t.date.max()}" if len(t) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
