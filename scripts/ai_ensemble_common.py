"""Shared settings and paths for the AI ensemble pattern products (scripts/ai_ensemble_*.py).

Added 2026-10-10, promoted from the incoming/2026-10-10_ai_ensembles sandbox (analyst: "move this
whole thing into the main project and set up a daily task"). Settings live in
config/ai_ensemble_settings.json; paths in it are relative to the repo root. See
docs/ai_ensembles/README.md for the whole tool.

Layout (data_dir, default data/ai_ensembles/):
    runs/<model>_<YYYYMMDDHH>.nc     fetched ensemble runs cut to the domain (keep_cycles per model)
    climo_r1_1991-2020.nc            R1 daily climatology for anomalies (ai_ensemble_climo.py)
    history/obs/                     ASOS / sounding / R1 daily history (ai_ensemble_history_obs.py)
    history/oscc/                    OSCC satellite growth days (ai_ensemble_history_oscc.py)
    history/calibration/             skill, cutoffs, regimes (ai_ensemble_calibrate.py)
output_dir (default data/outputs/ai_ensembles/): <YYYYMMDD_HHMM>/ per products run, latest/ (a
copy of the newest complete set) and logs/.
"""
import json
import pathlib

Path = pathlib.Path

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "config" / "ai_ensemble_settings.json"


def load_settings(path=None):
    """The settings dict with _comment keys dropped."""
    s = json.loads(pathlib.Path(path or SETTINGS_PATH).read_text(encoding="utf-8"))
    return {k: v for k, v in s.items() if not k.startswith("_")}


def forecast_dates(newest_cycle, n=15):
    """The products' Day 1..n dates. Day 1 = today (analyst 2026-10-10: show day 1 as "today"
    instead of tomorrow), as a Pacific calendar date. It's never before the newest cycle's UTC
    date, and at most one day after it, so a rebuild of an old run days later still starts at
    that run's own today. The 02:30 PDT run of the 00Z cycle gets today's date."""
    import datetime as dt
    import zoneinfo
    today = dt.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles")).date()
    first = newest_cycle.date()
    start = min(max(first, today), first + dt.timedelta(days=1))
    return [start + dt.timedelta(days=d) for d in range(n)]


def resolve(p):
    """A settings path (relative to the repo root, or absolute) -> Path."""
    p = pathlib.Path(p)
    return p if p.is_absolute() else ROOT / p


class Paths:
    def __init__(self, cfg):
        self.data = resolve(cfg["data_dir"])
        self.runs = self.data / "runs"
        self.climo = self.data / "climo_r1_1991-2020.nc"
        self.history = self.data / "history"
        self.obs = self.history / "obs"
        self.oscc = self.history / "oscc"
        self.cal = self.history / "calibration"
        self.output = resolve(cfg["output_dir"])
        self.latest = self.output / "latest"
        self.logs = self.output / "logs"
