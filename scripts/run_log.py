"""Run logs for scheduled jobs -- under Windows Task Scheduler stdout goes nowhere, so a failed
step leaves no trace unless the script writes one itself. Added 2026-10-03 for the interactive
FDRA charts' nightly publish; modeled on psa_danger_map_prep.py's own run log (which keeps its
copy, unchanged).

    log_path = run_log.start("data/outputs/fdra_charts_logs", "fdra_charts")
    ...
    run_log.summary("data/outputs/fdra_charts_logs", "fdra_charts", "ONCC site ok: 13 sent | GitHub ok")

start() tees stdout+stderr (tracebacks included) to <dir>/<name>_<yyyymmdd>.log, appending, so
a same-day manual rerun lands under the scheduled run's entry. summary() appends one dated line
to <dir>/<name>_history.log -- the file to skim for "did every night work?".
Both soft-fail: a log that can't be written never stops the run.
"""
import datetime as dt
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


class _Tee:
    """Mirrors a stream to the run log file."""
    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, data):
        self.stream.write(data)
        try:
            self.log_file.write(data)
            self.log_file.flush()
        except (OSError, ValueError):
            pass
        return len(data)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start(log_dir, name):
    """Returns the log path, or None if it can't be opened (the run proceeds unlogged)."""
    log_path = ROOT / log_dir / f"{name}_{dt.date.today():%Y%m%d}.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8")
    except OSError as e:
        print(f"WARNING: could not open run log {log_path}: {e}")
        return None
    log_file.write(f"\n===== run started {dt.datetime.now().isoformat(timespec='seconds')} "
                   f"argv={sys.argv[1:]} =====\n")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return log_path


def summary(log_dir, name, line):
    """Appends "<timestamp>  <line>" to <log_dir>/<name>_history.log."""
    path = ROOT / log_dir / f"{name}_history.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M}  {line}\n")
    except OSError as e:
        print(f"WARNING: could not write {path}: {e}")
