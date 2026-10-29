"""Teams card for the cloud AI ensemble run -- added 2026-10-29 (analyst: "get a success message
posted using the toolkit's TEAMS webhook").

    python scripts/ai_ensemble_teams_notify.py --status success --cycle 2026102912 \
        --site https://bah7-humboldt.github.io/ca-ai-ensemble-research/ --run <actions run url>

Run by the notify job of ai-ensemble-daily.yml after a build (skipped hours post nothing). The
webhook is the same Power Automate "Post to a channel when a webhook request is received" flow
teams_card.py posts to; on GitHub it comes from the repo secret TEAMS_WEBHOOK_URL (env), never a
file. No secret -> prints a note and exits 0. Standard library only, so the job needs no installs.
A failed post is logged, never raised: the site is already deployed.
"""
import argparse
import datetime as dt
import json
import os
import sys
import urllib.request

MODELS = "AIFS ENS · AIGEFS · IFS ENS · GEFS"


def card(status, cycle, site, run, started):
    ok = status == "success"
    when = ""
    if cycle and cycle[:10].isdigit():
        c = dt.datetime.strptime(cycle[:10], "%Y%m%d%H")
        when = f"{c:%d %b %Y %H}Z"
    facts = [{"title": "Cycle", "value": f"{when}, all four models" if when else (cycle or "n/a")}]
    if cycle == "partial":
        facts = [{"title": "Cycle", "value": "partial -- a model missed the newest cycle; next hour retries"}]
    if started:
        mins = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(started.replace("Z", "+00:00"))).total_seconds() / 60
        facts.append({"title": "Run time", "value": f"{mins:.0f} min"})
    facts.append({"title": "Models", "value": MODELS})
    title = "AI ENSEMBLE VIEWER — UPDATED" if ok else "AI ENSEMBLE VIEWER — RUN FAILED"
    headline = (when or "Cloud run") if ok else "Build or deploy failed; the site keeps its previous run"
    actions = [{"type": "Action.OpenUrl", "title": "Run log", "url": run}]
    if site:
        actions.insert(0, {"type": "Action.OpenUrl", "title": "Open viewer", "url": site})
    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.5",
        "body": [
            {"type": "TextBlock", "text": title, "size": "Small", "weight": "Bolder",
             "color": "Good" if ok else "Attention"},
            {"type": "TextBlock", "text": headline, "size": "Large", "weight": "Bolder", "wrap": True},
            {"type": "FactSet", "facts": facts},
            {"type": "TextBlock", "text": "Experimental research product (GitHub Actions cloud run).",
             "size": "Small", "isSubtle": True, "wrap": True},
        ],
        "actions": actions,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", required=True, choices=["success", "failure"])
    ap.add_argument("--cycle", default="")
    ap.add_argument("--site", default="")
    ap.add_argument("--run", required=True)
    ap.add_argument("--started", default="", help="run start, ISO UTC")
    a = ap.parse_args()
    url = os.environ.get("TEAMS_WEBHOOK_URL", "").strip()
    if not url:
        print("TEAMS_WEBHOOK_URL not set -- no Teams card")
        return 0
    body = json.dumps(card(a.status, a.cycle.strip(), a.site, a.run, a.started)).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"Teams card posted (HTTP {r.status}, {len(body)} bytes)")
    except Exception as e:  # noqa: BLE001 -- never fail the run over the card
        print(f"WARNING: Teams post failed: {type(e).__name__}: {str(e)[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
