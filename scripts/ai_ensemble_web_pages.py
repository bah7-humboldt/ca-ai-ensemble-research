"""Sub-pages for the AI ensemble research viewer: one "About" page per calibrated predictor and per
500 hPa regime, plus a "How to read this" guide for each section. Called by
ai_ensemble_web_export.run() after the data file is built; writes static pages (no scripts) under
<output_dir>/web/{predictors,regimes,guide}/ that share web/pages.css.

Added 2026-10-10 (analyst: "for each of the calibrated output scenarios can you provide a link to a
sub page that provides a description of everything in colored from patterns stations fire etc also
a How to interpret subpage for each of the sections").

Each predictor page: what it measures and why it matters for fire, a station/region locator map,
the color key, the historical evidence (seasonally adjusted lift vs cutoff, the distribution on big
growth days vs all days, the largest growth days with their fires and whether the cutoff caught
them) and today's forecast by model. Each regime page: the composite map, a plain description of
where the ridges/troughs sit (found from the composite), monthly frequency, lift by region and the
biggest growth days under it. Same framing as the main page: experimental research, public data.
"""
import datetime as dt
import html
import json
import re

import numpy as np
import pandas as pd

import ai_ensemble_common as common

THRESH = 5000
REGION = {"ONCC": "Northern California", "OSCC": "Southern California", "CA": "California", "HI": "Hawaii"}
GROWTH = {"ONCC": common.ROOT / "data" / "satellite_growth" / "oncc_satellite_growth_days.csv"}

STATIONS = {   # name, kind
    "SFO": ("San Francisco International Airport", "surface"), "SAC": ("Sacramento Executive Airport", "surface"),
    "WMC": ("Winnemucca, Nevada", "surface"), "RNO": ("Reno-Tahoe International, Nevada", "surface"),
    "RDD": ("Redding Municipal Airport", "surface"), "MFR": ("Medford, Oregon", "surface"),
    "LAX": ("Los Angeles International Airport", "surface"), "DAG": ("Daggett-Barstow", "surface"),
    "TPH": ("Tonopah, Nevada", "surface"),
}
SOUNDINGS = {"OAK": ("Oakland", (37.72, -122.22)), "VBG": ("Vandenberg Space Force Base", (34.75, -120.57)),
             "NKX": ("San Diego (Miramar)", (32.85, -117.12)), "REV": ("Reno, Nevada", (39.57, -119.80)),
             "MFR": ("Medford, Oregon", (42.37, -122.87))}
SEQ = ["#fdf0e6", "#f9d2b4", "#f2a679", "#e47740", "#c2531f", "#8e3812"]
SEQ_DARK = ["#2b1f18", "#4d2c19", "#7a3d1c", "#b2521f", "#e46c2c", "#ff9a5c"]
RDBU = ["#08306b", "#2166ac", "#4393c3", "#92c5de", "#c6dbef", "#e8f0f6", "#f7f7f7",
        "#fbe3d6", "#f4c4a8", "#f4a582", "#d6604d", "#b2182b", "#67001f"]

CSS = """:root { --bg:#eef1f0; --surface:#fff; --surface-2:#e2e6e4; --border:#c7cdc9; --text:#1b201e; --text-dim:#5c655f;
  --accent:#d9622a; --focus:#1b6de0; --ocean:#dfe6ea; --grid:#e4e8e6; --line:#3c3c3c; --line-2:#9aa29d;
  --seq-0:#fdf0e6; --seq-1:#f9d2b4; --seq-2:#f2a679; --seq-3:#e47740; --seq-4:#c2531f; --seq-5:#8e3812; color-scheme: light; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#12161a; --surface:#191f24; --surface-2:#212830;
  --border:#333d47; --text:#e8ecef; --text-dim:#8b98a3; --accent:#ff8a4c; --focus:#6fa8ff; --ocean:#1d2830; --grid:#2a323b;
  --line:#c9d1d8; --line-2:#5d6a75; --seq-0:#2b1f18; --seq-1:#4d2c19; --seq-2:#7a3d1c; --seq-3:#b2521f; --seq-4:#e46c2c; --seq-5:#ff9a5c; color-scheme: dark; } }
:root[data-theme="dark"] { --bg:#12161a; --surface:#191f24; --surface-2:#212830; --border:#333d47; --text:#e8ecef; --text-dim:#8b98a3;
  --accent:#ff8a4c; --focus:#6fa8ff; --ocean:#1d2830; --grid:#2a323b; --line:#c9d1d8; --line-2:#5d6a75;
  --seq-0:#2b1f18; --seq-1:#4d2c19; --seq-2:#7a3d1c; --seq-3:#b2521f; --seq-4:#e46c2c; --seq-5:#ff9a5c; color-scheme: dark; }
* { box-sizing: border-box; } html, body { margin: 0; }
body { background: var(--bg); color: var(--text); font: 14.5px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; }
.wrap { max-width: 960px; margin: 0 auto; padding: 12px 16px 40px; }
nav { font-size: 13px; margin-bottom: 8px; display: flex; gap: 14px; flex-wrap: wrap; }
a { color: var(--focus); }
h1 { font-size: 20px; margin: 4px 0 2px; } h2 { font-size: 16px; margin: 0 0 6px; } h3 { font-size: 14px; margin: 14px 0 4px; }
.sub { color: var(--text-dim); font-size: 13px; margin: 0 0 12px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px; margin-bottom: 14px; }
.banner { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 6px; padding: 8px 12px; margin-bottom: 14px; font-size: 13px; }
.cols { display: flex; gap: 16px; flex-wrap: wrap; } .cols > * { flex: 1 1 300px; min-width: 0; }
table { border-collapse: collapse; font-size: 13px; width: 100%; } th, td { padding: 4px 6px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
th { color: var(--text-dim); font-weight: 600; font-size: 12px; } td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
.stat { display: inline-block; margin: 0 18px 8px 0; } .stat b { display: block; font-size: 20px; font-variant-numeric: tabular-nums; } .stat span { color: var(--text-dim); font-size: 12px; }
.chip { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 12px; border: 1px solid var(--border); }
.yes { background: var(--seq-3); color: #fff; border-color: transparent; } .hi { background: var(--seq-5); color: #fff; border-color: transparent; }
.swatches { display: flex; gap: 0; border: 1px solid var(--border); border-radius: 4px; overflow: hidden; max-width: 520px; }
.swatches span { flex: 1; height: 16px; } .swlab { display: flex; justify-content: space-between; max-width: 520px; font-size: 11.5px; color: var(--text-dim); }
svg { width: 100%; height: auto; display: block; } svg text { fill: var(--text-dim); font-size: 11px; font-family: inherit; }
img.map { width: 100%; border-radius: 6px; border: 1px solid var(--border); background: #fff; }
.gtags { display: inline-flex; gap: 4px; margin-left: 8px; vertical-align: middle; flex-wrap: wrap; }
.gtag { display: inline-block; padding: 0 7px; border-radius: 9px; font-size: 11px; font-weight: 600; line-height: 18px; color: #fff; letter-spacing: .02em; }
.gt-ONCC { background: #1f6fb2; } .gt-OSCC { background: #a8481a; } .gt-HI { background: #2a7347; }
.gt-M1, .gt-M2 { background: var(--surface-2); color: var(--text); border: 1px solid var(--border); font-weight: 500; }
.note { color: var(--text-dim); font-size: 12.5px; } ul { padding-left: 20px; margin: 6px 0; } li { margin: 3px 0; }
footer { color: var(--text-dim); font-size: 12px; margin-top: 18px; }
"""


def esc(s):
    return html.escape(str(s))


def ordinal(v):
    """92 -> '92nd', 11 -> '11th'."""
    n = int(round(v))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def slug(*parts):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", "__".join(parts))


def page(title, body, depth=1, desc=""):
    up = "../" * depth
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} · California &amp; Hawaii AI Ensemble Research</title>
<meta name="robots" content="noindex, nofollow">
<meta name="description" content="{esc(desc or title)} -- experimental research, not a forecast.">
<link rel="stylesheet" href="{up}pages.css"></head><body><div class="wrap">
<nav><a href="{up}index.html">← California &amp; Hawaii AI Ensemble Research</a>
<a href="{up}guide/map.html">Reading the map</a><a href="{up}guide/weeks.html">Reading week 1 and week 2</a>
<a href="{up}guide/outlook.html">Reading the calibrated outlook</a>
<a href="{up}guide/plumes.html">Reading the plumes</a><a href="{up}guide/regimes.html">Reading the regimes</a>
<a href="{up}guide/patterns.html">Documented patterns</a>{f'<a href="{up}guide/pyroconvection.html">Pyroconvection</a>' if PYRO else ''}</nav>
<div class="banner"><b>Experimental research, not a forecast.</b> Built from publicly available data, plus a licensed lightning
history shown only as derived counts and maps. For official information
see the <a href="https://www.weather.gov/fire/">National Weather Service fire weather</a> pages and the
<a href="https://www.nifc.gov/nicc/predictive-services/outlooks">NIFC Predictive Services outlooks</a>.</div>
{body}
<footer>Experimental, non-commercial research; data provided as-is. See the main page for full data credits.</footer>
</div></body></html>"""


# ----------------------------------------------------------------------------- SVG helpers

def svg_line_chart(series, xlab, ylab, w=640, h=260, hlines=(), marks=(), xfmt="{:.0f}", yfmt="{:.0f}"):
    """series: [(label, color, dash, [(x, y), ...])]; hlines: [(y, label)]; marks: [(x, y, label)]."""
    xs = [x for _, _, _, pts in series for x, _ in pts]
    ys = [y for _, _, _, pts in series for _, y in pts] + [y for y, _ in hlines]
    if not xs:
        return ""
    x0, x1 = min(xs), max(xs)
    y0, y1 = 0.0, max(ys) * 1.08 if ys else 1
    m = dict(l=46, r=12, t=10, b=38)
    sx = lambda v: m["l"] + (v - x0) / ((x1 - x0) or 1) * (w - m["l"] - m["r"])
    sy = lambda v: m["t"] + (y1 - v) / ((y1 - y0) or 1) * (h - m["t"] - m["b"])
    out = [f'<svg viewBox="0 0 {w} {h}" role="img">']
    for t in np.linspace(y0, y1, 5):
        out.append(f'<line x1="{m["l"]}" x2="{w - m["r"]}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="var(--grid)"/>'
                   f'<text x="{m["l"] - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{yfmt.format(t)}</text>')
    for t in np.linspace(x0, x1, 6):
        out.append(f'<text x="{sx(t):.1f}" y="{h - m["b"] + 15}" text-anchor="middle">{xfmt.format(t)}</text>')
    out.append(f'<text x="{(m["l"] + w - m["r"]) / 2}" y="{h - 6}" text-anchor="middle">{esc(xlab)}</text>'
               f'<text x="12" y="{(m["t"] + h - m["b"]) / 2}" text-anchor="middle" transform="rotate(-90 12 {(m["t"] + h - m["b"]) / 2})">{esc(ylab)}</text>')
    for y, lab in hlines:
        out.append(f'<line x1="{m["l"]}" x2="{w - m["r"]}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="var(--text-dim)" stroke-dasharray="4 4"/>'
                   f'<text x="{w - m["r"] - 2}" y="{sy(y) - 4:.1f}" text-anchor="end">{esc(lab)}</text>')
    for lab, col, dash, pts in series:
        p = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts if np.isfinite(y))
        out.append(f'<polyline points="{p}" fill="none" stroke="{col}" stroke-width="2"{" stroke-dasharray=\"6 4\"" if dash else ""}/>')
    for x, y, lab in marks:
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="var(--accent)" stroke="var(--surface)" stroke-width="1.5"/>'
                   f'<text x="{sx(x) + 8:.1f}" y="{sy(y) - 6:.1f}" style="fill:var(--text)">{esc(lab)}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_bars(labels, values, w=640, h=200, color="var(--seq-3)", fmt="{:.0%}"):
    n = len(values)
    vmax = max([v for v in values if np.isfinite(v)] + [1e-9])
    m = dict(l=10, r=10, t=14, b=22)
    bw = (w - m["l"] - m["r"]) / n
    out = [f'<svg viewBox="0 0 {w} {h}" role="img">']
    for i, (lab, v) in enumerate(zip(labels, values)):
        x = m["l"] + i * bw
        hh = 0 if not np.isfinite(v) else (h - m["t"] - m["b"]) * v / vmax
        out.append(f'<rect x="{x + 2:.1f}" y="{h - m["b"] - hh:.1f}" width="{bw - 4:.1f}" height="{hh:.1f}" rx="3" fill="{color}"/>'
                   f'<text x="{x + bw / 2:.1f}" y="{h - 6}" text-anchor="middle">{esc(lab)}</text>')
        if np.isfinite(v) and v > 0:
            out.append(f'<text x="{x + bw / 2:.1f}" y="{h - m["b"] - hh - 3:.1f}" text-anchor="middle" style="fill:var(--text)">{fmt.format(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def svg_locator(lines, region_lines=(), stations=(), arrow=None, bounds=(-126.5, -113.5, 31.5, 44.5), w=420):
    """Small California map: state/country lines, an optional highlighted region, station dots,
    and an optional arrow drawn from arrow[0] to arrow[1] (inland high -> coast for a gradient)."""
    x0, x1, y0, y1 = bounds
    asp = (x1 - x0) * np.cos(np.radians((y0 + y1) / 2)) / (y1 - y0)
    h = w / asp
    P = lambda lon, lat: ((lon - x0) / (x1 - x0) * w, (y1 - lat) / (y1 - y0) * h)

    def path(lns):
        d = []
        for ln in lns:
            pts = [P(ln[i], ln[i + 1]) for i in range(0, len(ln), 2)]
            pts = [p for p in pts if -50 < p[0] < w + 50 and -50 < p[1] < h + 50]
            if len(pts) >= 2:
                d.append("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts))
        return " ".join(d)
    out = [f'<svg viewBox="0 0 {w} {h:.0f}" role="img" style="background:var(--ocean);border-radius:6px">']
    out.append(f'<path d="{path(lines.get("state", []))}" fill="none" stroke="var(--line-2)" stroke-width="0.8"/>')
    out.append(f'<path d="{path(lines.get("country", []))}" fill="none" stroke="var(--line)" stroke-width="1"/>')
    if region_lines:
        out.append(f'<path d="{path(region_lines)}" fill="none" stroke="var(--accent)" stroke-width="2.2"/>')
    if arrow:
        (ax, ay), (bx, by) = P(*arrow[0]), P(*arrow[1])
        out.append('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                   '<path d="M0,0 L10,5 L0,10 z" fill="var(--accent)"/></marker></defs>'
                   f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="var(--accent)" stroke-width="2" '
                   f'stroke-dasharray="5 3" marker-end="url(#ah)"/>')
    for code, (lat, lon), label in stations:
        x, y = P(lon, lat)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="var(--text)" stroke="var(--surface)" stroke-width="1.5"/>'
                   f'<text x="{x + 7:.1f}" y="{y + 4:.1f}" style="fill:var(--text);font-weight:600">{esc(code)}</text>')
    out.append("</svg>")
    return "".join(out)


def swatch_legend(colors, labels):
    return ('<div class="swatches">' + "".join(f'<span style="background:{c}"></span>' for c in colors) + "</div>"
            '<div class="swlab">' + "".join(f"<span>{esc(l)}</span>" for l in labels) + "</div>")


# ----------------------------------------------------------------------------- predictor texts

ERA5_NOTE = (" This version is computed from the ERA5 reanalysis (ECMWF, 0.25°, every 6 h) at the nearest grid cells, so it "
             "has no station gaps and matches what the models forecast.")
FP_FIELDS = {
    "vpd_wind": ("Hot-Dry-Windy", "units",
                 "The highest value during the fire day of the footprint-average 2 m vapour-pressure deficit (how much drier the air is "
                 "than saturated, hPa) times the 10 m wind speed (m/s), from ERA5 at 00 and 12 UTC (the models' times). A surface version of the Hot-Dry-Windy "
                 "index (Srock et al. 2018).",
                 "Fire spreads fastest when the air is hot, dry and windy at once; this one number captures all three across the "
                 "whole footprint rather than at one station."),
    "rh_min": ("area minimum humidity", "%",
               "The lowest footprint-average 2 m relative humidity of the fire day, from ERA5 at 00 and 12 UTC (the models' times).",
               "Area-wide afternoon dryness: low humidity everywhere in the footprint dries fine fuels and slows overnight recovery."),
    "blh_max": ("mixing height", "m",
                "The highest footprint-average boundary-layer (mixing) height of the fire day, from ERA5.",
                "A deep mixed layer mixes dry, windy air from aloft down to the fire and lets smoke columns grow; it marks "
                "unstable, plume-dominated fire days. No model in this set forecasts it, so it's history only."),
    "dry_tstorm": ("dry-thunderstorm setup", "share",
                   "The share of the footprint's grid cells with convective energy (CAPE ≥ 250 J/kg) and a dry layer at 700 hPa "
                   "(dewpoint depression ≥ 5 °C) and less than 0.1 inch of rain that fire day, from ERA5.",
                   "Thunderstorms over dry air drop lightning with little rain: new ignitions everywhere at once, the setup behind "
                   "California's lightning sieges. Forecast from IFS ENS and GEFS, the two models that publish CAPE."),
}


def describe(key, cfg):
    """{title, what, why, units, stations: [(code, (lat, lon), label)], arrow, region}"""
    pts = cfg["points"]
    m = re.match(r"^([a-z_]+)_(vpd_wind|blh_max|rh_min|dry_tstorm)(_anom)?$", key)
    if m and m.group(1) in cfg.get("footprints", {}):
        f = cfg["footprints"][m.group(1)]
        title, units, what, why = FP_FIELDS[m.group(2)]
        if m.group(3):
            title += " anomaly"
            what += (" Shown as the departure from the 2001–2025 ERA5 normal for the date (±15 days), so a "
                     "positive value means hotter-drier-windier (or, for humidity, a negative value drier) than "
                     "usual for that time of year. Scoring the raw value mostly rewarded the season itself.")
        return dict(title=f"{f['label'].split(' (')[0]} {title}", what=what, why=why,
                    units="hPa·m/s" if m.group(2) == "vpd_wind" else units, arrow=None, region=None, stations=[])
    if key.endswith("_era5_rhmin") or key.endswith("_era5"):
        base = describe(key.replace("_era5_rhmin", "_rhmin6").replace("_era5", "_min6"), cfg)
        base["title"] += " (ERA5)"
        base["what"] = base["what"] + ERA5_NOTE
        return base
    m = re.match(r"^(.+)-(.+)_min6?$", key)
    if m:
        a, b = m.groups()
        g = next((x for x in cfg["gradients"] if x["a"] == a and x["b"] == b), {})
        return dict(
            title=f"{a}–{b} surface pressure gradient",
            what=(f"Sea-level pressure at {STATIONS.get(a, (a,))[0]} minus {STATIONS.get(b, (b,))[0]}, in hPa, taken as the "
                  f"lowest (most negative) value of the fire day at the 00, 06, 12 and 18 UTC observations. "
                  f"<b>Negative means pressure is higher inland than at the coast</b>, which pushes air from the interior "
                  f"toward the coast: offshore flow."),
            why=(f"{esc(g.get('label', 'Offshore flow'))}. Offshore winds are dry (the air sinks and warms as it "
                 "comes down off the mountains), gusty, and blow toward the populated coastal hills. Strong offshore "
                 "gradients go with the fastest-spreading fires in California's record: Diablo and North winds in the north, "
                 "Santa Ana winds in the south."),
            units="hPa", arrow=((pts[b][1], pts[b][0]), (pts[a][1], pts[a][0])) if a in pts and b in pts else None,
            stations=[(c, tuple(pts[c]), STATIONS.get(c, (c,))[0]) for c in (a, b) if c in pts], region=None)
    m = re.match(r"^([A-Z]{3})_(rhmin6?|windmax)$", key)
    if m:
        st = m.group(1)
        ll = cfg.get("rh_stations", {}).get(st)
        names = {"SAC": "Sacramento Executive Airport", "RDD": "Redding Municipal Airport", "STS": "Santa Rosa (Sonoma County Airport)",
                 "BUR": "Burbank (Hollywood Burbank Airport)", "ONT": "Ontario International Airport", "PRB": "Paso Robles Municipal Airport"}
        return dict(title=f"{st} minimum relative humidity",
                    what=(f"The lowest relative humidity of the fire day at {names.get(st, st)}, from the airport's hourly observations, "
                          "taken at the 00, 06, 12 and 18 UTC reports (the late-afternoon 00 UTC report is usually the day's driest), "
                          "so a model's 12-hourly forecast can be compared with it."),
                    why=("Low afternoon humidity dries the fine fuels that carry fire, and a dry night with poor humidity recovery lets "
                         "fires keep running overnight. The models' 2 m humidity at the nearest 0.5° cell is scored against it."),
                    units="%", arrow=None, region=None, stations=[(st, tuple(ll), names.get(st, st))] if ll else [])
    m = re.match(r"^([A-Z]{3})(850|700|500)_(offshore_kt|t|z|dd)_(12z|00z)(_anom)?$", key)
    if m:
        st, lev, kind, tag, an = m.groups()
        name, ll = SOUNDINGS.get(st, (st, None))
        when = "early morning (12 UTC, ~5 AM PDT)" if tag == "12z" else "late afternoon (00 UTC, ~5 PM PDT)"
        base = dict(units={"offshore_kt": "kt", "t": "°C", "z": "m", "dd": "°C"}[kind], arrow=None, region=None,
                    stations=[(st, ll, name)] if ll else [])
        if kind == "offshore_kt":
            base.update(title=f"{st} 850 hPa offshore wind ({'morning' if tag == '12z' else 'afternoon'})",
                        what=(f"Wind speed at about 5,000 ft (850 hPa) from the {name} weather balloon, {when}, counted "
                              f"only when it blows from the northeast-to-east (0–110°); otherwise 0 kt."),
                        why=("Northeast-to-east wind just above the ridgetops is the offshore flow that surfaces as Diablo or "
                             "Santa Ana winds. The surface pressure gradients usually capture these events better, because "
                             "many strong north-wind days aloft are cool post-frontal days."))
        elif kind == "t":
            base.update(title=f"{st} 850 hPa temperature{' anomaly' if an else ''} ({'morning' if tag == '12z' else 'afternoon'})",
                        what=(f"Air temperature at about 5,000 ft (850 hPa) from the {name} weather balloon, {when}"
                              + (", <b>minus the normal for that date</b> (2001–2025)." if an else ".")),
                        why=("Warm air aloft marks a ridge of high pressure: sinking, drying air, a lid on the marine layer, "
                             "hot and dry days, and on offshore days a warm, dry wind. It's measured against the date's normal "
                             "so a warm August day isn't credited just for being August."))
        elif kind == "z":
            base.update(title=f"{st} 500 hPa height{' anomaly' if an else ''}",
                        what=(f"Height of the 500 hPa pressure surface (~18,000 ft) over {name}, {when}"
                              + (", minus the normal for the date." if an else ".")),
                        why="Higher than normal heights mean a ridge overhead: warm, dry, stable air and weak onshore flow.")
        else:
            base.update(title=f"{st} {lev} hPa dryness{' anomaly' if an else ''} ({'morning' if tag == '12z' else 'afternoon'})",
                        what=(f"Temperature minus dewpoint (the dewpoint depression) at {lev} hPa (about "
                              f"{'10,000' if lev == '700' else '5,000'} ft) over {name}, {when}"
                              + (", minus the date's normal." if an else ".")),
                        why=("A large dewpoint depression aloft means very dry air. Mixed down in the afternoon or brought down by offshore "
                             "winds, it drops surface humidity; with thunderstorms it means dry lightning (storms whose rain evaporates "
                             "before reaching the ground)."))
        return base
    m = re.match(r"^(ONCC|OSCC|CA)_(t850|z500)_anom$", key)
    if m:
        g, f = m.groups()
        return dict(
            title=f"{REGION[g]} {'850 hPa temperature' if f == 't850' else '500 hPa height'} anomaly",
            what=(f"The {'850 hPa (~5,000 ft) temperature' if f == 't850' else '500 hPa (~18,000 ft) height'} averaged over "
                  f"{REGION[g]}, as a daily mean, minus the normal for the date (NCEP reanalysis climatology 1991–2020)."),
            why=("A region-wide measure of how hot (or how strongly ridged) the air mass is, rather than a single balloon "
                 "site. Warm air masses dry fuels and make for hot, unstable or stagnant fire days."),
            units="°C" if f == "t850" else "m", arrow=None, region=g, stations=[])
    return dict(title=key, what=key, why="", units="", arrow=None, region=None, stations=[])


# ----------------------------------------------------------------------------- data

def load_growth(paths):
    files = dict(GROWTH, OSCC=paths.oscc / "satellite_growth" / "oscc_satellite_growth_days.csv")
    out = {}
    for g, p in files.items():
        if p.exists():
            out[g] = pd.read_csv(p, usecols=["event_id", "burn_date", "growth_acres", "fire_label", "psa"],
                                 parse_dates=["burn_date"])
    if len(out) == 2:
        out["CA"] = pd.concat(out.values())
    return out


NFDRS_DIR = common.ROOT / "data" / "historic_fires_research"


def load_nfdrs():
    """{PSA: DataFrame(ERC, BI, ERC_pct, BI_pct) by date} from the toolkit's reference-data build
    (Historical 2005-2025 + CurrentYear, PSA-average of member stations). Percentiles are against
    the PSA's whole history, the same empirical rank the toolkit's charts use. ONCC only so far."""
    out = {}
    for gacc in ("ONCC", "OSCC", "HI"):         # HI: island groups (ai_ensemble_history_hawaii.build_nfdrs)
        for period in ("Historical", "CurrentYear"):
            base = NFDRS_DIR / gacc / period
            if not base.is_dir():
                continue
            for f in base.glob("*/*_AvgDailyExtremes.csv"):
                psa = f.parent.name
                d = pd.read_csv(f, usecols=["ObservationTime", "ERC", "BI"], parse_dates=["ObservationTime"])
                out.setdefault(psa, []).append(d)
    res = {}
    for psa, parts in out.items():
        d = pd.concat(parts).drop_duplicates("ObservationTime", keep="last").set_index("ObservationTime").sort_index()
        hist = d[d.index.year <= 2025]
        for c in ("ERC", "BI"):
            srt = np.sort(hist[c].dropna().values)
            d[c + "_pct"] = np.searchsorted(srt, d[c].values, side="right") / max(1, len(srt)) * 100
        res[psa] = d
    return res


def run_miles(burn_dir, event_id, day):
    """How far a fire's burned area grew on one fire day: the farthest that day's newly burned cells
    lie from everything burned before (100 m day-of-burn grid). On a fire's first day: the span of
    that day's burned cells. None if the grid isn't there."""
    import rasterio
    from scipy import ndimage
    f = burn_dir / f"{event_id}.tif"
    if not f.exists():
        return None
    with rasterio.open(f) as r:
        a = r.read(1)
        day0 = pd.Timestamp(r.tags().get("day0"))
        res = abs(r.res[0])
    k = (pd.Timestamp(day) - day0).days + 1
    new = a == k
    if not new.any():
        return None
    prior = (a > 0) & (a < k)
    if prior.any():
        dist = ndimage.distance_transform_edt(~prior) * res
        return float(dist[new].max() / 1609.34)
    jj, ii = np.nonzero(new)
    return float(np.hypot(jj.max() - jj.min(), ii.max() - ii.min()) * res / 1609.34)


def daily_totals(df):
    return df.groupby("burn_date").growth_acres.sum()


def top_fires(df, day, n=3):
    d = df[df.burn_date == day].groupby("fire_label").growth_acres.sum().sort_values(ascending=False).head(n)
    # Upper case as the toolkit's growth charts show them (title case mangles LNU, SCU, SQF, CZU...)
    return ", ".join(f"{esc(k)} ({v:,.0f} ac)" for k, v in d.items())


def seq_color(p):
    if not np.isfinite(p):
        return "transparent"
    i = 0 if p < .05 else 1 if p < .2 else 2 if p < .4 else 3 if p < .6 else 4 if p < .8 else 5
    return f"var(--seq-{i})"


def starts_card(row, area, season):
    st = row.get("starts")
    if not st or st.get("days_past") is None:
        return ""
    lift = st.get("lift_adj")
    return f"""<div class="card"><h2>New large fires</h2>
<p>Growth isn't the only question: offshore and dry days also start the fires that become large. On the {st['days_past']:,} {season} days
past the elevated cutoff, a fire that went on to burn at least 1,000 acres started in {esc(area)} on {st['starts_past']:,} of them
({(st['p_past'] or 0):.1%} of those days, against {st['base']:.1%} on a typical {season} day){f"; about <b>×{lift:.1f}</b> the usual odds for the date" if lift else ""}.
That's {st['share_caught']:.0%} of the {st['total_starts']:,} large-fire starts there since 2001.</p>
<p class="note">Fires and start dates from the satellite fires table (MTBS / WFIGS perimeters ≥ 1,000 acres; discovery from FPA FOD / InFORM).</p></div>"""


def odds_table(row, models, days, target=None):
    """Weather + fuels odds by day: members past the cutoff carry the historical odds for 'past' on
    that day's forecast fuels (FEMS ERC percentile >= dry_pct or not), the rest the 'not past' odds."""
    J, F = row.get("joint"), row.get("fuels")
    if not J or not F or not any(v is not None for v in F.get("erc_pct", [])):
        return ""
    cut, tail = row["elevated"]["cutoff"], row["tail"]

    def frac(vals):
        a = [v for v in vals if v is not None and np.isfinite(v)]
        return np.nan if not a else np.mean([(v <= cut) if tail == "low" else (v >= cut) for v in a])
    cells_e, cells_o = "", ""
    for k, d in enumerate(days):
        e = F["erc_pct"][k]
        pers = F.get("persisted_from") and d >= F["persisted_from"]
        cells_e += f'<td class="n"{" style=\"opacity:.6\"" if pers else ""}>{"–" if e is None else round(e)}</td>'
        if e is None:
            cells_o += '<td class="n">–</td>'
            continue
        dry = e >= F["dry_pct"]
        lp, ln = J["past_dry" if dry else "past_wet"]["lift"], J["not_dry" if dry else "not_wet"]["lift"]
        os_ = []
        for m in models:
            mm = row["members"].get(m["key"])
            if mm and lp is not None and ln is not None:
                p = frac(mm[k])
                if np.isfinite(p):
                    os_.append(p * lp + (1 - p) * ln)
        o = np.mean(os_) if os_ else np.nan
        i = 0 if not np.isfinite(o) or o < 1 else 1 if o < 1.5 else 2 if o < 2 else 3 if o < 3 else 4 if o < 5 else 5
        cells_o += (f'<td class="n" style="background:var(--seq-{i});{"color:#fff;" if np.isfinite(o) and o >= 3 else ""}'
                    f'{"opacity:.7;" if pers else ""}">{"–" if not np.isfinite(o) else "×%.1f" % o}</td>')
    head = "<tr><th></th>" + "".join(f'<th class="n">{d[5:].replace("-", "/")}</th>' for d in days) + "</tr>"
    return f"""<h3>Weather + fuels odds</h3><div class="scroll"><table>{head}
<tr><td>Forecast ERC percentile (area mean, FEMS)</td>{cells_e}</tr>
<tr><td><b>× usual odds of {target or f"a ≥{THRESH:,}-acre day"}</b></td>{cells_o}</tr></table></div>
<p class="note">Odds combine this measure (each member past the cutoff or not) with the forecast fuels (ERC ≥ the {F['dry_pct']}th percentile =
critically dry), using the history in "Weather + fuels" above. Faded: past the 7-day FEMS forecast, the last day's fuels carried forward.</p>"""


def forecast_table(row, models, days):
    """% of each model's members past the elevated cutoff, by day (plus the equal-weight pool)."""
    cut, tail = row["elevated"]["cutoff"], row["tail"]
    def frac(vals):
        a = [v for v in vals if v is not None and np.isfinite(v)]
        return np.nan if not a else np.mean([(v <= cut) if tail == "low" else (v >= cut) for v in a])
    head = "<tr><th>Model</th>" + "".join(f'<th class="n">{d[5:].replace("-", "/")}</th>' for d in days) + "</tr>"
    body, pool = "", np.zeros(len(days)),
    pool, cnt = np.zeros(len(days)), np.zeros(len(days))
    for m in models:
        mm = row["members"].get(m["key"])
        if not mm:
            continue
        ps = [frac(col) for col in mm]
        for k, p in enumerate(ps):
            if np.isfinite(p):
                pool[k] += p
                cnt[k] += 1
        body += f"<tr><td>{esc(m['label'])}</td>" + "".join(
            f'<td class="n" style="background:{seq_color(p)};{"color:#fff;" if np.isfinite(p) and p >= .6 else ""}">'
            f'{"–" if not np.isfinite(p) else round(100 * p)}</td>' for p in ps) + "</tr>"
    pp = np.where(cnt > 0, pool / np.maximum(cnt, 1), np.nan)
    body = (f"<tr><td><b>All models</b></td>" + "".join(
        f'<td class="n" style="background:{seq_color(p)};{"color:#fff;" if np.isfinite(p) and p >= .6 else ""}"><b>'
        f'{"–" if not np.isfinite(p) else round(100 * p)}</b></td>' for p in pp) + "</tr>") + body
    return f'<div class="scroll"><table>{head}{body}</table></div>'


# ----------------------------------------------------------------------------- pages

def load_psas(paths):
    import geopandas as gpd
    parts = []
    for f in (common.ROOT / "config" / "oncc_psa" / "ONCC_PSA.shp", paths.oscc / "OSCC_PSA.shp"):
        if f.exists():
            parts.append(gpd.read_file(f).to_crs(3310)[["PSANationa", "PSANAME", "geometry"]])
    return pd.concat(parts, ignore_index=True) if parts else None


def footprint_map(out_png, past_days, other_days, gdf_growth, psas, cells, fp_psas, title_left):
    """Two panels on the CA PSAs (EPSG:3310): left = satellite burn area per day on the days past
    the cutoff (3 km cells) over PSAs shaded by how much more often each had a >= 1,000-acre day on
    those days than on other season days; right = burn area per day on the other season days.
    Footprint PSAs outlined heavy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    days_all = past_days.union(other_days)
    flags = (gdf_growth.groupby(["psa", "burn_date"]).growth_acres.sum() >= 1000)
    ratio = {}
    for code in psas.PSANationa:
        f = flags.xs(code, level="psa") if code in flags.index.get_level_values("psa") else pd.Series(dtype=bool)
        f = f.reindex(days_all).fillna(False).astype(bool)
        n_big = int(f.sum())
        if n_big < 5 or len(past_days) == 0:
            ratio[code] = np.nan
            continue
        p_past = f.reindex(past_days).mean()
        p_other = f.reindex(other_days).mean() if len(other_days) else np.nan
        ratio[code] = p_past / p_other if p_other and np.isfinite(p_other) else np.nan
    psas = psas.assign(ratio=psas.PSANationa.map(ratio).clip(upper=9.9))
    bins = [0, 1, 1.5, 2, 3, 5, 10]
    cols = ["#d9dcda", "#fdf0e6", "#f9d2b4", "#f2a679", "#e47740", "#8e3812"]
    cmap = mcolors.ListedColormap(cols)
    norm = mcolors.BoundaryNorm(bins, cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.2))
    for k, (ax, days, lab) in enumerate(zip(axes, (past_days, other_days),
                                            (title_left, "All other days in the season"))):
        if k == 0:
            psas.plot(ax=ax, column="ratio", cmap=cmap, norm=norm, missing_kwds={"color": "#f3f4f3"},
                      edgecolor="#9aa29d", linewidth=0.5)
        else:
            psas.plot(ax=ax, color="#f3f4f3", edgecolor="#9aa29d", linewidth=0.5)
        if fp_psas:
            psas[psas.PSANationa.isin(fp_psas)].boundary.plot(ax=ax, color="#1b201e", linewidth=2.0)
        c = cells[cells.date.isin(days.strftime("%Y-%m-%d"))]
        if len(c) and len(days):
            agg = c.groupby(["x_km", "y_km"]).acres.sum() / len(days)       # acres per day
            xy = np.array(agg.index.tolist()) * 1000
            ax.scatter(xy[:, 0], xy[:, 1], s=np.clip(agg.values, 0.5, 400) ** 0.6 * 1.6, c="#1b201e",
                       alpha=0.55, linewidths=0, zorder=3)
        ax.set_title(f"{lab}\n{len(days):,} days", fontsize=9.5)
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes[0], orientation="horizontal", fraction=0.045, pad=0.02, spacing="uniform",
                      ticks=[0.5, 1.25, 1.75, 2.5, 4, 7.5])
    cb.ax.set_xticklabels(["<1x", "1-1.5x", "1.5-2x", "2-3x", "3-5x", ">5x"], fontsize=7.5)
    cb.set_label("x as often (PSA days with >= 1,000 ac)", fontsize=7.5)
    fig.text(0.5, 0.02, "Dots: satellite-mapped fire growth per day (3 km cells; bigger = more acres per day). "
             "Heavy outline: this measure's footprint. Light gray PSAs: fewer than 5 days of >= 1,000 ac.",
             ha="center", fontsize=8, color="#5c655f")
    fig.savefig(out_png, dpi=105, bbox_inches="tight")
    plt.close(fig)


def predictor_page(row, cfg, cal, scan, derived, growth, data, region_lines, psas=None, cells=None, out_dir=None):
    ent, key = row["entity"], row["key"]
    g, season = ent.split("-")[0], ("offshore season" if ent.endswith("offshore") else "fire season")
    info = describe(key, cfg)
    op = "≤" if row["tail"] == "low" else "≥"
    el, hi = row["elevated"], row.get("high")
    base = row.get("base", np.nan)
    months = cal["seasons"][ent]
    mnames = ", ".join(dt.date(2000, m, 1).strftime("%b") for m in months)
    # history: season days, big days
    gdf_all = growth.get(g)
    fp = row.get("footprint")
    gdf = gdf_all[gdf_all.psa.isin(fp["psas"])] if (gdf_all is not None and fp) else gdf_all
    area = fp["label"] if fp else REGION.get(g, g)
    tot = daily_totals(gdf) if gdf is not None else pd.Series(dtype=float)
    idx = derived.index[derived.index.month.isin(months) & (derived.index <= (tot.index.max() if len(tot) else derived.index.max()))]
    x = derived.loc[idx, key] if key in derived else pd.Series(index=idx, dtype=float)
    acres = tot.reindex(idx).fillna(0)
    big = acres >= THRESH
    # lift curve
    sc = scan[(scan.gacc == ent) & (scan.predictor == key) & (scan.threshold_acres == THRESH) & (scan.days >= 25)]
    curve = svg_line_chart([("lift", "var(--accent)", False, list(zip(sc.cutoff, sc.lift_adj)))],
                           f"cutoff ({info['units']}; counts days {op} cutoff)", "× usual odds for the date",
                           hlines=[(1, "usual odds"), (2, "elevated ×2"), (4, "high ×4")],
                           marks=[(el["cutoff"], el["lift_adj"], f"elevated {op} {el['cutoff']:.1f}")]
                           + ([(hi["cutoff"], hi["lift_adj"], f"high {op} {hi['cutoff']:.1f}")] if hi else []),
                           xfmt="{:.1f}", yfmt="{:.1f}")
    # distributions
    v = x.dropna()
    if len(v) > 50:
        lo, hi_ = np.nanpercentile(v, [1, 99])
        bins = np.linspace(lo, hi_, 26)
        c_all, _ = np.histogram(v, bins)
        c_big, _ = np.histogram(v[big.reindex(v.index).fillna(False)], bins)
        mids = (bins[1:] + bins[:-1]) / 2
        dist = svg_line_chart([("all season days", "var(--text-dim)", True, list(zip(mids, 100 * c_all / max(1, c_all.sum())))),
                               (f"≥{THRESH:,}-acre days", "var(--accent)", False, list(zip(mids, 100 * c_big / max(1, c_big.sum()))))],
                              f"{info['title']} ({info['units']})", "% of days", xfmt="{:.1f}", yfmt="{:.0f}")
    else:
        dist = '<p class="note">Not enough history to draw.</p>'
    # biggest days
    rows, stories = [], []
    burn_dir = BURN_DIRS.get(g)
    if gdf is not None:
        for rank, (d, a) in enumerate(acres.sort_values(ascending=False).head(15).items()):
            val = x.get(d, np.nan)
            beyond = np.isfinite(val) and ((val <= el["cutoff"]) if row["tail"] == "low" else (val >= el["cutoff"]))
            high = hi and np.isfinite(val) and ((val <= hi["cutoff"]) if row["tail"] == "low" else (val >= hi["cutoff"]))
            chip = ('<span class="chip hi">high</span>' if high else '<span class="chip yes">elevated</span>' if beyond
                    else '<span class="chip">no</span>' if np.isfinite(val) else '<span class="note">no obs</span>')
            # the day's main fire inside the area, where it burned most, how far it ran, the fuels there
            # main fire = most acres inside the area that day; its total, PSA and run use ALL its growth
            # (a fire can burn mostly outside the footprint -- AUGUST COMPLEX on 2020-09-08 was in NC04)
            day = gdf[gdf.burn_date == d]
            ev = day.groupby(["event_id", "fire_label"]).growth_acres.sum().sort_values(ascending=False)
            (eid, fname), _ = ev.index[0], float(ev.iloc[0])
            whole = gdf_all[(gdf_all.burn_date == d) & (gdf_all.event_id == eid)]
            ev_ac = float(whole.growth_acres.sum())
            psa = whole.groupby("psa").growth_acres.sum().sort_values(ascending=False).index[0]
            miles = run_miles(burn_dir, eid, d) if burn_dir is not None else None
            nf = NFDRS.get(psa)
            erc_p = float(nf.at[d, "ERC_pct"]) if nf is not None and d in nf.index and np.isfinite(nf.at[d, "ERC_pct"]) else None
            erc_v = float(nf.at[d, "ERC"]) if erc_p is not None else None
            rows.append(f'<tr><td>{d:%Y-%m-%d}</td><td class="n">{a:,.0f}</td><td>{esc(fname)} ({ev_ac:,.0f} ac)</td>'
                        f'<td class="n">{"–" if miles is None else f"{miles:.1f}"}</td><td>{esc(psa)}</td>'
                        f'<td class="n">{"–" if erc_p is None else ordinal(erc_p)}</td>'
                        f'<td class="n">{"–" if not np.isfinite(val) else f"{val:.1f}"}</td><td>{chip}</td></tr>')
            if rank < 5:
                pname = PSA_NAMES.get(psa, psa)
                cond = (f"{info['title'].split(' (')[0]} was {val:.1f} {info['units']}"
                        + (" (past the high cutoff)" if high else " (past the elevated cutoff)" if beyond else
                           " (not past the cutoff)") if np.isfinite(val) else f"{info['title'].split(' (')[0]} wasn't observed")
                fuel = (f" With ERC at the {ordinal(erc_p)} percentile ({erc_v:.0f}) in {esc(psa)} {esc(pname)}," if erc_p is not None
                        else f" In {esc(psa)} {esc(pname)},")
                run = f" ran {miles:.0f} mile{'s' if round(miles) != 1 else ''} and" if miles is not None and miles >= 1 else ""
                stories.append(f"<li><b>{d:%a %b} {d.day}, {d.year}</b>: {cond}.{fuel} the <b>{esc(fname)}</b>{run} grew "
                               f"{ev_ac:,.0f} acres that fire day ({a:,.0f} acres across the {'footprint' if fp else 'region'}).</li>")
    # weather x fuels: odds when the measure is past its cutoff AND the area's ERC is >= 90th percentile
    joint = ""
    areas = fp["psas"] if fp else sorted({c for c in (gdf.psa.unique() if gdf is not None else []) if c in NFDRS})
    ercs = [NFDRS[c]["ERC_pct"] for c in areas if c in NFDRS]
    if ercs and gdf is not None:
        erc_area = pd.concat(ercs, axis=1).mean(axis=1).reindex(idx)
        past = ((x <= el["cutoff"]) if row["tail"] == "low" else (x >= el["cutoff"])).reindex(idx)
        hot = erc_area >= 90
        ok = past.notna() & erc_area.notna()
        bb = big.reindex(idx).fillna(False)
        def cell(mask):
            m = mask & ok
            n = int(m.sum())
            return (f'<td class="n">{n:,} days</td><td class="n">{(bb[m].mean() if n else float("nan")):.1%}</td>'
                    f'<td class="n">{"–" if not n or not base else "×%.1f" % (bb[m].mean() / base)}</td>')
        joint = f"""<div class="card"><h2>Weather + fuels</h2>
<p class="sub">How the odds of a ≥{THRESH:,}-acre day in {esc(area)} change when this measure is past its elevated cutoff and the fuels are
critically dry: ERC (energy release component) at or above the 90th percentile, averaged over {"the footprint's" if fp else "the region's"} PSAs
(NFDRS from the toolkit's RAWS reference data, 2005–present).</p>
<div class="scroll"><table><tr><th></th><th class="n">Days</th><th class="n">Share with ≥{THRESH:,} ac</th><th class="n">× season base ({base:.1%})</th></tr>
<tr><td>Past cutoff <b>and</b> ERC ≥ 90th</td>{cell(past.astype(bool) & hot)}</tr>
<tr><td>Past cutoff, ERC below 90th</td>{cell(past.astype(bool) & ~hot)}</tr>
<tr><td>Not past cutoff, ERC ≥ 90th</td>{cell(~past.astype(bool) & hot)}</tr>
<tr><td>Neither</td>{cell(~past.astype(bool) & ~hot)}</tr></table></div>
<p class="note">Weather and fuels multiply: the same wind event does far more on critically dry fuels. Days before 2005 have no NFDRS
history here and are left out of this table.</p></div>"""
    elif g == "OSCC":
        joint = """<div class="card"><h2>Weather + fuels</h2><p class="note">NFDRS (ERC/BI) history for Southern California
hasn't been built in this project yet, so the fuels side isn't shown here.</p></div>"""
    caught = big & ((x <= el["cutoff"]) if row["tail"] == "low" else (x >= el["cutoff"]))
    stations = "".join(f"<li><b>{esc(c)}</b>: {esc(n)} ({ll[0]:.2f}°N, {-ll[1]:.2f}°W)</li>" for c, ll, n in info["stations"])
    fmap = ""
    if psas is not None and cells is not None and out_dir is not None and gdf_all is not None:
        past = idx[((x <= el["cutoff"]) if row["tail"] == "low" else (x >= el["cutoff"])).reindex(idx).fillna(False).values]
        other = idx.difference(past)
        png = f"{slug(ent, key)}_footprint.png"
        g_psas = psas[psas.PSANationa.str.startswith("NC" if g == "ONCC" else "SC")]
        footprint_map(out_dir / png, past, other, gdf_all, g_psas, cells[cells.gacc == g], fp["psas"] if fp else [],
                      f"Days past the elevated cutoff ({op} {el['cutoff']:.1f} {info['units']})")
        names = dict(zip(psas.PSANationa, psas.PSANAME))
        fp_txt = (f"<p>This measure is scored against fire growth inside its <b>footprint</b>: "
                  f"{esc(fp['label'])}, PSAs " + ", ".join(f"{c} {esc(names.get(c, ''))}" for c in fp["psas"]) +
                  ". Only acres that satellites saw burning inside those PSAs count toward its big growth days.</p>"
                  if fp else f"<p>This region-wide measure is scored against fire growth anywhere in {esc(REGION.get(g, g))}.</p>")
        fmap = f"""<div class="card"><h2>Where fire grew: the satellite footprint</h2>{fp_txt}
<img class="map" src="{png}" alt="Map of satellite-mapped fire growth on days past the cutoff vs other days">
<p class="note">Left: fire growth per day on the {season} days past the elevated cutoff, with each PSA shaded by how much more often it
had a day of at least 1,000 acres on those days than on the other days of the season. Right: growth per day on all the other days. If the
measure matches where its weather acts, the left map's growth and shading should concentrate inside the outlined footprint.</p></div>"""
    locator = svg_locator(data["lines"], region_lines.get(info["region"], []) if info["region"] else (),
                          info["stations"], info["arrow"])
    models = data["meta"]["models"]
    body = f"""
<h1>{esc(info['title'])}</h1>
<p class="sub">Calibrated predictor · {esc(REGION.get(g, g))}, {season} ({mnames}){" · footprint: " + esc(fp["label"]) if fp else ""} · today's run {esc(data['meta']['generated'])} PT</p>
<div class="card"><div class="cols"><div>
<h2>What it measures</h2><p>{info['what']}</p>
<h2>Why it matters for fire</h2><p>{info['why']}</p>
{"<h3>Stations</h3><ul>" + stations + "</ul>" if stations else ""}
</div><div>{locator}<p class="note">{"Arrow: the direction the pressure difference pushes air (from high to low pressure) when the gradient is negative." if info["arrow"] else "Orange outline: the region averaged." if info["region"] else "An area-wide measure: no single station; the footprint's PSAs are outlined in the maps below." if not info["stations"] else "Dot: the airport weather station." if "rhmin" in key or "windmax" in key else "Dot: the weather-balloon site."}</p></div></div></div>

{fmap}
<div class="card"><h2>The calibrated cutoffs</h2>
<div class="stat"><b>{op} {el['cutoff']:.1f} {esc(info['units'])}</b><span>"elevated" cutoff</span></div>
<div class="stat"><b>×{el['lift_adj']:.1f}</b><span>usual odds of a ≥{THRESH:,}-acre day</span></div>
<div class="stat"><b>{el['share_of_large']:.0%}</b><span>of those days caught</span></div>
<div class="stat"><b>{el['days']:,}</b><span>days past it, 2001–present</span></div>
{f'<div class="stat"><b>{op} {hi["cutoff"]:.1f}</b><span>"high" cutoff, ×{hi["lift_adj"]:.1f}, catches {hi["share_of_large"]:.0%}</span></div>' if hi else ""}
<div class="stat"><b>{row['auc']:.2f}</b><span>AUC (0.5 = no skill, 1 = perfect)</span></div>
<p>On a typical {season} day, {base:.1%} of days have at least {THRESH:,} acres of satellite-mapped fire growth in
{esc(area)}. When this measure is past the elevated cutoff, such days are about <b>{el['lift_adj']:.1f} times as likely as usual for the
same time of year</b>. The cutoff is the most inclusive one that reaches ×2 with at least 25 days of history behind it.</p>
<h3>Lift vs cutoff</h3>{curve}
<p class="note">How the odds grow as the cutoff gets stricter ({"lower" if row["tail"] == "low" else "higher"} values). Cutoffs beyond the
"high" mark rest on fewer days, so that end of the curve is noisier.</p>
<h3>Big growth days vs all days</h3>{dist}
<p class="note">The share of {season} days at each value: all days (dashed) and days with ≥{THRESH:,} acres of growth (orange).
The further the orange curve sits toward the cutoff side, the more useful the measure.</p></div>

<div class="card"><h2>The biggest growth days and what this measure showed</h2>
<p class="sub">{esc(area)}, {season}, 2001–present. Acres are only those inside the footprint; fire names and acres from satellite
detections inside mapped perimeters.
{int(caught.sum())} of {int(big.sum())} days with ≥{THRESH:,} acres were past the elevated cutoff ("no obs": a station didn't report that day).</p>
<h3>The five biggest days</h3><ul>{''.join(stories)}</ul>
<div class="scroll"><table><tr><th>Fire day</th><th class="n">Acres ({"footprint" if fp else "region"})</th><th>Main fire (its acres)</th>
<th class="n">Run (mi)</th><th>PSA</th><th class="n">ERC pct</th><th class="n">Value</th><th>Past cutoff?</th></tr>
{''.join(rows)}</table></div>
<p class="note">Run: how far that day's newly burned area reached beyond everything the fire had burned before, from the 100 m satellite
day-of-burn grid (on a fire's first day, the span of that day's burn). ERC pct: energy release component in the PSA where the fire grew
most, ranked against that PSA's 2005–present history.</p></div>
{joint}

{starts_card(row, area, season)}
<div class="card"><h2>Today's forecast</h2>
<p class="sub">% of each ensemble's members past the elevated cutoff ({op} {el['cutoff']:.1f} {esc(info['units'])}), by fire day.</p>
{forecast_table(row, models, data['meta']['days'])}
{odds_table(row, models, data['meta']['days'])}
<h3>Color key</h3>{swatch_legend(SEQ, ['<5%', '5–20%', '20–40%', '40–60%', '60–80%', '≥80%'])}
<p class="note">Darker = more members past the cutoff. The models are weighted equally in "All models". How to use it:
<a href="../guide/outlook.html">Reading the calibrated outlook</a>.</p></div>
"""
    return page(info["title"], body, desc=f"{info['title']}: calibrated against large-fire growth days")


# ----------------------------------------------------------------------------- Hawaii rows (2026-10-13)

HI_ISLANDS = {"HI01": "Kauai", "HI02": "Oahu", "HI03": "Maui Nui (Maui, Molokai, Lanai)", "HI04": "Hawaii Island"}
HI_STATIONS = {"PHLI": "Lihue, Kauai", "PHNL": "Honolulu, Oahu", "PHJR": "Kalaeloa, Oahu", "PHOG": "Kahului, Maui",
               "PHNY": "Lanai", "PHMK": "Molokai", "PHKO": "Kona, Hawaii Island", "PHTO": "Hilo, Hawaii Island"}


def describe_hawaii(key, cfg):
    st = cfg.get("hawaii", {}).get("stations", {})
    stn = lambda c: [(c, tuple(st[c]), HI_STATIONS.get(c, c))] if c in st else []
    m = re.match(r"^(HI\d\dN)-(HI\d\dN)_max6$", key)
    if m:
        a, b = m.group(1)[2:4], m.group(2)[2:4]
        return dict(title=f"Trade-wind gradient, {a}°N to {b}°N", units="hPa", arrow=None, region=None, stations=[],
                    what=(f"The sea-level pressure at {a}°N minus {b}°N along 157°W, the highest of the fire day's 2 a.m. and 2 p.m. "
                          "(12 and 00 UTC) values: how hard the subtropical high to the north is pushing air toward the islands."),
                    why=("A big north-south difference means strong, gusty trades, and the strongest cases bring damaging downslope winds "
                         "to the leeward slopes. August 8, 2023 on Maui was a strong high to the north with Hurricane Dora passing far to "
                         "the south: an unusually steep gradient over very dry grass."))
    m = re.match(r"^(PH[A-Z]{2})-(PH[A-Z]{2})_(min|max)6$", key)
    if m:
        a, b, k = m.groups()
        return dict(title=f"{a}–{b} pressure gradient (fire-day {k})", units="hPa", arrow=None, region=None, stations=stn(a) + stn(b),
                    what=(f"Sea-level pressure at {HI_STATIONS.get(a, a)} minus {HI_STATIONS.get(b, b)}, the fire day's {k} over the "
                          "synoptic hours (the times the models are sampled)."),
                    why="Pressure differences along the chain track the trade winds' strength and direction across the islands.")
    m = re.match(r"^(PH[A-Z]{2})_rhmin6$", key)
    if m:
        c = m.group(1)
        return dict(title=f"{c} minimum humidity", units="%", arrow=None, region=None, stations=stn(c),
                    what=f"The lowest relative humidity at {HI_STATIONS.get(c, c)} airport over the fire day's synoptic hours.",
                    why=("Leeward lowlands dry out in the afternoon under the trade inversion; low humidity cures the guinea and buffel "
                         "grass that carries most of Hawaii's large fires."))
    m = re.match(r"^(PH[A-Z]{2})(850|700)_(spd|dd)_(12z|00z)$", key)
    if m:
        c, lev, kind, tag = m.groups()
        when = "2 a.m." if tag == "12z" else "2 p.m."
        if kind == "spd":
            return dict(title=f"{c} {lev} hPa trade wind ({when})", units="kt", arrow=None, region=None, stations=stn(c),
                        what=f"Wind speed at {lev} hPa (about 5,000 ft) in the {HI_STATIONS.get(c, c)} weather-balloon sounding at {when} HST.",
                        why="The trades' strength just under the inversion: strong winds here mix down as gusty surface winds on lee slopes.")
        return dict(title=f"{c} {lev} hPa dryness ({when})", units="°C", arrow=None, region=None, stations=stn(c),
                    what=f"Dewpoint depression (temperature minus dewpoint) at {lev} hPa in the {HI_STATIONS.get(c, c)} sounding at {when} HST.",
                    why="Dry air above the trade inversion can be mixed down on the leeward slopes, dropping humidity sharply.")
    m = re.match(r"^HI_(850spd|850east|700dd)_max$", key)
    if m:
        k = m.group(1)
        t = {"850spd": ("Trade-wind speed upstream (850 hPa)", "m/s", "the 850 hPa wind speed"),
             "850east": ("Easterly trade wind upstream (850 hPa)", "m/s", "the easterly part of the 850 hPa wind"),
             "700dd": ("700 hPa dryness upstream", "°C", "the 700 hPa dewpoint depression")}[k]
        return dict(title=t[0], units=t[1], arrow=None, region=None, stations=[],
                    what=(f"The fire day's highest value of {t[2]} averaged over the trade-wind box northeast of the islands "
                          "(20–24°N, 158–153°W), from ERA5 for the history and from each ensemble for the forecast."),
                    why="A single measure of how strong (or dry) the trades feeding the islands are, free of any one station's terrain.")
    m = re.match(r"^(HI0\d)_(vpd_wind|rh_min|wind_max)(_anom)?$", key)
    if m:
        fid, kind, anom = m.groups()
        isl = HI_ISLANDS.get(fid, fid)
        title, units, what, why = {
            "vpd_wind": ("Hot-Dry-Windy", "hPa·m/s", "The highest island-average 2 m vapour-pressure deficit times 10 m wind of the fire day (ERA5 / ensembles at 2 a.m. and 2 p.m.).",
                         "Heat, dryness and wind at once, over the whole island rather than one airport."),
            "rh_min": ("area minimum humidity", "%", "The lowest island-average 2 m relative humidity of the fire day.",
                       "Island-wide afternoon dryness: low humidity everywhere cures the grass and slows overnight recovery."),
            "wind_max": ("10 m wind", "m/s", "The strongest island-average 10 m wind of the fire day.",
                         "Strong trades drive Hawaii's fastest grass fires.")}[kind]
        if anom:
            title += " anomaly"
            what += " Shown as the departure from the 2001–2025 ERA5 normal for the date (±15 days)."
        return dict(title=f"{isl} {title}", units=units, arrow=None, region=None, stations=[], what=what, why=why)
    return dict(title=key, units="", arrow=None, region=None, stations=[], what="", why="")


def hawaii_page(row, cfg, cal, data, paths):
    key = row["key"]
    H = cal["hawaii"]
    info = describe_hawaii(key, cfg)
    t = int(row.get("thresh", H.get("main_threshold", 300)))
    op = "≤" if row["tail"] == "low" else "≥"
    el, hi = row["elevated"], row.get("high")
    base = row.get("base") or float("nan")
    fid = row.get("footprint_hi")
    area = HI_ISLANDS.get(fid, "Hawaii") if fid else "Hawaii"
    months = H["seasons"]["HI"]
    mnames = ", ".join(dt.date(2000, m_, 1).strftime("%b") for m_ in months)
    scan_p, der_p = paths.cal / "hawaii_threshold_scan.csv", paths.cal / "hawaii_predictors_derived.csv"
    curve, dist, rows_html, stories = '<p class="note">No history table.</p>', "", [], []
    if scan_p.exists():
        sc = pd.read_csv(scan_p)
        sc = sc[(sc.predictor == key) & (sc.threshold_acres == t) & (sc.days >= 25)]
        if len(sc):
            curve = svg_line_chart([("lift", "var(--accent)", False, list(zip(sc.cutoff, sc.lift_adj)))],
                                   f"cutoff ({info['units']}; counts days {op} cutoff)", "× usual odds for the date",
                                   hlines=[(1, "usual odds"), (2, "elevated ×2"), (4, "high ×4")],
                                   marks=[(el["cutoff"], el["lift_adj"], f"elevated {op} {el['cutoff']:.1f}")], xfmt="{:.1f}", yfmt="{:.1f}")
    occ_p = paths.history / "hawaii" / "fire_occurrence" / "hi_fire_occurrence.csv"
    if der_p.exists() and occ_p.exists():
        der = pd.read_csv(der_p, index_col=0, parse_dates=True)
        x = der[key] if key in der else pd.Series(dtype=float)
        occ = pd.read_csv(occ_p, usecols=["FireDiscoveryDate", "TotalAcres", "FDRA_Name", "FireName"], parse_dates=["FireDiscoveryDate"])
        occ = occ[occ.FDRA_Name.astype(str).str.startswith("HI") & (occ.TotalAcres >= t)]
        if fid:
            occ = occ[occ.FDRA_Name == fid]
        occ["day"] = occ.FireDiscoveryDate.dt.normalize()
        daily = occ.groupby("day").agg(acres=("TotalAcres", "sum"), fires=("FireName", "size"),
                                      names=("FireName", lambda v: ", ".join(str(n).title() for n in v if str(n) != "nan")[:90]),
                                      isl=("FDRA_Name", lambda v: ", ".join(sorted({HI_ISLANDS.get(i, i).split(' (')[0] for i in v}))))
        nf = NFDRS
        for rank, (d, r) in enumerate(daily.sort_values("acres", ascending=False).head(12).iterrows()):
            val = x.get(d, np.nan)
            past = np.isfinite(val) and ((val <= el["cutoff"]) if row["tail"] == "low" else (val >= el["cutoff"]))
            chip = ('<span class="chip yes">elevated</span>' if past else '<span class="chip">no</span>' if np.isfinite(val)
                    else '<span class="note">no obs</span>')
            ercs = [float(nf[c].at[d, "ERC_pct"]) for c in ([fid] if fid else list(HI_ISLANDS)) if c in nf and d in nf[c].index
                    and np.isfinite(nf[c].at[d, "ERC_pct"])]
            erc = np.mean(ercs) if ercs else None
            rows_html.append(f'<tr><td>{d:%Y-%m-%d}</td><td class="n">{r.acres:,.0f}</td><td>{esc(r.names or "unnamed")}</td>'
                             f'<td>{esc(r.isl)}</td><td class="n">{"–" if erc is None else ordinal(erc)}</td>'
                             f'<td class="n">{"–" if not np.isfinite(val) else f"{val:.1f}"}</td><td>{chip}</td></tr>')
            if rank < 5:
                cond = (f"{info['title']} was {val:.1f} {info['units']} ({'past' if past else 'not past'} the cutoff)"
                        if np.isfinite(val) else f"{info['title']} wasn't observed")
                fuel = f", with ERC at the {ordinal(erc)} percentile" if erc is not None else ""
                stories.append(f"<li><b>{d:%a %b} {d.day}, {d.year}</b>: {cond}{fuel}. Fires that day that reached {t:,}+ acres: "
                               f"<b>{esc(r.names or 'unnamed')}</b> on {esc(r.isl)}, {r.acres:,.0f} acres in all.</li>")
    J = row.get("joint")
    jt = ""
    if J:
        jr = lambda k, lab: (f"<tr><td>{lab}</td><td class='n'>{J[k]['days']:,}</td><td class='n'>{J[k]['big_days']:,}</td>"
                             f"<td class='n'>{'–' if J[k]['lift'] is None else '×%.1f' % J[k]['lift']}</td></tr>")
        jt = f"""<div class="card"><h2>Weather + fuels</h2>
<p class="sub">How the odds of a {t:,}+ acre fire day in {esc(area)} change with this measure and the fuels: ERC at or above the 90th percentile,
the island-average of Hawaii's RAWS (2005–present).</p>
<div class="scroll"><table><tr><th></th><th class="n">Days</th><th class="n">Large-fire days</th><th class="n">× usual</th></tr>
{jr("past_dry", "Past cutoff <b>and</b> ERC ≥ 90th")}{jr("past_wet", "Past cutoff, ERC below 90th")}
{jr("not_dry", "Not past, ERC ≥ 90th")}{jr("not_wet", "Neither")}</table></div></div>"""
    stations = "".join(f"<li><b>{esc(c)}</b>: {esc(n)} ({ll[0]:.2f}°N, {-ll[1]:.2f}°W)</li>" for c, ll, n in info["stations"])
    models = data["meta"]["models"]
    body = f"""
<h1>{esc(info['title'])}</h1>
<p class="sub">Calibrated predictor · Hawaii, {mnames} · {"island: " + esc(area) if fid else "all islands"} · today's run {esc(data['meta']['generated'])} PT</p>
<div class="card"><h2>What it measures</h2><p>{info['what']}</p>
<h2>Why it matters for fire</h2><p>{info['why']}</p>
{"<h3>Stations</h3><ul>" + stations + "</ul>" if stations else ""}
<p class="note">Hawaii has no Predictive Service Areas, so the islands are grouped as Kauai, Oahu, Maui Nui (Maui, Molokai, Lanai) and
Hawaii Island. A <b>large-fire day</b> is a day when satellite-mapped growth, or the final size of the fires found that day, reached
{t:,} acres. Hawaii's grass fires mostly burn out within a day or two, often between satellite passes, so the fire records (FPA FOD,
InFORM) carry most of the history.</p></div>
<div class="card"><h2>The calibrated cutoffs</h2>
<div class="stat"><b>{op} {el['cutoff']:.1f} {esc(info['units'])}</b><span>"elevated" cutoff</span></div>
<div class="stat"><b>×{el['lift_adj']:.1f}</b><span>usual odds of a {t:,}+ acre fire day</span></div>
<div class="stat"><b>{el['share_of_large']:.0%}</b><span>of those days caught</span></div>
<div class="stat"><b>{el['days']:,}</b><span>days past it, 2001–present</span></div>
<div class="stat"><b>{row['auc']:.2f}</b><span>AUC (0.5 = no skill, 1 = perfect)</span></div>
<p>On a typical day, {base:.1%} of days are {t:,}+ acre fire days in {esc(area)}. Past the elevated cutoff they're about
<b>{el['lift_adj']:.1f} times as likely as usual for the date</b>.</p>
<h3>Lift vs cutoff</h3>{curve}</div>
<div class="card"><h2>The biggest fire days and what this measure showed</h2>
<ul>{''.join(stories) or '<li>None in the record.</li>'}</ul>
<div class="scroll"><table><tr><th>Day</th><th class="n">Acres</th><th>Fires</th><th>Island</th><th class="n">ERC pct</th><th class="n">Value</th><th>Past cutoff?</th></tr>
{''.join(rows_html)}</table></div>
<p class="note">Fires, final sizes and discovery dates: FPA FOD (Short, USDA Forest Service) and NIFC InFORM. ERC: island average of the
Hawaii RAWS percentiles.</p></div>
{jt}
<div class="card"><h2>Today's forecast</h2>
<p class="sub">% of each ensemble's members past the elevated cutoff ({op} {el['cutoff']:.1f} {esc(info['units'])}), by fire day.</p>
{forecast_table(row, models, data['meta']['days'])}
{odds_table(row, models, data['meta']['days'], target=f"a {t:,}+ acre fire day")}
<h3>Color key</h3>{swatch_legend(SEQ, ['<5%', '5–20%', '20–40%', '40–60%', '60–80%', '≥80%'])}
<p class="note">Island cells are few on the models' grids (0.25–0.5°), so island measures are smoothed; airport cutoffs come from the
stations themselves and a model's bias at an airport isn't removed yet.</p></div>
"""
    return page(info["title"], body, desc=f"{info['title']}: calibrated against Hawaii large-fire days")


# ----------------------------------------------------------------------------- lightning rows

NLDN_CREDIT = ("Lightning: Vaisala National Lightning Detection Network (NLDN), 2000–2025, used under license. Shown only as "
               "derived counts, rates and maps; no strike data are published or offered for download.")


def strike_map(out_png, density, psas, fp_psas, title):
    """Mean May-Oct strikes per 100 km² per year on the 0.05° grid (derived from NLDN; no strikes
    shown), California PSAs, this footprint outlined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    c = density["counts"].astype(float)            # year, month, lat, lon
    yrs = density["years"]
    sel = (yrs >= 2001) & (yrs <= 2025)
    per_yr = c[sel][:, 4:10].sum(axis=1).mean(axis=0)          # May-Oct, mean over years
    lat0, lon0, res = float(density["lat0"]), float(density["lon0"]), float(density["res"])
    ny, nx = per_yr.shape
    lats = lat0 + (np.arange(ny) + 0.5) * res
    km2 = (res * 111.2) * (res * 111.2 * np.cos(np.radians(lats)))[:, None]
    dens = per_yr / km2 * 100
    pp = psas.to_crs(4326)
    fig, ax = plt.subplots(figsize=(6.4, 7.0))
    levels = [0.5, 1, 2, 4, 8, 16, 32, 64]
    cols = ["#f3f4f3", "#fdf0e6", "#f9d2b4", "#f2a679", "#e47740", "#c2531f", "#8e3812", "#4a1d08", "#2a0f03"]
    cmap = mcolors.ListedColormap(cols)
    norm = mcolors.BoundaryNorm([0] + levels + [1e9], cmap.N)
    ax.pcolormesh(lon0 + np.arange(nx + 1) * res, lat0 + np.arange(ny + 1) * res, dens, cmap=cmap, norm=norm, shading="flat")
    pp.boundary.plot(ax=ax, color="#5c655f", linewidth=0.4)
    if fp_psas:
        pp[pp.PSANationa.isin(fp_psas)].boundary.plot(ax=ax, color="#1b6de0", linewidth=2.0)
    b = pp.total_bounds
    ax.set_xlim(b[0] - 0.2, b[2] + 0.2)
    ax.set_ylim(b[1] - 0.2, b[3] + 0.2)
    ax.set_aspect(1 / np.cos(np.radians(37.5)))
    ax.set_axis_off()
    ax.set_title(title, fontsize=9.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02,
                      ticks=[0.25] + [(a + b_) / 2 for a, b_ in zip(levels, levels[1:])] + [80])
    cb.ax.set_xticklabels(["<0.5", "0.5-1", "1-2", "2-4", "4-8", "8-16", "16-32", "32-64", ">64"], fontsize=7)
    cb.set_label("strikes per 100 km² per year, May-Oct 2001-2025 (Vaisala NLDN, derived)", fontsize=7.5)
    fig.savefig(out_png, dpi=105, bbox_inches="tight")
    plt.close(fig)


def lightning_page(row, cfg, cal, data, psas=None, out_dir=None, strikes=None, density=None):
    ent, key = row["entity"], row["key"]
    g = ent.split("-")[0]
    fid = row["footprint"]["id"]
    L = cal["lightning"][fid]
    fp = row["footprint"]
    el, hi = row["elevated"], row.get("high")
    st, J = L["starts"], L["joint"]
    area = fp["label"]
    names = PSA_NAMES
    formula = key.endswith("_ltgformula")          # 2026-10-22: the lightning formula replaced the CAPE setup
    units = "strikes" if formula else "J/kg"
    measure = "lightning formula" if formula else "thunderstorm setup"
    title = f"{area.split(' (')[0]} {measure}"
    models = data["meta"]["models"]
    smap, months_svg = "", ""
    if psas is not None and density is not None and out_dir is not None:
        png = f"{slug(ent, key)}_strikes.png"
        strike_map(out_dir / png, density, psas, fp["psas"], f"Lightning, May-Oct: {area}")
        smap = f'<img class="map" src="{png}" alt="Map of mean lightning strike density by 0.05 degree cell, May to October">'
    if strikes is not None:
        n = strikes.reindex(columns=fp["psas"], fill_value=0).sum(axis=1)
        n = n[(n.index.year >= 2001) & (n.index.year <= 2025)]
        yrs = n.index.year.nunique()
        bym = n.groupby(n.index.month).sum() / max(1, yrs)
        dym = (n >= L["lightning_day"]).groupby(n.index.month).sum() / max(1, yrs)
        months_svg = svg_bars([dt.date(2000, m, 1).strftime("%b") for m in range(1, 13)],
                              [float(bym.get(m, 0)) for m in range(1, 13)], fmt="{:,.0f}")
        per_month = ", ".join(f"{dt.date(2000, m, 1).strftime('%b')} {dym.get(m, 0):.1f}" for m in range(5, 11))
    else:
        per_month = ""
    fmt_pct = lambda v: "–" if v is None else f"{v:.0%}"
    jrow = lambda k, lab: (f"<tr><td>{lab}</td><td class='n'>{J[k]['days']:,}</td><td class='n'>{J[k]['big_days']:,}</td>"
                           f"<td class='n'>{'–' if J[k]['lift'] is None else '×%.1f' % J[k]['lift']}</td></tr>")
    ev_items = []
    for e in L.get("events", [])[:6]:
        d = pd.Timestamp(e["date"])
        v = e.get("value", e.get("capep90"))
        cond = (f"the {measure} was {v:,.0f} {units[:-1] if formula and round(v) == 1 else units} ({'past' if e['past'] else 'below'} the cutoff)"
                if v is not None else f"the {measure} wasn't available (the formula's history is Jun–Sep)" if formula
                else f"the {measure} wasn't available")
        fuel = f", ERC at the {ordinal(e['erc_pct'])} percentile" if e.get("erc_pct") is not None else ""
        ev_items.append(f"<li><b>{d:%a %b} {d.day}, {d.year}</b>: {e['strikes']:,} strikes in the footprint, {cond}{fuel}. "
                        f"{e['fires']} large lightning fire{'s' if e['fires'] != 1 else ''} found: "
                        f"{esc(', '.join(e['names']))}{' and others' if e['fires'] > len(e['names']) else ''} "
                        f"({e['acres']:,} acres in all).</li>")
    wet_dry = ""
    if L.get("start_rate_dry") is not None and L.get("start_rate_wet") is not None:
        wet_dry = (f"<p>Dry versus wet lightning: on the {L['ltg_days_dry']:,} lightning days with under 0.1 inch of rain across the footprint "
                   f"(ERA5), a large lightning fire was found that day or the next on <b>{L['start_rate_dry']:.1%}</b> of them; on the "
                   f"{L['ltg_days_wet']:,} wetter lightning days, <b>{L['start_rate_wet']:.1%}</b>.</p>")
    fp_list = ", ".join(f"{c} {esc(names.get(c, ''))}" for c in fp["psas"])
    cmp_ = L.get("compare") or {}
    fit_m = L.get("fit_months") or L["months"]
    mname = lambda ms: f"{dt.date(2000, ms[0], 1):%b}–{dt.date(2000, ms[-1], 1):%b}"
    if formula:
        what = (f"""<h2>What it measures</h2><p>The number of cloud-to-ground lightning strikes the <a href="../guide/lightning_formula.html">lightning
formula</a> expects across the footprint on the fire day. The formula follows the shape of ECMWF's own lightning scheme (Lopez 2016): convective
rain, cloud ice and water, instability (CAPE) and cloud-base height, multiplied together, with its exponents and scale fit to 25 years of
observed strikes. It replaced the plain CAPE measure this row used before.</p>""")
        extra = [m for m in L["months"] if m not in fit_m]
        fitnote = (f"""<p>It was fit to observed strikes, {mname(fit_m)}{" only (the formula's reanalysis history)" if extra else
", every month the row is shown"}. On those days it ranks lightning days with an AUC of <b>{L['auc_100']:.2f}</b>{
f", against {cmp_['auc_100']:.2f} for the CAPE setup it replaced" if cmp_ else ""}.{f''' The row is still shown {mname(L['months'])}:
in {" and ".join(dt.date(2000, m, 1).strftime("%B") for m in extra)} its cutoffs are carried over from summer, untested. Spring
and fall storms are colder-based and lower-topped, so read those months with care.''' if extra else ""}</p>""")
        fc_note = ("Only ECMWF IFS ENS carries the convective fields the formula needs (50 members, 14 fire days); the other models "
                   "show no values. Past day 7 the outlook fades these boxes: thunderstorm forecasts that far out have little skill, "
                   "and they haven't been verified.")
        bias_note = ("The formula was fit on the ERA5 reanalysis, an older and coarser version of the IFS. Today's IFS makes sharper "
                     "convective rain, which can push the expected strikes higher than the history behind the cutoffs.")
    else:
        what = """<h2>What it measures</h2><p>CAPE (convective available potential energy, J/kg) is the energy a rising air parcel can gain: the fuel for
thunderstorms. This measure is the 90th-percentile CAPE across the footprint's grid cells, the higher of the fire day's 5 a.m. and 5 p.m.
(12 and 00 UTC) values. "90th percentile" means the most unstable tenth of the footprint, where storms would form first.</p>"""
        fitnote = """<p>It was fit to 25 years of lightning strikes. Of the rules tested, this plain CAPE measure matched real lightning days best. Requiring
dry air aloft made it worse: storms need moisture to form, and dryness matters for whether a strike starts a fire, which the fuels side
(ERC) handles.</p>"""
        fc_note = "Only ECMWF IFS ENS and NOAA GEFS forecast CAPE; the two AI models don't."
        bias_note = ("A model's CAPE can run higher or lower than the reanalysis this was fit to; the run archive is collecting the "
                     "forecasts to check that.")
    cut_fmt = lambda c: f"{c:,.0f} {units}" if not formula or c >= 10 else f"{c:.1f} {units}"
    season = mname(fit_m)                              # the months the history (and so these rates) covers
    setup = "Formula past the cutoff" if formula else "Thunderstorm setup"
    body = f"""
<h1>{esc(title)}</h1>
<p class="sub">Thunderstorms and new lightning fires · {esc(REGION.get(g, g))}, May–Oct · footprint: {esc(area)} · today's run {esc(data['meta']['generated'])} PT</p>
<div class="card"><div class="cols"><div>
{what}
<h2>Why it matters for fire</h2><p>Lightning starts most of California's largest fires. In Northern California 235 of the 462 fires since
2001 that burned at least 1,000 acres were lightning-caused. Lightning fires come in clusters, often in remote country, and most are found
on the day of the storm or the next. This row asks two things: will there be a lightning day, and how likely is a new large fire?</p>
{fitnote}
<h3>Footprint</h3><p>{fp_list}</p>
</div><div>{smap}<p class="note">{esc(NLDN_CREDIT)}</p></div></div></div>

<div class="card"><h2>How well it finds lightning days</h2>
<div class="stat"><b>≥ {cut_fmt(el['cutoff'])}</b><span>"elevated" cutoff</span></div>
<div class="stat"><b>×{el['lift_adj']:.1f}</b><span>usual odds of a ≥{L['lightning_day']}-strike day</span></div>
<div class="stat"><b>{fmt_pct(L['p_ltg_past'])}</b><span>of days past it had ≥{L['lightning_day']} strikes</span></div>
<div class="stat"><b>{fmt_pct(L['share_ltg_caught'])}</b><span>of all lightning days caught</span></div>
<div class="stat"><b>{L['auc_100']:.2f} / {L['auc_500']:.2f}</b><span>AUC, ≥{L['lightning_day']} / ≥500-strike days</span></div>
{f'<div class="stat"><b>≥ {cut_fmt(hi["cutoff"])}</b><span>"high" cutoff, ×{hi["lift_adj"]:.1f}</span></div>' if hi else ""}
<p>A <b>lightning day</b> here is a fire day with at least {L['lightning_day']} strikes inside the footprint's PSAs. On a typical {season} day
{L['base_ltg']:.1%} of days are lightning days{f"; per year that's about {per_month} by month" if per_month else ""}. Past the elevated
cutoff, {fmt_pct(L['p_ltg_past'])} are, and {fmt_pct(L['p_big_past'])} have 500 strikes or more. History {esc(L['years'])}{f' ({season})' if formula else ''}, {L['days']:,}
days, {L['days_past']:,} past the cutoff.</p>
{"<h3>Average strikes per year by month</h3>" + months_svg if months_svg else ""}
</div>

<div class="card"><h2>New large lightning fires</h2>
<p>On the {L['days_past']:,} days past the cutoff, a lightning fire that went on to burn at least 1,000 acres was found in the footprint
that day or the next on {st['starts_past']:,} of them ({(st['p_past'] or 0):.1%}, against {st['base']:.1%} on a typical {season} day):
about <b>×{(st['lift_adj'] or 0):.1f}</b> the usual odds for the date. That covers {st['share_caught']:.0%} of the {st['total']:,} days a
large lightning fire was found there.</p>
<h3>Weather + fuels</h3>
<div class="scroll"><table><tr><th></th><th class="n">Days</th><th class="n">Days with a new large lightning fire</th><th class="n">× usual</th></tr>
{jrow("past_dry", f"{setup} <b>and</b> ERC ≥ 90th percentile")}
{jrow("past_wet", f"{setup}, ERC below 90th")}
{jrow("not_dry", "No setup, ERC ≥ 90th")}
{jrow("not_wet", "Neither")}</table></div>
<p class="note">ERC is the footprint-average energy release component percentile from the toolkit's NFDRS reference data (2005–present).
Each cell's odds are pulled toward its row's overall odds by {40} days' worth of history, so thin cells can't swing the forecast.</p>
{wet_dry}
</div>

<div class="card"><h2>The biggest lightning-fire days</h2>
<p class="sub">Days the most large (≥1,000-acre) lightning fires were found in the footprint, 2001–2025, with the strikes that day or the
day before and what this measure showed.</p>
<ul>{''.join(ev_items) or '<li>None in the record.</li>'}</ul>
<p class="note">Fires and causes from the satellite fires table (MTBS / WFIGS perimeters; cause and discovery from FPA FOD / InFORM).
Strike counts are derived from Vaisala NLDN.</p></div>

<div class="card"><h2>Today's forecast</h2>
<p class="sub">% of each ensemble's members past the elevated cutoff (≥ {cut_fmt(el['cutoff'])}), by fire day. {fc_note}</p>
{forecast_table(row, models, data['meta']['days'])}
{odds_table(row, models, data['meta']['days'], target="a new large lightning fire")}
<h3>Color key</h3>{swatch_legend(SEQ, ['<5%', '5–20%', '20–40%', '40–60%', '60–80%', '≥80%'])}
<p class="note">{bias_note} How to use it: <a href="../guide/outlook.html">Reading the calibrated outlook</a>.</p>
{'<p class="note">Lightning formula: after Lopez, P., 2016, <i>Mon. Wea. Rev.</i> 144, 3057–3075 (<a href="https://doi.org/10.1175/MWR-D-16-0026.1">doi:10.1175/MWR-D-16-0026.1</a>), on ECMWF IFS ENS open data (CC BY 4.0), calibrated with ERA5 (Copernicus Climate Change Service) to Vaisala NLDN strikes. Not ECMWF&#39;s lightning product.</p>' if formula else ''}</div>
"""
    return page(title, body, desc=f"{title}: {measure} fit to lightning strikes and new large lightning fires")


REF_PLACES = [("the Gulf of Alaska", 57, -145), ("the northeast Pacific", 45, -145), ("the Pacific Northwest", 46, -121),
              ("British Columbia", 54, -125), ("the California coast", 36, -126), ("California", 37, -120),
              ("the Great Basin", 40, -115), ("the Desert Southwest", 33, -111), ("the northern Rockies", 47, -110),
              ("the subtropical Pacific", 28, -140), ("Alaska", 63, -150)]


def place(lat, lon):
    return min(REF_PLACES, key=lambda p: (p[1] - lat) ** 2 + ((p[2] - lon) * np.cos(np.radians(lat))) ** 2)[0]


def regime_text(comp, lats, lons):
    """Plain description of a composite: its strongest ridge and trough centers and what CA sits under."""
    a = np.where(np.isfinite(comp), comp, 0)
    jmax, imax = np.unravel_index(np.argmax(a), a.shape)
    jmin, imin = np.unravel_index(np.argmin(a), a.shape)
    parts = []
    if a[jmax, imax] > 30:
        parts.append(f"a ridge (heights +{a[jmax, imax]:.0f} m above normal) centered near {place(lats[jmax], lons[imax])}")
    if a[jmin, imin] < -30:
        parts.append(f"a trough ({a[jmin, imin]:.0f} m) centered near {place(lats[jmin], lons[imin])}")
    j = int(np.abs(lats - 37.5).argmin()); i = int(np.abs(lons + 120).argmin())
    ca = a[j, i]
    over = ("higher than normal heights over California (warm, dry, sinking air)" if ca > 20 else
            "lower than normal heights over California (cooler, more onshore flow, more chance of showers)" if ca < -20 else
            "near-normal heights over California")
    return ("This pattern has " + (" and ".join(parts) if parts else "only weak anomalies") + f", with {over}.")


def regime_page(c, L, reg, rstats, labels, growth, data, out_dir, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import ai_ensemble_products as P
    comp = reg.composite_anom.values[c]
    lats, lons = reg.lat.values, reg.lon.values
    bm = P.Basemap(cfg, lats, lons)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cf = ax.contourf(lons, lats, comp, levels=P.ANOM_LEVELS, cmap=P.anom_cmap(), extend="both")
    bm.draw(ax)
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
    cb.set_label("500 hPa height anomaly (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(out_dir / f"regime_{L}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    lab = labels
    season = sorted(json.loads(reg.attrs["season_months"]), key=lambda m: (m - 5) % 12)   # May ... Jan
    mons, share = [], []
    for m in season:
        sel = lab[lab.index.month == m]
        mons.append(dt.date(2000, m, 1).strftime("%b"))
        share.append(float((sel == c).mean()) if len(sel) else np.nan)
    col = "lift_adj" if "lift_adj" in rstats else "lift"
    tr = ""
    for g in ("ONCC", "OSCC", "CA"):
        cells = ""
        for t in (1000, 5000, 10000):
            r = rstats[(rstats.gacc == g) & (rstats.regime == L) & (rstats.threshold_acres == t)]
            cells += f'<td class="n">{"–" if r.empty else "×%.2f" % r[col].iloc[0]}</td>'
        tr += f"<tr><td>{REGION[g]}</td>{cells}</tr>"
    # biggest CA growth days in this regime
    rows = ""
    if "CA" in growth:
        tot = daily_totals(growth["CA"])
        days = [d for d in tot.sort_values(ascending=False).index if d in lab.index and lab.get(d) == c][:12]
        for d in days:
            rows += f'<tr><td>{d:%Y-%m-%d}</td><td class="n">{tot[d]:,.0f}</td><td>{top_fires(growth["CA"], d)}</td></tr>'
    # today's forecast probability (pooled)
    R = data["regimes"]
    fc = []
    for k in range(len(data["meta"]["days"])):
        ps = [R["probs"][m["key"]][c][k] for m in data["meta"]["models"] if R["probs"].get(m["key"]) and R["probs"][m["key"]][c][k] is not None]
        fc.append(np.mean(ps) if ps else np.nan)
    fct = ("<tr>" + "".join(f'<th class="n">{d[5:].replace("-", "/")}</th>' for d in data["meta"]["days"]) + "</tr><tr>"
           + "".join(f'<td class="n" style="background:{seq_color(p)};{"color:#fff;" if np.isfinite(p) and p >= .6 else ""}">'
                     f'{"–" if not np.isfinite(p) else round(100 * p)}</td>' for p in fc) + "</tr>")
    freq = float(rstats[rstats.regime == L].freq.iloc[0])
    # documented patterns this regime is over-represented on (ai_ensemble_cefa_patterns.py)
    doc = ""
    cf = common.Paths(cfg).cal / "cefa_patterns.json"
    if cf.exists():
        hits = []
        for pt in json.loads(cf.read_text())["patterns"]:
            e = next((r["enrichment"] for r in pt["regimes"] if r["regime"] == L), None)
            if e and e >= 1.4:
                hits.append(f"{REGION.get(pt['gacc'], pt['gacc'])} {pt['name'].lower()} (×{e:.1f} as common on those days)")
        if hits:
            doc = ('<p><b>Documented fire-weather patterns it goes with</b>: ' + esc("; ".join(hits)) +
                   '. <a href="../guide/patterns.html">How these line up with CEFA and the Synthesis of Knowledge of Extreme Fire Behavior</a>.</p>')
    if PYRO:
        rc = {k: PYRO["regime_by_class"][k][L]["days"] for k in ("pyroCb", "pyroCu", "none")}
        lift = PYRO["regime_by_class"]["pyroCb"][L]["lift"]
        doc += (f'<p><b>Pyroconvection</b> ({_nfires()} studied California fires, {PYRO["summary"]["days"]} fire days): '
                f'{rc["pyroCb"]} pyroCb and {rc["pyroCu"]} pyroCu days in Regime {L}'
                + (f' (pyroCb days ×{lift:.1f} as common as on all study days)' if lift else '')
                + '. <a href="#pyroconvection">Which days</a>.</p>')
    pyro_card = pyro_regime_card(L) if PYRO else ""
    body = f"""
<h1>Regime {L}</h1>
<p class="sub">One of {reg.sizes['regime']} recurring 500 hPa patterns over the eastern Pacific and western U.S. in California's fire
seasons, 2001–2025 · {freq:.0%} of season days</p>
<div class="card"><div class="cols"><div><img class="map" src="regime_{L}.png" alt="Regime {L} composite 500 hPa anomaly map"></div>
<div><h2>What the pattern looks like</h2><p>{regime_text(comp, lats, lons)}</p>
{doc}
<p class="note">The map is the average 500 hPa (~18,000 ft) height anomaly over every day assigned to this regime: red = higher than
normal (ridge), blue = lower than normal (trough). Upper-level ridges bring sinking, warming, drying air; troughs bring cooler air,
onshore flow and the chance of showers or thunderstorms, and the wind shifts and dry northerly flow that can follow them.</p>
{swatch_legend(RDBU, ['−240', '−120', '−60', '±30', '+60', '+120', '+240 m'])}</div></div></div>

<div class="card"><h2>How often it happens</h2>{svg_bars(mons, share)}
<p class="note">Share of each month's days (2001–2025) in Regime {L}.</p></div>

<div class="card"><h2>Fire growth under this pattern</h2>
<p class="sub">Odds of a fire day with at least this much satellite-mapped growth, compared with the usual odds for the same time of year.</p>
<table><tr><th>Region</th><th class="n">≥1,000 ac</th><th class="n">≥5,000 ac</th><th class="n">≥10,000 ac</th></tr>{tr}</table>
<p class="note">×1.00 = no different from usual; ×1.5 = half again as likely; ×0.6 = 40% less likely. A single 500 hPa pattern is only a modest
signal on its own: lightning, wind events and fuel dryness matter a lot.</p>
<h3>California's biggest growth days under Regime {L}</h3>
<div class="scroll"><table><tr><th>Fire day</th><th class="n">Acres (CA)</th><th>Largest fires that day</th></tr>{rows}</table></div></div>
{pyro_card}
<div class="card"><h2>Today's forecast</h2><p class="sub">% of all members (models weighted equally) in Regime {L}, by day.</p>
<div class="scroll"><table>{fct}</table></div>
<p class="note"><a href="../guide/regimes.html">Reading the regimes</a></p></div>
"""
    return page(f"Regime {L}", body, desc=f"500 hPa regime {L} and California fire growth")


# ----------------------------------------------------------------------------- guides

def guides(cfg):
    out = {}
    out["map"] = ("Reading the map", f"""
<h1>Reading the pattern map</h1><p class="sub">How to read the interactive map at the top of the main page.</p>
<div class="card"><h2>The fields</h2>
<h3>500 hPa height anomaly + height (the default)</h3>
<p>The 500 hPa surface sits around 18,000 ft. Its height traces the big ridges and troughs that steer the weather.
The <b>black contours</b> are the height itself in decameters (588 = 5,880 m). The <b>colors</b> show how far the height is from normal
for the date (1991–2020).</p>
{swatch_legend(RDBU, ['−240 m', '−120', '−60', 'near normal', '+60', '+120', '+240 m'])}
<ul><li><b>Red (ridge):</b> sinking, warming, drying air. Hot, dry fire days and poor overnight humidity recovery. A ridge over the
Great Basin with a trough offshore can also set up offshore (Diablo / Santa Ana) wind events.</li>
<li><b>Blue (trough):</b> cooler air, onshore flow, deeper marine layer, a chance of showers. A trough's leading edge can bring dry
lightning; the dry northerly flow behind it can bring wind events.</li>
<li>Contours packed close together mean strong winds aloft. Wavy contours mean an active pattern.</li></ul>
<h3>850 hPa temperature anomaly</h3><p>About 5,000 ft. Red means a warm air mass (hot days, often stable and dry); blue means a cool air mass.</p>
<h3>Wetting rain</h3><p>The share of each model's members with at least {cfg['wetting_rain_mm'] / 25.4:.2f} inch of precipitation from the
start of the run through the end of the day shown. Blue shades only; clear = under 5%.</p>
<h3>Above-normal heights (% of all members)</h3><p>How many members of all four models (weighted equally) put the 500 hPa height above
normal at each spot. Dark red = nearly all members agree on a ridge, dark blue = nearly all agree on a trough, white = they're split.
Agreement is the quickest way to tell a confident pattern from a toss-up.</p></div>
<div class="card"><h2>Models and days</h2><ul>
<li><b>All-model mean</b> averages the four ensembles. Pick one model to see its own ensemble mean.</li>
<li><b>Layout → All four models</b> puts the four ensemble means side by side (physics models on the top row, AI models below) on the same colors, day and field, so you can see
at a glance where they agree on a ridge or trough and where one model stands apart. Hover any panel to mark that point on all four.
(The "above-normal heights" field is already all models pooled, so it stays one map.)</li>
<li>Two are AI models (ECMWF AIFS ENS, NOAA AIGEFS) and two are physics-based (ECMWF IFS ENS, NOAA GEFS). When the AI and physics
models agree, confidence is higher; when they disagree, both outcomes are worth keeping in mind.</li>
<li><b>Day 1 is today</b>, as a California calendar day; each map is that day's average. Use Play to animate the 15 days.</li>
<li><b>Lightning potential</b> (California only, in the field menu): the chance of at least one cloud-to-ground strike, or of 10
or more, in each 0.5° cell (about 35 miles) on each fire day, from the two models that forecast instability (ECMWF IFS ENS, NOAA GEFS).
<a href="lightning.html">How it's made, how well it does, and its assumptions</a>.</li>{'''
<li><b>Lightning formula</b> (California only, 14 days, low skill past day 7): the same chances from a second method, the shape of ECMWF's own
lightning scheme (Lopez 2016) run on IFS ENS convective rain, cloud ice and water, CAPE and cloud base, calibrated to the strike record.
<a href="lightning_formula.html">How it's made and how well it does</a>.</li>''' if cfg.get("lightning_lopez", {}).get("enabled") else ""}
<li><b>Run</b> switches between today's run and the previous days' 00 UTC runs. The map stays on the same valid date (or 12-hourly
time), so stepping back through the runs shows how the forecast for that day has changed: a trough that keeps arriving later, or a
ridge that keeps getting stronger, is a trend worth noticing. An earlier run's Day 1 is the day it was made.</li>
<li><b>Time step → 12-hourly</b> shows each day's two model times instead of the daily average: 5 a.m. and 5 p.m. PDT (12 and 00 UTC).
Troughs and ridges move by hundreds of miles in a day, so the half-day maps show when a feature passes, not just where it averages out.</li>
<li>Hover or tap the map to read every model's value at that point.</li></ul>
<p class="note">Ensemble means smooth out details, and the further out the day, the smoother they get. A weak anomaly at day 12 can still
hide a strong event in some members; the calibrated outlook and the regime section show the spread.</p></div>""")
    out["weeks"] = ("Reading week 1 and week 2", f"""
<h1>Reading the week 1 and week 2 patterns</h1><p class="sub">The "Week 1 and week 2 patterns" section of the main page.</p>
<div class="card"><h2>The windows</h2>
<p>Week 1 is days {cfg['scenarios']['windows_days'][0][0]}–{cfg['scenarios']['windows_days'][0][1]} and week 2 is days
{cfg['scenarios']['windows_days'][1][0]}–{cfg['scenarios']['windows_days'][1][1]}, counting today as day 1. Each member's 500 hPa
(~18,000 ft) height is averaged over the whole window. Averaging over a week filters out the day-to-day timing differences that
make single days past about day 7 so uncertain, and leaves the broad pattern: where the ridges and troughs tend to sit.</p></div>
<div class="card"><h2>The probability map</h2>
{swatch_legend(["#08306b", "#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#d6604d", "#67001f"], ['0', '10', '20', '33–67', '80', '90', '100%'])}
<p>The share of all members whose week-average 500 hPa height is above normal for those dates, with the four models weighted equally.</p>
<ul><li><b>Dark red:</b> nearly every member, in every model, has a ridge there. Expect warmer and drier than normal, with sinking air
and poor overnight humidity recovery.</li>
<li><b>Dark blue:</b> nearly every member has a trough. Cooler, more onshore flow, better chances of precipitation.</li>
<li><b>White (33–67%):</b> the members are split. The week could go either way there.</li>
<li>Heights have trended upward with the warming climate, so above-normal is somewhat more common than 50% even in an
ordinary pattern. Look for the deep colors.</li></ul></div>
<div class="card"><h2>The scenario cards</h2>
<p>Every member's week-average pattern from all four models is pooled and sorted into the {cfg['scenarios']['k']} most distinct groups
(k-means clustering on the main pattern shapes). Each card is one group:</p>
<ul><li><b>The map</b> is the group's average 500 hPa anomaly (red = ridge, blue = trough) with its average heights as contours (dam).</li>
<li><b>"% of all members"</b> is the group's share with each model weighted equally. The cards are ordered from most to least likely.</li>
<li><b>The bars</b> show how much of each model's own ensemble falls in that group. Solid bars are AI models, hatched bars are
physics models. When one scenario holds most of every model's members, the week is fairly confident. When different models
favor different scenarios, the forecast hinges on which model is right. That's worth watching as the week gets closer.</li></ul>
<p class="note">These are this run's scenarios, so the letters mean different patterns each day. The 500 hPa regimes further down the
page are fixed patterns from history, with their fire records.</p></div>""")
    lg = common.Paths(cfg).cal / "lightning_grid.json"
    V = json.loads(lg.read_text())["targets"] if lg.exists() else {}
    v1, v10 = (V.get("p1") or {}).get("verify", {}), (V.get("p10") or {}).get("verify", {})
    rel = "".join(f"<tr><td>{esc(r['bin'])}</td><td class='n'>{r['forecast']:.0%}</td><td class='n'>{r['observed']:.0%}</td>"
                  f"<td class='n'>{r['n']:,}</td></tr>" for r in v1.get("reliability", []))
    lc = common.Paths(cfg).cal / "lightning_cases.json"
    LC = json.loads(lc.read_text()) if lc.exists() else {"sieges": [], "false_alarms": []}

    def case_row(c):
        g = c.get("ingredients") or {}
        f = lambda k, fmt: format(g[k], fmt) if k in g else "–"
        d = pd.Timestamp(c["date"])
        return (f"<tr><td>{d:%b} {d.day}, {d.year}</td><td class='n'>{c['observed_cells']}</td><td class='n'>{c['expected_cells']:.0f}</td>"
                f"<td class='n'>{c['high_p_cells']}</td><td class='n'>{f('cape', '.0f')}</td><td class='n'>{f('t700', '.1f')}</td>"
                f"<td class='n'>{f('td700', '.1f')}</td><td class='n'>{f('td2m', '.1f')}</td><td class='n'>{f('lapse', '.1f')}</td></tr>")

    def case_table(rows):
        return ("<div class='scroll'><table><tr><th>Fire day</th><th class='n'>Cells with strikes</th><th class='n'>Model expected</th>"
                "<th class='n'>Cells ≥30%</th><th class='n'>CAPE</th><th class='n'>700 hPa temp</th><th class='n'>700 hPa dewpt</th>"
                "<th class='n'>2 m dewpt</th><th class='n'>2 m − 700 hPa</th></tr>" + "".join(case_row(c) for c in rows) + "</table></div>"
                "<p class='note'>Ingredients are medians over the day's cells at ≥30% (J/kg, °C). \"Model expected\" is the sum of the cell "
                "chances, the number of cells the equation expected to see a strike.</p>")

    def case_imgs(rows):
        return "<div class='cols'>" + "".join(
            f"<div><img class='map' src='{c['png']}' alt='Lightning potential and observed strikes, {c['date']}'></div>" for c in rows) + "</div>"
    out["lightning"] = ("Lightning potential: how it's made", f"""
<h1>Lightning potential: how it's made and its assumptions</h1>
<p class="sub">The "Lightning potential" fields in the pattern map's field menu.</p>
<div class="card"><h2>What it shows</h2>
<p>For each 0.5° cell (about 35 miles across) over California and each fire day (6 a.m. to 6 a.m. Pacific), the chance that at least
one cloud-to-ground lightning strike, or at least ten, lands in the cell. It's the average over the members of the two ensembles that
forecast atmospheric instability, ECMWF IFS ENS and NOAA GEFS. The AI models don't output CAPE, so picking one of them, or the
all-model mean, shows that two-model average. The colors are smoothed between cell centres for display; the forecast itself is per cell.</p></div>
<div class="card"><h2>How it's made</h2>
<p>It follows the method of Nikolov, Bothwell and Snook (2024), <i>Probabilistic Forecasting of Lightning Strikes over the Continental
USA and Alaska</i> (Fire 7:111), which fits logistic equations to decades of observed strikes and reanalysis weather, with each
area's lightning climatology built in. Their coefficients aren't published, so this is a re-fit on this project's data, not their model:</p>
<ul><li><b>Strikes</b>: Vaisala NLDN cloud-to-ground strikes, 2001–2025, counted per cell per fire day.</li>
<li><b>Weather</b>: the ERA5 reanalysis at the same grid points and the models' two daily times (00 and 12 UTC, 5 p.m. and 5 a.m. PDT).</li>
<li><b>The equation's inputs</b>, in order of weight:
<ol><li>the cell's lightning frequency for that month (2001–2020), the baseline;</li>
<li>instability, CAPE (the higher of the day's two values);</li>
<li>700 hPa (about 10,000 ft) dewpoint, moisture aloft for storms to build on;</li>
<li>2 m dewpoint, low-level moisture;</li>
<li>700 hPa temperature, and the 2 m minus 700 hPa temperature difference (how fast it cools with height) at 5 p.m.</li></ol></li>
<li><b>Fitting</b>: 2001–2020. <b>Testing</b>: 2021–2025, years the equation never saw.</li></ul></div>
<div class="card"><h2>How well it does (2021–2025)</h2>
<div class="stat"><b>{v1.get('auc', float('nan')):.2f}</b><span>AUC, any strike (climatology alone {v1.get('auc_climatology', float('nan')):.2f})</span></div>
<div class="stat"><b>{v10.get('auc', float('nan')):.2f}</b><span>AUC, 10+ strikes (climatology alone {v10.get('auc_climatology', float('nan')):.2f})</span></div>
<div class="stat"><b>{v1.get('bss', 0):+.0%}</b><span>Brier skill over climatology, any strike</span></div>
<div class="stat"><b>{v1.get('base_rate', 0):.1%}</b><span>of cell-days had a strike</span></div>
<p>AUC 0.5 is no skill and 1 is perfect. Reliability, any strike (does a forecast of X% come true X% of the time?):</p>
<div class="scroll"><table><tr><th>Forecast</th><th class="n">Mean forecast</th><th class="n">Observed</th><th class="n">Cell-days</th></tr>{rel}</table></div>
<p class="note">It runs a little low between 10% and 30%, so read 15–20% as "about one in five". Most cell-days sit near zero: lightning is rare
in any one cell on any one day.</p></div>
<div class="card"><h2>Assumptions and limits</h2><ul>
<li><b>Trained on the reanalysis, run on forecasts.</b> The equation learned from ERA5; the forecasts feed it model fields. Each model's
CAPE is defined a little differently (IFS: most-unstable parcel; GEFS: a near-surface layer) and runs higher or lower than ERA5's, which
shifts the probabilities. The project's run archive keeps each model's values so this can be measured and corrected.</li>
<li><b>Coarse instability.</b> A 0.5° grid point stands for a 35-mile cell; the storms that make lightning are a few miles across.
The model can tell which areas and days are favorable, not which canyon gets hit.</li>
<li><b>California only.</b> The strike record covers California's Predictive Service Areas, so cells elsewhere have no equation.</li>
<li><b>Fire days, not hours.</b> Each value is for a whole 6 a.m.–6 a.m. day. Night-time elevated storms (like August 2020's)
count toward the day they start in.</li>
<li><b>Rare events run low.</b> The climatology term pulls unusual events (early-season lightning) toward the usual, so a siege can
show as 10–20% rather than 50%. See the June 2008 example below.</li>
<li><b>No lifting term.</b> CAPE and moisture say storms <i>can</i> grow; nothing in the equation says what sets them off. See the
false alarms and watch-outs below.</li>
<li><b>Dry or wet isn't in it.</b> It forecasts strikes, not whether they come with rain. The "Thunderstorms and new lightning fires" rows
in the calibrated outlook combine the storm setup with fuels and the record of lightning-caused fires.</li>
<li><b>Licensed data.</b> The strike history is Vaisala NLDN, used under license; only derived probabilities and maps are shown.</li></ul></div>
<div class="card" id="ltg-hindcast"><h2>Past sieges, as the model sees them</h2>
<p>The equation run on the reanalysis for five of the biggest lightning days of 2001–2025. Dots are the cells that had strikes, bigger for
more. 2025 is a year the equation never saw. On every one of these days more cells were hit than the equation expected: on the big
outbreaks it gets the area right and undersells the size. June 2008 is the worst case. It was an early-season siege where lightning
almost never happens, so the equation's climatology term held it to 10–20% and not one cell reached 30%. Read the layer as
"where storms can happen" more than "which cells will be hit", and treat any organized 10–20% area in June as worth a look.</p>
{case_table(LC["sieges"])}
{case_imgs(LC["sieges"])}</div>
<div class="card" id="ltg-false"><h2>When it called for lightning and little came</h2>
<p>The four fire days of 2001–2025 with the biggest gap: among the top 2% of days by expected strike area, the ones where a quarter
or less of that area was hit. All four are July–September monsoon days, with the highest chances over the southeastern deserts (and on September 4, 2017, the
southern Sierra and San Joaquin Valley too).</p>
{case_table(LC["false_alarms"])}
{case_imgs(LC["false_alarms"])}
<h2 style="margin-top:14px">Watch-outs: ingredients that can leave a high chance empty</h2>
<p>Across every cell-day of 2001–2025 the equation put at 30% or more, about half had no strike. That's roughly what a 30–50%
forecast should do. Which ones stayed empty wasn't random, though:</p>
<ul><li><b>Warm air near 10,000 ft (a cap).</b> In the warmest quarter of these cell-days (700 hPa above about 12.7 °C) 65% stayed dry,
against 36% in the coolest quarter. On three of the four false alarms the 700 hPa temperature was 12–13 °C, against 9–11 °C on
most of the sieges. Warm air aloft is a lid that CAPE may not get through without strong lift. The exception is elevated storms: on August
16, 2020 it was 16.6 °C at 700 hPa and storms fired anyway, on moisture and lift arriving above that layer.</li>
<li><b>A very hot surface under a deep dry layer.</b> The miss rate climbs from 34% to 73% from the smallest to the largest quarter of the
2 m − 700 hPa temperature difference (above about 25 °C). That's desert afternoons: high cloud bases, storms that form over the
mountains and drift or die before they reach the cell, or virga.</li>
<li><b>Moisture at the surface only.</b> Three of the false alarms had 2 m dewpoints of 15.4–17.5 °C, against 12–13 °C on most of
the sieges, and were no wetter than the sieges at 700 hPa. That's a Gulf of California moisture surge into the low deserts. It puts big CAPE on paper for a
surface parcel (1,100 J/kg on August 22, 2009), but the moisture is shallow and capped. Look for moisture through the mid-levels, not
just at the surface.</li>
<li><b>No trigger.</b> The equation has no lifting term: no shortwave trough, no upper-level divergence, no outflow from yesterday's
storms. CAPE is fuel, not ignition. Before trusting a high chance, look for something to set it off: an easterly wave or monsoon
impulse, a closed low or shortwave approaching, or difluent flow aloft.</li>
<li><b>Things the equation can't see.</b> Inhibition (CIN) isn't an input, and neither is mid-level moisture above 700 hPa or the
steering flow. Heavy smoke can also shade the surface enough to keep afternoon storms from firing.</li>
<li><b>Early season.</b> High chances in March and April came true least often (66% and 83% of cell-days at ≥30% stayed dry), and
June the most of the main months (26% dry).</li>
<li><b>CAPE alone is a poor guide.</b> Among cell-days at ≥30%, the miss rate is about the same (48–55%) whatever the CAPE. The forecast
already counts CAPE; what decides the outcome is whether the air releases it.</li></ul></div>
<div class="card"><h2>Reference</h2><p>Nikolov, N., P. Bothwell and J. Snook, 2024: Probabilistic Forecasting of Lightning Strikes over
the Continental USA and Alaska: Model Development and Verification. <i>Fire</i>, 7, 111.
<a href="https://doi.org/10.3390/fire7040111">doi:10.3390/fire7040111</a> (open access, CC BY 4.0).</p></div>""")
    out["outlook"] = ("Reading the calibrated outlook", f"""
<h1>Reading the calibrated outlook</h1><p class="sub">The grid of colored boxes on the main page.</p>
<div class="card"><h2>What each row is</h2>
<p>Each row is one weather measure (a surface pressure gradient, an upper-air wind or temperature, or a region-wide temperature
anomaly) with a cutoff learned from history: on days past that cutoff, California fire days with at least 5,000 acres of growth were
about twice as likely as usual for the time of year (the "elevated" cutoff). Some rows also have a "high" cutoff, about four times as
likely. Click a row's <b>About</b> link for its stations, history and the fires behind it.</p>
<p>Rows are grouped by region and season: <b>offshore season</b> (Northern California Sep–Nov, Southern California Oct–Jan) or the
general <b>fire season</b>. Only rows with real skill (AUC at least 0.65) are shown, so the rows change with the season.</p></div>
<div class="card"><h2>What the colors mean</h2>
{swatch_legend(SEQ, ['<5%', '5–20%', '20–40%', '40–60%', '60–80%', '≥80%'])}
<p>Each box is the share of all ensemble members (the four models weighted equally) past the cutoff on that fire day. Hover a box for
each model's share.</p>
<ul><li><b>A run of dark boxes several days out</b>, in several rows at once, is the strongest signal: many members, several measures.</li>
<li><b>Light boxes</b> don't mean "no fire weather". The measure isn't expected past its cutoff, and other drivers (lightning, fuels)
can still matter.</li>
<li>A fire day runs 6 AM to 6 AM Pacific, matching how the satellite growth days are counted.</li></ul></div>
<div class="card"><h2>The two views: members and weather + fuels odds</h2>
<p><b>% of members past the cutoff</b> is the weather alone. <b>Weather + fuels odds</b> adds the fuels: for each day it takes the
FEMS 7-day NFDRS forecast (ERC, averaged over the row's footprint PSAs) and asks whether fuels will be critically dry (ERC at or above
the 90th percentile of 2005–2025). Members past the cutoff then carry history's odds for "past cutoff on dry (or not dry) fuels", and
the rest carry the odds for "not past". The box shows the result as times the usual odds of a ≥5,000-acre growth day.</p>
<ul><li>History says weather and fuels multiply. Northern California warmth aloft on critically dry fuels has run about 8× the usual odds,
and the same warmth on fuels below the 90th percentile about 0.9×.</li>
<li>Faded boxes are past the 7-day FEMS forecast; the last forecast day's fuels are carried forward.</li>
<li>Thin history is pulled toward the weather-only odds, so a combination seen on only a few dozen days can't swing the result.</li></ul></div>
<div class="card"><h2>The measures</h2><ul>
<li><b>Pressure gradients</b> (SFO–WMC, LAX–TPH, LAX–DAG...): offshore wind.</li>
<li><b>Minimum relative humidity</b> at inland airports (Sacramento, Redding, Santa Rosa, Burbank, Ontario, Paso Robles): surface
dryness. Some of the strongest single measures in the record, e.g. Paso Robles and Ontario in the Santa Ana season.</li>
<li><b>Upper-air warmth and dryness</b> from weather balloons (850 and 700 hPa) and region-wide reanalysis temperatures.</li>
<li>Each is scored only against fire growth inside its own <b>footprint</b> PSAs, and each page also shows its record for <b>new large
fires</b>: days a fire that went on to burn 1,000+ acres started.</li></ul></div>
<div class="card"><h2>Thunderstorm rows (May–Oct)</h2>
{'''<p>The rows under <b>Thunderstorms and new lightning fires</b> work differently. Their measure is the number of lightning strikes the
<a href="lightning_formula.html">lightning formula</a> expects across a footprint (the shape of ECMWF's own lightning scheme, Lopez 2016,
on convective rain, cloud ice and water, CAPE and cloud base), and their cutoff was fit to observed lightning strikes: past it, days with at
least 100 strikes in the footprint were much more common than usual for the date. Lightning starts more large fires than it grows in the
week after, so the weather + fuels view for these rows shows the odds of a <b>new large (1,000+ acre) lightning fire</b> found that day or
the next.</p>
<ul><li>Only ECMWF IFS ENS carries the convective fields the formula needs, so only it feeds these rows, out to 14 fire days.
Days 8–14 are shown faded: thunderstorm forecasts that far out have little skill, and they haven't been verified.</li>
''' + ("" if _ltg_fit_full(cfg) else "<li>The cutoffs were fit on June–September; in May and October they're carried over untested.</li>") if cfg.get("lightning_lopez", {}).get("enabled") and (common.Paths(cfg).history / "lightning_lopez" / "footprint_daily.csv").exists() else '''<p>The rows under <b>Thunderstorms and new lightning fires</b> work differently. Their measure is instability (CAPE, the energy that feeds
thunderstorms) across a footprint, and their cutoff was fit to 25 years of lightning strikes: past it, days with at least 100 strikes in the
footprint were at least twice as common as usual for the date. Lightning starts more large fires than it grows in the week after, so the
weather + fuels view for these rows shows the odds of a <b>new large (1,000+ acre) lightning fire</b> found that day or the next.</p>
<ul><li>Only ECMWF IFS ENS and NOAA GEFS forecast CAPE, so only they feed these rows.</li>'''}
<li>History: in Northern California, a day with 500+ strikes on critically dry fuels has run several times to over twenty times the usual
odds of a new large lightning fire, depending on the area; on moister fuels much less.</li>
<li>The lightning history is licensed (Vaisala NLDN) and is only shown as derived counts, rates and maps.</li></ul></div>
<div class="card"><h2>The detail chart and the cutoff slider</h2>
<p>Below the grid, the chart shows each model's share by day for the selected row. Drag the <b>cutoff</b> to see how the forecast changes
with a stricter or looser threshold. The buttons jump back to the calibrated or "high" cutoff. A model that stays high at the "high"
cutoff is showing a strong event, not just a marginal one.</p></div>
<div class="card"><h2>Caveats</h2><ul>
<li>"× usual odds" describes history. It isn't a fire forecast: ignitions, fuels and suppression aren't in it.</li>
<li>The cutoffs come from station observations. Each model's own bias at a station isn't removed yet.</li>
<li>Each member counts once, and each model gets equal weight however many members it has.</li></ul></div>""")
    out["plumes"] = ("Reading the plumes", f"""
<h1>Reading the gradient and wind plumes</h1><p class="sub">The two stacked charts in "Offshore gradients and 850 hPa wind".</p>
<div class="card"><h2>Top chart: the plume</h2>
<p>Each colored line is one model's <b>median member</b>; the shaded band around it holds the middle 80% of its members (10th–90th
percentile). Solid lines are AI models, dashed are physics models. Times are Pacific, every 12 hours.</p>
<ul><li><b>Pressure gradients</b> (hPa) are coast-minus-inland sea-level pressure. Negative = higher pressure inland = offshore wind.
The more negative, the stronger the offshore push: SFO–WMC for Diablo/North wind events, LAX–DAG and LAX–TPH for Santa Anas,
RDD–MFR for north wind down the Sacramento Valley, SFO–SAC for the Bay Area–Valley gradient.</li>
<li><b>850 hPa offshore wind</b> (kt) is the wind about 5,000 ft up at a point, counted only when it blows from the northeast-to-east;
otherwise it's 0.</li>
<li>A narrow band means the members agree; a wide band means uncertainty. A band that dips well past the dashed line while the median
doesn't means a minority of members carry a strong event.</li></ul></div>
<div class="card"><h2>Bottom chart: the probability</h2>
<p>The share of each model's members past the <b>cutoff</b> at each time. Drag the cutoff slider to try your own threshold. The
starting values here are rule-of-thumb reference values; the calibrated outlook above uses cutoffs learned from history, measured as
the day's strongest (most offshore) value.</p></div>""")
    out["regimes"] = ("Reading the regimes", f"""
<h1>Reading the regimes</h1><p class="sub">The "500 hPa regimes" section of the main page.</p>
<div class="card"><h2>What a regime is</h2>
<p>Every fire-season day from 2001–2025 was sorted, by its 500 hPa (~18,000 ft) height anomaly pattern over the eastern Pacific and
western U.S., into one of {cfg['scenarios']['k'] + 2} groups that recur again and again (k-means clustering). Each thumbnail is the
average pattern of its group. Red = higher than normal heights (ridge), blue = lower (trough). Click a thumbnail's letter for its page:
how often it happens by month, and the fire growth that came with it.</p>
<p>Each regime is labeled with its <b>historical odds</b> of a ≥5,000-acre growth day in Northern and Southern California, compared with
the usual odds for the time of year (×1 = usual).</p></div>
<div class="card"><h2>The probability grid</h2>
{swatch_legend(SEQ, ['<5%', '5–20%', '20–40%', '40–60%', '60–80%', '≥80%'])}
<p>Each member's daily-mean pattern is matched to the closest regime. A box is the share of all members (models weighted equally) in that
regime on that day; hover it for each model. A column dominated by one dark box means the models agree on the pattern. Spread-out
columns mean several patterns are possible, which is common past about day 7.</p></div>
<div class="card"><h2>The odds line</h2>
<p>For each day, the regime probabilities times each regime's historical odds, added up: a pattern-only estimate of how the chance of a
big growth day compares with usual. Values near 1 are ordinary; 1.3–1.5 is a notably fire-friendly pattern by this measure. It's a
modest, broad-brush signal; the calibrated outlook rows capture the sharper wind and heat signals.</p></div>{pyro_guide_card()}""")
    formula = lightning_formula_guide(cfg)
    if formula:
        out["lightning_formula"] = formula
    return out


def _ltg_fit_full(cfg):
    """True once the lightning formula's history covers every month (2026-10-24 full-year refit)."""
    f = common.Paths(cfg).cal / "lightning_lopez.json"
    return f.exists() and len(json.loads(f.read_text()).get("months", [])) == 12


def lightning_formula_guide(cfg):
    """guide/lightning_formula.html: the lightning formula layer (ai_ensemble_lightning_lopez.py), its fit and its full
    verification from calibration/lightning_lopez.json, or None before it's built."""
    f = common.Paths(cfg).cal / "lightning_lopez.json"
    if not f.exists() or not cfg.get("lightning_lopez", {}).get("enabled", False):
        return None
    L = json.loads(f.read_text())
    p, V = L["params"], L["verify"]
    tr, te = L["train_years"], L["test_years"]
    full_year = len(L["months"]) == 12
    mon = "all months" if full_year else ", ".join(pd.Timestamp(2001, m, 1).strftime("%b") for m in L["months"])
    PS = L.get("per_season") or {}
    season_rows = "".join(
        f"<tr><td>{esc(n)}</td><td class='n'>{r['base_rate']:.1%}</td><td class='n'>{r['p1']['lopez']['auc']:.3f}</td>"
        f"<td class='n'>{r['p1']['grid']['auc']:.3f}</td><td class='n'>{r['p1']['lopez']['bss']:+.0%}</td>"
        f"<td class='n'>{r['p1']['grid']['bss']:+.0%}</td><td class='n'>{r['p10']['lopez']['auc']:.3f}</td>"
        f"<td class='n'>{r['p10']['grid']['auc']:.3f}</td></tr>" for n, r in PS.items())
    cool = next((r for n, r in PS.items() if n.startswith("cool")), None)

    def stat_rows(t):
        rows = ""
        for key, name in (("lopez", "Lightning formula (fitted)"), ("lopez_shape", "Plain Lopez shape (scale fitted)"),
                          ("grid", "Logistic layer (Nikolov et al. method)")):
            v = V[t][key]
            rows += (f"<tr><td>{name}</td><td class='n'>{v['auc']:.3f}</td><td class='n'>{v['bss']:+.0%}</td>"
                     f"<td class='n'>{v['brier']:.4f}</td></tr>")
        return rows + f"<tr><td>Cell-month climatology alone</td><td class='n'>{V[t]['climatology_auc']:.3f}</td><td class='n'>0%</td><td class='n'>{V[t]['lopez']['brier_climatology']:.4f}</td></tr>"

    def rel_rows(t):
        return "".join(f"<tr><td>{esc(r['bin'])}</td><td class='n'>{r['forecast']:.0%}</td><td class='n'>{r['observed']:.0%}</td>"
                       f"<td class='n'>{r['n']:,}</td></tr>" for r in V[t]["lopez"]["reliability"])
    year_rows = "".join(
        f"<tr{' class=\"dim\"' if y['fit'] else ''}><td>{y['year']}{' (fit)' if y['fit'] else ''}</td><td class='n'>{y['strikes']:,}</td>"
        f"<td class='n'>{y['expected']:,.0f}</td><td class='n'>{y['cell_days_hit']:,}</td><td class='n'>{y['auc1_lopez']:.3f}</td>"
        f"<td class='n'>{y['auc1_grid']:.3f}</td><td class='n'>{y['auc10_lopez']:.3f}</td><td class='n'>{y['auc10_grid']:.3f}</td></tr>"
        for y in L["per_year"])
    month_rows = "".join(f"<tr><td>{pd.Timestamp(2001, m['month'], 1):%B}</td><td class='n'>{m['base_rate']:.1%}</td>"
                         f"<td class='n'>{m['auc1_lopez']:.3f}</td><td class='n'>{m['auc1_grid']:.3f}</td></tr>" for m in L["per_month"])
    siege_rows = "".join(
        f"<tr><td>{pd.Timestamp(s['date']):%b %d, %Y}{' (fit year)' if s['fit_year'] else ''}</td><td class='n'>{s['observed_cells']}</td>"
        f"<td class='n'>{s['expected_cells_lopez']:.0f}</td><td class='n'>{s['expected_cells_grid']:.0f}</td><td class='n'>{s['strikes']:,}</td>"
        f"<td class='n'>{s['expected_strikes_lopez']:,.0f}</td></tr>" for s in L["sieges"])
    siege_imgs = "".join(
        f"<div class='cols'><div><img class='map' src='{s['png_lopez']}' alt='Lightning formula, {s['date']}'></div>"
        f"<div><img class='map' src='{s['png_grid']}' alt='Logistic layer, {s['date']}'></div></div>" for s in L["sieges"])
    D = L["daily"]
    edge = L["grid_search"]["edge"]
    PY = pd.DataFrame(L["per_year"])
    early, late = PY[PY.year <= PY.year.min() + 9], PY[PY.year >= PY.year.max() - 9]
    drift_early, drift_late = early.expected.sum() / early.strikes.sum(), late.expected.sum() / late.strikes.sum()
    early_span, late_span = f"{early.year.min()}–{early.year.max()}", f"{late.year.min()}–{late.year.max()}"
    return ("Lightning formula: how it's made and how well it does", f"""
<h1>Lightning formula (Lopez 2016, IFS ENS): how it's made and how well it does</h1>
<p class="sub">The "Lightning formula" fields in the pattern map's field menu. Experimental.</p>
<div class="card"><h2>What it shows</h2>
<p>For each 0.5° cell (about 35 miles across) over California and each fire day, the chance of at least one cloud-to-ground lightning
strike, or of ten or more: the same cells and targets as the "Lightning potential" fields, so the two can be compared directly. It comes
from ECMWF IFS ENS alone (50 members, 14 fire days), so it's the same map whichever model is picked. Past day 7 read it as a
rough signal at most: thunderstorm forecasts that far out have little skill, and the formula's skill was only tested on the
reanalysis, not on forecasts at any lead.</p></div>
<div class="card"><h2>Why it exists</h2>
<p>ECMWF's forecast model computes lightning itself, with a scheme by Philippe Lopez (2016) that ties flash rates to the storms its
convection scheme builds. Those lightning fields aren't in ECMWF's free open data, the public charts are images only, and the ERA5
reanalysis on the Copernicus Climate Data Store doesn't carry them either. Since 20 October 2026 the open data does carry the
ingredients the scheme rests on, so this layer rebuilds the scheme's shape from them.</p></div>
<div class="card"><h2>How it's made</h2>
<p>Lopez's scheme: flash density = α · Q<sub>R</sub> · √CAPE · min(z<sub>base</sub>, 1.8 km)², where Q<sub>R</sub> is a charging term
(graupel colliding with cloud water and snow between the 0 °C and −25 °C levels) and z<sub>base</sub> the convective cloud base. The open
data has no graupel or cloud-base fields, so stand-ins are used:</p>
<ul><li><b>Graupel</b> → the convective precipitation rate (mm/h).</li>
<li><b>Cloud water and snow in the charging zone</b> → column cloud ice + 0.7 × column cloud liquid (0.7 is Lopez's weight over land).</li>
<li><b>CAPE</b> → most-unstable CAPE.</li>
<li><b>Cloud base</b> → the lifting condensation level of a surface parcel (125 m per °C of 2 m dewpoint depression), capped at 1.8 km
as in Lopez.</li></ul>
<p>Lopez's exponents (1, 1, ½, 2) were tuned for his graupel and cloud-base fields, not these stand-ins, so the exponents were refit:
convective rain<sup>{p['a_cp']:g}</sup> · cloud ice/water<sup>{p['b_q']:g}</sup> · CAPE<sup>{p['c_cape']:g}</sup> · cloud
base<sup>{p['d_zb']:g}</sup>{f", convective rain above {p['cp_min']:g} mm/h" if p['cp_min'] else ""}.
{f"(The best values for {', '.join(edge)} sat at the edge of the range searched.) " if edge else ""}The plain Lopez shape is scored below too.</p>
<ul><li><b>Strikes</b>: Vaisala NLDN cloud-to-ground strikes per cell per fire day (6 a.m.–6 a.m. Pacific), the same counts as the
Lightning potential layer. The output is calibrated to these, so it means cloud-to-ground strikes, not ECMWF's total (in-cloud +
ground) flashes.</li>
<li><b>Weather</b>: ERA5 reanalysis versions of the same six fields, every 3 hours, {mon}.</li>
<li><b>Fitting</b>: {tr[0]}–{tr[1]}: the exponents by search (best ranking of strike days), a scale so expected strikes match observed,
and curves turning a cell's expected strikes into a chance of ≥1 and ≥10. <b>Testing</b>: {te[0]}–{te[1]}, years not used in fitting.</li>
<li><b>Forecasts</b>: each IFS ENS member's fields every 6 hours, the formula per 6-hour window, summed to fire days (windows
12–12 UTC, within an hour of the fire day), then the members' chances averaged.</li></ul></div>
<div class="card"><h2>How well it does ({te[0]}–{te[1]}, {mon})</h2>
<p>Both layers on the same cell-days, from the reanalysis (not from forecasts; see the limits below). AUC: 0.5 is no skill, 1 is perfect.
Brier skill: improvement over the cell's usual frequency for the month.</p>
<h3>Any strike in the cell</h3>
<div class="scroll"><table><tr><th></th><th class="n">AUC</th><th class="n">Brier skill</th><th class="n">Brier score</th></tr>{stat_rows('p1')}</table></div>
<h3>10 or more strikes in the cell</h3>
<div class="scroll"><table><tr><th></th><th class="n">AUC</th><th class="n">Brier skill</th><th class="n">Brier score</th></tr>{stat_rows('p10')}</table></div>
<p>Day by day across California ({D['test_days']} test days): the rank correlation between the number of cells that had strikes and the
sum of the cells' chances is <b>{D['rho_cells_lopez']:.2f}</b> for the formula and {D['rho_cells_grid']:.2f} for the logistic layer;
between total strikes and the formula's expected strikes, {D['rho_strikes_lopez']:.2f}.</p>
<img class="map" src="lightning_lopez_reliability.png" alt="Reliability diagrams for both lightning layers, test years">
<h3>Reliability, any strike (formula)</h3>
<div class="scroll"><table><tr><th>Forecast</th><th class="n">Mean forecast</th><th class="n">Observed</th><th class="n">Cell-days</th></tr>{rel_rows('p1')}</table></div></div>
<div class="card"><h2>Every year of the strike record</h2>
<img class="map" src="lightning_lopez_per_year.png" alt="AUC by year for both lightning layers">
<p class="note">Years marked "fit" were used to fit both layers, so their scores are flattering; the test years are the fair comparison.</p>
<div class="scroll"><table><tr><th>Year</th><th class="n">Strikes</th><th class="n">Formula expected</th><th class="n">Cell-days hit</th>
<th class="n">AUC ≥1, formula</th><th class="n">AUC ≥1, logistic</th><th class="n">AUC ≥10, formula</th><th class="n">AUC ≥10, logistic</th></tr>{year_rows}</table></div></div>
{f"""<div class="card"><h2>By season ({te[0]}–{te[1]})</h2>
<div class="scroll"><table><tr><th>Season</th><th class="n">Cell-days with a strike</th><th class="n">AUC ≥1, formula</th>
<th class="n">AUC ≥1, logistic</th><th class="n">Brier skill ≥1, formula</th><th class="n">Brier skill ≥1, logistic</th>
<th class="n">AUC ≥10, formula</th><th class="n">AUC ≥10, logistic</th></tr>{season_rows}</table></div></div>""" if season_rows else ""}
<div class="card"><h2>By month ({te[0]}–{te[1]})</h2>
<div class="scroll"><table><tr><th>Month</th><th class="n">Cell-days with a strike</th><th class="n">AUC, formula</th><th class="n">AUC, logistic</th></tr>{month_rows}</table></div></div>
<div class="card"><h2>Past sieges, both layers</h2>
<div class="scroll"><table><tr><th>Fire day</th><th class="n">Cells with strikes</th><th class="n">Formula expected</th><th class="n">Logistic expected</th>
<th class="n">Strikes</th><th class="n">Formula expected strikes</th></tr>{siege_rows}</table></div>
<p class="note">"Expected" cells = the sum of the cell chances. Left: the formula; right: the logistic layer. Dots: cells that had strikes, bigger for more.</p>
{siege_imgs}</div>
<div class="card"><h2>Assumptions and limits</h2><ul>
{f"""<li><b>Cool-season storms.</b> One set of exponents and curves serves the whole year. October–May storms are colder-based
and lower-topped, and lightning is rarer then ({cool['base_rate']:.1%} of cool-season cell-days had a strike); the formula still
ranks them with an AUC of {cool['p1']['lopez']['auc']:.2f} (logistic layer {cool['p1']['grid']['auc']:.2f}), but
there are far fewer of them to learn from. See the season table above.</li>""" if full_year and cool else
f"""<li><b>Summer only so far.</b> It was fit and tested on {mon}. In other months the curves are carried over untested; read
cool-season values with care (spring and fall storms are colder-based and lower-topped).</li>"""}
<li><b>Trained on the reanalysis, run on forecasts.</b> ERA5 is an older, coarser version of the same model. Today's IFS makes more and
sharper convective rain, which can push the chances higher than the history behind them. Live forecasts can only be scored from
20 October 2026 on, when the ingredients entered the open data.</li>
<li><b>Strike totals drift.</b> Expected strikes came to {drift_early:.1f}× the observed total in {early_span} and {drift_late:.1f}× in
{late_span}. The NLDN network was upgraded around 2013, which changed how weak strokes are classed, and the reanalysis's convection
changed too. Rankings and chances held up across the record (the AUC by year above), but treat the formula's strike <i>counts</i> as
relative, not literal.</li>
<li><b>Stand-ins for the physics.</b> Convective rain stands in for graupel, and column totals for the charging layer; a deep column of
cloud water below the freezing level counts the same as one up in the charging zone.</li>
<li><b>Daily, not hourly.</b> The reanalysis is sampled every 3 hours and the forecasts every 6, so short-lived storms between samples
are averaged in, not caught as peaks.</li>
<li><b>Land only.</b> Fit on California's land cells; nothing is shown offshore.</li>
<li><b>Not ECMWF's product.</b> A rebuild of the scheme's shape, calibrated here. ECMWF's own lightning forecast is on its
<a href="https://charts.ecmwf.int/products/medium-lightning">public charts</a>.</li>
<li><b>Licensed data.</b> The strike history is Vaisala NLDN, used under license; only derived probabilities and maps are shown.</li></ul></div>
<div class="card"><h2>Credits and references</h2><ul>
<li>Lopez, P., 2016: A lightning parameterization for the ECMWF Integrated Forecasting System. <i>Monthly Weather Review</i>, 144,
3057–3075. <a href="https://doi.org/10.1175/MWR-D-16-0026.1">doi:10.1175/MWR-D-16-0026.1</a>.</li>
<li>ECMWF IFS ENS open data © European Centre for Medium-Range Weather Forecasts, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
Modified: cut to California, combined by the formula above and calibrated. ECMWF doesn't accept any liability for errors or omissions here.</li>
<li>ERA5: contains modified Copernicus Climate Change Service information. Neither the European Commission nor ECMWF is responsible for any use of it here.</li>
<li>Vaisala National Lightning Detection Network (NLDN), 2000–2025, used under license; derived counts and probabilities only.</li>
<li>The comparison layer follows Nikolov, N., P. Bothwell and J. Snook, 2024, <i>Fire</i> 7, 111,
<a href="https://doi.org/10.3390/fire7040111">doi:10.3390/fire7040111</a>.</li></ul></div>""")


def pyro_rule_card():
    """Pyroconvection page: regime + 700 hPa moisture rules (regime_moisture.py), scored fire by fire."""
    rm = PYRO.get("regime_moisture") or {}
    rules = rm.get("rules") or {}
    if not rules:
        return ""
    def label(name, cut):
        name = name.replace("ERA5 ", "").replace("q700", "700 hPa moisture")
        return name.replace(">= cut", f"≥ {cut:.1f} g/kg") if cut is not None else name

    rows = "".join(f"<tr><td>{esc(label(k, v['pooled_q_cut']))}</td>"
                   f"<td class='n'>{v['pooled_pyroCb']}</td><td class='n'>{v['pooled_pyroCu']}</td><td class='n'>{v['pooled_none']}</td>"
                   f"<td class='n'>{v['lofo_hit']}</td><td class='n'>{v['lofo_fa']}</td><td class='n'>{v['lofo_bal_acc']:.2f}</td></tr>"
                   for k, v in rules.items())
    return f"""<div class="card"><h2>Regime plus 700 hPa moisture: a pattern-level signal</h2>
<p>The 500 hPa regime and the 700 hPa moisture at the fire (ERA5, the moister of 11 a.m. and 5 p.m.) are both fields the models on this
page forecast days ahead. Together they cut the regime's false alarms by about two thirds; they're still weaker than the same-day
fire-power test, so read them as "the pattern allows pyroCb", not a forecast of one.</p>
<div class="scroll"><table><tr><th>Rule</th><th class="n">pyroCb flagged</th><th class="n">pyroCu flagged</th><th class="n">no-plume flagged</th>
<th class="n">fire-by-fire caught</th><th class="n">fire-by-fire flagged</th><th class="n">balanced acc.</th></tr>{rows}</table></div>
<p class="note">Left columns: one cut fitted on all the fires. Right columns: cut fitted with one fire held out and scored on that fire, for each fire in turn, summed.
Days with ordinary storms over the fire are left out.</p></div>"""


def pyro_guide_card():
    """'Reading the regimes' card on pyroconvection: which regimes the study's pyroCb days fell in."""
    if not PYRO:
        return ""
    reg = PYRO["regime_by_class"]["pyroCb"]
    rows = ", ".join(f"<a href='../regimes/{L}.html'>{L}</a> {reg[L]['days']}" for L in "ABCDEF")
    rule = _pyro_rule()
    rule_txt = (f" Together with 700 hPa moisture at the fire of ~{rule['pooled_q_cut']:.1f} g/kg or more (a field the models on this page "
                f"forecast), A/C/D marked all {rule['pooled_pyroCb'].split('/')[1]} pyroCb days and {rule['pooled_none']} of the days without "
                f"deep plume cloud; fire by fire, {rule['lofo_hit']} caught and {rule['lofo_fa']} flagged." if rule else "")
    return (f"""
<div class="card"><h2>Regimes and pyroconvection</h2>
<p>In {_nfires()} studied California fires ({PYRO['summary']['days']} fire days, {PYRO['summary']['n_pyrocb']} pyrocumulonimbus days), pyroCb days by
regime: {rows}. Every pyroCb day was in A, C or D, and none came under B, E or F. The average map on the pyroCb days has an upper ridge
over British Columbia with a trough offshore, which turns mid-level flow southerly and brings moisture north under the heat.{rule_txt} Regime C is also the offshore-wind
regime, and its dry days made the biggest runs with no deep plume cloud at all, so the regime alone is a broad filter; the moisture is what
separates the two. Each regime's page lists its pyroCb days; details on <a href="pyroconvection.html">Pyroconvection under the scenarios</a>.</p></div>""")


NFDRS, BURN_DIRS, PSA_NAMES = {}, {}, {}


# ----------------------------------------------------------------------------- documented patterns (2026-10-13)

def schroeder_card(M, lit):
    if not M:
        return ""
    ex = [c for c in (lit or []) if "Schroeder" in c["source"]]
    ok = sum(1 for c in ex if c["match"])
    rows = "".join(f"<tr><td>{esc(m['type'])}</td><td><b>{esc(m['best'])}</b></td><td class='n'>{m['corr'][m['best']]:+.2f}</td>"
                   f"<td class='note'>{esc(', '.join(f'{k} {v:+.2f}' for k, v in sorted(m['corr'].items())))}</td></tr>" for m in M)
    return f"""<div class="card"><h2>Schroeder et al. (1964), tested on 1951–60</h2>
<p><b>As published</b> (paraphrased): Northern and Central California's critical fire weather in 1951–60 fell into four types,
the Subtropical High Aloft (the worst and longest), a meridional ridge with southwest flow ahead of a trough near 130° W, the Pacific
High behind a dry cold front (post-frontal north wind) and the Great Basin High (Mono winds); Southern California's into the
Subtropical High Aloft, southwest flow, post-frontal and Santa Ana types. Offshore LAX–Tonopah gradients of about 7 mb start
northeast winds and 11–12 mb make them strong, the same levels the NWS later published.</p>
<p><b>Dated examples</b>: {ok} of the {len(ex)} example days (1953–1956, classified from the R1 reanalysis) fall in the regimes this
crosswalk ties to their type: the Santa Ana and both post-frontal cases in A, the 1955 heat wave in D. The Great Basin High day lands
in E, a trough off the coast, which is where Schroeder puts the other end of its gradient.</p>
<p><b>Seasonal fingerprints</b>: each type's 1951–60 monthly day counts against each regime's 1951–60 monthly frequency (Spearman
correlation over 12 months).</p>
<div class="scroll"><table><tr><th>Type</th><th>Best regime</th><th class="n">ρ</th><th>All regimes</th></tr>{rows}</table></div>
<p class="note">Only the cool-season Santa Ana has a seasonal shape distinctive enough to point to one regime (C). The summer types all
peak in the same months, so they line up with whichever regimes are common in summer (A, B); this test can't separate them.</p></div>"""


def nws_card(N):
    if not N:
        return ""
    rows = "".join(f"<tr><td>{esc(o['office'])}</td><td>{esc(o['pattern'])}</td><td>{esc(o['rule'])}</td>"
                   f"<td class='n'>{o['days']:,}</td><td class='n'>{'–' if o['lift'] is None else '×%.1f' % o['lift']}</td></tr>" for o in N)
    return f"""<div class="card"><h2>NWS objective techniques, tested</h2>
<p class="sub">The fire-weather offices' published rules of thumb from "Critical Fire Weather Patterns of the United States", reproduced
with this project's observations and scored as × usual odds (for the date) of a 1,000+ acre growth day in the pattern's area, 2001–2025.</p>
<div class="scroll"><table><tr><th>Office</th><th>Pattern</th><th>Rule</th><th class="n">Days</th><th class="n">× usual</th></tr>{rows}</table></div>
<ul><li><b>Riverside's Santa Ana thresholds hold up and go further than this project's cutoffs</b>: LAX–TPH beyond 10 hPa runs about
×5, beyond 12 hPa about ×8. They're on the Santa Ana rows' cutoff slider as presets.</li>
<li><b>Redding's north-wind technique</b> (a northerly Medford-to-Sacramento gradient) separates Sacramento Valley growth better than
the RDD–MFR gradient, so Sacramento minus Medford (SAC–MFR) is now a calibrated measure.</li>
<li>Fresno's Sundowner rule (Santa Maria minus Santa Barbara ≥ 4 mb, with northerly low-level wind at Vandenberg) marks the wind
pattern but doesn't separate large-growth days near Santa Barbara in this record.</li></ul></div>"""


def hawaii_card(H):
    if not H:
        return ""
    rows = "".join(f"<tr><td>{esc(p_['name'])}</td><td class='note'>{esc(p_['rule'])}</td><td class='n'>{p_['days']:,}</td>"
                   f"<td class='n'>{p_['large_fire_days']:,}</td><td class='n'>{'–' if p_['lift'] is None else '×%.1f' % p_['lift']}</td></tr>"
                   for p_ in H["patterns"])
    a = H.get("aug_2023") or {}
    return f"""<div class="card"><h2>Hawaii (NWCG PMS 437)</h2>
<p><b>As documented</b> (paraphrased): dry season May–September, but critical growth any time live fuels are drought-stressed or cured;
critical patterns center on strong trade-wind periods with high pressure to the north and/or a significant tropical cyclone passing
south of the main islands; southwest frontal winds have also driven downslope fire on usually windward sides.</p>
<p><b>Against Hawaii's {H['large_fire_days']:,} large-fire days</b> (≥ 300 acres from satellite growth or the fire records, 2001–2025):</p>
<div class="scroll"><table><tr><th>Pattern</th><th>How we found its days</th><th class="n">Days</th><th class="n">Large-fire days</th>
<th class="n">× usual for the date</th></tr>{rows}</table></div>
<p>August 8, 2023 (Lahaina, Upcountry, Pulehu Road): strong trades {"yes" if a.get("trades") else "no"}, tropical cyclone to the south
{"yes (" + esc(a.get("storm") or "") + ")" if a.get("tc_south") else "no"}. The southwest-frontal pattern doesn't show up in this record.
Tropical cyclone positions: NOAA IBTrACS v04.</p></div>"""


def ridge_card(R):
    if not R:
        return ""
    f = lambda g, k: R[g][k]["lift"]
    return f"""<div class="card"><h2>Breakdown of the upper ridge (NWCG PMS 437)</h2>
<p><b>As documented</b> (paraphrased): three stages, warmer-drier-breezy and unstable, then stronger winds while still warm, dry and
unstable, then the cold front; in the West most significant growth comes in the second stage.</p>
<p><b>In this record</b>, at the daily resolution of the 500 hPa regimes (the last ridge day, C or D, before a trough regime, B or E,
taken as the second stage): Northern California ×{f('ONCC', 'stage2_last_ridge_day'):.2f} on that day, ×{f('ONCC', 'stage3_first_trough_day'):.2f}
on the first trough day, and ×{f('ONCC', 'ridge_persists'):.2f} on ridge days that continue; Southern California ×{f('OSCC', 'stage2_last_ridge_day'):.2f},
×{f('OSCC', 'stage3_first_trough_day'):.2f} and ×{f('OSCC', 'ridge_persists'):.2f}. The ordering of stages 2 and 3 matches in the north, but a daily regime
label is too coarse to separate the breakdown from the ridge itself; the offshore and thunderstorm rows carry that signal better.</p></div>"""


def patterns_page(paths, gdir):
    """guide/patterns.html: DRI CEFA's and the Synthesis's critical fire-weather patterns against this
    project's regimes and measures (history/calibration/cefa_patterns.json, ai_ensemble_cefa_patterns.py)."""
    import shutil
    f = paths.cal / "cefa_patterns.json"
    if not f.exists():
        return None
    J = json.loads(f.read_text())
    cards = []
    for pt in J["patterns"]:
        if pt.get("png") and (paths.cal / pt["png"]).exists():
            shutil.copy2(paths.cal / pt["png"], gdir / pt["png"])
        regs = sorted(pt["regimes"], key=lambda r: -(r["enrichment"] or 0))
        reg_txt = ", ".join(f'<a href="../regimes/{r["regime"]}.html">{r["regime"]}</a> ×{r["enrichment"]:.1f}'
                            for r in regs[:3] if r["enrichment"])
        def case_li(c):
            bits = [f"regime {esc(c['regime'] or '–')}"]
            if pt["gacc"] == "ONCC" and c["SFO-WMC"] is not None:
                bits.append(f"SFO–WMC {c['SFO-WMC']:.1f} hPa")
            if pt["gacc"] == "OSCC" and c["LAX-TPH"] is not None:
                bits.append(f"LAX–TPH {c['LAX-TPH']:.1f} hPa")
            if pt["id"] == "sc_sundowner" and c["SMX-SBA"] is not None:
                bits.append(f"Santa Maria–Santa Barbara {c['SMX-SBA']:.1f} hPa")
            if c["strikes_nc"]:
                bits.append(f"{c['strikes_nc']:,} Northern California strikes")
            return (f"<li>{esc(c['date'])} {esc(c['label'])}: {', '.join(bits)} "
                    f"({'counted' if c['event'] else 'not counted'} by the rule)</li>")
        cases = "".join(case_li(c) for c in pt.get("case_check", []))
        gl, sl = pt.get("growth_lift"), pt.get("start_lift")
        cards.append(f"""<div class="card"><h2>{esc(REGION.get(pt['gacc'], pt['gacc']))}: {esc(pt['name'])}</h2>
<div class="cols"><div>
<p><b>As documented</b> (paraphrased): {esc(pt['cefa'])}</p>
<p><b>How we found its days</b>: {esc(pt['rule'])}. {pt['days']:,} days, 2001–2025 (about {pt['per_year']} a year).</p>
<p><b>Our 500 hPa regimes on those days</b>: {reg_txt or '–'} (× how much more often than on the season's other days).</p>
{f"<p><b>Average map on those days</b>: {esc(pt['composite_text'])}</p>" if pt.get('composite_text') else ''}
<p><b>Fire</b>: days of 1,000+ acres of growth in the pattern's area ran <b>×{gl:.1f}</b> the usual odds for the date; new
large fires ×{(sl or 0):.1f}.</p>
{"<h3>Case dates</h3><ul>" + cases + "</ul>" if cases else ""}
</div><div>{f'<img class="map" src="{pt["png"]}" alt="Mean 500 hPa height anomaly on the pattern days">' if pt.get('png') else ''}</div></div></div>""")
    lead = J.get("sundowner_lead") or {}
    lit_rows = "".join(
        f"<tr><td>{esc(c['date'])}</td><td>{esc(c['case'])}</td><td>{esc(c['pattern'])}</td><td>{esc(c['source'])}</td>"
        f"<td>{'<a href=\"../regimes/' + esc(c['regime']) + '.html\">' + esc(c['regime']) + '</a>' if c['regime'] else '–'}</td>"
        f"<td>{esc('/'.join(c['expected']))}</td><td>{'✓' if c['match'] else '–' if c['match'] is None else '✗'}</td>"
        f"<td class='note'>{esc(c['from'])}</td></tr>" for c in J.get("literature_cases", []))
    pre = [c for c in J.get("literature_cases", []) if c["from"] != "2001-present record" and c["match"] is not None]
    pre_txt = (f"Of the {len(pre)} pre-2001 cases (regimes classified, not fit, on those days), {sum(c['match'] for c in pre)} land in "
               "the regimes the 2001–2025 crosswalk ties to their pattern." if pre else "")
    body = f"""
<h1>Documented critical fire-weather patterns</h1>
<p class="sub">How the patterns described by fire-weather research line up with what this project found from 2001–2025 data.</p>
<div class="card"><h2>The references</h2><ul>
<li>{esc(J['source']['name'])}, Program for Climate, Ecosystem and Fire Applications, Desert Research Institute:
<a href="{esc(J['source']['url'])}">{esc(J['source']['url'])}</a> (accessed {esc(J['source']['accessed'])}). Patterns for Northern and
Southern California; there's no Hawaii page.</li>
<li>{esc(J['source2']['name'])} <a href="{esc(J['source2']['url'])}">{esc(J['source2']['url'])}</a></li>
{f"<li>{esc(J['source3']['name'])}: <a href=\"{esc(J['source3']['url'])}\">{esc(J['source3']['url'])}</a> (accessed {esc(J['source3']['accessed'])}). It summarizes both of the above and adds Hawaii and the stages of an upper-ridge breakdown.</li>" if J.get('source3') else ''}
{f"<li>{esc(J['source4']['name'])}: <a href=\"{esc(J['source4']['url'])}\">{esc(J['source4']['url'])}</a>. One page per NWS fire-weather office with each critical pattern's surface and upper-air signature and its objective forecast techniques; California entries from Redding, Sacramento, Fresno, Riverside and Reno.</li>" if J.get('source4') else ''}
{f"<li>{esc(J['source5']['name'])} <a href=\"{esc(J['source5']['url'])}\">{esc(J['source5']['url'])}</a>. The original synoptic fire-weather typing, 1951–60, with day counts by month and dated examples.</li>" if J.get('source5') else ''}
<li>Potter, B.E. and D. McEvoy, 2021: Weather factors associated with extremely large fires and fire growth days. Earth
Interactions (AMS). <a href="https://research.fs.usda.gov/download/treesearch/63719.pdf">research.fs.usda.gov/download/treesearch/63719.pdf</a>.
Twenty fires over 36,400 ha (about 90,000 acres), each paired with a nearby 10,100–30,300 ha fire.</li>
<li>Work they cite for California: Krumm 1954 (Rattlesnake Fire), Countryman et al. 1956 (Inaja Fire), Ortel 1964 (five Northern
and Central California weather types), Blier 1998 (Sundowners), Raphael 2003 and Hughes and Hall 2009 (Santa Anas), Huang et al.
2009 (October 2003), Wallmann et al. 2010 and Nauslar et al. 2013 (dry lightning).</li></ul>
<p class="note">Descriptions below are short paraphrases; see the sources for the full text and maps.</p></div>
<div class="card"><h2>What lines up</h2><ul>
<li><b>Regime C (a ridge over or off British Columbia) is the offshore-wind regime for both regions</b>: it is the most
over-represented regime on Northern California northeast-foehn days and on Southern California Santa Ana days. The Synthesis puts
the Santa Ana's upper ridge "off the Pacific Northwest coast"; the strongest growth odds in this project's offshore rows come
from these days.</li>
<li><b>Dry lightning has two modes</b>, as CEFA says: under the ridge (regimes C and D) and with a trough approaching from the west
(B and E; CEFA's June 2008 case).</li>
<li><b>Sundowners lead Santa Anas</b>: a Santa Ana day followed within one to two days on
{(lead.get('p_after_sundowner') or 0):.0%} of Sundowner evenings, against {(lead.get('p_any_day') or 0):.0%} of all October–January days.</li>
<li><b>Fire-slowing patterns</b> (NWCG PMS 437: closed lows, deep troughs, frontal passages) match regime E, a deep trough off the
coast, which cuts the odds of large growth to about ×0.6 in Northern California.</li>
<li><b>What the data doesn't show</b>: strong, dry southwest flow aloft at Reno (the southwest-foehn setup) doesn't raise the
odds of large growth on the east side over 2001–2025, and the Santa Maria–Santa Barbara Sundowner gradient alone barely separates
large-growth days near Santa Barbara (AUC 0.53). Both patterns are real but rare; a 25-year record can't calibrate them, so they
aren't outlook rows.</li>
<li><b>The pre-2001 cases</b> (classified from the 1948–2000 reanalysis, not fit to them; the classifier agrees with the
calibration's own labels on 98% of 2001–2005 days): the Inaja Santa Ana (1956), the 49er northeast foehn (1988) and the Painted
Cave Sundowner (1990) fall in the expected regimes. The Fountain Fire (1992) is regime C on its first two days and B on the next
two, a ridge giving way to the onshore trough, the transition CEFA describes. The Rattlesnake Fire (1953) sits in D, a warm summer
ridge, all week, which fits Krumm's coastal high over a Sacramento Valley heat trough better than the offshore regimes listed for it.</li></ul></div>
{''.join(cards)}
{schroeder_card(J.get("schroeder_monthly"), J.get("literature_cases"))}
{nws_card(J.get("nws_objective"))}
<div class="card"><h2>Megafires: fuels before weather (Potter and McEvoy 2021)</h2>
<p><b>As published</b> (paraphrased): the unusually large fires stood out by antecedent dryness (evaporative demand drought
index) more than by their daily weather; their growth pulled ahead of the smaller fires' only on the most extreme days (80th–100th
percentile wind, Fosberg index and Hot-Dry-Windy).</p>
<p><b>In California's 2005–2025 fires of 10,000+ acres</b>: fires that reached 90,000+ acres started with ERC at or above the 90th percentile
65% of the time, against 37% for 10,000–25,000-acre fires and 42% for 25,000–90,000 (median 93rd against the 85th–86th percentile), the
same signal measured with fuels instead of evaporative demand, and the reason this project's outlook multiplies weather odds by the forecast
fuels. Daily growth (scaled by each fire's own average) rises from about half the average on the calmest 40% of Hot-Dry-Windy days to
0.7–0.9 on the top 20% for every size of fire; the larger response of the biggest fires on those days doesn't show with this project's
area-wide, reanalysis Hot-Dry-Windy measure.</p></div>
{hawaii_card(J.get("hawaii"))}
{ridge_card(J.get("ridge_breakdown"))}
<div class="card"><h2>Case dates from the literature</h2>
<p class="sub">Each date's 500 hPa regime. Dates since 2001 come from the calibration record; earlier ones are classified from the
NCEP R1 reanalysis (1948–2000) against the same regime centroids. Before 1979 the reanalysis has no satellite data and is less
reliable over the Pacific. {esc(pre_txt)}</p>
<div class="scroll"><table><tr><th>Date</th><th>Case</th><th>Pattern</th><th>Source</th><th>Regime</th><th>Expected</th><th></th><th>Regime from</th></tr>
{lit_rows}</table></div></div>
"""
    (gdir / "patterns.html").write_text(page("Documented critical fire-weather patterns", body,
                                             desc="Critical fire-weather patterns from the literature against this project's regimes"),
                                        encoding="utf-8")
    return "guide/patterns.html"


PYRO = {}   # pyroconvection research summary, loaded by generate() when cfg["pyroconvection_page"] is on


def _pyro_rule():
    """The regime + 700 hPa moisture rule from the pyroconvection study (regime_moisture.py), or None."""
    rm = PYRO.get("regime_moisture") or {}
    return (rm.get("rules") or {}).get("regime A, C or D AND q700 >= cut")


def pyro_regime_card(L):
    """Regime page card: the study's pyroCb / pyroCu days that fell in this regime, with 700 hPa moisture."""
    rm = PYRO.get("regime_moisture") or {}
    cases = (rm.get("cases_by_regime") or {}).get(L, [])
    cb = [c for c in cases if c["class"] == "pyroCb"]
    cu = [c for c in cases if c["class"] == "pyroCu"]
    med = rm.get("q700_median_by_regime_class") or {}
    q_cb, q_none = (med.get("pyroCb") or {}).get(L), (med.get("none") or {}).get(L)
    rule = _pyro_rule()
    li = "".join(f"<li><b>{esc(c['fire'])}</b>, {esc(c['date'])}: 700 hPa moisture {c['q700'] if c['q700'] is not None else '–'} g/kg"
                 + (f", {c['growth_ac']:,} acres" if c.get("growth_ac") else "") + "</li>" for c in cb)
    if cb:
        lead = f"<p>{len(cb)} of the study's {PYRO['summary']['n_pyrocb']} pyroCb days formed under Regime {L}:</p><ul>{li}</ul>"
    else:
        lead = (f"<p><b>None</b> of the study's {PYRO['summary']['n_pyrocb']} pyroCb days formed under Regime {L}"
                f"{f' ({len(cu)} pyroCu days did)' if cu else ''}.</p>")
    cu_txt = (f"<p>PyroCu (fire-rooted tops −20 to −40 °C) under Regime {L}: {len(cu)} days"
              + (f" ({', '.join(sorted({c['fire'] for c in cu}))})" if cu else "") + ".</p>")
    q_txt = (f"<p>Median 700 hPa moisture at the fire: {q_cb} g/kg on pyroCb days, {q_none} g/kg on days with no deep plume cloud.</p>"
             if q_cb is not None and q_none is not None else "")
    rule_txt = ""
    if rule and L in "ACD":
        rule_txt = (f"<p class='note'>Under A, C or D, 700 hPa moisture at or above ~{rule['pooled_q_cut']:.1f} g/kg marked every studied pyroCb "
                    f"day ({rule['pooled_pyroCb']}) while flagging {rule['pooled_none']} days without deep plume cloud "
                    f"(fire by fire: {rule['lofo_hit']} caught, {rule['lofo_fa']} flagged). A pattern-level hint, weaker than the "
                    "fire-power test on the pyroconvection page.</p>")
    return (f'<div class="card" id="pyroconvection"><h2>Pyroconvection in Regime {L}</h2>{lead}{cu_txt}{q_txt}{rule_txt}'
            f'<p class="note">{_nfires().capitalize()} studied California fires ({PYRO["summary"].get("years", "")}), GOES-classified and reviewed by eye. '
            '<a href="../guide/pyroconvection.html">Pyroconvection under the scenarios</a>.</p></div>')


def _nfires():
    """Number of studied fires as a word (pyroCb study export; 'seven' before 'n_fires' was exported)."""
    n = PYRO.get("summary", {}).get("n_fires", 7) if PYRO else 7
    words = "zero one two three four five six seven eight nine ten eleven twelve".split()
    return words[n] if n < len(words) else str(n)


def load_pyro(paths, cfg):
    PYRO.clear()
    f = paths.cal / "pyroconv" / "pyroconv_patterns.json"
    if cfg.get("pyroconvection_page") and f.exists():
        PYRO.update(json.loads(f.read_text()))


def pyroconvection_page(paths, gdir):
    """guide/pyroconvection.html: pyroCb / pyroCu days of the studied California fires (GOES-classified, analyst-reviewed)
    against this project's 500 hPa regimes and the documented patterns, and the Tory & Kepert (2021) PFT
    ingredient thresholds (docs/pyrocb/README.md)."""
    import shutil
    if not PYRO:
        return None
    src = paths.cal / "pyroconv"
    for f in src.glob("*.png"):
        shutil.copy2(f, gdir / f"pyro_{f.name}")
    J, S = PYRO, PYRO["summary"]
    cls_names = {"pyroCb": "pyroCb", "pyroCu": "pyroCu", "ambiguous": "cold tops, ordinary storms too", "none": "no deep plume cloud"}
    reg_rows = "".join(
        f"<tr><td><a href='../regimes/{L}.html'>{L}</a></td>"
        + "".join(f"<td class='n'>{J['regime_by_class'][c][L]['days']}</td>" for c in cls_names)
        + f"<td class='n'>{'–' if not J['regime_by_class']['pyroCb'][L]['lift'] else '×%.1f' % J['regime_by_class']['pyroCb'][L]['lift']}</td>"
        + f"<td class='note'>{esc(J['regime_descriptions'].get(L, ''))}</td></tr>" for L in "ABCDEF")
    pat_names = {"nc_dry_tstorm": "Northern California dry thunderstorms (≥ 500 strikes)", "nc_ne_foehn": "Northeast foehn (offshore gradient)",
                 "nc_sw_foehn": "Southwest foehn (dry slot at Reno)", "breakdown_stage2": "Ridge breakdown, stage 2 (last ridge day)",
                 "breakdown_stage3": "Ridge breakdown, stage 3 (first trough day)", "ridge_persists": "Ridge persists"}
    pat_rows = "".join(f"<tr><td>{esc(n)}</td>" + "".join(f"<td class='n'>{J['patterns_by_class'][k][c]}</td>" for c in cls_names)
                       + f"<td class='n'>{J['patterns_by_class'][k]['all_days']}</td></tr>" for k, n in pat_names.items())
    comp = "".join(f"""<div class="card"><div class="cols"><div><img class="map" src="pyro_{v['png']}" alt="composite"></div>
<div><h2>{esc(t)} ({v['n']} days)</h2><p>{esc(v['text'])}</p></div></div></div>"""
                   for k, t in (("pyroCb", "PyroCb days"), ("pyroCu", "PyroCu days"),
                                ("big_none", "Biggest growth days without deep plume cloud"))
                   for v in [J["composites"].get(k)] if v)
    th_rows = "".join(f"<tr><td>{esc(r['metric'])}</td><td>{esc(r['rule'])}</td><td class='n'>{esc(r['lofo_hit'])}</td>"
                      f"<td class='n'>{esc(r['lofo_false_alarm'])}</td><td class='n'>{r['lofo_balanced_acc']:.2f}</td></tr>"
                      for r in J.get("thresholds", []))
    case_rows = "".join(
        f"<tr><td>{esc(c['fire'])}</td><td>{esc(c['fire_day'])}</td><td>{esc(c['final_class'])}</td>"
        f"<td><a href='../regimes/{esc(c['regime'] or '')}.html'>{esc(c['regime'] or '–')}</a></td>"
        f"<td>{esc(', '.join(n for k, n in (('nc_dry_tstorm', 'dry tstorms'), ('nc_ne_foehn', 'NE foehn'), ('nc_sw_foehn', 'SW foehn'), ('breakdown_stage2', 'ridge breakdown 2'), ('breakdown_stage3', 'ridge breakdown 3')) if c.get(k)) or '–')}</td>"
        f"<td class='n'>{'no solution' if c['pft_min_gw'] is None else '%.0f' % c['pft_min_gw']}</td>"
        f"<td class='n'>{(c['ratio_max'] or 0):.2f}</td><td class='n'>{(c['q_ml_gkg'] or 0):.1f}</td>"
        f"<td class='n'>{(c['growth_ac'] or 0):,.0f}</td><td class='n'>{'' if c['bt_rooted_min'] is None else '%.0f' % c['bt_rooted_min']}</td></tr>"
        for c in sorted(J["cases"], key=lambda c: (c["final_class"] != "pyroCb", c["fire_day"])))
    imgs = "".join(f'<div class="card"><h2>{esc(t)}</h2><img class="map" src="pyro_{f}" alt="{esc(t)}"></div>'
                   for f, t in J.get("images", []) if (src / f).exists())
    body = f"""
<h1>Pyroconvection under the weather scenarios</h1>
<p class="sub">Which days {_nfires()} large California fires made pyrocumulonimbus (pyroCb) or pyrocumulus (pyroCu), and which of this page's
500 hPa regimes and documented patterns they fell in. Research, not a forecast product.</p>
<div class="card"><h2>What was studied</h2><ul>
<li>{esc(S['fires'])}: {S['days']} fire days ({S['n_pyrocb']} pyroCb, {S['n_pyrocu']} pyroCu, {S['n_ambiguous']} with cold cloud over the
fire from ordinary storms, {S['n_none']} without deep plume cloud).</li>
<li><b>PyroCb</b> = cloud tops colder than −40 °C rooted on the fire in GOES-17/16 infrared (10-minute), reviewed by eye; pyroCu = fire-rooted
tops between −20 and −40 °C. Usable NEXRAD radars confirm them: the plume's echo column reaches ≥ ~11 km on every pyroCb day.</li>
<li>Atmosphere: the pyroCb firepower threshold (PFT) of Tory and Kepert (2021, Weather and Forecasting 36, 439–456) from hourly
HRRR columns over each day's most active front. It is the least fire heat output the air needs for a pyroCb.</li></ul></div>
<div class="card"><h2>Findings</h2><ul>{''.join(f'<li>{esc(x)}</li>' for x in J.get('findings', []))}</ul></div>
<div class="card"><h2>PyroCb and pyroCu days by 500 hPa regime</h2>
<div class="scroll"><table><tr><th>Regime</th>{''.join(f'<th class="n">{esc(n)}</th>' for n in cls_names.values())}<th class="n">pyroCb ×</th><th>Pattern</th></tr>{reg_rows}</table></div>
<p class="note">Counts of the studied fire days. pyroCb × = share of pyroCb days in the regime ÷ share of all studied fire days.
With {S['n_pyrocb']} pyroCb days these are indicative only.</p></div>
{pyro_rule_card()}
<div class="card"><h2>Documented patterns on those days</h2>
<div class="scroll"><table><tr><th>Pattern (same rules as <a href="patterns.html">Documented patterns</a>)</th>{''.join(f'<th class="n">{esc(n)}</th>' for n in cls_names.values())}<th class="n">all days</th></tr>{pat_rows}</table></div></div>
{comp}
<div class="card"><h2>Ingredient thresholds (scored fire by fire)</h2>
<div class="scroll"><table><tr><th>Per fire day</th><th>pyroCb when</th><th class="n">pyroCb days caught</th><th class="n">other days flagged</th><th class="n">balanced acc.</th></tr>{th_rows}</table></div>
<p class="note">Each cut is fitted with one fire held out and scored on that fire, for each fire in turn, then the counts are added up. FP = fire power from GOES fire
radiative power; PFT from HRRR. Days with ordinary storms over the fire are left out. The radar row uses only radars without
terrain clutter at the fire (Beale, Eureka, Reno; not Hanford or Sacramento for Creek and Caldor), and radar observes the plume
rather than forecasting it. "Fuel ahead" rows use pre-fire LANDFIRE fuels within 5 km of the previous day's growth.</p></div>
<div class="card"><h2>The cases</h2><div class="scroll"><table><tr><th>Fire</th><th>Fire day</th><th>Class</th><th>Regime</th><th>Patterns</th>
<th class="n">min PFT (GW)</th><th class="n">FP/PFT</th><th class="n">q_ML (g/kg)</th><th class="n">Growth (ac)</th><th class="n">Coldest top (°C)</th></tr>{case_rows}</table></div></div>
{imgs}
<div class="card"><h2>Caveats</h2><ul>{''.join(f'<li>{esc(x)}</li>' for x in J.get('caveats', []))}</ul></div>
"""
    (gdir / "pyroconvection.html").write_text(page("Pyroconvection under the weather scenarios", body,
                                                   desc=f"PyroCb and pyroCu days of {_nfires()} California fires against 500 hPa regimes and documented patterns"),
                                              encoding="utf-8")
    return "guide/pyroconvection.html"


# ----------------------------------------------------------------------------- GACC / method tags (2026-10-23)
# Analyst request: tag every helper page and section by the area it covers (ONCC, OSCC, HI) and the
# lightning material by method. Applied as a last pass over the written pages, so pages built by other
# modules get tagged too. Sections not listed take their page's tags.
CA, ALL = ("ONCC", "OSCC"), ("ONCC", "OSCC", "HI")
GACC_TITLES = {"ONCC": "Northern California (ONCC)", "OSCC": "Southern California (OSCC)", "HI": "Hawaii (HI)"}
METHODS = {"M1": ("Method 1 · logistic / CAPE", "Method 1: the first lightning method. A logistic equation on CAPE and moisture "
                  "(after Nikolov et al. 2024) for the map layer, and CAPE thresholds for the outlook rows. IFS ENS + GEFS."),
           "M2": ("Method 2 · ECMWF formula", "Method 2: ECMWF's lightning formula (Lopez 2016) on IFS ENS convective fields, "
                  "calibrated to NLDN strikes.")}
GUIDE_TAGS = {
    "map": (ALL, {}), "weeks": (ALL, {}), "plumes": (CA, {}), "regimes": (CA, {}), "pyroconvection": (CA, {}),
    "outlook": (ALL, {"Thunderstorm rows": CA}),
    "lightning": (CA + ("M1",), {}), "lightning_formula": (CA + ("M2",), {"Past sieges, both layers": CA + ("M1", "M2")}),
    "patterns": (ALL, {"What lines up": CA, "Northern California": ("ONCC",), "Southern California": ("OSCC",),
                       "Schroeder": CA, "NWS objective": CA, "Megafires": CA, "Hawaii": ("HI",), "Breakdown": CA}),
}


def gtags(tags):
    out = []
    for t in tags:
        if t in METHODS:
            out.append(f'<span class="gtag gt-{t}" title="{esc(METHODS[t][1])}">{esc(METHODS[t][0])}</span>')
        else:
            out.append(f'<span class="gtag gt-{t}" title="{esc(GACC_TITLES[t])}">{t}</span>')
    return f'<span class="gtags">{"".join(out)}</span>' if out else ""


def lightning_method(key):
    return "M2" if key.endswith("_ltgformula") else "M1" if re.search(r"_(capep90|dry_tstorm)$", key) else None


def tag_html(html_, page_tags, sections=None):
    """Chips after the <h1> text and inside every <h2> (section tags by heading prefix, else the page's)."""
    if 'class="gtags"' in html_:
        return html_
    html_ = re.sub(r"(<h1>)(.*?)(</h1>)", lambda m: m.group(1) + m.group(2) + gtags(page_tags) + m.group(3), html_, count=1, flags=re.S)
    gacc_only = tuple(t for t in page_tags if t not in METHODS)

    def h2(m):
        text = re.sub("<[^>]+>", "", m.group(2)).strip()
        tags = next((v for k, v in (sections or {}).items() if text.startswith(k)), gacc_only)
        return m.group(1) + m.group(2) + gtags(tags) + m.group(3)
    return re.sub(r"(<h2[^>]*>)(.*?)(</h2>)", h2, html_, flags=re.S)


def tag_pages(web):
    for name, (tags, sections) in GUIDE_TAGS.items():
        f = web / "guide" / f"{name}.html"
        if f.exists():
            f.write_text(tag_html(f.read_text(encoding="utf-8"), tags, sections), encoding="utf-8")
    for f in (web / "predictors").glob("*.html"):
        ent, key = f.stem.split("__", 1)
        g = ent.split("-")[0]
        tags = (g,) if g in GACC_TITLES else CA
        m = lightning_method(key)
        f.write_text(tag_html(f.read_text(encoding="utf-8"), tags + ((m,) if m else ())), encoding="utf-8")
    for f in (web / "regimes").glob("*.html"):
        f.write_text(tag_html(f.read_text(encoding="utf-8"), CA), encoding="utf-8")


def generate(web, data, cfg, outlook_rows, region_lines):
    """Writes every sub-page; returns {"predictors": {row index: href}, "regimes": {letter: href}}."""
    import xarray as xr
    paths = common.Paths(cfg)
    load_pyro(paths, cfg)
    NFDRS.clear()
    NFDRS.update(load_nfdrs())
    BURN_DIRS.update({"ONCC": common.ROOT / "data" / "satellite_growth" / "burnday",
                      "OSCC": paths.oscc / "satellite_growth" / "burnday"})
    (web / "pages.css").write_text(CSS, encoding="utf-8")
    cal = json.loads((paths.cal / "calibration.json").read_text())
    scan = pd.read_csv(paths.cal / "threshold_scan.csv")
    derived = pd.read_csv(paths.cal / "predictors_derived.csv", index_col=0, parse_dates=True)
    growth = load_growth(paths)
    psas = load_psas(paths)
    if psas is not None:
        PSA_NAMES.update(dict(zip(psas.PSANationa, psas.PSANAME)))
    cpath = paths.cal / "growth_cells.csv.gz"
    cells = pd.read_csv(cpath) if cpath.exists() else None
    links = {"predictors": {}, "regimes": {}}
    import shutil
    for sub in ("predictors", "regimes"):          # rebuilt from scratch: rows come and go with the season
        if (web / sub).is_dir():
            shutil.rmtree(web / sub)
    pdir = web / "predictors"
    pdir.mkdir(exist_ok=True)
    strikes = density = None
    lpath = paths.history / "lightning"
    if any(r.get("kind") == "lightning" for r in outlook_rows) and (lpath / "strikes_psa_fireday.csv").exists():
        strikes = pd.read_csv(lpath / "strikes_psa_fireday.csv", index_col=0, parse_dates=True)
        density = dict(np.load(lpath / "density_0p05.npz")) if (lpath / "density_0p05.npz").exists() else None
    for i, row in enumerate(outlook_rows):
        name = slug(row["entity"], row["key"]) + ".html"
        if row.get("kind") == "lightning":
            html_ = lightning_page(row, cfg, cal, data, psas, pdir, strikes, density)
        elif row["entity"] == "HI":
            html_ = hawaii_page(row, cfg, cal, data, paths)
        else:
            html_ = predictor_page(row, cfg, cal, scan, derived, growth, data, region_lines, psas, cells, pdir)
        (pdir / name).write_text(html_, encoding="utf-8")
        links["predictors"][i] = f"predictors/{name}"
    if data.get("regimes") and (paths.cal / "regimes.nc").exists():
        reg = xr.open_dataset(paths.cal / "regimes.nc").load()
        rstats = pd.read_csv(paths.cal / "regime_stats.csv")
        labels = pd.read_csv(paths.cal / "regime_labels.csv", index_col=0, parse_dates=True)["regime"]
        rdir = web / "regimes"
        rdir.mkdir(exist_ok=True)
        for c, L in enumerate(data["regimes"]["letters"]):
            (rdir / f"{L}.html").write_text(regime_page(c, L, reg, rstats, labels, growth, data, rdir, cfg), encoding="utf-8")
            links["regimes"][L] = f"regimes/{L}.html"
    gdir = web / "guide"
    gdir.mkdir(exist_ok=True)
    for name, (title, body) in guides(cfg).items():
        (gdir / f"{name}.html").write_text(page(title, body), encoding="utf-8")
    links["patterns"] = patterns_page(paths, gdir)
    links["pyroconvection"] = pyroconvection_page(paths, gdir)
    if not PYRO and (gdir / "pyroconvection.html").exists():   # toggled off: don't leave a stale page behind
        (gdir / "pyroconvection.html").unlink()
    for f in [*gdir.glob("lightning_hindcast_*.png"), *gdir.glob("lightning_case_*.png")]:   # stale examples
        f.unlink()
    for f in paths.cal.glob("lightning_case_*.png"):            # lightning page's sieges and false alarms
        shutil.copy2(f, gdir / f.name)
    for f in [*gdir.glob("lightning_lopez_*.png"), gdir / "lightning_formula.html"]:
        if f.name.endswith(".png") or not cfg.get("lightning_lopez", {}).get("enabled", False):
            f.unlink(missing_ok=True)                           # toggled off: no stale formula page
    if cfg.get("lightning_lopez", {}).get("enabled", False):
        for f in paths.cal.glob("lightning_lopez_*.png"):       # lightning formula page's charts and sieges
            shutil.copy2(f, gdir / f.name)
    tag_pages(web)
    return links
