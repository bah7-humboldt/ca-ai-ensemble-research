# California & Hawaii AI Ensemble Research

Experimental research only, built from publicly available data. It is not an official forecast.
For official fire weather information, use your local National Weather Service office and the
National Interagency Fire Center.

This repository builds the research viewer at
https://bah7-humboldt.github.io/ca-ai-ensemble-research/ once a day on a GitHub-hosted runner.
The workflow is `.github/workflows/ai-ensemble-daily.yml`. Each run:

1. Refreshes this year's NFDRS observations from FEMS.
2. Downloads the newest runs of four open ensembles:
   - ECMWF AIFS ENS and IFS ENS (open data, CC BY 4.0)
   - NOAA AIGEFS and GEFS (public domain)
3. Scores them against the calibration in `data/ai_ensembles/history/calibration/`.
4. Publishes the viewer.

**Inputs in this repo:**
- the calibration
- NFDRS station history from FEMS
- PSA boundaries
- Natural Earth basemaps (public domain)

The calibration is built from satellite fire detections and ERA5, outside this repository. The
lightning layer uses derived counts only; no raw strike data is stored here.

**Archive:** each model run's regime-box fields are attached to that month's `archive-YYYY-MM`
release.
