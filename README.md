# California & Hawaii AI Ensemble Research — 1–15 Day Patterns

**Experimental research, not a forecast.** This site compares openly available AI and physics-based
weather ensembles for large-scale California and Hawaii fire-weather patterns, 1–15 days ahead. It isn't an
official forecast, outlook, watch or warning, and it hasn't been reviewed or endorsed by any agency.
Don't use it for operational decisions. For official information see the
[National Weather Service fire weather pages](https://www.weather.gov/fire/) and the
[NIFC Predictive Services outlooks](https://www.nifc.gov/nicc/predictive-services/outlooks).

The page (`index.html`, data in `aie_data.js`) is regenerated once a day. Each update replaces this
repository's history, so only the current day is kept here.

## Data and credits

Everything comes from publicly available sources:

- ECMWF AIFS ENS and IFS ENS open data © European Centre for Medium-Range Weather Forecasts (ECMWF),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Modified: cut to the region, averaged
  and compared against thresholds. ECMWF doesn't accept any liability for errors or omissions here.
- NOAA NCEP AIGEFS and GEFS; NCEP/NCAR Reanalysis 1 data provided by NOAA PSL, Boulder, Colorado,
  USA. U.S. Government works, public domain.
- Active-fire detections: NASA FIRMS (LANCE / ESDIS). Fire perimeters: MTBS (USGS/USFS) and NIFC
  WFIGS. Fire records: FPA FOD (Short, USDA Forest Service Research Data Archive) and NIFC InFORM.
  Region boundaries: NIFC.
- Surface and upper-air observations: Iowa Environmental Mesonet (Iowa State University). Map lines:
  Natural Earth.
- Lightning formula layer: after the ECMWF IFS lightning parameterization, Lopez, P., 2016: A lightning
  parameterization for the ECMWF Integrated Forecasting System. Mon. Wea. Rev., 144, 3057-3075,
  [doi:10.1175/MWR-D-16-0026.1](https://doi.org/10.1175/MWR-D-16-0026.1). Rebuilt on IFS ENS open data and
  calibrated with ERA5 (contains modified Copernicus Climate Change Service information) to Vaisala NLDN
  strikes (licensed; only derived probabilities are shown). Not ECMWF's lightning product.

Independent, non-commercial research. Data are provided as-is with no warranty.
