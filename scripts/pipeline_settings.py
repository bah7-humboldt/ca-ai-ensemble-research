"""Central default-settings loader for the Fire Analyst Toolkit.

Single source of truth for pipeline-wide defaults (GACC polling scope, clip radius,
conditioning days, burn hours, ignition snap distance, LANDFIRE version, regional
landscape cache mode/build settings, all FARSITE advanced settings, the two
machine-specific paths, and the PrintKMZ/PrintPDF/SendTeams/SendEmail/ExternalFolder
toggles)
-- read from config/default_settings.json, layered over the fallback values below so
the pipeline still runs with sane behavior on a fresh checkout or if that file is
missing/unparseable. scripts/run_fire.py, scripts/irwin_watcher.py, and
arcpro/IrwinFarsiteTools.pyt all call load_settings() and use its values as their own
argparse/arcpy.Parameter defaults -- an explicit CLI flag or toolbox parameter edit
still overrides the settings file per-run.

Pure stdlib (json/pathlib only) so this is safe to import from ArcGIS Pro's own
arcgispro-py3 environment too, same constraint scripts/assemble_inputs.py already
satisfies.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "config" / "default_settings.json"

_FALLBACK = {
    "gacc": ["ONCC"],
    "farsite_simulation_enabled": True,
    "ia_pyrocb_potential_enabled": False,
    "ia_pyrocb_watch_gw": 360,
    "ia_pyrocb_warning_ratio": 0.13,
    "ia_pyrocb_frp_radius_km": 10,
    "pyrocb_data_dir": "data/pyrocb",
    "pyrocb_site_dir": "data/outputs/pyrocb_site",
    "clip_radius_miles": 10,
    "clip_radius_max_miles": 30,
    "clip_radius_max_retries": 2,
    "conditioning_days": 7,
    "burn_hours": 6,
    "ignition_snap_max_ft": 500,
    "lf_version": "LF 2025",
    "regional_cache_mode": "auto",
    "regional_cache_gacc": "ONCC",
    "regional_cache_boundary_geojson": "config/oncc_boundary.geojson",
    "regional_cache_tile_size_deg": 3.5,
    "regional_cache_tile_overlap_deg": 0.05,
    "regional_cache_output_projection": "5070",
    "gpkg": "data/outputs/LIVE_FARSITE_Runs.gpkg",
    "pipeline_python": r"C:\Users\benah\AppData\Local\Python\pythoncore-3.14-64\python.exe",
    "timestep": 60,
    "distance_res": 60.0,
    "perimeter_res": 60.0,
    "spot_grid_resolution": 30.0,
    "spot_probability": 0.01,
    "spot_ignition_delay": 0,
    "minimum_spot_distance": 60,
    "acceleration_on": True,
    "fill_barriers": True,
    "foliar_moisture_content": 80,
    "crown_fire_method": "ScottReinhardt",
    "gridded_winds_generate": False,
    "gridded_winds_resolution": 60.0,
    "gridded_winds_direction_bin": 20,
    "gridded_winds_speed_bin": 5,
    "spotting_seed": None,
    "print_kmz": True,
    "print_pdf": True,
    "send_teams": True,
    "send_email": True,  # soft-skips if config/email_config.json is missing -- see scripts/email_report.py
    "external_folder_enabled": True,
    "external_folder_path": "",
    "shared_folder_link": "",
    "weather_mode": "automatic",
    "weather_source": "automatic",
    "psa_shapefile": "config/oncc_psa/ONCC_PSA.shp",
    "psa_sample_points_geojson": "config/psa_sample_points.geojson",
    "psa_points_per_psa": 10,
    "psa_start_hour": 14,
    "psa_burn_hours": 6,
    "psa_clip_radius_miles": 10,
    "psa_landscape_dir": "data/lcp/psa",
    "psa_gpkg": "data/outputs/PSA_Daily_Runs.gdb",
    "psa_output_dir": "data/outputs/psa_reports",
    "psa_raws_csv": "config/oncc_raws_stations.csv",
    "psa_retention_days": 14,
    "psa_retention_dry_run": True,
    "psa_run_timeout_seconds": 300,
    "ati_station_search_radius_miles": 25,
    "ati_max_station_candidates": 5,
    "ati_fallback_speed_mph": 25,
    "ati_road_search_pad_miles": 2.0,
    "ati_fs_road_fallback_speed_mph": 15,
    "ati_road_search_pad_max_miles": 20.0,
    "ati_road_search_pad_max_retries": 3,
    "nfdrs_ovf_gacc": "ONCC",
    "nfdrs_ovf_raws_csv": "config/oncc_raws_stations.csv",
    "nfdrs_ovf_db_dir": "data/analysis/nfdrs_ovf",
    "nfdrs_ovf_forecast_days": 7,
    "nfdrs_ovf_observation_lookback_days": 10,
    "nfdrs_ovf_output_dir": "data/outputs/nfdrs_ovf",
    "nfdrs_ovf_indices": ["BI", "ERC"],
    "nfdrs_ovf_window_days": 30,
    "nfdrs_ovf_task_hour_local": 3,
    "nfdrs_ovf_notify_enabled": False,
    "nfdrs_ovf_shared_folder_link": "",
    "climatology_station_csv": "config/oncc_raws_stations.csv",
    "climatology_index_list": ["100HrFM", "1000HrFM", "ERC", "BI", "KBDI"],
    "climatology_dir": "data/historic_fires_research",
    "climatology_hist_startdate": "2005-03-01",
    "climatology_hist_enddate": "2025-12-31",
    "climatology_season_enabled": True,
    "climatology_season_startdate": "01-01",
    "climatology_season_enddate": "12-31",
    "pocket_charts_station_charts": False,
    "pocket_charts_gacc_charts": False,
    "pocket_charts_current_year": False,
    "pocket_charts_forecast": False,
    "pocket_charts_overlay_year": "",
    "pocket_charts_fire_events_csv": "config/oncc_fire_events.csv",
    "pocket_charts_plot_fires": True,
    "pocket_charts_label_fires": False,
    "pocket_charts_label": "ONCC PSA Fires",
    "fdra_charts_historical_dir": "data/historic_fires_research",
    "fdra_charts_web_output_dir": "data/outputs/nfdrs_ovf_web/fdra-charts",
    "fdra_charts_web_forecast_days": 7,
    "fdra_charts_web_publish_enabled": False,
    "fdra_charts_oncc_publish_enabled": False,
    "fdra_charts_oncc_remote_dir": "predictive/weather/fdra_charts",
    "fdra_charts_oncc_local_root": "",
    "fdra_charts_web_apply_raws_qc": True,
    "fdra_charts_web_min_station_coverage": 0.5,
    "fdra_charts_web_fires_enabled": True,
    "fdra_charts_web_fire_events_csv": "config/oncc_fire_events.csv",
    "fdra_charts_web_fire_min_size_class": "C",
    "fire_occurrence_gacc": "ONCC",
    "fire_occurrence_dir": "data/fire_occurrence",
    "fire_occurrence_psa_shapefile": "config/oncc_psa/ONCC_PSA.shp",
    "fire_occurrence_fdra_field": "PSANationa",
    "fire_occurrence_timezone": "US/Pacific",
    "fire_occurrence_start_date": "1992-01-01",
    "fire_occurrence_fod_min_edition": 7,
    "fire_occurrence_fod_auto_update": True,
    "fire_occurrence_fodr_url": "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/InFORM_FireOccurrence_Public/FeatureServer/0",
    "fdra_charts_web_growth_enabled": True,
    "fdra_charts_web_growth_csv": "data/imsr_growth/oncc_imsr_growth_days.csv",
    "fdra_charts_web_growth_thresholds_acres": [1000, 5000, 10000, 50000, 100000],
    "fdra_charts_web_large_fire_growth_source": "IMSR",
    "fdra_charts_web_satellite_growth_csv": "data/satellite_growth/oncc_satellite_growth_days.csv",
    "satellite_growth_dir": "data/satellite_growth",
    "satellite_growth_first_year": 2001,
    "satellite_growth_cell_m": 100,
    "satellite_growth_fire_day_start_hour": 6,
    "satellite_growth_max_gap_days": 7,
    "satellite_growth_wfigs_min_acres": 1000,
    "satellite_growth_mtbs_url": "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip",
    "satellite_growth_wfigs_url": "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0",
    "fire_activity_skill_gacc": "ONCC",
    "fire_activity_skill_key_raws_csv": "config/oncc_7day_key_raws.csv",
    "fire_activity_skill_breakpoints_csv": "config/oncc_7day_breakpoints.csv",
    "fire_activity_skill_compare_breakpoints_csv": "config/oncc_7day_breakpoints_fy2026_refit.csv",
    "fire_activity_skill_compare_label": "Refit to FY2026 calls",
    "fire_activity_skill_output_dir": "data/outputs/fire_activity_skill",
    "fire_activity_skill_start_date": "2005-03-01",
    "fire_activity_skill_season": "05-01:10-31",
    "fire_activity_skill_growth_acres": 1000,
    "fire_activity_skill_test_start_year": 2019,
    "fire_activity_skill_min_station_share": 0.5,
    "imsr_growth_accdb": "../IMSRtool-1.09/IMSR_Activity.accdb",
    "imsr_growth_dir": "data/imsr_growth",
    "imsr_growth_lag_days": 1,
    "imsr_growth_first_report_max_age_days": 3,
    "imsr_growth_thresholds_acres": [1000, 5000, 10000],
    "imsr_growth_season": "",
    "psa_danger_map_output_dir": "data/outputs/psa_danger_maps",
    "psa_danger_map_forecast_days": 7,
    "psa_danger_map_grid_resolution_deg": 0.03,
    "psa_danger_map_notify_enabled": False,
    "psa_danger_map_task_hour_local": 4,
    "psa_danger_map_min_por_years": 8,
    "psa_danger_map_raws_exclude": {
        "40732": "HIDDEN VALLEY -- bad RH sensor, confirmed live 2026-08-01 (analyst: pulling "
                 "down values in the danger-map charts). Excluded from the RAWS roster until "
                 "the sensor is fixed or replaced.",
    },
    "psa_danger_map_kriging_range_ceiling": 2.0,
    "psa_danger_map_raws_exclude_since": {},
    "raws_qc_restart_spinup_days": 42,
    "psa_danger_map_spatial_outlier_zscore": 3.5,
    "psa_danger_map_shared_folder_link": "",
    "psa_danger_map_web_viewer_link": "",
    "psa_danger_map_boundary_buffer_deg": 0.5,
    "psa_danger_map_publish_to_web": False,  # soft-skips if config/ftp_config.json is missing -- see scripts/ftp_publish.py
    "raster_retention_days": 7,
    "raster_retention_dry_run": True,
    "raster_retention_task_hour_local": 0,
    "lfmc_map_enabled": False,
    "lfmc_map_output_dir": "data/outputs/lfmc_maps",
}


def load_settings():
    """Reads config/default_settings.json over top of the built-in fallback values
    above. Missing file or unreadable/invalid JSON is a soft-skip (warn, use
    fallback), matching the repo's existing config/teams_config.json convention
    (see scripts/teams_card.py). "_comment"/"_comments" are documentation only
    (JSON has no native comment syntax) and are stripped before merging -- never
    returned as real settings keys."""
    settings = dict(_FALLBACK)
    if SETTINGS_PATH.exists():
        try:
            raw = json.loads(SETTINGS_PATH.read_text())
            raw.pop("_comment", None)
            raw.pop("_comments", None)
            settings.update(raw)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: could not parse {SETTINGS_PATH}: {e} -- using built-in defaults")
    return settings
