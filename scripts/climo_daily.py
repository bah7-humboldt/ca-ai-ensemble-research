"""Reference data, step 2: daily extremes + day-of-year (DOY) climatology for every station,
PSA (FDRA) and the GACC. Reads step 1's raw FEMS downloads (scripts/climo_download.py);
step 3 (scripts/climo_percentiles.py) reads what this writes.

Moved VERBATIM 2026-10-03 from scripts/FDRACharts.py's fems_calculate_dailyextremes(): the
PSA danger maps, reports and interactive charts all read these products, so they must not
shift. The only changes: the globals that function read are now parameters with the same
names, and arcpy.AddMessage() is log(). Output checked byte-for-byte against FDRACharts.py's
own on the full ONCC roster (see CHANGELOG.md, 2026-10-03).

Writes into <climatology_dir>/<GACC>/<Period>/ (layout: scripts/climo_common.py):
    <FDRA>/<StationID>[_Season]_DOY.csv
    <FDRA>/<FDRA>[_Season]_AvgDailyExtremes.csv, <FDRA>/<FDRA>[_Season]_DOY.csv
    <GACC>[_Season]_AvgDailyExtremes.csv, <GACC>[_Season]_DOY.csv
"_Season" files exist only for Historical with climatology_season_enabled.

    python scripts/climo_daily.py                       # Historical, every GACC in the roster
    python scripts/climo_daily.py --period CurrentYear  # (the static Pocket Card charts use these)
"""
import argparse
import os
from datetime import datetime, timedelta

import pandas as pd

from climo_common import PERIODS, climo_settings, load_roster, log, period_dir, toggle


def calculate_daily_extremes(df_stations, gacc, datadir, index_list, toggle_season):
    r'''
    This function processes the data downloaded by the "fems_data_download" function. It cycles through each FDRA folder,
    and for each station, determines the index daily extremes then writes these to an output CSV ("_DailyExtremes.csv").
    It then calculates the FDRA's average daily extreme by averaging the station daily extremes together ("_AvgDailyExtremes.csv").

    This function rolls data "up" through three levels, marked below with
    "# ===== STATION/FDRA/GACC ======" banners:
      STATION - group each station's raw daily rows by Day-Of-Year (1-365)
                across every year in the historical record, to get that
                station's typical min/max/avg for each index on that day.
      FDRA    - average all of an FDRA's stations together, day by day, then
                repeat the same min/max/avg-by-DayOfYear grouping.
      GACC    - same idea one level up: average every FDRA's numbers together
                for the whole GACC.
    Also note the leap-year handling used throughout: Feb 29 is dropped from
    the DayOfYear count (by decrementing every day after Feb 28 in leap
    years) so DayOfYear always tops out at 365 and lines up across years.
    '''
    period = os.path.basename(datadir)
    log("\u200B")
    log(f"CALCULATING '{period}' AVERAGE DAILY EXTREMES")
    #Get list of FDRAs from CSV
    FDRA_list = list(df_stations['FDRA'].unique())
    FDRAname_list = list(df_stations['FDRAName'].unique())
    log("..CALCULATING AVERAGE DAILY EXTREMES")
    gacc_station_df_season_list = []
    gacc_station_df_year_list = []
    for i in range(0, len(FDRA_list)):
        FDRA = FDRA_list[i]
        FDRA_name = FDRAname_list[i]
        FDRA_dir = f"{datadir}/{FDRA}"
        log(f"....{FDRA}")
        #Build list of paths to CSV based on user provided table
        station_list = list(df_stations[df_stations['FDRA'] == FDRA]["StationID"])
        FDRA_dir_csv_year_paths = [f"{FDRA_dir}/{item}.csv" for item in station_list]
        FDRA_dir_csv_season_paths = [f"{FDRA_dir}/{item}_Season.csv" for item in station_list]
        #If seasonal filter applied, process both regular and seasonal filter data
        if(toggle_season == "true" and period == "Historical"):
            csv_paths_list = [FDRA_dir_csv_year_paths, FDRA_dir_csv_season_paths]
            season_list = ["false", "true"]
        else:
            csv_paths_list = [FDRA_dir_csv_year_paths]
            season_list = ["false"]
        for j in range(0, len(csv_paths_list)):
            csv_paths = csv_paths_list[j]
            season = season_list[j]
            if(season == "true"):
                log(f"......Season")
            if(season == "false"):
                log(f"......Year")
            #Read-in CSVs to a list of dataframes, then adjust leap days and create DayOfYear field
            FDRA_station_df_list = []
            for k in range(0, len(csv_paths)):
                try:
                    FDRA_dir_csv_path = csv_paths[k]
                    station_df = pd.read_csv(FDRA_dir_csv_path)
                    #FEMS may have produced a CSV for the station, but there might not be any data in it. Skip if no data.
                    if(len(station_df) <= 1):
                        station_basename = os.path.basename(csv_paths[k]).replace(".csv", "")
                        log(f"........NO DATA AVAILABLE FOR STATION ({station_basename})")
                        continue
                    station = station_df["stationId"].iloc[0]
                    log(f"........{station}")
                    #Replace any leap days (Feb 29) with Feb 28
                    station_df["ObservationTime"] = station_df['ObservationTime'].apply(lambda d: d[:5] + '02-28' if d[5:10] == '02-29' else d)
                    #Create DayOfYear field, and adjust any leap years so all days after Feb 28th are reduced by 1 to yield a max of 365 days/yr
                    station_df["DayOfYear"] = pd.to_datetime(station_df["ObservationTime"]).dt.dayofyear
                    isleapyear_test = pd.to_datetime(station_df['ObservationTime']).dt.is_leap_year
                    ispastfeb_test = pd.to_datetime(station_df['ObservationTime']).dt.month > 2
                    station_df["DayOfYear"] = station_df["DayOfYear"] - (isleapyear_test & ispastfeb_test).astype(int)
                    #Rearrange fields
                    index_cols = [col for col in station_df.columns if col not in ["GACC", "FDRA", "FDRAName", "stationId", "StationName", "ObservationTime", "DayOfYear"]]
                    station_df = station_df[["GACC", "FDRA", "FDRAName", "stationId", "StationName", "ObservationTime", "DayOfYear"] + index_cols]
                    #Add dataframe to lists
                    FDRA_station_df_list.append(station_df)
                    if(season == "true"):
                        gacc_station_df_season_list.append(station_df)
                    if(season == "false"):
                        gacc_station_df_year_list.append(station_df)
                except:
                    station_basename = os.path.basename(csv_paths[k]).replace(".csv", "")
                    log(f"........NO DATA AVAILABLE FOR STATION ({station_basename})")
                    continue
            # ===== STATION ======
            #Process DOY (1-365) metrics for each individual station
            for k in range(0, len(FDRA_station_df_list)):
                station_df = FDRA_station_df_list[k]
                station = station_df["stationId"].loc[0]
                stationname = station_df["StationName"].loc[0]
                #Group the "station_df" dataframe by Day of Year (1-365), and determine the min/max/avg for each
                station_doy_df = station_df.groupby('DayOfYear').agg(
                    Hr1FM_min=('1HrFM', 'min'),
                    Hr1FM_max=('1HrFM', 'max'),
                    Hr1FM_avg=('1HrFM', 'mean'),
                    Hr10FM_min=('10HrFM', 'min'),
                    Hr10FM_max=('10HrFM', 'max'),
                    Hr10FM_avg=('10HrFM', 'mean'),
                    Hr100FM_min=('100HrFM', 'min'),
                    Hr100FM_max=('100HrFM', 'max'),
                    Hr100FM_avg=('100HrFM', 'mean'),
                    Hr1000FM_min=('1000HrFM', 'min'),
                    Hr1000FM_max=('1000HrFM', 'max'),
                    Hr1000FM_avg=('1000HrFM', 'mean'),
                    KBDI_min=('KBDI', 'min'),
                    KBDI_max=('KBDI', 'max'),
                    KBDI_avg=('KBDI', 'mean'),
                    GSI_min=('GSI', 'min'),
                    GSI_max=('GSI', 'max'),
                    GSI_avg=('GSI', 'mean'),
                    WoodyFM_min=('WoodyFM', 'min'),
                    WoodyFM_max=('WoodyFM', 'max'),
                    WoodyFM_avg=('WoodyFM', 'mean'),
                    HerbFM_min=('HerbFM', 'min'),
                    HerbFM_max=('HerbFM', 'max'),
                    HerbFM_avg=('HerbFM', 'mean'),
                    IC_min=('IC', 'min'),
                    IC_max=('IC', 'max'),
                    IC_avg=('IC', 'mean'),
                    ERC_min=('ERC', 'min'),
                    ERC_max=('ERC', 'max'),
                    ERC_avg=('ERC', 'mean'),
                    SC_min=('SC', 'min'),
                    SC_max=('SC', 'max'),
                    SC_avg=('SC', 'mean'),
                    BI_min=('BI', 'min'),
                    BI_max=('BI', 'max'),
                    BI_avg=('BI', 'mean')
                    ).reset_index()
                #The groupby() above will not allow the field names to begin with numbers. Adjust the Fuel Moisture fields here to match FEMS
                station_doy_df.rename(columns={"Hr1FM_min":"1HrFM_min", "Hr1FM_max":"1HrFM_max", "Hr1FM_avg":"1HrFM_avg",
                                            "Hr10FM_min":"10HrFM_min", "Hr10FM_max":"10HrFM_max", "Hr10FM_avg":"10HrFM_avg",
                                            "Hr100FM_min":"100HrFM_min", "Hr100FM_max":"100HrFM_max", "Hr100FM_avg":"100HrFM_avg",
                                            "Hr1000FM_min":"1000HrFM_min", "Hr1000FM_max":"1000HrFM_max", "Hr1000FM_avg":"1000HrFM_avg"}, inplace=True)
                #Filter the dataframe down to only the components of interest
                field_keep_list = ["DayOfYear"] + index_list
                station_doy_df = station_doy_df[[col for col in station_doy_df.columns if any(substr in col for substr in field_keep_list)]]
                #Add additional fields to the dataframe
                station_doy_df["GACC"] = gacc
                station_doy_df["FDRA"] = FDRA
                station_doy_df["FDRAName"] = FDRA_name
                station_doy_df["stationId"] = station
                station_doy_df["StationName"] = stationname
                station_doy_df['MonthDay'] = station_doy_df['DayOfYear'].apply(lambda x: (datetime(2021, 1, 1) + timedelta(days=x-1)).strftime('%m/%d'))
                index_cols = [col for col in station_doy_df.columns if col not in ["GACC", "FDRA", "FDRAName", "stationId", "StationName", "DayOfYear", "MonthDay"]]
                station_doy_df = station_doy_df[["GACC", "FDRA", "FDRAName", "stationId", "StationName", "DayOfYear", "MonthDay"] + index_cols]
                #Save Station dataframe
                if(season == "true"):
                    csv_filename = f"{station}_Season_DOY.csv"
                if(season == "false"):
                    csv_filename = f"{station}_DOY.csv"
                csv_outpath = f"{FDRA_dir}/{csv_filename}"
                station_doy_df.to_csv(csv_outpath, index=False)
            # ===== FDRA ======
            #Concatenate all station dataframes together, then calculate the FDRA averages
            #This will produce a dataframe of the average daily maximums, across all FDRA stations, for each day in the historical record
            if len(FDRA_station_df_list) == 0:
                log("......NO VALID STATION DATA AVAILABLE FOR FDRA, SKIPPING")
                continue
            FDRA_dailymax_full_df = pd.concat(FDRA_station_df_list, ignore_index=True)
            FDRA_avgdailymax_df = FDRA_dailymax_full_df.groupby('ObservationTime').agg({
                                        '1HrFM':'mean',
                                        '10HrFM':'mean',
                                        '100HrFM':'mean',
                                        '1000HrFM':'mean',
                                        'KBDI':'mean',
                                        'GSI':'mean',
                                        'WoodyFM':'mean',
                                        'HerbFM':'mean',
                                        'IC':'mean',
                                        'ERC':'mean',
                                        'SC':'mean',
                                        'BI':'mean'
                                         }).reset_index()
            #Create DayOfYear field, and adjust any leap years so all days after Feb 28th are reduced by 1 to yield a max of 365 days/yr
            FDRA_avgdailymax_df["DayOfYear"] = pd.to_datetime(FDRA_avgdailymax_df["ObservationTime"]).dt.dayofyear
            isleapyear_test = pd.to_datetime(FDRA_avgdailymax_df['ObservationTime']).dt.is_leap_year
            ispastfeb_test = pd.to_datetime(FDRA_avgdailymax_df['ObservationTime']).dt.month > 2
            FDRA_avgdailymax_df["DayOfYear"] = FDRA_avgdailymax_df["DayOfYear"] - (isleapyear_test & ispastfeb_test).astype(int)
            #Process DOY (1-365) metrics for FDRA
            #Group the "FDRA_avgdailymax_df" dataframe by Day of Year (1-365), and determine the min/max/avg for each
            FDRA_doy_df = FDRA_avgdailymax_df.groupby('DayOfYear').agg(
                Hr1FM_min=('1HrFM', 'min'),
                Hr1FM_max=('1HrFM', 'max'),
                Hr1FM_avg=('1HrFM', 'mean'),
                Hr10FM_min=('10HrFM', 'min'),
                Hr10FM_max=('10HrFM', 'max'),
                Hr10FM_avg=('10HrFM', 'mean'),
                Hr100FM_min=('100HrFM', 'min'),
                Hr100FM_max=('100HrFM', 'max'),
                Hr100FM_avg=('100HrFM', 'mean'),
                Hr1000FM_min=('1000HrFM', 'min'),
                Hr1000FM_max=('1000HrFM', 'max'),
                Hr1000FM_avg=('1000HrFM', 'mean'),
                KBDI_min=('KBDI', 'min'),
                KBDI_max=('KBDI', 'max'),
                KBDI_avg=('KBDI', 'mean'),
                GSI_min=('GSI', 'min'),
                GSI_max=('GSI', 'max'),
                GSI_avg=('GSI', 'mean'),
                WoodyFM_min=('WoodyFM', 'min'),
                WoodyFM_max=('WoodyFM', 'max'),
                WoodyFM_avg=('WoodyFM', 'mean'),
                HerbFM_min=('HerbFM', 'min'),
                HerbFM_max=('HerbFM', 'max'),
                HerbFM_avg=('HerbFM', 'mean'),
                IC_min=('IC', 'min'),
                IC_max=('IC', 'max'),
                IC_avg=('IC', 'mean'),
                ERC_min=('ERC', 'min'),
                ERC_max=('ERC', 'max'),
                ERC_avg=('ERC', 'mean'),
                SC_min=('SC', 'min'),
                SC_max=('SC', 'max'),
                SC_avg=('SC', 'mean'),
                BI_min=('BI', 'min'),
                BI_max=('BI', 'max'),
                BI_avg=('BI', 'mean')
                ).reset_index()
            #The groupby() above will not allow the field names to begin with numbers. Adjust the Fuel Moisture fields here to match FEMS
            FDRA_doy_df.rename(columns={"Hr1FM_min":"1HrFM_min", "Hr1FM_max":"1HrFM_max", "Hr1FM_avg":"1HrFM_avg",
                                        "Hr10FM_min":"10HrFM_min", "Hr10FM_max":"10HrFM_max", "Hr10FM_avg":"10HrFM_avg",
                                        "Hr100FM_min":"100HrFM_min", "Hr100FM_max":"100HrFM_max", "Hr100FM_avg":"100HrFM_avg",
                                        "Hr1000FM_min":"1000HrFM_min", "Hr1000FM_max":"1000HrFM_max", "Hr1000FM_avg":"1000HrFM_avg"}, inplace=True)
            #Save AvgDailyMax dataframe
            #Filter the dataframe down to only the components of interest
            field_keep_list = ["ObservationTime", "DayOfYear"] + index_list
            FDRA_avgdailymax_df = FDRA_avgdailymax_df[[col for col in FDRA_avgdailymax_df.columns if any(substr in col for substr in field_keep_list)]].copy()
            #Add additional fields
            FDRA_avgdailymax_df["GACC"] = gacc
            FDRA_avgdailymax_df["FDRA"] = FDRA
            FDRA_avgdailymax_df["FDRAName"] = FDRA_name
            FDRA_avgdailymax_df['MonthDay'] = FDRA_avgdailymax_df['DayOfYear'].apply(lambda x: (datetime(2021, 1, 1) + timedelta(days=x-1)).strftime('%m/%d'))
            index_cols = [col for col in FDRA_avgdailymax_df.columns if col not in ["GACC", "FDRA", "FDRAName", "ObservationTime", "DayOfYear", "MonthDay"]]
            FDRA_avgdailymax_df = FDRA_avgdailymax_df[["GACC", "FDRA", "FDRAName", "ObservationTime", "DayOfYear", "MonthDay"] + index_cols]
            if(season == "true"):
                csv_filename = f"{FDRA}_Season_AvgDailyExtremes.csv"
            if(season == "false"):
                csv_filename = f"{FDRA}_AvgDailyExtremes.csv"
            csv_outpath = f"{FDRA_dir}/{csv_filename}"
            FDRA_avgdailymax_df.to_csv(csv_outpath, index=False)
            #Save DOY dataframe
            #Filter the dataframe down to only the components of interest
            field_keep_list = ["DayOfYear"] + index_list
            FDRA_doy_df = FDRA_doy_df[[col for col in FDRA_doy_df.columns if any(substr in col for substr in field_keep_list)]]
            #Add additional fields
            FDRA_doy_df["GACC"] = gacc
            FDRA_doy_df["FDRA"] = FDRA
            FDRA_doy_df["FDRAName"] = FDRA_name
            FDRA_doy_df['MonthDay'] = FDRA_doy_df['DayOfYear'].apply(lambda x: (datetime(2021, 1, 1) + timedelta(days=x-1)).strftime('%m/%d'))
            index_cols = [col for col in FDRA_doy_df.columns if col not in ["GACC", "FDRA", "FDRAName", "DayOfYear", "MonthDay"]]
            FDRA_doy_df = FDRA_doy_df[["GACC", "FDRA", "FDRAName", "DayOfYear", "MonthDay"] + index_cols]
            #Save FDRA dataframe
            if(season == "true"):
                csv_filename = f"{FDRA}_Season_DOY.csv"
            if(season == "false"):
                csv_filename = f"{FDRA}_DOY.csv"
            csv_outpath = f"{FDRA_dir}/{csv_filename}"
            FDRA_doy_df.to_csv(csv_outpath, index=False)
    # ===== GACC ======
    #If seasonal filter applied, process both regular and seasonal filter data
    if(toggle_season == "true" and period == "Historical"):
        gacc_station_df_list = [gacc_station_df_year_list, gacc_station_df_season_list]
        season_list = ["false", "true"]
    else:
        gacc_station_df_list = [gacc_station_df_year_list]
        season_list = ["false"]
    for i in range(0, len(gacc_station_df_list)):
        curr_gacc_station_df_list = gacc_station_df_list[i]
        season = season_list[i]
        #Concatenate all station dataframes together
        gacc_dailymax_full_df = pd.concat(curr_gacc_station_df_list, ignore_index=True)
        #Some FDRAs use the same station, which introduces duplicates into the GACC station list.
        #Need to account for that here by removing duplicate records
        # Step 1: Count number of unique FDRAs per stationId + ObservationTime
        # Step 2: Identify which rows are part of a conflict group (i.e. FDRA differs within the group)
        # Step 3: For those rows, mark duplicates across stationId + ObservationTime (keep first)
        # Only mark duplicates where there are multiple FDRAs
        # Step 4: Drop rows that are:
        # - Part of a conflicting FDRA group AND
        # - Duplicates (i.e. later in sorted FDRA order)
        # Step 5: Remove them
        FDRA_counts = gacc_dailymax_full_df.groupby(['stationId', 'ObservationTime'])['FDRA'].transform('nunique')
        group_keys = ['stationId', 'ObservationTime']
        first_FDRA = gacc_dailymax_full_df.groupby(group_keys)['FDRA'].transform('first')  # first because df is sorted by FDRA
        gacc_dailymax_full_df_nodupes = gacc_dailymax_full_df[(gacc_dailymax_full_df['FDRA'] == first_FDRA)]
        #Calculate GACC averages
        #This will produce a dataframe of the average daily maximums, across all GACC stations, for each day in the historical record
        gacc_avgdailymax_df = gacc_dailymax_full_df_nodupes.groupby('ObservationTime').agg({
                                    '1HrFM':'mean',
                                    '10HrFM':'mean',
                                    '100HrFM':'mean',
                                    '1000HrFM':'mean',
                                    'KBDI':'mean',
                                    'GSI':'mean',
                                    'WoodyFM':'mean',
                                    'HerbFM':'mean',
                                    'IC':'mean',
                                    'ERC':'mean',
                                    'SC':'mean',
                                    'BI':'mean'
                                     }).reset_index()
        #Create DayOfYear field, and adjust any leap years so all days after Feb 28th are reduced by 1 to yield a max of 365 days/yr
        gacc_avgdailymax_df["DayOfYear"] = pd.to_datetime(gacc_avgdailymax_df["ObservationTime"]).dt.dayofyear
        isleapyear_test = pd.to_datetime(gacc_avgdailymax_df['ObservationTime']).dt.is_leap_year
        ispastfeb_test = pd.to_datetime(gacc_avgdailymax_df['ObservationTime']).dt.month > 2
        gacc_avgdailymax_df["DayOfYear"] = gacc_avgdailymax_df["DayOfYear"] - (isleapyear_test & ispastfeb_test).astype(int)
        #Process DOY (1-365) metrics for FDRA
        #Group the "gacc_avgdailymax_df" dataframe by Day of Year (1-365), and determine the min/max/avg for each
        gacc_doy_df = gacc_avgdailymax_df.groupby('DayOfYear').agg(
            Hr1FM_min=('1HrFM', 'min'),
            Hr1FM_max=('1HrFM', 'max'),
            Hr1FM_avg=('1HrFM', 'mean'),
            Hr10FM_min=('10HrFM', 'min'),
            Hr10FM_max=('10HrFM', 'max'),
            Hr10FM_avg=('10HrFM', 'mean'),
            Hr100FM_min=('100HrFM', 'min'),
            Hr100FM_max=('100HrFM', 'max'),
            Hr100FM_avg=('100HrFM', 'mean'),
            Hr1000FM_min=('1000HrFM', 'min'),
            Hr1000FM_max=('1000HrFM', 'max'),
            Hr1000FM_avg=('1000HrFM', 'mean'),
            KBDI_min=('KBDI', 'min'),
            KBDI_max=('KBDI', 'max'),
            KBDI_avg=('KBDI', 'mean'),
            GSI_min=('GSI', 'min'),
            GSI_max=('GSI', 'max'),
            GSI_avg=('GSI', 'mean'),
            WoodyFM_min=('WoodyFM', 'min'),
            WoodyFM_max=('WoodyFM', 'max'),
            WoodyFM_avg=('WoodyFM', 'mean'),
            HerbFM_min=('HerbFM', 'min'),
            HerbFM_max=('HerbFM', 'max'),
            HerbFM_avg=('HerbFM', 'mean'),
            IC_min=('IC', 'min'),
            IC_max=('IC', 'max'),
            IC_avg=('IC', 'mean'),
            ERC_min=('ERC', 'min'),
            ERC_max=('ERC', 'max'),
            ERC_avg=('ERC', 'mean'),
            SC_min=('SC', 'min'),
            SC_max=('SC', 'max'),
            SC_avg=('SC', 'mean'),
            BI_min=('BI', 'min'),
            BI_max=('BI', 'max'),
            BI_avg=('BI', 'mean')
            ).reset_index()
        #The groupby() above will not allow the field names to begin with numbers. Adjust the Fuel Moisture fields here to match FEMS
        gacc_doy_df.rename(columns={"Hr1FM_min":"1HrFM_min", "Hr1FM_max":"1HrFM_max", "Hr1FM_avg":"1HrFM_avg",
                                    "Hr10FM_min":"10HrFM_min", "Hr10FM_max":"10HrFM_max", "Hr10FM_avg":"10HrFM_avg",
                                    "Hr100FM_min":"100HrFM_min", "Hr100FM_max":"100HrFM_max", "Hr100FM_avg":"100HrFM_avg",
                                    "Hr1000FM_min":"1000HrFM_min", "Hr1000FM_max":"1000HrFM_max", "Hr1000FM_avg":"1000HrFM_avg"}, inplace=True)
        #Save AvgDailyMax dataframe
        #Filter the dataframe down to only the components of interest
        field_keep_list = ["ObservationTime", "DayOfYear"] + index_list
        gacc_avgdailymax_df = gacc_avgdailymax_df[[col for col in gacc_avgdailymax_df.columns if any(substr in col for substr in field_keep_list)]].copy()
        #Rearrange fields, then save as CSV
        gacc_avgdailymax_df["GACC"] = gacc
        gacc_avgdailymax_df['MonthDay'] = gacc_avgdailymax_df['DayOfYear'].apply(lambda x: (datetime(2021, 1, 1) + timedelta(days=x-1)).strftime('%m/%d'))
        index_cols = [col for col in gacc_avgdailymax_df.columns if col not in ["GACC", "ObservationTime", "DayOfYear", "MonthDay"]]
        gacc_avgdailymax_df = gacc_avgdailymax_df[["GACC", "ObservationTime", "DayOfYear"] + index_cols]
        if(season == "true"):
            csv_filename = f"{gacc}_Season_AvgDailyExtremes.csv"
        if(season == "false"):
            csv_filename = f"{gacc}_AvgDailyExtremes.csv"
        csv_outpath = f"{datadir}/{csv_filename}"
        gacc_avgdailymax_df.to_csv(csv_outpath, index=False)
        #Save DOY dataframe
        #Filter the dataframe down to only the components of interest
        field_keep_list = ["DayOfYear"] + index_list
        gacc_doy_df = gacc_doy_df[[col for col in gacc_doy_df.columns if any(substr in col for substr in field_keep_list)]].copy()
        #Add additional fields
        gacc_doy_df["GACC"] = gacc
        gacc_doy_df['MonthDay'] = gacc_doy_df['DayOfYear'].apply(lambda x: (datetime(2021, 1, 1) + timedelta(days=x-1)).strftime('%m/%d'))
        index_cols = [col for col in gacc_doy_df.columns if col not in ["GACC","DayOfYear", "MonthDay"]]
        gacc_doy_df = gacc_doy_df[["GACC","DayOfYear", "MonthDay"] + index_cols]
        #Save GACC dataframe
        if(season == "true"):
            csv_filename = f"{gacc}_Season_DOY.csv"
        if(season == "false"):
            csv_filename = f"{gacc}_DOY.csv"
        csv_outpath = f"{datadir}/{csv_filename}"
        gacc_doy_df.to_csv(csv_outpath, index=False)

def main():
    s = climo_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", choices=PERIODS, default="Historical")
    ap.add_argument("--gacc", action="append", help="only this GACC (repeatable); default: every GACC in the roster")
    args = ap.parse_args()
    for gacc, df_stations in load_roster(s["station_csv"], args.gacc).items():
        calculate_daily_extremes(df_stations, gacc, period_dir(s["outdir"], gacc, args.period),
                                 s["index_list"], toggle(s["season_enabled"]))


if __name__ == "__main__":
    main()
