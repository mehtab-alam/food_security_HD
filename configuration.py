#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:16:59 2024

@author: syed
"""

# =============================================================================#
# Data Directories Structure                                                   #
# =============================================================================#

DATA_DIRECTORY = "data/" # Main data directory 
VAR_DIRECTORY = DATA_DIRECTORY+"data_explicatives/" # Variables directory 
SPATIAL_DIRECTORY = DATA_DIRECTORY+"spatial/" # Spatial data e.g. TIF files
PREPROCESS_DATA_DIR = "preprocessed_data/"
PREPROCESS_FILES ={"timeseries": PREPROCESS_DATA_DIR+"data_timeseries.xlsx", "conjunctural": PREPROCESS_DATA_DIR+"data_conjunctural.xlsx",
                        "structural": PREPROCESS_DATA_DIR+"data_structural.xlsx", "all": PREPROCESS_DATA_DIR+"data_preprocessed.xlsx"}

# =============================================================================#
# Output: Prediction Variables                                                 #
# =============================================================================#

OUTPUT_VARIABLES = ['sca', 'sda']

# =============================================================================#
# Survey Response File                                                         #         
# =============================================================================#
RESPONSE_FILE = "rep_epa_2009-2018.xlsx" # Name of the response file 

# =============================================================================#
# Features Directory                                                           #
# =============================================================================#

FEATURES_DIRECTORY = "features/"

# =============================================================================#
# Spatio-temporal Granularity                                                  #
# =============================================================================#

SPATIAL_TEMPORAL_GRANULARITY = ['REGION', 'PROVINCE', 'COMMUNE', 'ANNEE']

# =============================================================================#
# Variables Lists: Time Series, Conjuctural and structural variables           #
# =============================================================================#
vars_timeseries = ['rainfall', 'maize', 'smt', 'tmax', 'tmin'] # Features used in the code : ['rainfall', 'maize', 'smt', 'tmax', 'tmin', 'ndvi','grains']
vars_conjuctral = [ 'world_bank','weather', 'population', 'ndvi'] # conjunctual variables = ['world_bank', 'meteo', 'pop', 'ndvi']
vars_structural = ['hospital_education', 'voilence_events', 'quality_soil', 'elevation', 'waterways'] # structural variables = ['hosp_educ', 'acled', 'quality_soil', 'elevation', 'waterways']
   
# =============================================================================#
# Spatial Data files                                                           #
# =============================================================================#
PIXEL = "100m"
SPATIAL_TIF_VARS =  {"epa": 'epa_' + PIXEL + '_com.tif', "crop": 'crop_mean_' + PIXEL + '.tif', "forest": 'forest_mean_' + PIXEL + '.tif', 
                     "zones": 'built_mean_' + PIXEL + '.tif'} 

# =============================================================================#
# Temporal Range                                                               #
# =============================================================================#

TIME_RANGE =[2009, 2018]