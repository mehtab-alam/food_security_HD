#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:16:59 2024

@author: syed
"""

import os
import argparse, sys
# =============================================================================#
# Data Directories Structure                                                   #
# =============================================================================#



DATA_DIRECTORY = "data" # Main data directory
VAR_DIRECTORY =   "data_explicatives"  # Variables directory
SPATIAL_DIRECTORY = "spatial"  # Spatial data e.g. TIF files
PREPROCESS_DATA_DIR = "staging_data" #data_intermediate, data_pipeline
OUTPUT_DIR = "output" #data_intermediate, data_pipeline
PREPROCESS_FILES = {
    "timeseries":  "data_timeseries.xlsx",
    "conjunctural_structural":  "data_conjunctural_structural.xlsx",
    "all": "data_preprocessed.xlsx",
}

# =============================================================================#
# Datasets: Countries Data                                                     #
# =============================================================================#
countries = {"burkina_faso":"burkina_faso", "rwanda":"rwanda", "tanzania": "tanzania"}

# =============================================================================#
# Output: Prediction Variables                                                 #
# =============================================================================#

OUTPUT_VARIABLES = {"burkina_faso":{"regression":["sca", "sda"],"classification":["class_sca", "class_sda"]}, 
                    "rwanda": {"regression":["fcs", "hdds"],"classification":["class_fcs", "class_hdds"]}, 
                    "tanzania": {"regression":["fcs", "hdds"],"classification":["class_fcs", "class_hdds"]}}

# =============================================================================#
# Survey Response File                                                         #
# =============================================================================#
RESPONSE_FILE = {"burkina_faso":"rep_epa_2009-2018.xlsx","rwanda":"rep_epa_2006-2021.xlsx",
                 "tanzania":"rep_epa_2011-2023.xlsx"}  # Name of the response file

# =============================================================================#
# Features Directory                                                           #
# =============================================================================#

FEATURES_DIRECTORY =  "features/"

# =============================================================================#
# Spatio-temporal Granularity                                                  #
# =============================================================================#

SHAPE_FILE = {"burkina_faso": os.path.join("shape","bf_shape.shp"),"rwanda": os.path.join("shape","rwanda_shape.shp"),
                 "tanzania":os.path.join("shape","tz_shape.shp")} 

SPATIAL_TEMPORAL_GRANULARITY = {"burkina_faso":["REGION", "PROVINCE", "COMMUNE", "ANNEE"], 
                                "rwanda": [ "province", "district", "year"],
                                "tanzania": [ "region", "district", "year"]}

SPATIAL_GRANULARITY = {"burkina_faso":SPATIAL_TEMPORAL_GRANULARITY['burkina_faso'][:-1], 
                                "rwanda": SPATIAL_TEMPORAL_GRANULARITY['rwanda'][:-1],
                                "tanzania": SPATIAL_TEMPORAL_GRANULARITY['tanzania'][:-1]}


FINE_SP_GRANULARITY = {"burkina_faso": SPATIAL_TEMPORAL_GRANULARITY['burkina_faso'][2], 
                                "rwanda": SPATIAL_TEMPORAL_GRANULARITY['rwanda'][1],
                                "tanzania": SPATIAL_TEMPORAL_GRANULARITY['tanzania'][1]}

ID_REGIONS = {"burkina_faso":"ID_COM", "rwanda":"DISTRICT_ID", "tanzania": "DISTRICT_ID"}

TEMPORAL_GRANULARITY = {"burkina_faso":"ANNEE", "rwanda":"year", "tanzania": "year"}

time_window = {"burkina_faso": {'start': 'may', 'end':'November', 'applied_year': 'same'} , "rwanda":{'start': 'march', 'end':'april', 'applied_year': 'previous'},
               "tanzania": {'start': 'march', 'end':'april', 'applied_year': 'previous'}}

cnn_settings = {"burkina_faso": {'length': 10, 'step': 30} , "rwanda":{'length': 6, 'step': 3},
               "tanzania": {'length': 8, 'step': 4} }


# =============================================================================#
# Variables Lists: Time Series, Conjuctural and structural variables           #
# =============================================================================#
vars_timeseries = [
    "rainfall",
    "maize",
    "smt",
    "tmax",
    "tmin",
    #"beans",
    #"rice"
]  # Features used in the code : ['rainfall', 'maize', 'smt', 'tmax', 'tmin', 'ndvi','grains']
vars_conjuctral = [
    "world_bank",
    "weather",
    "population",
    "ndvi",
]  # conjunctual variables = ['world_bank', 'meteo', 'pop', 'ndvi']
vars_structural = [
    "hospital_education",
    "voilence_events",
    "quality_soil",
    "elevation",
    #"waterways",
]  # structural variables = ['hosp_educ', 'acled', 'quality_soil', 'elevation', 'waterways']

# =============================================================================#
# Spatial Data files                                                           #
# =============================================================================#
PIXEL = "100m"
SPATIAL_TIF_VARS = {
    "epa": "epa_" + PIXEL + "_com.tif",
    "crop": "crop_mean_" + PIXEL + ".tif",
    "forest": "forest_mean_" + PIXEL + ".tif",
    "zones": "built_mean_" + PIXEL + ".tif",
}

# =============================================================================#
# Temporal Range                                                               #
# =============================================================================#

TIME_RANGE = [2009, 2018]
