#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 00:02:35 2024

@author: syed
"""


"""
Installation libraries required

pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install xlrd
pip install openpyxl
"""


"""
Importing necessary libraries
"""
from logger import log
from logger import Logs
import calendar
import pandas as pd
import numpy as np
import os
import re
import configuration as conf
from osgeo import gdal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from shape_to_features import shape_to_raster
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
warnings.simplefilter("ignore")

def get_temporal_indices(df, column, value):
    # Get the indices of the rows where the column matches the value
    matching_indices = df.index[df[column] == value].tolist()
    
    # Get the indices of the rows where the column does not match the value
    non_matching_indices = df.index[df[column] != value].tolist()
    
    return matching_indices, non_matching_indices

def get_saptio_temporal_indices(country, df, column, id_list):
    # Determine the split point (85% of the original list length)
    split_point = int(0.85 * len(id_list))
    
    # Randomly sample 85% of the values
    values_train = random.sample(id_list, split_point)
    log(country , f"Commune/Districts for Training Data : {values_train}")
    # Calculate the remaining 15% of the values
    values_test = list(set(id_list) - set(values_train))
    log(country , f"Commune/Districts for Test Data : {values_test}")
    # Get the indices of the rows where the column matches the value
    train_matching_indices = df.index[df[column].isin(values_train)].tolist()
    
    # Get the indices of the rows where the column does not match the value
    test_matching_indices = df.index[df[column].isin(values_test)].tolist()
    
    log(country , f"Train Indices Size : {len(train_matching_indices)}, Test Indices Size : {len(test_matching_indices)}")
    
    return test_matching_indices, train_matching_indices

def get_month_number(month):
    month = month.capitalize()  
    if month in list(calendar.month_abbr): 
        return list(calendar.month_abbr).index(month)
    if month in list(calendar.month_name):
        return list(calendar.month_name).index(month)

def get_time_range(time_window):
  time_window_ret =[]
  if time_window['applied_year'] == 'same':
    start_c, end_c = get_month_number(time_window['start']), get_month_number(time_window['end'])
    time_window_ret.append(list(range(start_c, end_c+1)))
    time_window_ret.append(list(range(start_c, end_c+1)))
    return time_window_ret
  if time_window['applied_year'] == 'previous':
    start_c, end_c = get_month_number(time_window['start']), get_month_number('December')
    start_p, end_p = get_month_number('January'), get_month_number(time_window['end'])
    time_window_ret.append(list(range(start_p, end_p+1)))
    time_window_ret.append(list(range(start_c, end_c+1)))
    return time_window_ret

# Function to rename columns
def rename_columns(country, df, year, column_name, is_current):
    renamed_columns = {}
    for col in df.columns:
        if col not in conf.SPATIAL_GRANULARITY[country]:
            match_term = re.search(re.escape(column_name) + r'_*(\d+)', col) 
            
            if match_term:
              month = int(match_term.group(1))
              renamed_columns[col] = f'{column_name}_{month}(t-{1 if is_current else 0})'
    return df.rename(columns=renamed_columns)


def visualize_distribution(country, df, variable):
    log(country, f"Summary Statistics of variable {variable}")
    os.makedirs(os.path.join(conf.plots[country]), exist_ok=True)
    
    # Visualizations
    num_columns = len(df.columns)
    num_rows = (num_columns + 1) // 2
    
    plt.figure(figsize=(15, 5 * num_rows))
    # Histograms for each feature
    for i, column in enumerate(df.columns, 1):
        plt.subplot(num_rows, 2, i)
        sns.histplot(df[column], kde=True)
        plt.title(f'{variable} Distribution')
    
    # plt.savefig(os.path.join(conf.plots[country]), f'{variable}_histogram.png'))
    # plt.close()
    
    # Box plots for the dataframe
    plt.figure(figsize=(15, 5))
    sns.boxplot(data=df)
    plt.title(f'Box plot of features {variable}')
    plt.savefig(os.path.join(conf.plots[country], f'{variable}_boxplots.png'))
    plt.close()
    
    # Checking overall skewness and standard deviation
    overall_skewness = df.skew().mean()
    overall_std_dev = df.std().mean()
    
    log(country, f'Overall Skewness of {variable}: {overall_skewness}')
    log(country, f'Overall Standard Deviation of {variable}: {overall_std_dev}')
    
    # Choosing a scaler based on overall distribution
    if abs(overall_skewness) > 1:
        scaler = RobustScaler()
        log(country, f'Using RobustScaler for {variable} (due to high overall skewness)')
    elif overall_std_dev > 1:
        scaler = StandardScaler()
        log(country, f'Using StandardScaler for {variable} (due to high overall standard deviation)')
    else:
        scaler = MinMaxScaler()
        log(country, f'Using MinMaxScaler for {variable} (due to relatively low overall standard deviation and normal distribution)')
    
    # Applying the chosen scaler to the entire dataframe
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    # Visualize scaled data distributions
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(scaled_df.columns, 1):
        plt.subplot(num_rows, 2, i)
        sns.histplot(scaled_df[column], kde=True)
        plt.title(f'Scaled {variable} Distribution')
    
    plt.savefig(os.path.join(conf.plots[country], f'{variable}_hostogram_scaled.png'))
    plt.close()
    return scaler

# ======================================================================================#
# Variables transformation for Tanzania/Rawanda for year t-0, t-1 from May-November     #
# ======================================================================================#
def rename_timeseries_variables(country, df, years, column_name):
    merged_df = pd.DataFrame()
    for year in years:
        year_t_minus_1, year_t = year-1, year
       
    
    
        # Filter columns using regex
        cols_to_keep = conf.SPATIAL_TEMPORAL_GRANULARITY[country] + [col for col in df.columns if re.match(re.escape(column_name) + r'_*(\d+)$', col)] # We have to change it as to which months for time-series
        df_filtered = df[cols_to_keep]
        
        # Split the data into year_t_minus_1 and year_t
        df_t_minus_1 = df_filtered[df_filtered[conf.TEMPORAL_GRANULARITY[country]] == year_t_minus_1].drop(columns=[conf.TEMPORAL_GRANULARITY[country]])
        df_t = df_filtered[df_filtered[conf.TEMPORAL_GRANULARITY[country]] == year_t].drop(columns=[conf.TEMPORAL_GRANULARITY[country]])
        
        
        # Rename columns
        df_t_minus_1_renamed = rename_columns(country, df_t_minus_1, year_t_minus_1, column_name, False)
        df_t_renamed = rename_columns(country, df_t, year_t, column_name, True)
       
        # Merge the two DataFrames on 'province' and 'district'
        
        merged_pair_df = pd.merge(df_t_renamed, df_t_minus_1_renamed, on=conf.SPATIAL_GRANULARITY[country])
        # Add the year column back
       
        merged_pair_df.insert(loc=0, column=conf.TEMPORAL_GRANULARITY[country], value=year_t)
        # Append the merged result to the main DataFrame
        merged_df = pd.concat([merged_df, merged_pair_df], ignore_index=True)
    return merged_df

# =============================================================================#
# Data (variables) loading method: Load data (variables) from Excel files      #
# =============================================================================#
def read_add_variables(data_rep, country, var_type, criteria):

    for D in var_type:
        log(country, f"Reading Variable {D}...")
        if not os.path.exists(os.path.join(conf.DATA_DIRECTORY , country, conf.VAR_DIRECTORY , "data_" + D + ".xlsx")):
            log(country, "*****File doesn't exists: "+ os.path.join(conf.DATA_DIRECTORY , country, conf.VAR_DIRECTORY , "data_" + D + ".xlsx"), Logs.WARNING)
        else:    
            data_temp = pd.read_excel(
               os.path.join(conf.DATA_DIRECTORY , country, conf.VAR_DIRECTORY , "data_" + D + ".xlsx")
            )# Impoort variables
            #log(country, f"Path of Reading variable {D} with columns: "+ str(list(data_temp.columns)), Logs.INFO)
            if country != conf.countries['burkina_faso'] and var_type == conf.vars_timeseries:
                #print('variables check:', data_rep[conf.TEMPORAL_GRANULARITY[country]].unique(), D)
                data_temp = rename_timeseries_variables(country, data_temp, data_rep[conf.TEMPORAL_GRANULARITY[country]].unique(), D)
            data_rep = pd.merge(data_rep, data_temp, how="left", on=criteria)
    log(country, "criteria of Merging:"+ str(criteria))    
    return data_rep

# =======================================================================================#
# format time series variables and its renaming for Rawanda and Tanzania                 #
# =======================================================================================#
def filter_variables_(country, data_aggr, var_type):
    time_window = get_time_range(conf.time_window[country])
    cols_to_keep = []
    for i in [0,1]:  # année t-1 et t
        log(country, "Time window: "+ str(time_window[i]))
        for j in time_window[i-1]:  # 
            for D in var_type:
               pattern = r"{0}.*{1}\(t-{2}".format(D, j, i)
               # Filter columns using the regex and compute max across these columns
               filtered_df = data_aggr.filter(regex=pattern)
               new_columns = "{0}{1}t-{2}".format(D, j, i)

               # Compute max across the filtered columns
               data_aggr[new_columns] = filtered_df.max(axis=1)
               cols_to_keep.append(new_columns)
    data_aggr = data_aggr[cols_to_keep]
    return data_aggr

# =======================================================================================#
# Standardization/Formating Method: Extract,format time series variables and its renaming#
# =======================================================================================#
def filter_variables(country, data_aggr, var_type):
    if country != conf.countries['burkina_faso']:
        return filter_variables_(country, data_aggr, var_type)
    time_window = get_time_range(conf.time_window[country])
    cols_to_keep = []
    for i in [0,1]:  # année t-1 et t
       log(country, "Time window: "+ str(time_window[i]))
       for j in time_window[i-1]:  # 
           for D in var_type:
               if os.path.exists(os.path.join(conf.DATA_DIRECTORY , country, conf.VAR_DIRECTORY , "data_" + D + ".xlsx")):
                    if D == 'maize':
                        pattern = r"{1}{0}-.*\(.*t-{2}".format(D, j, i)
                    elif D == "tmin" or D == "tmax":
                        t_var = D.replace("t", "t_")
                        pattern = r"{0}.*{1}\(t-{2}".format(t_var, j, i)
                    else:
                        pattern = r"{0}.*{1}\(.*t-{2}".format(D, j, i)
                    
                    filtered_df = data_aggr.filter(regex=pattern)
                    new_columns = "{0}{1}t-{2}".format(D, j, i)
                    data_aggr[new_columns] = filtered_df.max(axis=1)
                    cols_to_keep.append(new_columns)
    
    data_aggr = data_aggr[cols_to_keep]
    return data_aggr


# ====================================================================================================#
# Normalization Method: Normalization of variables using mean and standard deviation: (X-X.mean)/X.std#
# ====================================================================================================#


def normalize(country, data, var_type):
    log(country, f"{var_type} variables to Normalize:"+ str(list(data.columns)))
    for v in var_type:
        # Get the columns that match the substring
        columns_to_scale = [col for col in data.columns if col.startswith(v)]
        # data[columns_to_scale] = (
        #    data[columns_to_scale]
        #    - data[columns_to_scale].stack().mean()
        #    ) / data[columns_to_scale].stack().std()
        
        # columns_to_scale = data.columns.str.startswith(v)
        # # Initialize the MinMaxScaler
        if len(columns_to_scale) > 0:
            log(country, "Column to scale:"+ str(columns_to_scale))
            df = data[columns_to_scale]
            #visualize_distribution(country, df, v)
            scaler = MinMaxScaler(feature_range=(-1,1))
            # Apply the scaler to the selected columns and update the dataframe
            data.loc[:, columns_to_scale] = scaler.fit_transform(data.loc[:, columns_to_scale])

    return data


# ==========================================================================================#
# Return output (Y) Numpy Array                                                             #
# ==========================================================================================#
def output_Y(country, data_rep, output):
    if os.path.exists(os.path.join(conf.np_processed[country], output + ".npy")):
        return np.load(os.path.join(conf.np_processed[country], output + ".npy"))
    data_Y = []
    for i in range(len(data_rep)):
        data_Y.append([data_rep[output][i]])
    data_Y = np.array(data_Y)
    os.makedirs(os.path.join(os.path.join(conf.np_processed[country])), exist_ok=True)
    np.save(os.path.join(conf.np_processed[country], output + ".npy"), data_Y)
    return data_Y

# ==========================================================================================#
# Function to extract the year from the date string                                         #
# ==========================================================================================#

def transform_year(date_str):
    # Split the date string by either '/' or '-' and return the first part (year)
    if isinstance(date_str, str):
        return int(date_str.split('/')[0].split('-')[1])
    return int(date_str)


# ==========================================================================================#
# Export the preprocessed variables                                                         #
# ==========================================================================================#
def export(country, data_timeseries, data_conjunctural_structural, data_rep_columns):
    log(country, f"Saving of {country} time-series variables at: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.PREPROCESS_FILES["timeseries"]),Logs.INFO)
    data_timeseries.to_excel(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.PREPROCESS_FILES["timeseries"]))
    log(country, f"Saving of {country} conjunctural/structural variables at: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.PREPROCESS_FILES["conjunctural_structural"]),Logs.INFO)
    data_conjunctural_structural.to_excel(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.PREPROCESS_FILES["conjunctural_structural"]))
    
    return


# ==========================================================================================#
# Create directories for features in case if doesn't exists                                 #
# ==========================================================================================#
def create_directories(r_split, country, algorithm):
    os.makedirs(
        os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_" + conf.OUTPUT_VARIABLES[country]['classification'][0]), exist_ok=True
    )
    os.makedirs(
        os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY , "features_" + conf.OUTPUT_VARIABLES[country]['classification'][1]), exist_ok=True
    )
    os.makedirs(
        os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_" + conf.OUTPUT_VARIABLES[country]['regression'][0]), exist_ok=True
    )
    os.makedirs(
        os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY , "features_" + conf.OUTPUT_VARIABLES[country]['regression'][1]), exist_ok=True
    )

# Define a helper function to extract patches
def extract_patch(data, i, j, length):
    half_length = int(length // 2)
    return data[i - half_length:i + half_length , j - half_length:j + half_length]


# Split the Train/Test data 
def train_test_data_split(country, df_response, tt_split, data_X_timeseries, data_X_CS, data_Y_SCA, data_Y_SDA, data_Y_CLASS_SCA, data_Y_CLASS_SDA, data_W, dataInfo, year, r_split):
    if tt_split == 'percentage':
        log(country, 'Percentage train/test is selected...')
        (X_train_timeseries, X_test_timeseries, X_train_CS, X_test_CS, y_train_SCA, y_test_SCA, y_train_SDA, y_test_SDA, y_train_class_SCA, y_test_class_SCA, y_train_class_SDA, 
         y_test_class_SDA, w_train, w_test, info_train, info_test) = train_test_split(data_X_timeseries, data_X_CS, data_Y_SCA, data_Y_SDA, data_Y_CLASS_SCA, data_Y_CLASS_SDA, data_W, dataInfo, test_size=0.15)#, random_state=r_split)
    else: 
        if tt_split == 'temporal':
            log(country, 'Temporal train/test is selected...')
            matching_indices, non_matching_indices = get_temporal_indices(df_response, conf.TEMPORAL_GRANULARITY[country], year)
        else:
            log(country, 'Spatio-temporal train/test is selected...')
            id_regions = df_response[conf.ID_REGIONS[country]].unique().tolist()
            matching_indices, non_matching_indices = get_saptio_temporal_indices(country, df_response, conf.ID_REGIONS[country], id_regions)
       
        X_train_timeseries = np.take(data_X_timeseries, non_matching_indices, axis = 0)
        X_test_timeseries = np.take(data_X_timeseries, matching_indices, axis = 0)
        X_train_CS = np.take(data_X_CS, non_matching_indices, axis = 0)
        X_test_CS = np.take(data_X_CS, matching_indices, axis = 0)
        y_train_SCA = np.take(data_Y_SCA, non_matching_indices, axis = 0)
        y_test_SCA = np.take(data_Y_SCA, matching_indices, axis = 0)
        y_train_SDA = np.take(data_Y_SDA, non_matching_indices, axis = 0)
        y_test_SDA = np.take(data_Y_SDA, matching_indices, axis = 0)
        y_train_class_SCA = np.take(data_Y_CLASS_SCA, non_matching_indices, axis = 0)
        y_test_class_SCA = np.take(data_Y_CLASS_SCA, matching_indices, axis = 0)
        y_train_class_SDA = np.take(data_Y_CLASS_SDA, non_matching_indices, axis = 0)
        y_test_class_SDA = np.take(data_Y_CLASS_SDA, matching_indices, axis = 0)
        w_train = np.take(data_W, non_matching_indices, axis = 0)
        w_test = np.take(data_W, matching_indices, axis = 0)
        info_train = np.take(dataInfo, non_matching_indices, axis = 0)
        info_test = np.take(dataInfo, matching_indices, axis = 0)
    
    return (X_train_timeseries, X_test_timeseries, X_train_CS, X_test_CS, y_train_SCA, y_test_SCA, y_train_SDA, y_test_SDA, y_train_class_SCA, y_test_class_SCA, y_train_class_SDA, 
         y_test_class_SDA, w_train, w_test, info_train, info_test) 
  

def processed_timeseries(country, data_rep, algorithm, data_rep_columns):
     if os.path.exists(os.path.join(conf.np_processed[country], "timeseries.npy")):
         return np.load(os.path.join(conf.np_processed[country], "timeseries.npy"))
     os.makedirs(os.path.join(conf.np_processed[country]), exist_ok=True)
     data_aggregation_timeseries = read_add_variables(data_rep, country, conf.vars_timeseries, conf.SPATIAL_TEMPORAL_GRANULARITY[country])  
     data_aggregation_timeseries.dropna(subset=conf.OUTPUT_VARIABLES[country][algorithm], how="all", inplace=True)  # Remove rows with response variable Nan
     data_aggregation_timeseries = filter_variables(country,data_aggregation_timeseries, conf.vars_timeseries )  # Standardization of variables and column names
     data_aggregation_timeseries = normalize(country,data_aggregation_timeseries, conf.vars_timeseries)# Normalize variables by mean/STD data_X_timeseries = np.array(
     data_aggregation_timeseries.to_excel(os.path.join(conf.np_processed[country], "timeseries.xlsx"))
     data_X_timeseries = np.array(data_aggregation_timeseries.loc[:, ~data_aggregation_timeseries.columns.isin(data_rep_columns)])
     os.makedirs(os.path.join(os.path.join(conf.np_processed[country])), exist_ok=True)
     np.save(os.path.join(conf.np_processed[country], "timeseries.npy"), data_X_timeseries)
     return data_X_timeseries
 
def processed_conjunctural_structural(country, data_rep, algorithm, data_rep_columns):
     if os.path.exists(os.path.join(conf.np_processed[country], "conjunctural.npy")):
         return np.load(os.path.join(conf.np_processed[country], "conjunctural.npy"))
     data_aggregation = data_rep
     data_temp = pd.read_excel(os.path.join(conf.DATA_DIRECTORY, country, conf.VAR_DIRECTORY , "data_world_bank.xlsx"))  # Conjunctural variable handle explicitly
     data_aggregation = pd.merge(data_aggregation, data_temp, how="left", on=[conf.TEMPORAL_GRANULARITY[country]]) # read conjunctural variables
    
     data_aggregation = read_add_variables(data_aggregation, country, [i for i in conf.vars_conjuctral if i != "world_bank"],conf.SPATIAL_TEMPORAL_GRANULARITY[country],)
     data_aggregation = read_add_variables(data_aggregation, country, conf.vars_structural,[i for i in conf.SPATIAL_GRANULARITY[country]],)
     column_to_normalize = list(set(data_aggregation.columns) - set(data_rep_columns))
     for v in column_to_normalize:
        data_aggregation[v] = (data_aggregation[v] - data_aggregation[v].mean())/data_aggregation[v].std()
     log(country, f"Columns shape {len(column_to_normalize)}:"+ str(column_to_normalize))
     data_X_CS = np.array(data_aggregation.loc[:, column_to_normalize])
     
     os.makedirs(os.path.join(os.path.join(conf.np_processed[country])), exist_ok=True)
     np.save(os.path.join(conf.np_processed[country], "conjunctural.npy"), data_X_CS)
     return data_X_CS
 
def process_spatial(country, data_rep):
    if os.path.exists(os.path.join(conf.np_processed[country], "info_spatial.npy")):
         return np.load(os.path.join(conf.np_processed[country], "info_spatial.npy"))
    dataInfo = []
    for i in range(len(data_rep)):
        dataInfo.append([data_rep[conf.TEMPORAL_GRANULARITY[country]][i],data_rep[conf.ID_REGIONS[country]][i],])
    dataInfo = np.array(dataInfo)
    os.makedirs(os.path.join(os.path.join(conf.np_processed[country])), exist_ok=True)
    np.save(os.path.join(conf.np_processed[country], "info_spatial.npy"), dataInfo)
    return dataInfo

def process_weight(country, data_rep):
    if os.path.exists(os.path.join(conf.np_processed[country], "data_W.npy")):
       return np.load(os.path.join(conf.np_processed[country], "data_W.npy")) 
    data_rep["count"] = (np.sqrt(data_rep["count"]) / np.sqrt(data_rep["count"]).sum())
    data_W = []  # Weight Array
    for i in range(len(data_rep)):
        data_W.append([data_rep["count"][i]])
    data_W = np.array(data_W)
    os.makedirs(os.path.join(os.path.join(conf.np_processed[country])), exist_ok=True)
    np.save(os.path.join(conf.np_processed[country], "data_W.npy"), data_W)
    return data_W



def process_rasters(country, algorithm , data_rep, rep, years):
    if os.path.exists(os.path.join(conf.np_processed[country], "info_pix_cnn.npy")):
       info_pix_cnn = np.load(os.path.join(conf.np_processed[country], "info_pix_cnn.npy")) 
       dataX_CNN = np.load(os.path.join(conf.np_processed[country], "dataX_CNN.npy")) 
       dataY_CNN = np.load(os.path.join(conf.np_processed[country], "dataY_CNN.npy")) 
       log(country, f'Shape of output: {dataY_CNN.shape}')
       return info_pix_cnn, dataX_CNN, dataY_CNN
       
    crop = gdal.Open(os.path.join(conf.DATA_DIRECTORY, country, conf.SPATIAL_DIRECTORY, conf.SPATIAL_TIF_VARS["crop"]))  # import crop Data
    log(country, "Loaded Rasters: "+os.path.join(conf.DATA_DIRECTORY, country, conf.SPATIAL_DIRECTORY, conf.SPATIAL_TIF_VARS["crop"]), Logs.INFO)
    forest = gdal.Open(os.path.join(conf.DATA_DIRECTORY, country, conf.SPATIAL_DIRECTORY, conf.SPATIAL_TIF_VARS["forest"]))  # import forest data
    log(country, "Loaded Rasters: "+os.path.join(conf.DATA_DIRECTORY, country, conf.SPATIAL_DIRECTORY, conf.SPATIAL_TIF_VARS["forest"]), Logs.INFO)
    zones = gdal.Open(os.path.join(conf.DATA_DIRECTORY, country, conf.SPATIAL_DIRECTORY, conf.SPATIAL_TIF_VARS["zones"]))  # import constructed zones data
    log(country, "Loaded Rasters: "+os.path.join(conf.DATA_DIRECTORY, country, conf.SPATIAL_DIRECTORY, conf.SPATIAL_TIF_VARS["zones"]), Logs.INFO)
  
    # =============================================================================#
    #     Numpy Arrays of Tif Files: epa, crop, forest and zones                   #
    # =============================================================================#
    
    
   
    #raster_com = np.array(raster_com.ReadAsArray())
    crop = np.array(crop.ReadAsArray())
    forest = np.array(forest.ReadAsArray())
    zones = np.array(zones.ReadAsArray())
    log(country, "Resulting Numpy Array for Raster replacement:"+ str(crop.shape))
    #log(country, "ID COM/DISTRICTS:"+ str(list(data_response_src[conf.ID_REGIONS[country]].unique())))
    raster_com = shape_to_raster(country, data_rep, conf.FINE_SP_GRANULARITY[country], conf.ID_REGIONS[country], crop.shape)
    dictrep = dict()  # dictionary of answers for each year
    dictpop = dict()  # dictionary of population for each year
    
    unique_values, counts = np.unique(raster_com, return_counts=True)

    # import des rasters réponse et population de chaque année
    for annee in years:  # import des rasters réponse et population
        data_response_year = data_rep[data_rep[conf.TEMPORAL_GRANULARITY[country]] == annee]
        dictrep[annee] = shape_to_raster(country, data_response_year, 
                                          conf.FINE_SP_GRANULARITY[country], 
                                          rep, crop.shape)
        dictpop[annee] = gdal.Open(os.path.join(conf.DATA_DIRECTORY, country, conf.SPATIAL_DIRECTORY, 
             "population_"
            + conf.PIXEL
            + "/population_"
            + str(annee)
            + ".tif"
        ))
        log(country, "Loaded Rasters: "+os.path.join(conf.DATA_DIRECTORY, country, conf.SPATIAL_DIRECTORY, 
             "population_"
            + conf.PIXEL
            + "/population_"
            + str(annee)
            + ".tif"
        ), Logs.INFO)
        # Conversion into Numpy Array per year
        dictpop[annee] = np.array(dictpop[annee].ReadAsArray())
        
        # Replacement of Zero-value pixel to NAN value
        dictrep[annee][dictrep[annee] <= 0] = np.nan
        dictpop[annee][dictpop[annee] < 0] = np.nan
       
        # Normalization of pixelated poulation (X-mean/std)
        # dictpop[annee] = dictpop[annee] - np.nanmean(dictpop[annee]) / np.nanstd(
        #     dictpop[annee]
        # )
        #scaler = MinMaxScaler(feature_range=(-1,1))
        scaler = RobustScaler()
        dictpop[annee] = scaler.fit_transform(dictpop[annee])

    raster_com[raster_com <= 0] = np.nan
    # ================================================================================================================#
    #     Listing the variables for CNN                                                                               #
    # =============================================================================================================== #
   
    info_pix_cnn = (
        []
    )  # data on the pairs (municipality, year) associated with each pixel
    dataX_CNN = []  # list of population pixel patches
    dataY_CNN = []  # list of response pixels
    length = conf.cnn_settings[country]['length']  # length of patches
    #step = conf.cnn_settings[country]['step']  # distance between 2 selected pixels
    no_of_features = 50000
    #feature_reduction = 8 if algorithm == 'classification' else 4
    feature_reduction = 4
    no_of_pixels = crop.shape[0] * crop.shape[1]
    step = int(math.sqrt(feature_reduction * no_of_pixels/no_of_features))
    log(country, f"Step selected for {country} is: {step}")
    # ================================================================================================================#
    #     Filling the variables declared for CNN                                                                      #
    # =============================================================================================================== #
    rows = range(int(length / 2), dictrep[annee].shape[0] - int(length / 2), step)
    cols = range(int(length / 2), dictrep[annee].shape[1] - int(length / 2), step)
    count = 0
    for year in years:
        for row in rows:
            for col in cols:
                if np.isnan(dictrep[year][row, col]):
                    continue
                if np.isnan(extract_patch(raster_com, row, col, length)).any():
                    continue
                if np.isnan(extract_patch(dictpop[annee], row, col, length)).any():
                    continue
                
                info_pix_cnn.append([year, raster_com[row, col], len(info_pix_cnn)])
                dataX_CNN.append([
                    extract_patch(dictpop[annee], row, col, length),
                    extract_patch(crop, row, col, length),
                    extract_patch(forest,row, col, length),
                    extract_patch(zones, row, col, length)
                ])
                dataY_CNN.append([dictrep[year][row, col]])
                count = count+1
    log(country, "No. of CNN features:"+ str(count))   
    np.save(os.path.join(conf.np_processed[country], "info_pix_cnn.npy"), info_pix_cnn)
    np.save(os.path.join(conf.np_processed[country], "dataX_CNN.npy"), dataX_CNN)
    np.save(os.path.join(conf.np_processed[country], "dataY_CNN.npy"), dataY_CNN)
    return info_pix_cnn, dataX_CNN, dataY_CNN

# ==========================================================================================#
# Main Preprocessing Method: Data processing for processing of numerical and spatial data   #
# ==========================================================================================#
def preprocess(rep, r_split, country, algorithm, tt_split):

    pd.set_option("display.max_columns", None)
    log(country, f"Preprocessing of {country} data started ...", Logs.INFO)
    log(country, "Path of Ground Truth file: " + os.path.join(
        conf.DATA_DIRECTORY, country, conf.RESPONSE_FILE[country]), Logs.INFO)
    data_response_src = pd.read_excel(os.path.join(
        conf.DATA_DIRECTORY, country, conf.RESPONSE_FILE[country]))  # Read Survey Data
    data_response_src[conf.TEMPORAL_GRANULARITY[country]] = data_response_src[conf.TEMPORAL_GRANULARITY[country]].apply(transform_year) # Transform year
    
    data_rep_columns = list(
        data_response_src.columns
    )  # Get the name of Response columns
    #log(country, "Column Names: " + str(list(data_rep_columns)), Logs.INFO)
    
    data_response_src.loc[:, conf.ID_REGIONS[country]] = pd.factorize(data_response_src[conf.FINE_SP_GRANULARITY[country]])[0] + 1 if conf.ID_REGIONS[country] not in data_rep_columns else data_response_src[conf.ID_REGIONS[country]]
    
    data_rep_columns = list(
        data_response_src.columns
    ) 
    log(country, 'Columns of Ground Truth Data:'+ str(data_rep_columns))
    data_rep = data_response_src  # Keeping the original response data as it is...
    
    
    
    # =============================================================================#
    #     Preprocessing of Time-series variables data                              #
    # =============================================================================#
    
    data_X_timeseries = processed_timeseries(country, data_rep, algorithm, data_rep_columns)
    log(country, f"Time-series variables Numpy array with shape: {data_X_timeseries.shape}", Logs.INFO)
    
    # =============================================================================#
    #     Preprocessing of Conjuntural+structural variables data                              #
    # =============================================================================#
    
    data_X_CS = processed_conjunctural_structural(country, data_rep, algorithm, data_rep_columns)
    log(country, f"Conjunctural+Structural variables Numpy array with shape: {data_X_CS.shape}", Logs.INFO)
    
    # =============================================================================#
    #     Storing Commune information per year for CNN Processing                  #
    # =============================================================================#
    dataInfo = process_spatial(country, data_rep)
    log(country, f"Spatial Info variables Numpy array with shape: {dataInfo.shape}", Logs.INFO)
    
    
    # =============================================================================#
    #     Storing output information (Y) in numpy array = ['sca','sda']            #
    # =============================================================================#
    create_directories(r_split, country, algorithm)
    data_Y_SCA = output_Y(country, data_rep, conf.OUTPUT_VARIABLES[country]['regression'][0])  # Return output variable (SCA) Numpy Array
    data_Y_SDA = output_Y(country, data_rep, conf.OUTPUT_VARIABLES[country]['regression'][1])  # Return output variable (SDA) Numpy Array
    data_Y_CLASS_SCA = output_Y(country, data_rep, conf.OUTPUT_VARIABLES[country]['classification'][0])  # Return output variable (SCA Class) Numpy Array
    data_Y_CLASS_SDA = output_Y(country, data_rep, conf.OUTPUT_VARIABLES[country]['classification'][1])  # Return output variable (SDA Class) Numpy Array
    log(country, f"Output variables Numpy array with shape: {data_Y_SCA.shape}", Logs.INFO)
   
   
    # =================================================================================================#
    # Weight storage (W) (for each observation: W = root(number of households)/root(total households)) #
    # =================================================================================================#
    data_W = process_weight(country, data_rep)
    log(country, f"Weight Numpy array with shape: {data_W.shape}", Logs.INFO)
    # =================================================================================================#
    # Train/Test data split                                                                            #
    # =================================================================================================#
    years= list(data_response_src[conf.TEMPORAL_GRANULARITY[country]].unique())
    log(country, "Years list: "+ str(list(data_response_src[conf.TEMPORAL_GRANULARITY[country]].unique())))
    
    #tt_split = "temporal" # percentage
    
    (X_train_timeseries,
    X_test_timeseries,
    X_train_CS,
    X_test_CS,
    y_train_SCA,
    y_test_SCA,
    y_train_SDA,
    y_test_SDA,
    y_train_class_SCA,
    y_test_class_SCA,
    y_train_class_SDA,
    y_test_class_SDA,
    w_train,
    w_test,
    info_train,
    info_test) = train_test_data_split(country, data_response_src, tt_split, data_X_timeseries, data_X_CS, data_Y_SCA, data_Y_SDA, data_Y_CLASS_SCA, data_Y_CLASS_SDA, data_W, dataInfo, max(years), r_split)
    log(country, f'Shape of train/test data: {X_train_timeseries.shape}, {X_test_timeseries.shape}, {info_train.shape}, {info_test.shape}, {y_train_SCA.shape},{y_test_SCA.shape}')   
    
    
    
    # save explicative variables X
    
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_train.npy"), X_train_timeseries)
    log(country, "Saving data: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_train.npy"), Logs.INFO)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_test.npy"), X_test_timeseries)
    log(country, "Saving data: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_test.npy"), Logs.INFO)
    
    # save response Y (FCS+ HDDS with Classes)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['regression'][0],"y_train.npy"), y_train_SCA)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['regression'][0],"y_train.npy"), Logs.INFO)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['regression'][0],"y_test.npy"), y_test_SCA)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['regression'][0],"y_test.npy"), Logs.INFO)
    
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['classification'][0],"y_train.npy"), y_train_class_SCA)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['classification'][0],"y_train.npy"), Logs.INFO)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['classification'][0],"y_test.npy"), y_test_class_SCA)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['classification'][0],"y_test.npy"), Logs.INFO)
   
    
    
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['regression'][1],"y_train.npy"), y_train_SDA)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['regression'][1],"y_train.npy"), Logs.INFO)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['regression'][1],"y_test.npy"), y_test_SDA)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['regression'][1],"y_test.npy"), Logs.INFO)
    
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['classification'][1],"y_train.npy"), y_train_class_SDA)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['classification'][1],"y_train.npy"), Logs.INFO)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['classification'][1],"y_test.npy"), y_test_class_SDA)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_"+ conf.OUTPUT_VARIABLES[country]['classification'][1],"y_test.npy"), Logs.INFO)
   
    # save weights W
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY + "w_train.npy"), w_train)
    log(country, "Saving weights: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "w_train.npy"), Logs.INFO)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY + "w_test.npy"), w_test)
    log(country, "Saving weights: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "w_test.npy"), Logs.INFO)
    
    # save infos (spatial, year) for train and test data
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY + "info_train.npy"), info_train)
    log(country, "Saving Spatial Information: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_train.npy"), Logs.INFO)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY + "info_test.npy"), info_test)
    log(country, "Saving Spatial Information: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_test.npy"), Logs.INFO)
    
    
    
        
    
    # ================================================================================================================#
    #     Conversion of CNN spatial variables into Numpy array                                                        #
    # =============================================================================================================== #

   
    
    
    
    info_pix_cnn, dataX_CNN, dataY_CNN = process_rasters(country, algorithm, data_response_src, rep, years)
    dataX_CNN = np.array(dataX_CNN)
    dataY_CNN = np.array(dataY_CNN)
    info_pix_cnn = np.array(info_pix_cnn, dtype=int)
    
    log(country, f"dataX_CNN shape is: {dataX_CNN.shape}")
    log(country, f"info_pix_cnn shape is: {info_pix_cnn.shape}")
    log(country, f"dataY_CNN shape is: {dataY_CNN.shape}")
    
    # ================================================================================================================#
    #  transformation of info data into a dataframe for merging pixel info and info (municipality, year) train / test #
    # =============================================================================================================== #

    RASTER_GRANULARITY = conf.ID_REGIONS[country]
    
    info_pix_cnn = pd.DataFrame(info_pix_cnn, columns=[conf.TEMPORAL_GRANULARITY[country], RASTER_GRANULARITY, "line"]) # CODE_COM RASTER_GRANULARITY, "line"]
    info_train = pd.DataFrame(info_train, columns=[conf.TEMPORAL_GRANULARITY[country], RASTER_GRANULARITY])
    info_test = pd.DataFrame(info_test, columns=[conf.TEMPORAL_GRANULARITY[country], RASTER_GRANULARITY])
    
    # ================================================================================================================#
    #  Merge pixel info and info (town, year) train and test                                                          #
    # =============================================================================================================== #

    info_pix_cnn_train = pd.merge(
        info_pix_cnn, info_train, how="inner", on=[conf.TEMPORAL_GRANULARITY[country], RASTER_GRANULARITY]
    )
    info_pix_cnn_test = pd.merge(
        info_pix_cnn, info_test, how="inner", on=[conf.TEMPORAL_GRANULARITY[country], RASTER_GRANULARITY]
    )

    # ================================================================================================================#
    #  Conversion into Numpy array                                                                                    #
    # =============================================================================================================== #

    info_pix_cnn_train = np.array(info_pix_cnn_train)
    info_pix_cnn_test = np.array(info_pix_cnn_test)

    # ================================================================================================================#
    #  Train/Test split                                                                                               #
    # =============================================================================================================== #

    X_pix_train_cnn = dataX_CNN[info_pix_cnn_train[:, 2].astype(int)]
    X_pix_test_cnn = dataX_CNN[info_pix_cnn_test[:, 2].astype(int)]
    y_pix_train_cnn = dataY_CNN[info_pix_cnn_train[:, 2].astype(int)]
    y_pix_test_cnn = dataY_CNN[info_pix_cnn_test[:, 2].astype(int)]

    # no more need for information on pixel locations, they're removed
    info_pix_cnn_train = np.delete(info_pix_cnn_train, 2, 1)
    info_pix_cnn_test = np.delete(info_pix_cnn_test, 2, 1)

    log(country, "X-Train Shape: "+  str(X_pix_train_cnn.shape), Logs.INFO)
    log(country, "X-Test Shape: "+   str(X_pix_test_cnn.shape), Logs.INFO)

    # save cnn elements
    # np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/cnn_info_pix_train.npy", info_pix_cnn_train)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cnn_info_pix_train.npy"), info_pix_cnn_train)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cnn_info_pix_train.npy"), Logs.INFO)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cnn_info_pix_test.npy"), info_pix_cnn_test)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cnn_info_pix_test.npy"), Logs.INFO)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cnn_x_pix_train.npy"), X_pix_train_cnn)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cnn_x_pix_train.npy"), Logs.INFO)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cnn_x_pix_test.npy"), X_pix_test_cnn)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cnn_x_pix_test.npy"), Logs.INFO)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cnn_y_pix_train.npy"), y_pix_train_cnn)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cnn_y_pix_train.npy"), Logs.INFO)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cnn_y_pix_test.npy"), y_pix_test_cnn)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cnn_y_pix_test.npy"), Logs.INFO)

    #loaded_train = np.load(conf.FEATURES_DIRECTORY + "cnn_x_pix_train.npy")
    #print("X-Train Load", type(loaded_train), loaded_train.shape)

    
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cs_x_train.npy"), X_train_CS)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cs_x_train.npy"), Logs.INFO)
    np.save(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cs_x_test.npy"), X_test_CS)
    log(country, "Saving Rasters: "+ os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "cs_x_test.npy"), Logs.INFO)
    
   
  
   
    log(country, 'Variables and features saved in folder "features" ', Logs.INFO)
    log(country, "End preprocessings", Logs.INFO)
