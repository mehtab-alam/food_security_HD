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
import pandas as pd
import numpy as np
import os
from osgeo import gdal
from sklearn.model_selection import train_test_split
import configuration as conf





# =============================================================================#
# Data (variables) loading method: Load data (variables) from Excel files      #
# =============================================================================#
def read_add_variables(data_rep, var_type, criteria):
    
    for D in var_type:
        data_temp = pd.read_excel(conf.VAR_DIRECTORY + "data_"+ D + ".xlsx") # Impoort variables 
        data_rep = pd.merge(data_rep, data_temp, how="left", on=criteria)
        
    return data_rep


# =======================================================================================#
# Standardization/Formating Method: Extract,format time series variables and its renaming#
# =======================================================================================#
def filter_variables(data_aggr, data_rep, var_type):
    for i in range(2): # année t-1 et t
        for j in range(5, 12): # mois de mai (5) à novembre (11)
            for D in var_type:
                if D == 'maize' or D == 'grains':
                    data_aggr['{0}{1}t-{2}'.format(D,j,1-i)] = data_rep.filter(regex='{1}{0}-.*\(.*t-{2}'.format(D,j,1-i))
                elif D == 'tmin' or D == 'tmax':
                    t_var = D.replace('t', 't_')
                    data_aggr['{0}{1}t-{2}'.format(D,j,1-i)] = data_rep.filter(regex='{0}.*{1}\(t-{2}'.format(t_var,j,1-i))
                else:
                    data_aggr['{0}{1}t-{2}'.format(D,j, 1 - i)] = data_rep.filter(regex= '{0}.*{1}\(.*t-{2}'.format(D,j, 1 - i)).max(axis=1)
    return data_aggr


# ====================================================================================================#
# Normalization Method: Normalization of variables using mean and standard deviation: (X-X.mean)/X.std#
# ====================================================================================================#

def normalize(data, var_type):
    for v in conf.vars_timeseries:
      data.loc[:, data.columns.str.startswith(v)]=(data.loc[:, data.columns.str.startswith(v)]-
                                                              data.loc[:, data.columns.str.startswith(v)].stack().mean())/\
                                                            data.loc[:, data.columns.str.startswith(v)].stack().std()
    return data


# ==========================================================================================#
# Return output (Y) Numpy Array                                                             #
# ==========================================================================================#
def output_Y(data_aggregate, output):
    data_Y = []
    for i in range(len(data_aggregate)):
        data_Y.append([data_aggregate[output][i]])
    data_Y = np.array(data_Y)
    return data_Y

# ==========================================================================================#
# Export the preprocessed variables                                                         #
# ==========================================================================================#
def export(data_timeseries, data_conjunctural, data_structural, data_rep_columns):
    os.makedirs(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY , exist_ok=True)
    data_timeseries.to_excel(conf.PREPROCESS_FILES['timeseries'])
    data_conjunctural.to_excel(conf.PREPROCESS_FILES['conjunctural'])
    data_structural.to_excel(conf.PREPROCESS_FILES['structural'])
    data_all = pd.merge(data_timeseries, data_conjunctural, how="left", on=data_rep_columns)
    data_all = pd.merge(data_all,  data_structural, how="left", on=data_rep_columns)
    data_all.to_excel(conf.PREPROCESS_FILES['all'])
    return


# ==========================================================================================#
# Main Preprocessing Method: Data processing for processing of numerical and spatial data   #
# ==========================================================================================#

def preprocess(rep, r_split):
    
    pd.set_option('display.max_columns', None)
    print("Preprocessing of data started ...")
    data_response_src = pd.read_excel(conf.DATA_DIRECTORY + conf.RESPONSE_FILE) # Read Survey Data
    data_rep_columns = list(data_response_src.columns) # Get the name of Response columns 
    data_rep = data_response_src         # Keeping the original response data as it is...
    
    # =============================================================================#
    #     Preprocessing of Time-series variables data                              #
    # =============================================================================#
        
    data_rep = read_add_variables(data_rep, conf.vars_timeseries, conf.SPATIAL_TEMPORAL_GRANULARITY) # read time-series variables
    data_aggregation_timeseries = data_rep.loc[:, data_rep_columns]
    data_aggregation_timeseries = filter_variables(data_aggregation_timeseries, data_rep, conf.vars_timeseries) # Standardization of variables and column names 
    data_aggregation_timeseries = normalize(data_aggregation_timeseries, conf.vars_timeseries) # Normalize variables by mean/STD
    data_aggregation_timeseries.dropna(subset = conf.OUTPUT_VARIABLES, how='all', inplace=True)# Remove rows with response variable Nan
    data_X_timeseries = np.array(data_aggregation_timeseries.loc[:, ~data_aggregation_timeseries.columns.isin(data_rep_columns)])# Store the data of time-series variables in NUMPY Array
      
    # =============================================================================#
    #     Preprocessing of Conjuntural variables data                              #
    #     TODO: Need to restructure the code for world bank                        #
    # =============================================================================#
        
    data_aggregation_conjunctural = data_response_src
    data_temp = pd.read_excel(conf.VAR_DIRECTORY +"data_world_bank.xlsx")  # Conjunctural variable handle explicitly 
    data_aggregation_conjunctural = pd.merge(data_aggregation_conjunctural, data_temp, how="left", on=['ANNEE']) # Merge on Temporal aspect
    # read conjunctural variables
    data_aggregation_conjunctural = read_add_variables(data_aggregation_conjunctural, [i for i in conf.vars_conjuctral if i!='world_bank'], conf.SPATIAL_TEMPORAL_GRANULARITY) 
    data_aggregation_conjunctural = normalize(data_aggregation_conjunctural, conf.vars_conjuctral) # Normalize variables by mean/STD
    data_aggregation_conjunctural.dropna(subset = conf.OUTPUT_VARIABLES, how='all', inplace=True)# Remove rows with response variable Nan
    data_X_conjunctural = np.array(data_aggregation_conjunctural.loc[:, ~data_aggregation_conjunctural.columns.isin(data_rep_columns)])# Store the data of conjunctural variables in NUMPY Array
    #print(data_X_conjunctural)

    # =============================================================================#
    #     Preprocessing of Structural variables data                               #
    # =============================================================================#
    data_aggregation_structural = data_response_src
    # read structural variables
    data_aggregation_structural = read_add_variables(data_aggregation_structural, conf.vars_structural, [i for i in conf.SPATIAL_TEMPORAL_GRANULARITY if i!='ANNEE']) 
    data_aggregation_structural = normalize(data_aggregation_structural, conf.vars_structural) # Normalize variables by mean/STD
    data_aggregation_structural.dropna(subset = conf.OUTPUT_VARIABLES, how='all', inplace=True)# Remove rows with response variable Nan
    data_X_structural = np.array(data_aggregation_structural.loc[:, ~data_aggregation_structural.columns.isin(data_rep_columns)])# Store the data of structural variables in NUMPY Array
    #print(data_aggregation_structural)
    
    
    # =============================================================================#
    #     Storing Commune information per year for CNN Processing                  #
    # =============================================================================#
    dataInfo = []
    for i in range(len(data_aggregation_structural)):
        dataInfo.append([data_aggregation_structural['ANNEE'][i], data_aggregation_structural['ID_COM'][i]])
    dataInfo = np.array(dataInfo)
    
    # =============================================================================#
    #     Storing output information (Y) in numpy array = ['sca','sda']            #
    # =============================================================================#
    data_Y_SCA = output_Y(data_aggregation_structural, conf.OUTPUT_VARIABLES[0]) # Return output variable (SCA) Numpy Array
    data_Y_SDA = output_Y(data_aggregation_structural, conf.OUTPUT_VARIABLES[1]) # Return output variable (SDA) Numpy Array
    
    # =================================================================================================#
    # Combine Conjunctural and Structural Variables                                                                     #    
    # =================================================================================================#
    data_aggregate_CS = pd.merge(data_aggregation_conjunctural, data_aggregation_structural, how="left", on=data_rep_columns)
    data_X_CS = np.array(data_aggregate_CS.loc[:, ~data_aggregate_CS.columns.isin(data_rep_columns)])# Store the data of Conjunctural+structural variables in NUMPY Array
    
    # =================================================================================================#
    # EXPORT : Export the variables data                                                               #   
    # =================================================================================================#
    
    export(data_aggregation_timeseries, data_aggregation_conjunctural, data_aggregation_structural, data_rep_columns)
    
    # =================================================================================================#
    # Weight storage (W) (for each observation: W = root(number of households)/root(total households)) #   
    # =================================================================================================#
   
    data_aggregate_CS['Count'] = np.sqrt(data_aggregate_CS['Count']) / np.sqrt(data_aggregate_CS['Count']).sum()
    data_W = [] # Weight Array 
    for i in range(len(data_aggregate_CS)):
        data_W.append([data_aggregate_CS['Count'][i]])
    data_W = np.array(data_W)
    
    # =================================================================================================#
    # Train/Test data split                                                                            #    
    # =================================================================================================#
    
    X_train_timeseries, X_test_timeseries, X_train_CS, X_test_CS, y_train_SCA, y_test_SCA,  y_train_SDA, y_test_SDA, w_train, w_test, info_train, info_test = train_test_split(
        data_X_timeseries, data_X_CS, data_Y_SCA, data_Y_SDA, data_W, dataInfo, test_size=0.15, random_state=r_split)


    # =============================================================================#
    #     Preprocessing of Spatial Data                                            #
    # =============================================================================#
    
    raster_com = gdal.Open(conf.SPATIAL_DIRECTORY + conf.SPATIAL_TIF_VARS['epa'])
    crop = gdal.Open(conf.SPATIAL_DIRECTORY + conf.SPATIAL_TIF_VARS['crop']) # import crop Data
    forest = gdal.Open(conf.SPATIAL_DIRECTORY + conf.SPATIAL_TIF_VARS['forest'])  # import forest data
    zones = gdal.Open(conf.SPATIAL_DIRECTORY + conf.SPATIAL_TIF_VARS['zones'])  # import constructed zones data

    
    # =============================================================================#
    #     Numpy Arrays of Tif Files: epa, crop, forest and zones                   #
    # =============================================================================#

    raster_com = np.array(raster_com.ReadAsArray())
    crop = np.array(crop.ReadAsArray())
    forest = np.array(forest.ReadAsArray())
    zones = np.array(zones.ReadAsArray())
    
   
    dictrep = dict() # dictionary of answers for each year
    dictpop = dict() # dictionary of population for each year

    # ================================================================================================================#
    #     Importing Rasters for response and population for each year                                                 #
    #     **TODO: Reduce the redundant data folders e.g. (sca_100m, sda_sda100m)-> should be one folder               #
    # =============================================================================================================== #

    # import des rasters réponse et population de chaque année
    for annee in range(conf.TIME_RANGE[0], conf.TIME_RANGE[1]): # import des rasters réponse et population
        dictrep[annee] = gdal.Open(conf.SPATIAL_DIRECTORY + rep + '_' +conf.PIXEL+ '/epa_' + rep + '_' + str(annee) + '.tif')
        dictpop[annee] = gdal.Open(conf.SPATIAL_DIRECTORY + 'population_' + conf.PIXEL + '/population_' + str(annee) + '.tif')

        #Conversion into Numpy Array per year
        dictrep[annee] = np.array(dictrep[annee].ReadAsArray())
        dictpop[annee] = np.array(dictpop[annee].ReadAsArray())

        #Replacement of Zero-value pixel to NAN value
        dictrep[annee][dictrep[annee] <= 0] = np.nan
        dictpop[annee][dictpop[annee] <= 0] = np.nan

        #Normalization of pixelated poulation (X-mean/std)
        dictpop[annee] = dictpop[annee] - np.nanmean(dictpop[annee]) / np.nanstd(dictpop[annee])
    
    # ================================================================================================================#
    #     Listing the variables for CNN                                                                               #
    # =============================================================================================================== #
    
    info_pix_cnn = [] # data on the pairs (municipality, year) associated with each pixel
    dataX_CNN = [] # list of population pixel patches
    dataY_CNN = [] # list of response pixels
    length = 10 # length of patches
    step = 30 # distance between 2 selected pixels
    
   
    # ================================================================================================================#
    #     Filling the variables declared for CNN                                                                      #
    # =============================================================================================================== #
    for annee in range(conf.TIME_RANGE[0], conf.TIME_RANGE[1]): # pour chaque année
        for i in range(int(length/2), dictrep[annee].shape[0] - int(length/2), step): # pixels are scanned horizontally in steps of 30 with a margin = int(length/2)
            for j in range(int(length/2), dictrep[annee].shape[1] - int(length/2), step): # pixels are scanned vertically in steps of 30 with a margin = int(length/2)
                """
                conditions for integrating a response pixel + a population patch + 3 oqp_sol patches into the CNN dataset:
                - the response pixel must not be Nan
                - none of the pixels in the associated population patch must be Nan
                - all the pixels in the associated population patch must belong to the same municipality as the response pixel               
                """
                if not ((np.isnan(dictrep[annee][i, j])) |
                        (np.isnan(dictpop[annee][i - int(length/2):i + int(length/2), j - int(length/2):j + int(length/2)]).any()) |
                        (raster_com[i - int(length/2):i + int(length/2), j - int(length/2):j + int(length/2)] != raster_com[i, j]).any()):

                    info_pix_cnn.append([annee, raster_com[i, j], len(info_pix_cnn)])
                    dataX_CNN.append([dictpop[annee][i - int(length/2):i + int(length/2), j - int(length/2):j + int(length/2)],
                                      crop[i - int(length/2):i + int(length/2), j - int(length/2):j + int(length/2)],
                                      forest[i - int(length/2):i + int(length/2), j - int(length/2):j + int(length/2)],
                                      zones[i - int(length/2):i + int(length/2), j - int(length/2):j + int(length/2)]])
                    dataY_CNN.append([dictrep[annee][i, j]])
   
    # ================================================================================================================#
    #     Conversion of CNN spatial variables into Numpy array                                                        #
    # =============================================================================================================== #

    dataX_CNN = np.array(dataX_CNN)
    dataY_CNN = np.array(dataY_CNN)
    info_pix_cnn = np.array(info_pix_cnn, dtype=int)

    # ================================================================================================================#
    #  transformation of info data into a dataframe for merging pixel info and info (municipality, year) train / test #
    # =============================================================================================================== #

    info_pix_cnn = pd.DataFrame(info_pix_cnn, columns=['ANNEE', 'CODE_COM', 'line'])
    info_train = pd.DataFrame(info_train, columns=['ANNEE', 'CODE_COM'])
    info_test = pd.DataFrame(info_test, columns=['ANNEE', 'CODE_COM'])
    
    # ================================================================================================================#
    #  Merge pixel info and info (town, year) train and test                                                          #
    # =============================================================================================================== #

    info_pix_cnn_train = pd.merge(info_pix_cnn, info_train, how='inner', on=['ANNEE', 'CODE_COM'])
    info_pix_cnn_test = pd.merge(info_pix_cnn, info_test, how='inner', on=['ANNEE', 'CODE_COM'])
    
    # ================================================================================================================#
    #  Conversion into Numpy array                                                                                    #
    # =============================================================================================================== #

    info_pix_cnn_train = np.array(info_pix_cnn_train)
    info_pix_cnn_test = np.array(info_pix_cnn_test)

    # ================================================================================================================#
    #  Train/Test split                                                                                               #
    # =============================================================================================================== #

    X_pix_train_cnn = dataX_CNN[info_pix_cnn_train[:, 2]]
    X_pix_test_cnn = dataX_CNN[info_pix_cnn_test[:, 2]]
    y_pix_train_cnn = dataY_CNN[info_pix_cnn_train[:, 2]]
    y_pix_test_cnn = dataY_CNN[info_pix_cnn_test[:, 2]]

    # no more need for information on pixel locations, they're removed
    info_pix_cnn_train = np.delete(info_pix_cnn_train, 2, 1)
    info_pix_cnn_test = np.delete(info_pix_cnn_test, 2, 1)
    
    
    # Create directory in case if doesn't exists
    os.makedirs(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split), exist_ok=True)
    os.makedirs(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_sca_" + str(r_split), exist_ok=True)
    os.makedirs(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_sda_" + str(r_split), exist_ok=True)
    
    
    # save cnn elements
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/cnn_info_pix_train.npy", info_pix_cnn_train)
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/cnn_info_pix_test.npy", info_pix_cnn_test)
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/cnn_x_pix_train.npy", X_pix_train_cnn)
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/cnn_x_pix_test.npy", X_pix_test_cnn)
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/cnn_y_pix_train.npy", y_pix_train_cnn)
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/cnn_y_pix_test.npy", y_pix_test_cnn)

    # save explicative variables X
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/timeseries_x_train.npy", X_train_timeseries)
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/timeseries_x_test.npy", X_test_timeseries)

    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/cs_x_train.npy", X_train_CS)
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/cs_x_test.npy", X_test_CS)

    # save response Y (SCA)
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_sca_" + str(r_split) + "/y_train.npy", y_train_SCA)
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_sca_" + str(r_split) + "/y_test.npy", y_test_SDA)
    
    # save response Y (SDA)
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_sda_" + str(r_split) + "/y_train.npy", y_train_SDA)
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_sda_" + str(r_split) + "/y_test.npy", y_test_SDA)

    # save weights W
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/w_train.npy", w_train)
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/w_test.npy", w_test)

    # save infos (commune, année) des données train et test
    np.save(conf.PREPROCESS_DATA_DIR+ conf.FEATURES_DIRECTORY + "features_" + str(r_split) + "/info_train.npy", info_train)
    print("Variables and features saved in folder \"features\" ")
    print("End preprocessings")

    
    
    
   