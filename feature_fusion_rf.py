#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:11:23 2024

@author: syed
"""
from logger import log
from logger import Logs
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score
import configuration as conf
from sklearn.impute import SimpleImputer
from visualization import save_classification_map, save_region_classification_map, plot_confusion_matrix, plot_roc_auc, plot_regression_results 



def train_evaluate(task_type, params, x_train, y_train, x_test, y_test, w_train, w_test, description, country):
    # Select model based on task type
    if task_type == 'regression':
        model = RandomForestRegressor(**params)
    elif task_type == 'classification':
        model = RandomForestClassifier(**params)
    else:
        raise ValueError("Unsupported task type. Use 'regression' or 'classification'.")
    
   
    x_train = np.nan_to_num(x_train, nan=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0)
    
    # Fit the model
    model.fit(x_train, y_train.ravel(), sample_weight=w_train.ravel())

    # Make predictions
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    # Evaluate and log performance
    if task_type == 'regression':
        r2 = r2_score(y_test, predictions, sample_weight=w_test)
        log(country, f"RF({description}) TEST R2: {r2:.6f}")
    elif task_type == 'classification':
        #print(y_test.shape, predictions.shape, w_test.shape)
        acc = accuracy_score(y_test.ravel(), predictions, sample_weight=w_test.ravel())
        f1 = f1_score(y_test.ravel(), predictions, sample_weight=w_test.ravel(), average='weighted')
        log(country, f"RF({description}) TEST Accuracy: {acc:.6f}, F1 Score: {f1:.6f}")
    
   
    return y_test.ravel() , predictions.ravel(), probabilities

#Save Results
def save_results(country, algorithm, rep, tt_split, test_targets, test_predictions, y_test_com, description, test_probabilities):
   
    
    data = pd.read_excel(os.path.join(
        conf.DATA_DIRECTORY, country, conf.RESPONSE_FILE[country]))
    output = pd.DataFrame({conf.TEMPORAL_GRANULARITY[country]: y_test_com[:, 0].tolist(),  conf.ID_REGIONS[country].upper(): y_test_com[:, 1].tolist(), 
                           'label': test_targets, 'prediction': test_predictions})
    
    results = pd.merge(output, data[[conf.TEMPORAL_GRANULARITY[country], conf.SPATIAL_GRANULARITY[country][-1], conf.SPATIAL_GRANULARITY[country][-2],
                          conf.ID_REGIONS[country].upper()]], on= [conf.TEMPORAL_GRANULARITY[country], conf.ID_REGIONS[country].upper()], how='inner')
    rearranged_columns = [conf.TEMPORAL_GRANULARITY[country], conf.SPATIAL_GRANULARITY[country][-1], conf.SPATIAL_GRANULARITY[country][-2],
                          conf.ID_REGIONS[country].upper(), 'label', 'prediction']
    results = results[rearranged_columns]
    os.makedirs(os.path.join(conf.OUTPUT_DIR, country, "results", "RF", description, tt_split, algorithm), exist_ok=True)
    results.to_excel(os.path.join(conf.OUTPUT_DIR, country, "results", "RF", description, tt_split, algorithm,  rep + '.xlsx'), index=False)
    if algorithm == 'classification':
        save_classification_map(country, algorithm, tt_split, os.path.join("RF", description), rep, max(y_test_com[:, 0].tolist()))
        save_region_classification_map(country, algorithm, tt_split, os.path.join("RF", description), rep, max(y_test_com[:, 0].tolist()))
        plot_confusion_matrix(country, algorithm, tt_split, os.path.join("RF", description), rep, max(y_test_com[:, 0].tolist()))
        plot_roc_auc(country, algorithm, tt_split, os.path.join("RF", description), rep, max(y_test_com[:, 0].tolist()), test_probabilities)
    else:
        plot_regression_results(country, algorithm, tt_split, os.path.join("RF", description), rep, max(y_test_com[:, 0].tolist()))
        
def ml(rep, algorithm, r_split, country, tt_split):
    log(country, "Begin Machine Learning on initial variables and NN features")
    PATH = os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY)

    y_train = np.load(os.path.join( PATH, "features_" + rep, "y_train.npy"))
    log(country, "Loading y_train from: "+ os.path.join( PATH, "features_" + rep, "y_train.npy"))
    y_test = np.load(os.path.join( PATH, "features_" + rep , "y_test.npy"))
    log(country, "Loading y_test from: "+ os.path.join( PATH, "features_" + rep, "y_test.npy"))
    
    # import weights
    w_train = np.load(os.path.join( PATH,  "w_train.npy"))
    log(country, "Loading w_train from: "+ os.path.join( PATH, "y_test.npy"))
    w_test = np.load(os.path.join( PATH,  "w_test.npy"))
    log(country, "Loading w_test from: "+ os.path.join( PATH, "y_test.npy"))

    info_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_train.npy"))
    log(country, f"Loading info_train from with {info_train.shape}: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_train.npy"))
    info_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_test.npy"))
    log(country, f"Loading info_test from with {info_test.shape}: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "info_test.npy"))

    
    data_list = dict()
    for data in ['lstm', 'cnn']:
        data_list['feat_x_train_{0}'.format(data)] = np.load(os.path.join(PATH, data, tt_split, algorithm, rep, data + "_x_train.npy"))
        log(country, "Loading feat_x_train_{0}:"+ os.path.join(PATH, data, tt_split, algorithm, rep, data + "_x_train.npy"))
        data_list['feat_x_test_{0}'.format(data)] = np.load(os.path.join(PATH, data, tt_split, algorithm, rep, data + "_x_test.npy"))
        log(country, "Loading feat_x_test_{0}"+ os.path.join(PATH, data, tt_split, algorithm, rep, data + "_x_test.npy"))
        
    data_list['x_train_lstm'] = np.load(os.path.join(PATH, "timeseries_x_train.npy"))
    data_list['x_test_lstm'] = np.load(os.path.join(PATH, "timeseries_x_test.npy"))

    data_list['x_train_cs'] = np.load(os.path.join(PATH, "cs_x_train.npy"))
    data_list['x_test_cs'] = np.load(os.path.join(PATH, "cs_x_test.npy"))
    
    
     #concaténation des variables initiales
    x_train_tot = np.concatenate((data_list['x_train_lstm'], data_list['x_train_cs']), axis=1)
    x_test_tot = np.concatenate((data_list['x_test_lstm'], data_list['x_test_cs']), axis=1)
    
    
    # concaténation des features
    feat_x_train_cnn = data_list['feat_x_train_cnn']
    feat_x_train_lstm = data_list['feat_x_train_lstm']
    
    trimmed_train_cnn = feat_x_train_cnn.flatten()[:data_list['x_train_cs'].shape[0] * data_list['x_train_cs'].shape[1]] # Reshaping DL CNN features into same conjunctural shape
    feat_x_train_cnn = trimmed_train_cnn.reshape(data_list['x_train_cs'].shape)
   
    
    feat_x_train_cnn_cs = np.concatenate((feat_x_train_cnn, data_list['x_train_cs']), axis=1)

    feat_x_train_tot = np.concatenate((feat_x_train_cnn, data_list['x_train_cs']), axis=1)
    feat_x_train_tot = np.concatenate((feat_x_train_tot, data_list['feat_x_train_lstm']), axis=1)

    feat_x_test_cnn = data_list['feat_x_test_cnn']
    feat_x_test_lstm = data_list['feat_x_test_lstm']

    trimmed_test_cnn = feat_x_test_cnn.flatten()[:data_list['x_test_cs'].shape[0] * data_list['x_test_cs'].shape[1]] # Reshaping DL CNN features into same conjunctural shape
    feat_x_test_cnn = trimmed_test_cnn.reshape(data_list['x_test_cs'].shape)
   
    feat_x_test_cnn_cs = np.concatenate((feat_x_test_cnn, data_list['x_test_cs']), axis=1)

    feat_x_test_tot = np.concatenate((feat_x_test_cnn, data_list['x_test_cs']), axis=1)
    feat_x_test_tot = np.concatenate((feat_x_test_tot, data_list['feat_x_test_lstm']), axis=1)
    
    # Parameters for the model
    params = {'n_estimators': 900, 'max_depth': 20, 'random_state': 1}
    
    # Datasets for evaluation
    datasets = [
        ('time series', data_list['x_train_lstm'], data_list['x_test_lstm']),
        ('init. variables', x_train_tot, x_test_tot),
        ('lstm features', feat_x_train_lstm, feat_x_test_lstm),
        ('cnn features', feat_x_train_cnn, feat_x_test_cnn),
        ('cnn features + cs vars', feat_x_train_cnn_cs, feat_x_test_cnn_cs),
        ('all features', feat_x_train_tot, feat_x_test_tot),
    ]
    
    # Apply for different variables
    for description, x_train, x_test in datasets:
        test_targets,  test_predictions, test_probabilities = train_evaluate(algorithm, params, x_train, y_train, x_test, y_test, w_train, w_test, description, country)
        save_results(country, algorithm, rep, tt_split, test_targets, test_predictions, info_test, description, test_probabilities)
   