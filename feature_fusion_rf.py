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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import configuration as conf
from sklearn.impute import SimpleImputer


def dynamic_concatenate(array1, array2):
   
    
    # Calculate the number of dimensions to expand
    max_dim = max(len(array1.shape), len(array2.shape))
    
    # Expand dimensions of both arrays to match the maximum dimension
    array1_expanded = np.reshape(array1, array1.shape + (1,) * (max_dim - len(array1.shape)))
    array2_expanded = np.reshape(array2, array2.shape + (1,) * (max_dim - len(array2.shape)))
    
    # Calculate the broadcast shape for concatenation
    broadcast_shape = np.broadcast_shapes(array1_expanded.shape, array2_expanded.shape)
    
    # Broadcast arrays to the target shape
    array1_broadcasted = np.broadcast_to(array1_expanded, broadcast_shape)
    array2_broadcasted = np.broadcast_to(array2_expanded, broadcast_shape)
    
    # Concatenate along the last axis
    result = np.concatenate((array1_broadcasted, array2_broadcasted), axis=-1)
    
    return result

def ml(rep, r_split, country):
    log(country, "Begin Machine Learning on initial variables and NN features")
    PATH = os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY)
    # import variable réponse Y train et test
    y_train = np.load(os.path.join( PATH, "features_" + rep, "y_train.npy"))
    log(country, "Loading y_train from: "+ os.path.join( PATH, "features_" + rep, "y_train.npy"))
    y_test = np.load(os.path.join( PATH, "features_" + rep , "y_test.npy"))
    log(country, "Loading y_test from: "+ os.path.join( PATH, "features_" + rep, "y_test.npy"))
    
    # import weights
    w_train = np.load(os.path.join( PATH,  "w_train.npy"))
    log(country, "Loading w_train from: "+ os.path.join( PATH, "y_test.npy"))
    w_test = np.load(os.path.join( PATH,  "w_test.npy"))
    log(country, "Loading w_test from: "+ os.path.join( PATH, "y_test.npy"))

    
    OUTPUT_FEATURES_PATH = os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY, "features_" + rep + "_" + str(r_split))
     # import des variables explicatives initiales (x_train, x_test) et features en sortie des NN (feat_x_train, feat_x_test)
    data_list = dict()
    for data in ['lstm', 'cnn']:
        
        data_list['feat_x_train_{0}'.format(data)] = np.load(os.path.join(OUTPUT_FEATURES_PATH, data + "_feat_x_train.npy"))
        log(country, "Loading feat_x_train_{0}"+ os.path.join(OUTPUT_FEATURES_PATH, data + "_feat_x_train.npy"))
        data_list['feat_x_test_{0}'.format(data)] = np.load(os.path.join(OUTPUT_FEATURES_PATH, data + "_feat_x_test.npy"))
        log(country, "Loading feat_x_test_{0}"+ os.path.join(OUTPUT_FEATURES_PATH, data + "_feat_x_test.npy"))
    for data in ['timeseries', 'cs']:
        data_list['x_train_{0}'.format(data)] = np.load(os.path.join(PATH, data + "_x_train.npy"))
        log(country, "Loading x_train_{0}"+ os.path.join(PATH, data + "_x_train.npy"))
        data_list['x_test_{0}'.format(data)] = np.load(os.path.join(PATH, data + "_x_test.npy"))
        log(country, "Loading x_test_{0}"+ os.path.join(PATH, data + "_x_test.npy"))
        
    #concaténation des variables initiales
    x_train_tot = np.concatenate((data_list['x_train_timeseries'], data_list['x_train_cs']), axis=1)
    x_test_tot = np.concatenate((data_list['x_test_timeseries'], data_list['x_test_cs']), axis=1)

    # concaténation des features
    feat_x_train_cnn = data_list['feat_x_train_cnn']
    feat_x_train_lstm = data_list['feat_x_train_lstm']

    feat_x_train_cnn_cs = np.concatenate((feat_x_train_cnn, data_list['x_train_cs']), axis=1)
    
    feat_x_train_tot = np.concatenate((feat_x_train_cnn, data_list['x_train_cs']), axis=1)
    print(feat_x_train_tot.shape, data_list['feat_x_train_lstm'].shape)
    feat_x_train_tot = np.concatenate((feat_x_train_tot, data_list['feat_x_train_lstm']), axis=1)
    #feat_x_train_tot = dynamic_concatenate(feat_x_train_tot, data_list['feat_x_train_lstm'])   
    


    feat_x_test_cnn = data_list['feat_x_test_cnn']
    feat_x_test_lstm = data_list['feat_x_test_lstm']

    feat_x_test_cnn_cs = np.concatenate((feat_x_test_cnn, data_list['x_test_cs']), axis=1)

    feat_x_test_tot = np.concatenate((feat_x_test_cnn, data_list['x_test_cs']), axis=1)
    feat_x_test_tot = np.concatenate((feat_x_test_tot, data_list['feat_x_test_lstm']), axis=1)
    
    
     ### RF
    ## sur les variables initiales
    params = {'n_estimators': 900, 'max_depth': 20, 'random_state': 1}

    rf = RandomForestRegressor(**params)
    rf.fit(data_list['x_train_timeseries'], y_train.ravel(), sample_weight=w_train.ravel())
    predictions = rf.predict(data_list['x_test_timeseries'])
    log(country, "RF(time series) TEST R2: %f" % r2_score(y_test, predictions, sample_weight=w_test))

    rf = RandomForestRegressor(**params)
    rf.fit(x_train_tot, y_train.ravel(), sample_weight=w_train.ravel())
    predictions = rf.predict(x_test_tot)
    log(country, "RF(init. variables) TEST R2: %f" % r2_score(y_test, predictions, sample_weight=w_test))

    ## sur les features
    # vars lstm seules
    rf = RandomForestRegressor(**params)
    rf.fit(feat_x_train_lstm, y_train.ravel(), sample_weight=w_train.ravel())
    predictions = rf.predict(feat_x_test_lstm)
    log(country, "RF(lstm features) TEST R2: %f" % r2_score(y_test, predictions, sample_weight=w_test))

    # vars cnn seules
    rf = RandomForestRegressor(**params)
    
    imputer = SimpleImputer(strategy='mean')
    feat_x_train_cnn_imputed = imputer.fit_transform(feat_x_train_cnn)
    feat_x_test_cnn_imputed = imputer.fit_transform(feat_x_test_cnn)
    
    rf.fit(feat_x_train_cnn_imputed, y_train.ravel(), sample_weight=w_train.ravel())
    predictions = rf.predict(feat_x_test_cnn_imputed)
    log(country, "RF(cnn features) TEST R2: %f" % r2_score(y_test, predictions, sample_weight=w_test))

    # vars cnn + cs
    rf = RandomForestRegressor(**params)
    rf.fit(imputer.fit_transform(feat_x_train_cnn_cs), y_train.ravel(), sample_weight=w_train.ravel())
    predictions = rf.predict(imputer.fit_transform(feat_x_test_cnn_cs))
    log(country, "RF(cnn features + cs vars) TEST R2: %f" % r2_score(y_test, predictions, sample_weight=w_test))

    # toutes les features
    rf = RandomForestRegressor(**params)
    rf.fit(imputer.fit_transform(feat_x_train_tot), y_train.ravel(), sample_weight=w_train.ravel())
    predictions = rf.predict(imputer.fit_transform(feat_x_test_tot))
    log(country, "RF(all features) TEST R2: %f" % r2_score(y_test, predictions, sample_weight=w_test))



  