# -*- coding: utf-8 -*-
from logger import log
from logger import Logs
import configuration as conf
from preprocessor import preprocess
from timeseries_lstm import timeseries_lstm
from spatial_cnn import cnn
from feature_fusion_rf import ml
import argparse, sys






"""
from cnn import cnn
from ML_feature_fusion import ml
from ML_late_fusion import ml_late
"""


## variables explicatives :
# Séries temporelles de smt, pluie, prix mais, temperature_max, temperature_min
# occupation du sol
# nombres de centres de santé et d'écoles pour 1000 habitants
# nombre d'événements violents pour 1000 habitants
# variables météo
# variables population
# ndvi moyens de l'année n et n-1
# variables économiques annuelles World Bank
# altitude
# qualité des sols
# cours d'eau
# densités de population par pixel de 100m


def get_country():
    parser=argparse.ArgumentParser()
    parser.add_argument("-country", help="Please select the country \n" +
                        "python3 main.py -country=burkinafaso")
    args=parser.parse_args()
    
    if args.country is not None:
        
        log(args.country, f"\n\nCountry Selected: {args.country}", Logs.INFO)
        return args.country.lower()
    else:
        log(args.country, "Run the following command e.g.," +
                        "'python3 main.py -country=Burkina Faso'", Logs.INFO)
      
country = get_country()
'''
reps = conf.OUTPUT_VARIABLES[country]
for rep in reps:
    preprocess(rep, 1, country)
'''
if country:
    for r_split in [1]:  # [1, 2, 3, 4, 5]
        for rep in ['sca']:  # ['sda', 'sca']
            # print(rep, " / ", r_split)
            # preprocessing des variables
            preprocess(rep, r_split, country)
            # création des features avec 2 réseaux de neurones
            timeseries_lstm(rep, r_split, country) # Timeseries modeling using RNN (LSTM)
            #cnn(rep, r_split, country)  # CNN sur les pixels de densités de population et occupation du sol (cultures, forêts, constructions)
            # Random forest sur les variables initiales et sur les features
            #ml(rep, r_split, country)
            # Régression ridge sur les réponses des 3 modèles
            # ml_late(rep, r_split)
