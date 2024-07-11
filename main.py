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


def get_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some arguments.")
    
    # Add arguments
    parser.add_argument('-country', type=str, required=True, help='The name of the country (burkina_faso/rwanda/tanzania)')
    parser.add_argument('-algorithm', type=str, required=True, help='The name of the algorithm(classification/regression)')
    parser.add_argument('-tt_split', type=str, required=False, help='The name of the algorithm(temporal/percentage)')
    
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Access arguments
    country = args.country
    algorithm = args.algorithm
    if args.tt_split is not None:
        tt_split = args.tt_split
    else:
        tt_split = 'percentage'
    # Ensure arguments are not None
    if not country or not algorithm:
        print("Error: Both -country,-algorithm must be provided.")
        parser.print_help()
        sys.exit(1)
    return country, algorithm, tt_split
      
country, algorithm, tt_split = get_arguments()

if country:
    for r_split in [1]:  # [1, 2, 3, 4, 5]
        for rep in conf.OUTPUT_VARIABLES[country][algorithm]:  # ['sda', 'sca']
            # print(rep, " / ", r_split)
            # preprocessing des variables
            preprocess(rep, r_split, country, algorithm, tt_split)
            # création des features avec 2 réseaux de neurones
            timeseries_lstm(rep, algorithm, r_split, country) # Timeseries modeling using RNN (LSTM)
            #cnn(rep, r_split, country)  # CNN sur les pixels de densités de population et occupation du sol (cultures, forêts, constructions)
            # Random forest sur les variables initiales et sur les features
            #ml(rep, r_split, country)
            # Régression ridge sur les réponses des 3 modèles
            # ml_late(rep, r_split)
