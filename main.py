# -*- coding: utf-8 -*-

from preprocessor import preprocess

'''
from lstm import lstm
from cnn import cnn
from ML_feature_fusion import ml
from ML_late_fusion import ml_late
'''


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

for r_split in [1]: # [1, 2, 3, 4, 5]
    for rep in ['sda']: # ['sda', 'sca']
        print(rep, " / ", r_split)
        # preprocessing des variables
        preprocess(rep, r_split)

        # création des features avec 2 réseaux de neurones
        #lstm(rep, r_split) # LSTM sur les séries temporelles
        #cnn(rep, r_split) # CNN sur les pixels de densités de population et occupation du sol (cultures, forêts, constructions)
        # Random forest sur les variables initiales et sur les features
        #ml(rep, r_split)
        # Régression ridge sur les réponses des 3 modèles
        #ml_late(rep, r_split)


