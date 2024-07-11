#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 00:52:15 2024

@author: syed
"""



import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
import configuration as conf
import re

def save_classification_map(country, algorithm, rep, year):
    # Load the shapefile
    shapefile_path = os.path.join(
        conf.DATA_DIRECTORY, country, conf.SHAPE_FILE[country])
    gdf = gpd.read_file(shapefile_path)
    
    spatial_granularity = conf.FINE_SP_GRANULARITY[country]
    #attribute = 'attribute'
    #gdf[attribute] = pd.cut(gdf[attribute], bins=3, labels=['Low', 'Medium', 'High'])
    df = pd.read_excel(os.path.join(conf.OUTPUT_DIR, country, "results", algorithm,  rep + '.xlsx'))
    gdf[spatial_granularity] = gdf[spatial_granularity].str.upper()
    df[spatial_granularity] = df[spatial_granularity].str.upper()
    
    gdf = gdf.merge(df, left_on=spatial_granularity, right_on=spatial_granularity)
    # Define the color mapping
    color_mapping = {1: '#D05C47', 2: '#FEB264', 3: '#AADFAA'}
    
    # Create the subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    country_capital = re.sub(r'_', ' ', country)
    rep_rem_class = re.sub(r'class\_', '', rep)
    title_inital = country_capital.title() + ' '+ rep_rem_class.upper() 
    # Plot the 'prediction' map
    gdf['color'] = gdf['prediction'].map(color_mapping)
    gdf.plot(ax=ax1, color=gdf['color'])
    ax1.set_title( title_inital + r' - $\bf{Predicted}$ ('+str(year)+')')
    ax1.axis('off')
    
    # Plot the 'label' map
    gdf['color'] = gdf['label'].map(color_mapping)
    gdf.plot(ax=ax2, color=gdf['color'])
    ax2.set_title(title_inital + r' - $\bf{Actual}$ ('+str(year)+')')
    ax2.axis('off')
    
    # Create a legend
    custom_labels = {1: 'Poor', 2: 'Borderline', 3: 'Acceptance'}
    patches = [mpatches.Patch(color=color, label=custom_labels[label]) for label, color in color_mapping.items()]
    
    # Add the legend to the first subplot (ax1)
    ax1.legend(handles=patches, loc='upper right')
    
    # Show the plot
    plt.savefig(os.path.join(conf.OUTPUT_DIR, country, "results", algorithm,  rep + '.png'))
    plt.close()