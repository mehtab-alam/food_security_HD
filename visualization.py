#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 00:52:15 2024

@author: syed
"""


import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
import numpy as np
import configuration as conf
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, recall_score
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import re

def get_upper_granularity_dataframe(country, algorithm, df):
    #df = pd.read_excel(os.path.join(conf.OUTPUT_DIR, country, "results", "lstm", algorithm,  rep + '.xlsx'))
    # Group by 'PROVINCE' and count occurrences of each label and prediction
    
    label = df.groupby(conf.upper_sp_granularity[country])['label'].value_counts().unstack().fillna(0)
    prediction = df.groupby(conf.upper_sp_granularity[country])['prediction'].value_counts().unstack().fillna(0)
    #difference = df.groupby(conf.upper_sp_granularity[country])['difference'].value_counts().unstack()#.fillna(0)
    
    
    # Find the max count for each 'PROVINCE'
    max_label_counts = label.idxmax(axis=1)
    max_prediction_counts = prediction.idxmax(axis=1)
    #difference_counts = difference.idxmax(axis=1)
    granularity = conf.upper_sp_granularity[country]
    # Combine the results
    result = pd.DataFrame({
         granularity : label.index,
        'label': max_label_counts,
        'prediction': max_prediction_counts,
        #'difference': difference_counts
    })
    
    return result

def get_cnn_dataframe(country, algorithm, rep, tt_split):
    df = pd.read_excel(os.path.join(conf.OUTPUT_DIR, country, "results", "cnn", tt_split, algorithm,  rep + '.xlsx'))
    # Group by 'PROVINCE' and count occurrences of each label and prediction
    label = df.groupby(conf.FINE_SP_GRANULARITY[country])['label'].value_counts().unstack().fillna(0)
    prediction = df.groupby(conf.FINE_SP_GRANULARITY[country])['prediction'].value_counts().unstack().fillna(0)
    #difference = df.groupby(conf.FINE_SP_GRANULARITY[country])['difference'].value_counts().unstack().fillna(0)
    upper_granularity = df.groupby(conf.FINE_SP_GRANULARITY[country])[conf.SPATIAL_GRANULARITY[country][-2]].first().reset_index()
    
    # Find the max count for each 'PROVINCE'
    max_label_counts = label.idxmax(axis=1)
    max_prediction_counts = prediction.idxmax(axis=1)
    #difference_counts = difference.idxmax(axis=1)
    granularity = conf.FINE_SP_GRANULARITY[country]
    # Combine the results
    result = pd.DataFrame({
        
         granularity : label.index,
        'label': max_label_counts,
        'prediction': max_prediction_counts,
        #'difference': difference_counts
        
    }).reset_index(drop=True)
    result = result.merge(upper_granularity, on=conf.FINE_SP_GRANULARITY[country], how='inner')

    result = result.reset_index(drop=True)
    return result

def save_classification_map(country, algorithm, tt_split, nn, rep, year):
    # Load the shapefile
    shapefile_path = os.path.join(
        conf.DATA_DIRECTORY, country, conf.SHAPE_FILE[country])
    gdf = gpd.read_file(shapefile_path)
    
    spatial_granularity = conf.FINE_SP_GRANULARITY[country]

    if nn == 'cnn':
        df = get_cnn_dataframe(country, algorithm, rep, tt_split)
    else:
        df = pd.read_excel(os.path.join(conf.OUTPUT_DIR, country, "results", nn, tt_split, algorithm,  rep + '.xlsx'))
    
        
    #print("df columns:", df.shape, list(df.columns))
   
    gdf[spatial_granularity] = gdf[spatial_granularity].str.upper()
    df[spatial_granularity] = df[spatial_granularity].str.upper()
   
   
    gdf = gdf.merge(df, left_on=spatial_granularity, right_on=spatial_granularity)
    # Define the color mapping
    #gdf = gdf.dropna()
    color_mapping = {1: '#D05C47', 2: '#FEB264', 3: '#AADFAA'}
    
    # Create the subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 10))
    
    country_capital = re.sub(r'_', ' ', country)
    rep_rem_class = re.sub(r'class\_', '', rep)
    if tt_split == 'temporal':
        title_inital = country_capital.title() + ' '+ rep_rem_class.upper() + '-' + nn.upper() + ' ('+ str(year)+ ')'
    else:
        title_inital = country_capital.title() + ' '+ rep_rem_class.upper() + '-' + nn.upper()
    # Plot the 'prediction' map
    
    gdf['difference'] = np.where(gdf['label'] != gdf['prediction'], gdf['label'], np.nan)
    
    gdf['color'] = gdf['prediction'].map(color_mapping)
   
    
    
    gdf.plot(ax=ax1, color=gdf['color'], edgecolor = 'black')
    ax1.set_title( title_inital + r' - $\bf{Predicted}$') #('+str(year)+')')
    ax1.axis('off')
    
    # Plot the 'label' map
    gdf['color'] = gdf['label'].map(color_mapping)
    gdf.plot(ax=ax2, color=gdf['color'], edgecolor = 'black')
    ax2.set_title(title_inital + r' - $\bf{Actual}$ ') #('+str(year)+')')
    ax2.axis('off')
    
      # Plot the 'difference' map
    gdf['color'] = gdf['difference'].map(color_mapping)
    gdf['color'].fillna('white', inplace=True)
    gdf.to_excel(os.path.join(conf.OUTPUT_DIR, country, "results", nn, tt_split, algorithm,  'gdf_'+rep + '.xlsx'))
    gdf.plot(ax=ax3, color=gdf['color'], edgecolor = 'black')
    ax3.set_title( r'$\bf{Geographic\ Distribution\ of\ Prediction\ Errors}$ ') #('+str(year)+')')
    ax3.axis('off')
    
    # Create a legend
    custom_labels = {1: 'Poor', 2: 'Borderline', 3: 'Acceptance'}
    patches = [mpatches.Patch(color=color, label=custom_labels[label]) for label, color in color_mapping.items()]
    
    # Add the legend to the first subplot (ax1)
    ax1.legend(handles=patches, loc='upper right')
    ax2.legend(handles=patches, loc='upper right')
    ax3.legend(handles=patches, loc='upper right')
    # Show the plot
    plt.savefig(os.path.join(conf.OUTPUT_DIR, country, "results", nn, tt_split, algorithm,  rep + '.png'))
    plt.close()


def save_region_classification_map(country, algorithm, tt_split, nn, rep, year):
    # Load the shapefile
    shapefile_path = os.path.join(
        conf.DATA_DIRECTORY, country, conf.SHAPE_FILE[country])
    gdf = gpd.read_file(shapefile_path)
    
    spatial_granularity = conf.upper_sp_granularity[country]
    province_groups = gdf.groupby(spatial_granularity)
    
    if nn == 'cnn':
        df = get_cnn_dataframe(country, algorithm, rep, tt_split)  
    else:
        df = pd.read_excel(os.path.join(conf.OUTPUT_DIR, country, "results", nn, tt_split, algorithm,  rep + '.xlsx'))
    
    df = get_upper_granularity_dataframe(country, algorithm, df)
    df.reset_index(drop=True, inplace=True)
    #print(spatial_granularity, list(df.columns)) 
    gdf[spatial_granularity] = gdf[spatial_granularity].str.upper()
    df[spatial_granularity] = df[spatial_granularity].str.upper()
    
    gdf = gdf.merge(df, left_on=spatial_granularity, right_on=spatial_granularity)
    
    
    # Define the color mapping
    color_mapping = {1: '#D05C47', 2: '#FEB264', 3: '#AADFAA'}
    
    # Create the subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    
    country_capital = re.sub(r'_', ' ', country)
    rep_rem_class = re.sub(r'class\_', '', rep)
    if tt_split == 'temporal':
        title_inital = country_capital.title() + ' '+ rep_rem_class.upper() + '-' + nn.upper() + ' ('+ str(year)+ ')'
    else:
        title_inital = country_capital.title() + ' '+ rep_rem_class.upper() + '-' + nn.upper()
    # Plot the 'prediction' map
    gdf['difference'] = np.where(gdf['label'] != gdf['prediction'], gdf['label'], np.nan)
    
    gdf['color'] = gdf['prediction'].map(color_mapping)
    gdf.plot(ax=ax1, color=gdf['color'])
    ax1.set_title( title_inital + r' - $\bf{Predicted}$ ') #('+str(year)+')')
    ax1.axis('off')
    
    for province, group in province_groups:
        # Plot districts of the current province with solid fill
        group.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=0.5)
        province_boundary = group.unary_union.boundary
        if province_boundary.geom_type == 'MultiLineString':
            for geom in province_boundary.geoms:
                x, y = geom.xy
                ax1.plot(x, y, color='black', linewidth=2)
        elif province_boundary.geom_type == 'LineString':
            x, y = province_boundary.xy
            ax1.plot(x, y, color='black', linewidth=2)
    
    # Plot the 'label' map
    gdf['color'] = gdf['label'].map(color_mapping)
    gdf.plot(ax=ax2, color=gdf['color'])
    ax2.set_title(title_inital + r' - $\bf{Actual}$ ') #('+str(year)+')')
    ax2.axis('off')
    
    for province, group in province_groups:
      # Plot districts of the current province with solid fill
      group.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=0.5)
      province_boundary = group.unary_union.boundary
      if province_boundary.geom_type == 'MultiLineString':
          for geom in province_boundary.geoms:
              x, y = geom.xy
              ax2.plot(x, y, color='black', linewidth=2)
      elif province_boundary.geom_type == 'LineString':
          x, y = province_boundary.xy
          ax2.plot(x, y, color='black', linewidth=2)
    
       # Plot the 'difference' map
    gdf['color'] = gdf['difference'].map(color_mapping)
    gdf['color'].fillna('white', inplace=True)
    gdf.to_excel(os.path.join(conf.OUTPUT_DIR, country, "results", nn, tt_split, algorithm,  'gdf_'+rep + '.xlsx'))
    gdf.plot(ax=ax3, color=gdf['color'], edgecolor = 'black')
    ax3.set_title( r'$\bf{Geographic\ Distribution\ of\ Prediction\ Errors}$ ') #('+str(year)+')')
    ax3.axis('off')
    
    for province, group in province_groups:
       # Plot districts of the current province with solid fill
       group.plot(ax=ax3, facecolor='none', edgecolor='black', linewidth=0.5)
       province_boundary = group.unary_union.boundary
       if province_boundary.geom_type == 'MultiLineString':
           for geom in province_boundary.geoms:
               x, y = geom.xy
               ax3.plot(x, y, color='black', linewidth=2)
       elif province_boundary.geom_type == 'LineString':
           x, y = province_boundary.xy
           ax3.plot(x, y, color='black', linewidth=2)
    # Create a legend
    custom_labels = {1: 'Poor', 2: 'Borderline', 3: 'Acceptance'}
    patches = [mpatches.Patch(color=color, label=custom_labels[label]) for label, color in color_mapping.items()]
    
    # Add the legend to the first subplot (ax1)
    ax1.legend(handles=patches, loc='upper right')
    ax2.legend(handles=patches, loc='upper right')
    ax3.legend(handles=patches, loc='upper right')
    
    # Show the plot
    plt.savefig(os.path.join(conf.OUTPUT_DIR, country, "results", nn, tt_split, algorithm, spatial_granularity + "_" + rep + '.png'))
    plt.close()
    

    
############################################################################################################################
#             PLOT AND SAVE CONFUSION MATRIX
############################################################################################################################


def plot_confusion_matrix(country, algorithm, tt_split, nn, rep, year): #(y_true, y_pred, country, rep, arch, r_split):
    if nn == 'cnn':
        df = get_cnn_dataframe(country, algorithm, rep, tt_split)
       
    else:
         df = pd.read_excel(os.path.join(conf.OUTPUT_DIR, country, "results", nn, tt_split, algorithm,  rep + '.xlsx'))
    #df = df.dropna()
    y_true, y_pred = df['label'], df['prediction']
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Compute metrics
    accuracy = round(accuracy_score(y_true, y_pred), 3)
    precision = round(precision_score(y_true, y_pred, average='weighted'), 3)
    recall = round(recall_score(y_true, y_pred, average='weighted'), 3)
    f1 = round(f1_score(y_true, y_pred, average='weighted'), 3)
    specificity = round(np.sum(cm[1:, 1:]) / (np.sum(cm[1:, 1:]) + np.sum(cm[1:, 0])), 3)
    
    # Create a figure with a grid of 1 row and 2 columns, for the matrix and the metrics table
    fig, ax = plt.subplots(1, 2, figsize=(8,5), gridspec_kw={'width_ratios': [4, 1]})
    
    
    country_capital = re.sub(r'_', ' ', country)
    rep_rem_class = re.sub(r'class\_', '', rep)
    if tt_split == 'temporal':
        title_inital = country_capital.title() + ' '+ rep_rem_class.upper() + '-' + nn.upper() + ' ('+ str(year)+ ')'
    else:
        title_inital = country_capital.title() + ' '+ rep_rem_class.upper() + '-' + nn.upper()
    # Plot heatmapclear
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Poor', 'Borderline', 'Acceptance'], 
                yticklabels=['Poor', 'Borderline', 'Acceptance'], ax=ax[0])
    
    # Add labels and title
    ax[0].set_xlabel('Predicted', fontsize=12, fontweight='bold', fontname='Arial')
    ax[0].set_ylabel('True', fontsize=12, fontweight='bold', fontname='Arial')
    ax[0].set_title(f'Confusion Matrix for '+ title_inital, fontsize=14, fontweight='bold', fontname='Arial')

    # Create metrics DataFrame
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy','F1 Score', 'Precision', 'Recall', 'Specificity'],
        'Score': [accuracy, f1, precision, recall, specificity]
    })

    # Hide axes for the metrics table
    ax[1].axis('off')
    
    # Create the table
    metrics_table = ax[1].table(cellText=metrics_df.values,
                                colLabels=metrics_df.columns,
                                cellLoc='center',
                                loc='center',
                                colColours=['#1f77b4', '#1f77b4'])

    metrics_table.auto_set_font_size(False)
    metrics_table.set_fontsize(12)
    metrics_table.scale(1, 2)

    # Styling the table
    metrics_table.auto_set_column_width([0, 1])
    for (i, j), cell in metrics_table.get_celld().items():
        if (i == 0):
            cell.set_text_props(fontproperties={'weight': 'bold', 'size': 12})
            cell.set_fontsize(14)
            cell.set_facecolor('#1f77b4')
            cell.set_text_props(color='white')
        else:
            cell.set_fontsize(12)
            cell.set_edgecolor('black')

    # Add a vertical line for scaling
    ax[0].axvline(x=3, color='black', linewidth=1)

    # Add Metrics heading
    #plt.figtext(0.65, 0.9, 'Metrics', fontsize=14, fontweight='bold', fontname='Arial', ha='center')

    # Ensure the output directory exists
    output_path = os.path.join(conf.OUTPUT_DIR, country, "results", nn, tt_split, algorithm, rep+ "_confusion_matrix.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    #print(f"Confusion matrix saved to {output_path}")



############################################################################################################################
#             PLOT AND SAVE ROC-AUC CURVE                                                                                  #
############################################################################################################################
def plot_roc_auc(country, algorithm, tt_split, nn, rep, year, y_prob):#(y_true, y_prob, country, rep, arch, r_split):
    # Extract the actual variable name from rep (e.g., 'class_fcs' -> 'fcs')
    #rep_name = rep.split('_', 1)[1]
    #df = pd.read_excel(os.path.join(conf.OUTPUT_DIR, country, "results", "lstm", algorithm,  rep + '.xlsx'))
    
    if nn == 'cnn':
        df = get_cnn_dataframe(country, algorithm, rep, tt_split)
    else:
        df = pd.read_excel(os.path.join(conf.OUTPUT_DIR, country, "results", nn, tt_split, algorithm,  rep + '.xlsx'))
    
    y_true, y_pred = df['label'], df['prediction']
    num_classes = 3
    class_labels = {1: 'Poor', 2: 'Borderline', 3: 'Acceptance'}
    colors = {1: '#D05C47', 2: '#FEB264', 3: '#AADFAA'}
    
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=[1, 2, 3])
    
    # Compute ROC curve and ROC area for each class
    fprs, tprs, roc_aucs = [], [], []
    for y_true_col, y_prob_col in zip(y_true_bin.T, y_prob.T):
        fpr, tpr, _ = roc_curve(y_true_col, y_prob_col)
        roc_auc = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)
    
    # Compute micro-average ROC curve and ROC area
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate(fprs))
    
    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in zip(fprs, tprs):
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    
    fpr_macro, tpr_macro = all_fpr, mean_tpr
    roc_auc_macro = auc(fpr_macro, tpr_macro)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    country_capital = re.sub(r'_', ' ', country)
    rep_rem_class = re.sub(r'class\_', '', rep)
    if tt_split == 'temporal':
        title_initial = country_capital.title() + ' '+ rep_rem_class.upper() + '-' + nn.upper() + ' ('+ str(year)+ ')'
    else:
        title_initial = country_capital.title() + ' '+ rep_rem_class.upper() + '-' + nn.upper()
    # Plot all ROC curves
    for fpr, tpr, roc_auc, color, label in zip(fprs, tprs, roc_aucs, colors, class_labels):
        plt.plot(fpr, tpr, color=colors[color], lw=3,
                 label=f'{class_labels[label]} (area = {roc_auc:.2f})')
    
    plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', linewidth=4,
             label=f'micro-average ROC curve (area = {roc_auc_micro:.2f})')
    
    plt.plot(fpr_macro, tpr_macro, color='navy', linestyle=':', linewidth=4,
             label=f'macro-average ROC curve (area = {roc_auc_macro:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold', fontname='Arial')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold', fontname='Arial')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {title_initial}', fontsize=16, fontweight='bold', fontname='Arial')
    plt.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True, framealpha=0.7, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
  
    # Ensuring the output directory exists
    output_path = os.path.join(conf.OUTPUT_DIR, country, "results", nn, tt_split, algorithm, rep + "_roc_auc_curve_.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Saving the plot
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    #print(f"ROC-AUC curve saved to {output_path}")


def plot_regression_results(country, algorithm, tt_split, nn, rep, year):
    # Load the data
    df = pd.read_excel(os.path.join(conf.OUTPUT_DIR, country, "results", nn, tt_split, algorithm, rep + '.xlsx'))
    #df = df.dropna()
    y_true, y_pred = df['label'], df['prediction']
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Calculate R² and MSE
    r2 = round(r2_score(y_true, y_pred), 3)
    mse = round(mean_squared_error(y_true, y_pred), 3)
    
    # Plotting the actual and predicted values
    plt.scatter(y_true, y_pred, alpha=0.7, color='darkorange', edgecolor='k', label='Predicted vs Actual', s=100)
    
    # Plotting the ideal line
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label='Ideal Line')
    
    # Customizing the axis labels based on the variable provided
    rep_label = rep.upper()
    plt.xlabel(f'Actual {rep_label}', fontsize=14, fontweight='bold', fontname='Arial')
    plt.ylabel(f'Predicted {rep_label}', fontsize=14, fontweight='bold', fontname='Arial')
    
    # Formatting the title
    country_capital = re.sub(r'_', ' ', country)
    rep_rem_class = re.sub(r'class\_', '', rep)
    if tt_split == 'temporal':
        title_initial = country_capital.title() + ' '+ rep_rem_class.upper() + '-' + nn.upper() + ' ('+ str(year)+ ')'
    else:
        title_initial = country_capital.title() + ' '+ rep_rem_class.upper() + '-' + nn.upper()
    
    # Adding title and legend
    plt.title(f'Actual vs Predicted {title_initial}', fontsize=16, fontweight='bold', fontname='Arial')
    plt.legend(fontsize=12, loc='lower right', frameon=True, fancybox=True, framealpha=0.7, shadow=True, borderpad=1)
    
    # Adding grid
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)

    # Adding R² and MSE inside the graph
    plt.text(0.05, 0.95, f'R² :  {r2}\nMSE :  {mse}', transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', fontname='Arial', verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

    # Ensuring the output directory exists
    output_path = os.path.join(conf.OUTPUT_DIR, country, "results", nn, tt_split, algorithm, rep + "_regression_plot.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Saving the plot
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    #print(f"Regression plot saved to {output_path}")
    
