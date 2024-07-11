#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 02:43:31 2024

@author: syed
"""
import geopandas as gpd
import rasterio
import configuration as conf
import os
from rasterio.transform import from_origin
from rasterio.features import rasterize
import numpy as np
import matplotlib.pyplot as plt
import pyproj
from shapely.geometry import Point
from pyproj import CRS
import pandas as pd
from logger import log
import warnings
warnings.simplefilter("ignore")




def shape_to_raster(country, df, spatial_granularity, pixel_column, shape):
    log(country, f"SP Granularity: {spatial_granularity}, Pixel Column : {pixel_column}, Shape: {shape} ")
    # Step 1: Load shape file of country for desired spatial granularity    
    shapefile_path = os.path.join(
        conf.DATA_DIRECTORY, country, conf.SHAPE_FILE[country])
    log(country, "Shape File path: "+ str(os.path.join(
        conf.DATA_DIRECTORY, country, conf.SHAPE_FILE[country])))
    
    gdf = gpd.read_file(shapefile_path)
    
    # Check the columns in the CSV and shapefile to ensure they can be merged correctly
    log(country, "Shapefile columns:"+ str(list(gdf.columns)))
    log(country, "CSV columns:"+ str(list(df.columns)))
    
    # Step 2: Convert names to uppercase to ensure case-insensitive merging
    gdf[spatial_granularity] = gdf[spatial_granularity].str.upper()
    df[spatial_granularity] = df[spatial_granularity].str.upper()
   
    
    # Step 3: Merge the shapefile and CSV file based on the lowercase commune names
    merged_gdf = gdf.merge(df, left_on=spatial_granularity, right_on=spatial_granularity)
    
    # Step 4: Assign IDs from the CSV to the shapefile
    # Assuming the ID column in CSV is 'commune_id'
    merged_gdf[pixel_column] = merged_gdf[pixel_column]
    
    # Print the list of commune names and their IDs
    commune_names_ids = merged_gdf[[pixel_column, spatial_granularity]]
    #log(country, f"{pixel_column} and Names:"+ str(commune_names_ids))
    
    
    # centroid = merged_gdf.geometry.centroid.unary_union.centroid
    # utm_zone = int((centroid.x + 180) / 6) + 1
    # utm_crs = CRS.from_epsg(32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone)
    # merged_gdf = merged_gdf.to_crs(utm_crs)
    
    # Step 6: Define the raster properties with precise dimensions
    # Desired output shape
    desired_height, desired_width = shape[0], shape[1]
    
    # Calculate the pixel size based on desired dimensions and bounding box
    minx, miny, maxx, maxy = merged_gdf.total_bounds
    x_resolution = (maxx - minx) / desired_width
    y_resolution = (maxy - miny) / desired_height
    
    # Ensure the width and height match the desired dimensions
    transform = from_origin(minx, maxy, x_resolution, y_resolution)
    
    # Prepare the shapes and their corresponding values (commune IDs)
    shapes = ((geom, value) for geom, value in zip(merged_gdf.geometry, merged_gdf[pixel_column]))
    
    # Rasterize the shapes with the precise dimensions
    raster_array = rasterize(shapes, out_shape=(desired_height, desired_width), transform=transform, fill=0, dtype='float32')
    return raster_array