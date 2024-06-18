#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 00:09:24 2024

@author: syed
"""

import rasterio
import numpy as np

def linear_interpolate_rasters(raster1, raster2, year1, year2, target_years):
    """
    Linearly interpolate rasters between two years.
    
    Parameters:
    raster1 (str): Path to the first raster file (earlier year).
    raster2 (str): Path to the second raster file (later year).
    year1 (int): The year corresponding to the first raster file.
    year2 (int): The year corresponding to the second raster file.
    target_years (list of int): List of target years for interpolation.

    Returns:
    dict: Dictionary with target years as keys and interpolated rasters as values.
    """
    PATH = './data/rawanda/spatial/population_100m/'
    with rasterio.open(raster1) as src1, rasterio.open(raster2) as src2:
        data1 = src1.read(1)
        data2 = src2.read(1)
        profile = src1.profile
        
        interpolated_rasters = {}
        for target_year in target_years:
            fraction = (target_year - year1) / (year2 - year1)
            interpolated_data = data1 + (data2 - data1) * fraction
            interpolated_rasters[target_year] = interpolated_data

            # Write the interpolated raster to a new file
            output_file = f"{PATH}population_{target_year}.tif"
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(interpolated_data, 1)
                
    return interpolated_rasters

PATH = './data/rawanda/spatial/population_100m/'
year_1 = 2020
year_2 = 2025
# Paths to the raster files for the years 2005 and 2010
raster_1 = PATH+ 'rw_'+str(year_1)+'_population_data.tif'
raster_2 = PATH+ 'rw_'+str(year_2)+'_population_data.tif'

# Years for interpolation
#target_years = [2006, 2007, 2008, 2009]
target_years = [2021,2022,2023]

print("Population Rasters Running...")
# Perform interpolation
interpolated_rasters = linear_interpolate_rasters(raster_1, raster_2, year_1, year_2, target_years)
print(interpolated_rasters[2021].shape)

# Output files will be saved as interpolated_2006.tif, interpolated_2007.tif, etc.
