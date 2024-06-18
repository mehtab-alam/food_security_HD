#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 23:38:08 2024

@author: syed
"""

import numpy as np
import rasterio

# Define the input LULC file path
lulc_file = 'data/rawanda/spatial/rw_lulc_100m.tif'

# Define the output file paths
crop_file = 'data/rawanda/spatial/crop_mean_100m.tif'
forest_file = 'data/rawanda/spatial/forest_mean_100m.tif'
built_file = 'data/rawanda/spatial/built_mean_100m.tif'

# Define the land cover codes for crops, forest, and built-up areas
# Adjust these codes based on the LULC classification in your dataset
CROP_CODE = 1
FOREST_CODE = 2
BUILT_CODE = 3

# Open the LULC file
with rasterio.open(lulc_file) as src:
    # Read the LULC data
    lulc_data = src.read(1)

    # Create masks for each land cover type
    crop_mask = (lulc_data == CROP_CODE).astype(np.uint8)
    forest_mask = (lulc_data == FOREST_CODE).astype(np.uint8)
    built_mask = (lulc_data == BUILT_CODE).astype(np.uint8)

    # Define the metadata for the output files
    profile = src.profile
    profile.update(dtype=rasterio.uint8, count=1)

    # Save the crop mask
    with rasterio.open(crop_file, 'w', **profile) as dst:
        dst.write(crop_mask, 1)

    # Save the forest mask
    with rasterio.open(forest_file, 'w', **profile) as dst:
        dst.write(forest_mask, 1)

    # Save the built mask
    with rasterio.open(built_file, 'w', **profile) as dst:
        dst.write(built_mask, 1)

print("Conversion completed.")
