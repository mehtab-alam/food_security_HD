#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 22:42:35 2024

@author: syed
"""

import os
import re
import pandas as pd

# Specify the directory path
directory_path = 'data/rawanda/data_explicatives/'
#directory_path = 'data/rawanda/data_explicatives/'

# Define your regular expression pattern to capture the relevant part of the file name
#pattern = r'^rw_(.*?)(?:_data|_price)\.csv$'



pattern = r'^(?:tz_|rw_)(.*?)(?:_data|_price)\.csv$'

# Define the replacement pattern for the new file name
replacement = r'data_\1.xlsx'

# Get a list of files in the directory
file_names = os.listdir(directory_path)

# Loop through the files, rename them, and convert them to Excel
for file_name in file_names:
    if file_name.endswith('.csv'):
        new_name = re.sub(pattern, replacement, file_name)
        if new_name != file_name:
            old_file = os.path.join(directory_path, file_name)
            new_file = os.path.join(directory_path, new_name)
            
            # Read the CSV file
            df = pd.read_csv(old_file)
            
            # Write to Excel file
            df.to_excel(new_file, index=False)
            
            # Optionally, remove the old CSV file
            os.remove(old_file)
            
            print(f'Converted and renamed: {file_name} -> {new_name}')
        else:
            print(f'No change: {file_name}')