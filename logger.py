#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:20:42 2024

@author: syed
"""

#!pip3 install --upgrade pandas pyarrow cloudpickle

import logging
from datetime import datetime
import os
from enum import Enum


IS_LOGGER = True

class Logs(Enum):
    INFO = 1
    DEBUG = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5

def apply_log_config(country, today):
    if os.path.exists(os.path.join("logs", country, today.strftime('%Y-%m-%d'))):
        return                  
    os.makedirs(os.path.join("logs", country), exist_ok=True)
    # Set up logging
    logging.basicConfig(filename=os.path.join("logs", country, today.strftime('%Y-%m-%d')), level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filemode='a')  

def log(country, message, message_type = Logs.INFO):
    # Get today's date
    today = datetime.today()
    print("{0}\t\t{1}".format(today.strftime('%Y-%m-%d %H:%M'), message))
    if not IS_LOGGER:
        return
    apply_log_config(country, today)
    
    if message_type == Logs.INFO:
        logging.info(message)
    if message_type == Logs.DEBUG:
        logging.debug(message)
    if message_type == Logs.WARNING:
        logging.warning(message)
    if message_type == Logs.ERROR:
        logging.error(message)
    if message_type == Logs.CRITICAL:
        logging.critical(message)
   