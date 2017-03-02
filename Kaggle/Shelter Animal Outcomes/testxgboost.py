# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 00:40:56 2017

@author: Teerapat
"""

import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'

os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb