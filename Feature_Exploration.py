#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:22:47 2021

@author: aott
"""

import os
os.getcwd()
#feature engineering: https://alphascientist.com/feature_engineering.html
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import gc

STONK_DIRECTORY = '/home/aott/Documents/Stonks'

stonk_dict = {}
for dirname, _, filenames in os.walk(os.path.join(STONK_DIRECTORY, 'stonk_folder')):
    for filename in filenames:
        stonk_dict[filename] = os.path.join(dirname, filename)
        
with open(os.path.join(STONK_DIRECTORY, 'stonk_info', 'STONK_RECORDS.pkl'), 'rb') as f: #../input/nasdaq-huge-v2/stonk_info/STONK_RECORD.pkl
    STONK_RECORDS = pickle.load(f)
    

from multiprocessing import Pool
from time import time
start = time()
key_dir = [[key, stonk_dict[key]] for key in stonk_dict.keys()]
def data_manip(df):
    pass

def load(data):
    key, dir_ = data
    df = pd.read_feather(dir_)
    df['ticker'] = key
    data_manip(df)
    return df

p = Pool(os.cpu_count())
df = p.map(load, key_dir)
df = pd.concat(df)
print(f'Took {time() - start :.2f}')


import stonk_helper
df = stonk_helper.reduce_mem_usage(df,obj_to_cat=True)
df.sort_values(['ticker', 'day'], ascending='True', inplace=True)
df.info(memory_usage='deep')

mini = df[df.ticker.isin(['JFIN','BBH'])].copy()
mini['ticker'] = mini.ticker.cat.remove_unused_categories()
mini = mini.groupby('ticker').head(5)
mini

indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=3)
mini['2week_high']=mini.groupby('ticker').high.rolling(window=indexer).max().values
#mini['2week_high']=mini.groupby('ticker')['2week_high'].shift(-1).values
mini