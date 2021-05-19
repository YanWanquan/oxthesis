# --- --- ---
# build_csv.py
# Sven Giegerich / 18.05.2021
# --- --- ---

"""Read one data frame per instrument and merge them by datetime column"""

import sys
sys.path.append('../../')

import pandas as pd
import numpy as np
import os
import glob
from functools import reduce

ROOT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
RAW_PATH = os.path.join(DATA_FOLDER, 'raw')

# get *.csv files
extension = 'CSV'
os.chdir(RAW_PATH)
FILE_PATHS = glob.glob('*.{}'.format(extension))

cols = ['time', 'x1', 'x2', 'x3', 'x4', 'x5', 'x7']
sel_cols = ['x1', 'x2']
dfs = []
for i, path in enumerate(FILE_PATHS):
    df_i = pd.read_csv(path, index_col=0, names=cols)
    df_i = df_i[sel_cols]
    df_i.columns = [f"{i}_" + df_i.columns]
    dfs.append(df_i)

df = reduce(lambda  left,right: pd.merge(left, right, left_index=True, right_index=True,
                                            how='outer'), dfs)
print(df)
df.to_csv(os.path.join(DATA_FOLDER, 'data_merged.csv'))
