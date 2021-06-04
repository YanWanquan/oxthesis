# --- --- ---
# build_csv.py
# Sven Giegerich / 18.05.2021
# --- --- ---

"""Read one data frame per instrument and merge them by datetime column"""

from functools import reduce
import glob
import os
import numpy as np
import pandas as pd


ROOT_FOLDER = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "../..")
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
RAW_PATH = os.path.join('/nfs/data/files/DAILY/PINNACLE/CLCDATA')

# get *.csv files
extension = 'CSV'
os.chdir(RAW_PATH)
FILE_PATHS = glob.glob('*_RAD.{}'.format(extension))

cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'open_interest']
index_col = 0
sel_cols = ['close']
sel_inst = ["CC", "DA", "GI", "JO", "KC", "KW", "LB", "NR", "SB", "ZA", "ZC", "ZF", "ZG", "ZH", "ZI", "ZK", "ZL", "ZN", "ZO", "ZP", "ZR", "ZT", "ZU", "ZW",
            "ZZ", "CA", "EN", "ER", "ES", "LX", "MD", "SC", "SP", "XU", "XX", "YM", "DT", "FB", "TY", "UB", "US", "AN", "BN", "CN", "DX", "FN", "JN", "MP", "NK", "SN"]

dfs = []
for i, path in enumerate(FILE_PATHS):
    inst = path.split("_")[0]
    df_i = pd.read_csv(path, index_col=index_col, names=cols)
    df_i.index = pd.to_datetime(df_i.index, format="%m/%d/%Y")
    df_i = df_i[sel_cols]
    df_i.columns = [f"{inst}_" + df_i.columns]
    if len(df_i.columns) == 1:
        df_i_columns = f"{inst}"
    if sel_inst is not None:
        if inst in sel_inst:
            dfs.append(df_i)
        else:
            pass
    else:
        dfs.append(df_i)

df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True,
                                         how='outer'), dfs)

print(df)
print(df.shape)

df.to_csv(os.path.join(DATA_FOLDER, 'data_clc_merged.csv'))
