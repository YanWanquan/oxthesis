# --- --- ---
# run_tsmom.py
# Sven Giegerich / 05.05.2021
# --- --- ---

import pandas as pd
import numpy as np

from libs.data_loader_interface import BaseDataLoader


def run_tsmom():
    data_loader = BaseDataLoader(
        filename="futures_prop.csv", index_col=0, start_date="01/01/1990", end_date="01/01/2021")
    df = data_loader.df


if __name__ == "__main__":
    run_tsmom()
