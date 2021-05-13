# --- --- ---
# futures_data_loader.py
# Sven Giegerich / 03.05.2021
# --- --- ---

import numpy as np
import pandas as pd

import utils
from data_loader import BaseDataLoader

# --- --- ---


class FuturesDataLoader(BaseDataLoader):
    """Load the entire future data"""

    def __init__(self):
        self._name = "futures"

        self.raw_df = super().load_raw_df(
            filename='futures_prop.csv',
            index_col=0
        )
        self.idfs = super().get_identifiers(self.raw_df)


# --- --- ---
# Test the loader
# --- --- ---

if __name__ == "__main__":

    futures_loader = FuturesDataLoader()
    print(futures_loader._name)
