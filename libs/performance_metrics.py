# --- --- ---
# performance_metrics.py
# Sven Giegerich / 03.05.2021
# --- --- ---

import numpy as np
import pandas as pd


class PerformanceMetrics:

    def __init__(self, trts=None, scale_hrz=252):
        """
        Args
            trts: total returns
            scale_hrz: scaling value for the averaged returns (e.g., 252 for annualized returns from a daily series) 
        """
        self.trts = trts
        self.rts = self.trts / self.trts.shift(1) - 1
        self.scale_hrz = scale_hrz
        self.means = self.rts.mean()
        self.vols = self.rts.std()

    def get_metrics(self):
        return pd.DataFrame({
            'scaled_returns': calc_scaled_returns(),
            'scaled_volatility': calc_scaled_vols(),
            'sharpe': calc_sharpe_ratio()
        })

    def calc_scaled_returns(self):
        return self.mean * self.scale_hrz

    def calc_scaled_vols(self):
        return self.vols * np.sqrt(self.scale_hrz)

    def calc_sharpe_ratio(self):
        return self.mean / self.vols * np.sqrt(self.scale_hrz)