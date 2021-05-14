# --- --- ---
# position_sizing.py
# Sven Giegerich / 14.05.2021
# --- --- ---

import numpy as np

class PositionSizingHelper():

    @staticmethod
    def sign_sizing_fn(signals):
        return signals.apply(np.sign)
