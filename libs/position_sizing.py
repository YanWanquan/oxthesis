# --- --- ---
# position_sizing.py
# Sven Giegerich / 14.05.2021
# --- --- ---

import numpy as np
from libs.losses import LossTypes

class PositionSizingHelper():

    @staticmethod
    def sign_sizing_fn(signals, loss_type=None):
        if loss_type == LossTypes.MSE:
            # capped & floored @ 2 stds
            original_shape = signals.shape
            positions = np.reshape(np.array([min(max(x, -2), 2) / 2 for x in signals.flatten()]),
                                   original_shape) 
        else: 
            positions = signals.apply(np.sign)

        return positions
