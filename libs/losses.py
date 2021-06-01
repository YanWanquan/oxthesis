# --- --- ---
# losses.py
# Sven Giegerich / 03.05.2021
# --- --- ---

import numpy as np
import torch
from enum import IntEnum


class LossTypes(IntEnum):
    AVG_RETURNS = 1
    SHARPE = 2
    MSE = 3


def calc_loss_avg_returns(pred_position, returns_scaled, freq='d'):

    if freq == 'd':
        scale_factor = 252
    else:
        return ValueError("Other frequencies in the loss function currently not supported.")

    avg_returns = torch.mean(
        calc_captured_return(pred_position, returns_scaled))

    return -avg_returns * scale_factor


def calc_loss_sharpe(pred_position, returns_scaled, freq='d'):
    captured_rts = calc_captured_return(pred_position, returns_scaled)
    avg_rts = torch.mean(captured_rts)
    avg_sqrt_rts = torch.mean(torch.square(captured_rts))
    var = avg_sqrt_rts - torch.square(avg_rts)
    std = torch.sqrt(var + 1e-9)

    if freq == 'd':
        scale_factor = 252
    else:
        return ValueError("Other frequencies in the loss function currently not supported.")

    sharpe = avg_rts / std * np.sqrt(scale_factor)
    return -sharpe


def calc_captured_return(pred_position, returns_scaled):
    return pred_position * returns_scaled


class LossHelper:

    @staticmethod
    def get_valid_losses():
        return [LossTypes.AVG_RETURNS, LossTypes.SHARPE, LossTypes.MSE]

    @staticmethod
    def get_name(loss_type):
        return {LossTypes.AVG_RETURNS: 'avg-returns',
                LossTypes.SHARPE: 'sharpe', LossTypes.MSE: 'mse'}[loss_type]

    @classmethod
    def get_loss_type(cls, loss_name):
        return {cls.get_name(lt): lt for lt in cls.get_valid_losses()}[loss_name]

    @staticmethod
    def get_loss_function(loss_type):
        return {
            LossTypes.AVG_RETURNS: calc_loss_avg_returns,
            LossTypes.SHARPE: calc_loss_sharpe,
            LossTypes.MSE: torch.nn.MSELoss()
        }[loss_type]

    @staticmethod
    def get_prediction_type(loss_type):
        return {
            LossTypes.AVG_RETURNS: 'position',
            LossTypes.SHARPE: 'position',
            LossTypes.MSE: 'trend'
        }[loss_type]

    @staticmethod
    def get_output_activation(loss_type):
        if loss_type == LossTypes.MSE:
            return torch.tanh  # was lambda x: x before
        else:
            return torch.tanh

    @staticmethod
    def use_returns_for_loss(loss_type):
        if loss_type == LossTypes.MSE:
            return False
        else:
            return True
