# --- --- ---
# losses.py
# Sven Giegerich / 03.05.2021
# --- --- ---

import numpy as np
import torch
from enum import IntEnum


def calc_loss_avg_returns(pred_position, return_normalized):
    return -torch.mean(calc_captured_return(pred_position, return_normalized))


def calc_loss_sharpe(pred_position, return_normalized):
    return 1


def calc_captured_return(pred_position, return_normalized):
    return pred_position * return_normalized


class LossTypes(IntEnum):
    AVG_RETURNS = 1
    SHARPE = 2
    MSE = 3


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
            return lambda x: x
        else:
            return torch.tanh
