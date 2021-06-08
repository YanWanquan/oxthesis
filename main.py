# --- --- ---
# main.py
# Sven Giegerich / 03.05.2021
# --- --- ---

import argparse
from libs.models.tsmom import *
from libs.models.transformer import TransformerEncoder
from libs.models.mlp import MLP
from libs.models.conv_transformer import ConvTransformerEncoder
from libs.models.lstm import LSTM
from libs.losses import LossTypes, LossHelper
from train import run_training_window

models = {
    'long': LongOnlyStrategy,
    'tsmom': BasicMomentumStrategy,
    'transformer': TransformerEncoder,
    'lstm': LSTM,
    'conv_transformer': ConvTransformerEncoder
    # 'mlp': MLP
}


def get_args():
    parser = argparse.ArgumentParser(
        description='Time Series Momentum with Attention-based Models')

    # choices
    loss_dict = {LossHelper.get_name(loss): loss
                 for loss in LossHelper.get_valid_losses()}

    # main params ----
    parser.add_argument('--arch', type=str, nargs='?', choices=list(
        models.keys()), default="transformer", help="Learning architecture")
    parser.add_argument('--loss_type', type=str, nargs='?', choices=list(
        loss_dict.keys()), default="mse", help="Loss function")

    args = parser.parse_args()
    return args


def main():

    pass

# --- --- ---


if __name__ == "__main__":
    main()
