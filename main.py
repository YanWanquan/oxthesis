# --- --- ---
# main.py
# Sven Giegerich / 03.05.2021
# --- --- ---

import argparse
import torch
from torch.utils.data import DataLoader

import libs.utils
from libs.data_loader import BaseDataLoader, DataTypes
from libs.futures_dataset import FuturesDataset
from libs.models.tsmom import *
from libs.models.transformer import *
from libs.losses import LossTypes, LossHelper
from train import *
from evaluate import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {
    'long': LongOnlyStrategy,
    'tsmom': BasicMomentumStrategy,
    'transformer': TransformerEncoder
}

parser = argparse.ArgumentParser(description='Time Series Momentum with Attention-based Models')

def main():
    arg_model = 'transformer'
    # arguments & hyperparameter ----
    filename = "futures_prop.csv"
    index_col = 0
    start_date = "01/01/1990"
    end_date = "01/01/2000"
    test_date = "01/01/1995"
    lead_target = 1
    # ---
    scaler = 'minmax'
    win_size = 60
    tau = 1
    step = 10
    # ---
    epochs = 1
    lr = 0.005
    # ---
    loss_type = LossTypes.MSE
    # ---
    train_batch_size = 32
    val_batch_size = 128
    test_batch_size = 128

    # (1) load data ----
    base_loader = BaseDataLoader(
        filename=filename, index_col=0, start_date=start_date, end_date=end_date, test_date=test_date, lead_target=lead_target)
    
    if arg_model in ['tsmom', 'long']:
        train_dataloader = base_loader.df[DataTypes.TRAIN]
        val_dataloader = base_loader.df[DataTypes.VAL]
        test_dataloader = base_loader.df[DataTypes.TEST]
    else:
        dataset_train = FuturesDataset(
            base_loader, DataTypes.TRAIN, win_size=win_size, tau=tau, step=step, scaler=scaler)
        dataset_val = FuturesDataset(
            base_loader, DataTypes.VAL, win_size=win_size, tau=tau, step=step, scaler=scaler)
        dataset_test = FuturesDataset(
            base_loader, DataTypes.TEST, win_size=win_size, tau=tau, step=step, scaler=scaler)
        train_dataloader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset_val, batch_size=val_batch_size, shuffle=False)
        test_dataloader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False)

    # (2) setup model ----
    print("> Build model")

    if arg_model in ['tsmom', 'long']:
        model = models[arg_model]()
    elif arg_model == 'transformer':
        d_model = 256
        d_input = len(dataset_train.INP_COLS)
        d_output = 1
        n_head = 8
        n_layer = 1
        n_hidden = d_model * 2
        dropout = 0.1
        len_input_window = win_size
        len_output_window = win_size
        model = models[arg_model](
            d_model, d_input, d_output, n_head, n_layer, n_hidden, dropout, device, len_input_window, len_output_window
        )
    else: 
        raise NotImplementedError("To be done!")

    # (3) train & validate ----
    train_details = {
        'epochs': epochs,
        'lr': lr,
        'loss_type': loss_type,
        'loss_label': LossHelper.get_name(loss_type),
        'loss_fn': LossHelper.get_loss_function(loss_type)
    }
    train(model, train_data=train_dataloader, val_data=val_dataloader, train_details=train_details)

    # (4) evaluate ----
    eval_info = {
        'time_start': start_date,
        'time_end': end_date
    }
    evaluate(model, test_iter=test_dataloader, base_df=base_loader.df[DataTypes.TEST], data_info=eval_info, train_details=train_details)

# --- --- ---

if __name__ == "__main__":  
    main()


