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
from train import *
from evaluate import *

models = {
    'long': LongOnlyStrategy,
    'tsmom': BasicMomentumStrategy
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Time Series Momentum with Attention-based Models')

def main():
    arg_model = 'tsmom'
    # arguments & hyperparameter ----
    filename = "futures_prop.csv"
    index_col = 0
    start_date = "01/01/1995"
    end_date = "01/01/2015"
    test_date = "01/01/1995"
    lead_target = 1
    # ---
    scaler = 'minmax'
    win_size = 60
    tau = 1
    step = 10
    # ---
    epoch = 20
    lr = 0.005
    # ---
    loss = 'shape'
    # ---
    train_batch_size = 254
    val_batch_size = 64
    test_batch_size = 64

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
    model = models[arg_model]()

    # (3) train & validate ----
    train(model, train_data=train_dataloader, val_data=val_dataloader)

    # (4) evaluate ----
    eval_info = {
        'time_start': start_date,
        'time_end': end_date
    }
    evaluate(model, test_data=test_dataloader, data_info=eval_info)

# --- --- ---

if __name__ == "__main__":  
    main()


