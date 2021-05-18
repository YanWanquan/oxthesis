# --- --- ---
# futures_dataset.py
# Sven Giegerich / 11.05.2021
# --

import sys
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime

import libs.utils as utils
from libs.data_loader import BaseDataLoader, DataTypes

# TBD:
# device


class FuturesDataset(Dataset):

    INP_COLS = ['MACD_8', 'MACD_16', 'MACD_32',
                  'annual_rts', 'semianual_rts',
                  'quarterly_rts', 'monthly_rts',
                  'daily_rts']
    TRG_COLS = ['trg']
    PLOT_COLS = ['rts_scaled']

    def __init__(self, dataloader, dat_type, win_size, tau, step, scaler):
        """
        Args:
            dataload (object): the corresponding dataloader object to load the dataset
            dat_type (object): see the corresponding class
            win_size (int): the size of the time window (input plus label window)
            tau (int): prediction (label) length for each window (input window = win_size - tau)
            step (int): ..
            scaler (str): either None, sklearn's StandardScaler ('scaler'), or MinMax ('minmax')
        """
        self.dat_type = dat_type
        self.win_size = win_size
        self.tau = tau
        self.step = step
        self.scaler = scaler

        data_scal = self.do_scale_df(
           dataloader.df, do_scaler=self.scaler)
        self.data, self.time_index, self.inst_index = self.make_dataset(
            data_scal[self.dat_type], win_size=win_size, step=step, dim=0)

        self.cov_lookup = {label: i for i, label in enumerate(
            utils.order_set(dataloader.df[self.dat_type].columns.get_level_values(1)))}
        self.cov_indexes = [
            i for k, i in self.cov_lookup.items() if k in self.INP_COLS]
        self.trg_indexes = [
            i for k, i in self.cov_lookup.items() if k in self.TRG_COLS]
        self.plot_indexes = [
            i for k, i in self.cov_lookup.items() if k in self.PLOT_COLS]

    def make_dataset(self, df, win_size, step, dim=0, filter_na=True):
        dat_inst_list = []
        time_ids_list = []
        inst_index_lookup = []

        # time stamp ----
        time_lookup_d2i = {d:i for i, d in enumerate(df.index)}

        for instrument, df_inst in df.groupby(level=0, axis=1):
            ten_inst = torch.from_numpy(df_inst.to_numpy())
            df_inst_win = self.slice_torch(
                ten=ten_inst, win_size=win_size, step=step, dim=dim)
            dat_inst_list.append(df_inst_win)

            # attributes lookups (timestamp, instrument) ----
            ## .. time stamp (need to convert to an id to be able to slice it)
            ten_time_ids = torch.IntTensor([time_lookup_d2i[d] for i, d in enumerate(df_inst.index)])
            ten_time_ids = ten_time_ids.unsqueeze(-1)
            time_ids_win = self.slice_torch(
                ten=ten_time_ids, win_size=win_size, step=step, dim=0).squeeze(-1)
            time_ids_list.append(time_ids_win)
            ## .. instrument
            inst_index_lookup.extend(
                [instrument for i in range(len(df_inst_win))])

        data = torch.cat(dat_inst_list)
        time_ids = torch.cat(time_ids_list)
        inst_index = pd.Series(inst_index_lookup)

        if filter_na:
            mask = self.mask_na_rows(data)
            data = data[~mask]

            # attributes lookup
            inst_index = inst_index[~mask.numpy()].reset_index()
            time_ids = time_ids[~mask]

        time_index = time_ids

        return data, time_index, inst_index

    def do_scale_df(self, df, do_scaler=None):
        if do_scaler is None:
            print("> No scaling used")
            return df 
        elif do_scaler == 'standard':
            scaler = StandardScaler().fit(df[DataTypes.TRAIN])
        elif do_scaler == 'minmax':
            scaler = MinMaxScaler(feature_range=(
                -1, 1)).fit(df[DataTypes.TRAIN])
        else:
            raise ValueError("Either use the standard or min/max scaler!")

        print(f"> Scaling data with {do_scaler} scaler")
        df_norm = {}
        for data_type, x in df.items():
            df_norm[data_type] = pd.DataFrame(scaler.transform(x), index=df[data_type].index, columns=df[data_type].columns)

        return df_norm

    def slice_torch(self, ten, win_size, step, dim=0):
        # Assumes ten is a tensor (dim: T x (instruments x covariates))
        # new dim: B x T x Covariates
        return ten.unfold(dimension=dim, size=win_size, step=step).permute(0, 2, 1)

    @staticmethod
    def mask_na_rows(ten):
        """Filter all windows that contain at least one NA (in the dimensionality of T & all covariates"""
        mask = torch.any(torch.any(ten.isnan(), dim=-1), dim=-1)
        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  

        # covariates, target var, instrument label
        return {
            'inp': self.data[idx, :, self.cov_indexes],
            'trg': self.data[idx, :, self.trg_indexes],
            'time': self.time_index[idx],
            'inst': self.inst_index.iloc[idx].tolist()[1], # index 0 are the numerical ids
            'plot': self.data[idx, :, self.plot_indexes]
        }

    def plot_example(self, id, model=None, base_df=None):
        """Plots a sample of the dataset"""
        # tbd: replace x-axis by 'time' (can not just add one day as this will confuse with weekends/bank holidays)
        inp_col = self.plot_indexes.index(self.cov_lookup['rts_scaled'])

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 1, 1)
        plt.plot(range(0, self.win_size-self.tau+1), self[id]['plot']
                 [:, inp_col], label="Inputs", marker='.', zorder=-10)
        plt.scatter(range(self.tau, self.win_size+1),
                    self[id]['trg'], label="Targets", marker='.', c='#2ca02c', s=64, edgecolors='k')
        if model is not None:
            with torch.no_grad():
                pred = model(self[id]['inp'].unsqueeze(0))
                pred = pred.squeeze().numpy()
            plt.scatter(range(self.tau, self.win_size+1), pred, marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)
        #plt.title(",".join(self[id]['inst']))
        plt.legend()
        plt.show()

# --- --- ---


if __name__ == "__main__":
    # Testing
    from torch.utils.data import DataLoader

    base_loader = BaseDataLoader(
        filename="futures_prop.csv", index_col=0, start_date="01/01/1990", end_date="01/01/2021", test_date="01/01/2015", lead_target=1)
    dataset_train = FuturesDataset(
        base_loader, DataTypes.TRAIN, win_size=60, tau=1, step=10, scaler='standard')

    train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    x = next(iter(train_dataloader))
    print("Batch example:")
    print("> Shape:")
    print(x['inp'].shape)
    print(x['trg'].shape)
    print(len(x['inst'])) # index 0 contains the corresponding indexes of the dataset
    print("---")
    print("> Values:")
    print(x['inp'][0])
    print(x['trg'][0])
    print(x['inst'][0]) # index 0 contains the corresponding indexes of the dataset

    dataset_train.plot_example(100)

    

