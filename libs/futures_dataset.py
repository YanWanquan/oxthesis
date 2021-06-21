# --- --- ---
# futures_dataset.py
# Sven Giegerich / 11.05.2021
# --

from torch._C import Value, device
from libs.data_loader import BaseDataLoader, DataTypes
import libs.utils as utils
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import sys
import pickle
from libs.models.informer import time_features as informer_time_features
from libs.losses import LossTypes
sys.path.append('../')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FuturesDataset(Dataset):

    INP_COLS = ['MACD_8', 'MACD_16', 'MACD_32',
                'annual_rts', 'semianual_rts',
                'quarterly_rts', 'monthly_rts',
                'daily_rts']
    TRG_COLS = ['trg']
    RTS_COLS = ['rts_scaled_lead']
    PRS_COLS = ['prs', 'prs_lead']

    def __init__(self, dataloader, dat_type, win_size, tau, step, scaler_type=None, scaler=None, freq="d"):
        """
        Args:
            dataload (object): the corresponding dataloader object to load the dataset
            dat_type (object): see the corresponding class
            win_size (int): the size of the time window (input plus label window)
            tau (int): prediction (label) length for each window (input window = win_size - tau)
            step (int): ..
            scaler_type (str): either None, sklearn's StandardScaler ('scaler'), or MinMax ('minmax')
        """
        self.dat_type = dat_type
        self.win_size = win_size
        self.tau = tau
        self.step = step
        self.scaler_type = scaler_type
        self.scaler = scaler

        # id refers to numeric representation of the actual index/col
        data_scal, self.scaler = self.do_scale_df(
            dataloader.df, scaler_type=self.scaler_type, scaler=scaler)
        self.data, self.time_id, self.inst_index, self.time_embd = self.make_dataset(
            data_scal[self.dat_type], win_size=win_size, step=step, slice_dim=0, freq=freq)

        # lookups
        self.cov_lookup = {label: i for i, label in enumerate(
            utils.order_set(data_scal[self.dat_type].columns.get_level_values(1)))}
        tmp_inst = list(dataloader.df[DataTypes.TRAIN].columns.unique(
            level=0))
        self.inst_lookup = {inst: i for i, inst in enumerate(tmp_inst)}
        self.cov_indexes = [
            i for k, i in self.cov_lookup.items() if k in self.INP_COLS]
        self.trg_indexes = [
            i for k, i in self.cov_lookup.items() if k in self.TRG_COLS]
        self.rts_indexes = [
            i for k, i in self.cov_lookup.items() if k in self.RTS_COLS]
        self.prs_indexes = [
            i for k, i in self.cov_lookup.items() if k in self.PRS_COLS]

    def make_dataset(self, df, win_size, step, slice_dim=0, filter_na=True, freq='d'):
        dat_inst_list = []
        time_ids_list = []
        inst_index_lookup = []

        # time stamp ----
        time_lookup_d2i = {d: i for i, d in enumerate(df.index)}

        for instrument, df_inst in df.groupby(level=0, axis=1):
            ten_inst = torch.from_numpy(df_inst.to_numpy())
            df_inst_win = self.slice_torch(
                ten=ten_inst, win_size=win_size, step=step, dim=slice_dim)
            dat_inst_list.append(df_inst_win)

            # attributes lookups (timestamp, instrument) ----
            # .. time stamp (need to convert to an id to be able to slice it)
            ten_time_ids = torch.IntTensor(
                [time_lookup_d2i[d] for i, d in enumerate(df_inst.index)])
            ten_time_ids = ten_time_ids.unsqueeze(-1)
            time_ids_win = self.slice_torch(
                ten=ten_time_ids, win_size=win_size, step=step, dim=0).squeeze(-1)
            time_ids_list.append(time_ids_win)
            # .. instrument
            inst_index_lookup.extend(
                [instrument for i in range(len(df_inst_win))])

        data = torch.cat(dat_inst_list)
        time_ids = torch.cat(time_ids_list)
        inst_index = pd.Series(inst_index_lookup)

        if filter_na:
            mask = self.mask_na_rows(data)
            data = data[~mask]

            # attributes lookup
            inst_index = inst_index[~mask.cpu().numpy()].reset_index()
            time_ids = time_ids[~mask]

        # calculate time embedding
        time_embedding = np.apply_along_axis(
            self.calc_time_embedding, 0, time_ids.numpy(), time_index=df.index, freq=freq)  # W x T -> W x T x E
        time_embedding = torch.tensor(time_embedding).permute(0, 2, 1)

        return data, time_ids, inst_index, time_embedding

    def do_scale_df(self, df, scaler_type=None, scaler=None):
        if scaler_type is None or scaler_type == 'none':
            print("> No additional scaling used")
            return df, None

        if scaler is not None:
            print("> Use fitted scaler")
            scaler_inp = scaler['inp']
            scaler_trg = scaler['trg']
        else:
            # stack data for scaler
            df_train_stack = df[DataTypes.TRAIN].stack(level=0)

            if scaler_type == 'standard':
                scaler_inp = StandardScaler().fit(
                    df_train_stack[self.INP_COLS])
                scaler_trg = StandardScaler().fit(
                    df_train_stack[self.TRG_COLS])
            elif scaler_type == 'minmax':
                scaler_inp = MinMaxScaler(feature_range=(
                    -1, 1)).fit(df_train_stack[self.INP_COLS])
                scaler_trg = MinMaxScaler(feature_range=(
                    -1, 1)).fit(df_train_stack[self.TRG_COLS])
            else:
                raise ValueError("Either use the standard or min/max scaler!")

        print(f"> Scaling data with {type(scaler_inp)} scaler")
        df_norm = {}
        for data_type, df_i in df.items():  # here unnecessary to scale every type, but useful to keep it general
            # input ----
            df_i_inp_stack = df_i.stack(level=0)[self.INP_COLS]
            df_i_inp_scaled = scaler_inp.transform(df_i_inp_stack)
            df_i_inp_norm = pd.DataFrame(
                df_i_inp_scaled, index=df_i_inp_stack.index, columns=df_i_inp_stack.columns)

            # target ----
            df_i_trg_stack = df_i.stack(level=0)[self.TRG_COLS]
            df_i_trg_scaled = scaler_trg.transform(df_i_trg_stack)
            df_i_trg_norm = pd.DataFrame(
                df_i_trg_scaled, index=df_i_trg_stack.index, columns=df_i_trg_stack.columns)

            # combine input & target columns with unscaled other columns
            df_i_norm = pd.concat([df_i_inp_norm, df_i_trg_norm], axis=1)

            # .. unscaled columns
            df_i_stack = df_i.stack(level=0)
            not_norm_cols = set(df_i_stack.columns) - \
                set(self.INP_COLS + self.TRG_COLS)
            df_i_not_norm = df_i_stack.loc[:, not_norm_cols]
            df_norm[data_type] = pd.concat([df_i_norm, df_i_not_norm], axis=1)
            df_norm[data_type] = df_norm[data_type].unstack(
            ).swaplevel(axis=1)  # unstack

        scaler = {'inp': scaler_inp, 'trg': scaler_trg}
        return df_norm, scaler

    def slice_torch(self, ten, win_size, step, dim=0):
        # Assumes ten is a tensor (dim: T x (instruments x covariates))
        # new dim: B x T x Covariates
        return ten.unfold(dimension=dim, size=win_size, step=step).permute(0, 2, 1)

    @staticmethod
    def calc_time_embedding(time_win, time_index, timeenc=1, freq='d'):
        # Helper function to calculate time embeddings for positional information
        return informer_time_features(pd.DataFrame({'date': time_index[time_win]}), timeenc=timeenc, freq=freq)

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
            'inp': self.data[idx, :, self.cov_indexes],  # B x T x C (tensor)
            'trg': self.data[idx, :, self.trg_indexes],  # B x T x 1 (tensor)
            'time': self.time_id[idx],  # B x T (tensor)
            'time_embd': self.time_embd[idx],  # B x T x E (tensor)
            # additional vars for plotting ----
            'inst': self.inst_index.iloc[idx].tolist()[1],  # len: B (list)
            'inst_id': self.inst_lookup[self.inst_index.iloc[idx].tolist()[1]],
            'rts': self.data[idx, :, self.rts_indexes],
            'prs': self.data[idx, :, self.prs_indexes]  # prs, prs_lead
        }

    def plot_example(self, id, model=None, loss_type=None, scaler="none"):
        """Plots a sample of the dataset"""
        # tbd: replace x-axis by 'time' (can not just add one day as this will confuse with weekends/bank holidays)
        inp_col = self.rts_indexes.index(self.cov_lookup['rts_scaled_lead'])

        if scaler != "none":
            raise ValueError("Scaler not properly implemented yet!")
            trg = self[id]['trg'].cpu().numpy()
            if scaler != "none":
                trg = scaler['trg'].inverse_transform(
                    trg.reshape(-1, 1)).flatten()
        else:
            trg = self[id]['trg']

        fig, ax1 = plt.subplots()

        # observed returns
        ax1.plot(range(0, self.win_size-self.tau+1), self[id]['rts']
                 [:, inp_col], label="Inputs", marker='.', zorder=-10)

        # target
        ax1.scatter(range(self.tau, self.win_size+1),
                    trg, label="Targets", marker='.', c='#2ca02c', s=64, edgecolors='k')

        if model is not None:
            with torch.no_grad():
                inp = self[id]['inp'].unsqueeze(
                    0).to(device)
                emb = self[id]['time_embd'].unsqueeze(0).to(device)

                if model.name == 'informer':
                    pred = model(inp, emb)
                    if len(pred) > 1:
                        pred = pred[0]  # 1 would be attention
                elif model.name == 'transformer':
                    pred = model(inp)
                else:
                    pred = model(inp)

                pred = pred.squeeze().cpu().numpy()

                if scaler != "none":
                    raise ValueError("Scaler option not implemented yet!")

            if loss_type == LossTypes.SHARPE or loss_type == LossTypes.AVG_RETURNS:
                # positions

                ax1.tick_params(axis='y')
                ax2 = ax1.twinx()
                ax2.set_ylabel('pos')

                ax2.scatter(range(self.tau, self.win_size+1), pred, marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

                ax2.tick_params(axis='y')

                fig.tight_layout()
            else:
                # predictions
                plt.scatter(range(self.tau, self.win_size+1), pred, marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

        ax1.set_title(self[id]['inst'])
        ax1.legend()
        plt.show()

    def get_attention(self, model, batch=None, id=None):
        if model.name not in ['transformer', 'conv_transformer']:
            raise ValueError("Model not supported")

        if batch is not None:
            inputs = batch['inp'].double().to(device)
            x_time = batch['time_embd'].double().to(device)
            x_static = batch['inst_id'].to(device)
        elif id is not None:
            inputs = self[id]['inp'].unsqueeze(0).to(device)
            x_time = self[id]['time_embd'].unsqueeze(0).to(device)
            x_static = torch.tensor(
                self[id]['inst_id']).unsqueeze(0).to(device)
        else:
            raise ValueError(
                "Either input the entire bacht or just a single id.")

        if model.name == 'transformer':
            attn = model.get_attention(
                inputs, x_time, x_static)  # B x n_layer x L x L
            # currently just outputs multi-head
            attn = attn.unsqueeze(2)  # -> B x n_layer x H x L x L
        elif model.name == 'conv_transformer':
            # B x n_layer x H x L x L
            _, attn = model(inputs, x_time, x_static)

        # elif model.name == 'informer':
        #    emb = self[id]['time_embd'].unsqueeze(0).to(device)
        #       _, attn = model(inp, emb)

        return attn.detach().cpu().numpy()

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
    # index 0 contains the corresponding indexes of the dataset
    print(len(x['inst']))
    print("---")
    print("> Values:")
    print(x['inp'][0])
    print(x['trg'][0])
    # index 0 contains the corresponding indexes of the dataset
    print(x['inst'][0])

    dataset_train.plot_example(100)
