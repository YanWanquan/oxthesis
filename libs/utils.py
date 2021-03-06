# --- --- ---
# utils.py
# Sven Giegerich / 03.05.2021
# --- --- ---

import os
import numpy as np
from numpy.lib import isin
import pandas as pd
import datetime
from pathlib import Path
from libs.losses import LossHelper
import pickle
from pathlib import Path
import torch
import math
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_

# includes  '<path>/..'
ROOT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
LIB_FOLDER = os.path.join(ROOT_FOLDER, 'libs')
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')

MODEL_FOLDER = os.path.join(ROOT_FOLDER, 'model')
RESULTS_FOLDER = os.path.join(ROOT_FOLDER, 'results')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- --- ---


def calc_returns_srs(prs, offset, drop_na=False, verb=False):
    if drop_na:
        prs = prs.dropna()
    elif prs.isna().sum() > 0 and verb:
        print("> calc_returns: There are at least some empty prices!")

    prs = prs.replace({0: np.nan})
    prs = prs.dropna()
    rts = prs / prs.shift(offset) - 1.0
    rts = rts.replace({np.inf: np.nan, -np.inf: np.nan})
    return rts[:]  # first entry is per definition np.nan


def calc_returns_df(df, offset, drop_na=False):
    return df.apply(lambda prs: calc_returns_srs(prs, offset=offset, drop_na=drop_na))


def calc_normalized_returns_srs(prs, offset, vol_lookback, vol_target=1, vol_scaling_factor=252, drop_na=False):
    """
    Calculates the normalized returns based on Lim et al. (2020).
    "It takes the returns (offset) normalized by a measure of daily volatility
    scaled to an appropiate time scale."
    """
    rts = calc_returns_srs(prs, offset=offset, drop_na=drop_na)
    vol = rts.ewm(span=vol_lookback, min_periods=vol_lookback).std().fillna(
        method='bfill')
    vol = vol.shift(1)  # avoid look-ahead bias, ! diff to Lim et al. (2020)
    return (rts * vol_target) / (vol * np.sqrt(vol_scaling_factor))


def calc_normalized_returns_df(df, offset, vol_lookback, vol_target=1, vol_scaling_factor=252, drop_na=False):
    return df.apply(lambda prs: calc_normalized_returns_srs(prs, offset=offset, vol_lookback=vol_lookback, vol_target=vol_target, vol_scaling_factor=vol_scaling_factor, drop_na=drop_na))


def calc_volatility_df(df, ex_ante=True, halflife=None, vol_lookback=None, vol_min_periods=None):
    if halflife is not None:
        ewm = df.ewm(halflife=halflife)
    elif vol_lookback is not None:
        if vol_min_periods is None:
            vol_min_periods = vol_lookback
        ewm = df.ewm(span=vol_lookback, min_periods=vol_min_periods)
    else:
        raise ValueError(
            "No valid parameters provided to calculate rolling window!")

    vol = ewm.std().fillna(method='bfill')
    vol = vol.shift(1) if ex_ante else vol
    return vol


def calc_total_returns_df(df, vol_scaling, is_prices=True, vol_lookback=None, vol_target=None, vol_scaling_factor=252):
    """
    Calculate the total returns of a price series data frame
    Args
        df (pd.DataFrame): T x prices (idealy cleaned)
        vol_scaling (bool): scale the returns according to a ex_ante vol forecast
    """

    if is_prices:
        if vol_scaling:
            if vol_lookback is None or vol_target is None:
                raise ValueError("Invalid parameters!")
            rts = calc_normalized_returns_df(
                df=df, offset=1, vol_lookback=vol_lookback, vol_target=vol_target, vol_scaling_factor=vol_scaling_factor)
        else:
            rts = calc_returns_df(df=df, offset=1)
    else:
        rts = df

    trs = (1 + rts).cumprod()
    return trs


def order_set(seq):
    """Imitates an ordered set (http://www.peterbe.com/plog/uniqifiers-benchmark)"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def calc_strategy_returns(positions, realized_returns, aggregate_by=None, lead=1):
    """Calculates the returns of a strategy across time and instruments

    Args:
        positions (pd.DataFrame): (dim: T x instruments)
        realized_returns (pd.DataFrame): (dim: same)
        aggregate_by (str)
    """
    str_rts = positions * realized_returns.shift(-lead)

    if aggregate_by == 'instrument':
        return str_rts.mean(axis=0, skipna=True)
    elif aggregate_by == 'time':
        return str_rts.mean(axis=1, skipna=True)
    else:
        return str_rts


def get_save_path(file_label, model, time_test, setting=None, file_type="csv"):
    if isinstance(time_test, datetime.datetime):
        time_test = time_test.strftime("%Y-%m-%d")
    elif isinstance(time_test, str):
        time_test = time_test.replace(
            '/', '-').replace(':', '-').replace(' ', '-')

    Path(os.path.join(RESULTS_FOLDER, model)).mkdir(
        parents=True, exist_ok=True)
    file_name = f"{model}/{file_label}"
    if setting is not None:
        file_name = file_name + f"_{setting}"
    file_name = file_name + f"_||_{time_test}.{file_type}"
    return os.path.join(RESULTS_FOLDER, file_name)


def get_scaler_path(filename, start_date, test_date):
    os.makedirs(os.path.join(RESULTS_FOLDER, 'scaler'), exist_ok=True)
    filename = filename.replace('.csv', '')
    path = f"scaler_{filename}_st-{pd.to_datetime(start_date).year}_te-{pd.to_datetime(test_date).year}" + ".scl"
    return os.path.join(RESULTS_FOLDER, 'scaler', path)


def save_scaler(scaler, filename, start_date, test_date):
    """
    Args:
        scaler (sklearn.object)
    """
    if scaler is not None:
        path = get_scaler_path(filename, start_date, test_date)
        pickle.dump(scaler, open(path, 'wb'))
        return path
    else:
        return False


# Currently not used
def load_scaler(scaler_path):
    if isinstance(scaler_path, str):
        return pickle.load(open(scaler_path, 'rb'))
    else:
        print("> No scaler loaded")
        return "none"


def inverse_scale_tensor(df, scaler_dict):
    """
    Args:
        labels (pytorch.tensor): (dim: B x T)
    """
    def inverse_scale_per_feature(series):
        array = series.to_numpy().reshape(-1, 1)
        x = scaler_dict['trg'].inverse_transform(array)
        return pd.Series(x.flatten(), index=series.index)

    df_inv = df.apply(inverse_scale_per_feature)
    return df_inv


def load_model(path):
    train_dict = pickle.load(open(path, 'rb'))
    arch = train_dict['arch']
    model = train_dict['model'].double()
    train_manager = train_dict['train_manager']

    print(
        f"> Load model with arch {arch} and loss type {train_manager['args']['loss_type']}")
    print(f"> Setting: {train_manager['setting']}")
    return (arch, model, train_manager)

# --- --- ---
# --- --- ---

# See: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.do_save_model = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.do_save_model = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.do_save_model = False
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.do_save_model = True
            self.counter = 0

    def save_checkpoint(self, val_loss, dict):
        '''Saves model when validation loss decrease.'''
        # SVEN
        # save dict via pickle instead of a model
        # only if path is given else the logx is used
        if self.path is not None:
            if self.verbose:
                self.trace_func(
                    f'>> Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            pickle.dump(dict, open(self.path, 'wb'))
            self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, lr):
    lr_adjust = {epoch: lr * (0.9 ** ((epoch-1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'> updating learning rate to {lr}')

# --- --- ---
# --- --- ---

# See https://github.com/aparo/pyes/issues/183


class DotDict(dict):
    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)

    __setattr__ = dict.__setitem__

    __delattr__ = dict.__delitem__

# --- --- ---
# --- --- ---


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}


class BERTAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay_rate=0.01,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError(
                "Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError(
                "Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError(
                "Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError(
                "Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError(
                "Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(BERTAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(
                        state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

        return loss
