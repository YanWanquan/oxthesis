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

# includes  '<path>/..'
ROOT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
LIB_FOLDER = os.path.join(ROOT_FOLDER, 'libs')
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')

MODEL_FOLDER = os.path.join(ROOT_FOLDER, 'model')
RESULTS_FOLDER = os.path.join(ROOT_FOLDER, 'results')

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


def calc_strategy_returns(positions, realized_returns, aggregate_by=None):
    """Calculates the returns of a strategy across time and instruments

    Args:
        positions (pd.DataFrame): (dim: T x instruments)
        realized_returns (pd.DataFrame) (dim: same)
        aggregate_by (str)
    """
    str_rts = positions * realized_returns

    if aggregate_by == 'instrument':
        return str_rts.mean(axis=0)
    elif aggregate_by == 'time':
        return str_rts.mean(axis=1)
    else:
        return str_rts


def get_save_path(file_label, model, time_test, file_type="csv", loss_type=None):
    if isinstance(time_test, datetime.datetime):
        time_test = time_test.strftime("%Y-%m-%d")
    elif isinstance(time_test, str):
        time_test = time_test.replace(
            '/', '-').replace(':', '-').replace(' ', '-')

    Path(os.path.join(RESULTS_FOLDER, model)).mkdir(
        parents=True, exist_ok=True)
    file_name = f"{model}/{file_label}"
    if loss_type is not None:
        file_name = file_name + f"_{LossHelper.get_name(loss_type)}"
    file_name = file_name + f"_{time_test}.{file_type}"
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


def load_scaler(filename, start_date, test_date):
    path = get_scaler_path(filename, start_date, test_date)
    return pickle.load(open(path, 'rb'))
